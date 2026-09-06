---
title: Workspaces and File Tools
description: Let an LLM read, write, edit, and search files inside a sandboxed per-job directory
---

Some work does not fit through a prompt and a structured response: reviewing a repository, drafting a long report one section at a time, or editing a document that another worker produced. `WorkspaceLLMTaskWorker` gives the LLM a set of file tools that are jailed to a per-job working directory. The worker returns a small structured result while the large text lives on disk, and many jobs can run concurrently without any of them reaching outside the directory assigned to it.

## Overview

PlanAI exports four pieces that work together:

- **`Workspace`** is the sandbox: a root directory that every path is resolved against.
- **`make_file_tools()`** builds `read_file`, `write_file`, `edit_file`, `list_files`, and `grep_files` tools bound to one workspace.
- **`WorkspaceTask`** carries the workspace path through the graph.
- **`WorkspaceLLMTaskWorker`** is a `CachedLLMTaskWorker` that finds the workspace in the task's provenance, hands the tools to the LLM, and keeps the cache honest about files on disk.

## A Minimal Example

```python
from typing import List, Type
from planai import Graph, Task, WorkspaceLLMTaskWorker, WorkspaceTask, llm_from_config

class ReviewResult(Task):
    summary: str
    issues_found: int

class CodeReviewer(WorkspaceLLMTaskWorker):
    prompt = "Review the code in this workspace and write your findings to review.md."
    llm_input_type: Type[Task] = WorkspaceTask
    output_types: List[Type[Task]] = [ReviewResult]

    def expected_output_files(self, task: WorkspaceTask) -> List[str]:
        # a cache hit whose files are missing (e.g. a fresh checkout) is re-executed
        return ["review.md"]

llm = llm_from_config(provider="anthropic", model_name="claude-sonnet-5")
reviewer = CodeReviewer(llm=llm, max_tool_rounds=40)

graph = Graph(name="Review Workflow")
graph.add_workers(reviewer)
graph.set_entry(reviewer)
graph.set_exit(reviewer)
graph.run(initial_tasks=[(reviewer, WorkspaceTask(workspace="/jobs/job-123"))])
```

The model reads and writes files with the tools during the request, and the structured `ReviewResult` is what flows to downstream workers.

## How the Workspace Is Found

`WorkspaceLLMTaskWorker.get_workspace()` walks the task and then its input provenance, nearest first, and returns the first task with a non-empty string `workspace` attribute. Publish a `WorkspaceTask` (or your own `Task` subclass with a `workspace: str` field) once, and every workspace worker downstream operates on the same directory:

```python
from pathlib import Path
from planai import TaskWorker, WorkspaceTask

class JobSetup(TaskWorker):
    output_types: List[Type[Task]] = [WorkspaceTask]

    def consume_work(self, task: JobRequest):
        job_dir = Path("work") / task.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.md").write_text(task.text)
        self.publish_work(
            WorkspaceTask(workspace=str(job_dir.resolve())), input_task=task
        )
```

Override `get_workspace()` when the directory should come from worker configuration instead of the provenance chain.

## The File Tools

| Tool | What it does |
| --- | --- |
| `read_file(path, offset=0, limit=0)` | Returns the file with 1-based line numbers, like `cat -n`. `offset` and `limit` page through large files; output is truncated at `max_read_chars`. |
| `write_file(path, content)` | Creates or overwrites a file. Parent directories are created as needed. |
| `edit_file(path, old_string, new_string, replace_all=False)` | Replaces an exact snippet. `old_string` must occur exactly once unless `replace_all` is true. |
| `list_files(path=".", pattern="**/*")` | Lists files with their sizes, sorted by relative path. |
| `grep_files(pattern, path=".", glob="**/*.md")` | Searches each line of the matching files with a Python regular expression. |

Every path is workspace-relative. Absolute paths, `..` components, and symlinks that resolve outside the root are rejected. The tools return an `Error: ...` string instead of raising, so the model sees what went wrong and can correct itself. Set `read_only=True` on the worker to omit `write_file` and `edit_file`.

## Caching and Files on Disk

`WorkspaceLLMTaskWorker` extends `CachedLLMTaskWorker`, and files introduce two problems the normal cache key does not cover:

1. **Changed inputs.** The cache key is built from the input task and the prompt, not from the files the model will read. Set `input_globs` to fold the content of the matching files into the key through `hash_files()`.
2. **Missing outputs.** A cache hit replays the published output tasks, but not the files the tools wrote on the earlier run. Override `expected_output_files()` to list the workspace-relative paths the worker must produce. When any of them is missing, the hit is treated as a miss and the worker runs again.

```python
class SectionWriter(WorkspaceLLMTaskWorker):
    prompt = "Read the notes and write the requested section to the given file."
    llm_input_type: Type[Task] = SectionRequest
    output_types: List[Type[Task]] = [SectionDraft]
    input_globs: List[str] = ["notes/*.md"]

    def expected_output_files(self, task: SectionRequest) -> List[str]:
        return [task.output_file]
```

The worker also forwards its cache key to `llm-interface` as `cache_salt`. The library's own response cache is keyed on the initial prompt, so without the salt a second run could receive a stale answer after the files changed.

## Validating What the Model Wrote

Use `extra_validation()` to check the files and not only the structured response. Returning a string sends it back to the model as feedback and the request is retried:

```python
from typing import Optional

class SectionWriter(WorkspaceLLMTaskWorker):
    ...

    def extra_validation(
        self, response: SectionDraft, input_task: SectionRequest
    ) -> Optional[str]:
        path = self.get_workspace(input_task).resolve(input_task.output_file)
        if not path.exists():
            return f"Write the section to {input_task.output_file} with write_file."
        if len(path.read_text()) < 800:
            return "The section is too short; expand it to at least 800 characters."
        return None
```

This keeps the expensive judgement with the model and the cheap, deterministic checks in Python.

## Configuration

| Field | Default | Purpose |
| --- | --- | --- |
| `read_only` | `False` | Expose only `read_file`, `list_files`, and `grep_files`. |
| `max_tool_rounds` | `40` | Maximum tool-calling rounds per request. When the limit is reached the model is asked for its final answer with tools disabled. |
| `input_globs` | `[]` | Glob patterns whose file content is folded into the cache key. |
| `max_read_chars` | `100000` | Truncation limit for `read_file` output. |
| `tools` | `None` | Additional tools, appended after the file tools. |

## Using the Pieces Directly

`Workspace`, `make_file_tools()`, and `hash_files()` are exported from `planai` and work with any `LLMTaskWorker` through the `get_tools()` hook:

```python
from planai import LLMTaskWorker, Workspace, make_file_tools

class Summarizer(LLMTaskWorker):
    prompt = "Summarize every file in the workspace."
    llm_input_type: Type[Task] = JobTask
    output_types: List[Type[Task]] = [Summary]
    max_tool_rounds: int = 20

    def get_tools(self, task: JobTask):
        return make_file_tools(Workspace(task.job_dir), read_only=True)
```

`make_file_tools()` also accepts `max_read_chars`, `max_list_entries`, and `max_grep_matches` to bound the size of tool results. `Workspace.resolve(rel_path)` is the same check the tools use, so Python code can validate a model-supplied path before touching it.

## Testing

The tools are plain functions wrapped as `Tool` objects, so they can be exercised without an LLM:

```python
def test_edit_requires_unique_match(tmp_path):
    (tmp_path / "a.md").write_text("one two one")
    tools = {t.name: t for t in make_file_tools(tmp_path)}

    result = tools["edit_file"].execute(path="a.md", old_string="one", new_string="1")

    assert result.startswith("Error: old_string is not unique")
    assert tools["read_file"].execute(path="a.md") == "     1\tone two one"
```

For the worker itself, point a `WorkspaceTask` at `tmp_path`, drive it with [`InvokeTaskWorker`](/guide/testing/), and assert on both the published task and the files left in the directory.

## Next Steps

- See [LLM Integration](/features/llm-integration/) for the `get_tools()` hook and `max_tool_rounds`
- Read about [Caching](/features/caching/) to understand how file hashes and validity checks extend the cache key
- Review the [TaskWorker API](/api/taskworker/) for the full list of hooks
