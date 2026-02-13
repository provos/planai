---
title: Testing
description: Testing PlanAI workflows with MockLLM, InvokeTaskWorker, and other utilities
---

PlanAI provides a dedicated `planai.testing` module with utilities for testing workers and workflows without making real LLM calls or touching the filesystem.

## Import

```python
from planai.testing import (
    MockLLM,
    MockLLMResponse,
    MockCache,
    InvokeTaskWorker,
    inject_mock_cache,
    add_input_provenance,
    unregister_output_type,
)
```

## InvokeTaskWorker

`InvokeTaskWorker` lets you test a single worker in isolation. It mocks the graph context and captures all published tasks.

### Testing a TaskWorker

```python
from planai import Task, TaskWorker
from planai.testing import InvokeTaskWorker
from typing import List, Type

class InputTask(Task):
    data: str

class OutputTask(Task):
    result: str

class MyWorker(TaskWorker):
    output_types: List[Type[Task]] = [OutputTask]

    def consume_work(self, task: InputTask):
        self.publish_work(OutputTask(result=f"Processed: {task.data}"), task)


def test_my_worker():
    worker = InvokeTaskWorker(MyWorker)
    published = worker.invoke(InputTask(data="hello"))

    worker.assert_published_task_count(1)
    worker.assert_published_task_types([OutputTask])
    assert published[0].result == "Processed: hello"
```

### Testing a JoinedTaskWorker

Use `invoke_joined()` instead of `invoke()` for joined workers:

```python
from planai import JoinedTaskWorker
from planai.testing import InvokeTaskWorker

class AggregatorWorker(JoinedTaskWorker):
    join_type: Type[TaskWorker] = MyWorker
    output_types: List[Type[Task]] = [SummaryTask]

    def consume_work_joined(self, tasks: List[OutputTask]):
        combined = ", ".join(t.result for t in tasks)
        self.publish_work(SummaryTask(summary=combined), tasks[0])


def test_aggregator():
    worker = InvokeTaskWorker(AggregatorWorker)
    inputs = [
        OutputTask(result="first"),
        OutputTask(result="second"),
    ]

    published = worker.invoke_joined(inputs)

    worker.assert_published_task_count(1)
    assert published[0].summary == "first, second"
```

### Passing Constructor Arguments

Pass any keyword arguments the worker expects:

```python
worker = InvokeTaskWorker(ChatTaskWorker, llm=mock_llm)
worker = InvokeTaskWorker(MyConfigurableWorker, threshold=0.5, name="test")
```

## MockLLM

`MockLLM` replaces a real LLM provider. You define patterns that match against the prompt text and return either a Pydantic model or a plain string.

### String Responses

```python
from planai.testing import MockLLM, MockLLMResponse

mock_llm = MockLLM(responses=[
    MockLLMResponse(
        pattern=r"What is the capital of France",
        response_string="The capital of France is Paris.",
    ),
])
```

### Structured (Pydantic) Responses

When your worker expects a structured output, return a Pydantic model:

```python
from planai.testing import MockLLM, MockLLMResponse

class PlanDraft(Task):
    plan: str

mock_llm = MockLLM(responses=[
    MockLLMResponse(
        pattern="Create a detailed plan.*",
        response=PlanDraft(plan="# My Plan\n1. Step one\n2. Step two"),
    ),
])
```

### Multiple Responses

Define multiple patterns — the first matching pattern wins:

```python
mock_llm = MockLLM(responses=[
    MockLLMResponse(
        pattern=r"Hello, how are you\?$",
        response_string="I'm doing well!",
    ),
    MockLLMResponse(
        pattern=r".*Analyze how well.*",
        response=ScoreOutput(score=0.8),
    ),
    MockLLMResponse(
        pattern="Create a refined.*plan.*",
        response=FinalPlan(plan="Refined plan", rationale="Improved"),
    ),
])
```

### Combining MockLLM with InvokeTaskWorker

This is the typical pattern for testing LLM-powered workers:

```python
from planai.testing import MockLLM, MockLLMResponse, InvokeTaskWorker

mock_llm = MockLLM(responses=[
    MockLLMResponse(
        pattern=r"Hello, how are you\?$",
        response_string="I'm doing well, thank you for asking!",
    ),
])

worker = InvokeTaskWorker(ChatTaskWorker, llm=mock_llm)

chat_task = ChatTask(
    messages=[ChatMessage(role="user", content="Hello, how are you?")]
)
published = worker.invoke(chat_task)

assert published[0].content == "I'm doing well, thank you for asking!"
```

## MockCache

`MockCache` is an in-memory replacement for the disk-based cache used by `CachedTaskWorker` and `CachedLLMTaskWorker`. It also tracks access statistics.

### Basic Usage

```python
from planai.testing import MockCache

cache = MockCache()
cache.set("key1", "value1")
assert cache.get("key1") == "value1"
assert cache.get("missing") is None
```

### Disabling Storage

Use `dont_store=True` to simulate a cache that always misses — useful for forcing workers to recompute every time:

```python
cache = MockCache(dont_store=True)
cache.set("key1", "value1")
assert cache.get("key1") is None  # always returns None
```

### Access Statistics

Track how often keys are read and written:

```python
cache = MockCache()
cache.set("key1", "value1")
cache.get("key1")
cache.get("key1")

assert cache.set_stats["key1"] == 1
assert cache.get_stats["key1"] == 2

cache.clear_stats()  # reset counters
```

## Graph-Level Testing

For integration tests that run an entire graph, combine `MockLLM`, `MockCache`, and `inject_mock_cache`.

### inject_mock_cache

Replaces the cache on every `CachedTaskWorker` in a graph, including workers inside subgraphs:

```python
from planai import Graph
from planai.testing import MockCache, MockLLM, MockLLMResponse, inject_mock_cache

mock_cache = MockCache(dont_store=True)
mock_llm = MockLLM(responses=[
    MockLLMResponse(pattern="Create a detailed plan.*", response=PlanDraft(plan="...")),
    MockLLMResponse(pattern=".*Score each criterion.*", response=PlanCritique(overall_score=0.8)),
    MockLLMResponse(pattern="Create a refined.*plan.*", response=FinalPlan(plan="...", rationale="...")),
])

graph = Graph(name="TestGraph")
planning = create_planning_worker(llm=mock_llm, name="TestPlanning")
graph.add_workers(planning)
graph.set_sink(planning, FinalPlan)

inject_mock_cache(graph, mock_cache)

graph.run(
    initial_tasks=[(planning, PlanRequest(request="Create a test plan"))],
    run_dashboard=False,
    display_terminal=False,
)

output_tasks = graph.get_output_tasks()
assert len(output_tasks) == 1
assert isinstance(output_tasks[0], FinalPlan)
```

## Helper Functions

### add_input_provenance

Manually inject provenance into a test task when a worker depends on inspecting the provenance chain:

```python
from planai.testing import add_input_provenance

parent = InputTask(data="original")
child = OutputTask(result="derived")

add_input_provenance(child, parent)
# child._input_provenance now contains [parent]
```

### unregister_output_type

Remove an output type from a worker's consumers. Useful for intercepting intermediate tasks in graph tests:

```python
from planai.testing import unregister_output_type

# Capture PlanDraft tasks instead of letting them flow downstream
planner = graph.get_worker_by_output_type(PlanDraft)
unregister_output_type(planner, PlanDraft)
graph.set_sink(planner, PlanDraft)

graph.run(initial_tasks=initial_work, run_dashboard=False, display_terminal=False)
drafts = [t for t in graph.get_output_tasks() if isinstance(t, PlanDraft)]
assert len(drafts) == 3
```

## Tips

- Always pass `run_dashboard=False, display_terminal=False` when running graphs in tests to avoid starting the monitoring UI.
- `InvokeTaskWorker` validates input task types — passing the wrong type raises `TypeError`.
- Calling `invoke()` on a `JoinedTaskWorker` (or `invoke_joined()` on a regular worker) raises `TypeError`.
- `InvokeTaskWorker` automatically injects a `MockCache` if the worker has a `_cache` attribute.
- `MockLLMResponse.pattern` is matched as a regex against the full prompt text.
