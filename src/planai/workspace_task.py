# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A CachedLLMTaskWorker that lets an LLM read, write, and search files inside a
per-job working directory carried through task provenance."""

from typing import List, Optional, Tuple

from pydantic import Field

from .llm_task import CachedLLMTaskWorker
from .task import Task
from .tools.filesystem import Workspace, hash_files, make_file_tools


class WorkspaceTask(Task):
    """A Task that carries the absolute path of a per-job working directory.

    Graphs that want their LLM workers to operate on files should include a
    WorkspaceTask (or a Task subclass with a ``workspace: str`` attribute)
    somewhere in the provenance chain; WorkspaceLLMTaskWorker.get_workspace()
    will find it automatically.
    """

    workspace: str = Field(
        ..., description="Absolute path to the per-job working directory."
    )


class WorkspaceLLMTaskWorker(CachedLLMTaskWorker):
    """A CachedLLMTaskWorker that gives the LLM file tools jailed to a workspace.

    The workspace directory is discovered from the input task's provenance chain
    (see get_workspace()) unless overridden. Because a cache hit replays only the
    published output tasks and not any files the tools wrote on a prior run,
    subclasses can declare expected_output_files() so that a cache hit whose
    files are missing is treated as a cache miss and re-executed.
    """

    read_only: bool = Field(
        default=False,
        description="If true, only read/list/grep tools are exposed; write_file and edit_file are omitted.",
    )
    max_tool_rounds: int = Field(
        default=40,
        description="Maximum number of tool-calling rounds to allow the LLM.",
    )
    input_globs: List[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns (relative to the workspace root) whose matched files' "
            "content is folded into the cache key, so the cache is invalidated "
            "when those input files change."
        ),
    )
    max_read_chars: int = Field(
        default=100_000,
        description="Maximum number of characters the read_file tool returns before truncating.",
    )

    def get_workspace(self, task: Task) -> Workspace:
        """
        Finds the Workspace for this task by looking for the nearest task in the
        provenance chain -- the task itself first, then its input provenance,
        walked nearest-first the same way Task.find_input_task() does -- that has
        a string ``workspace`` attribute. Subclasses may override this, e.g. to
        pull the directory from worker configuration instead.

        Args:
            task (Task): The input task.

        Returns:
            Workspace: The workspace for this task.

        Raises:
            ValueError: If no task with a string ``workspace`` attribute is found.
        """
        candidates = [task] + list(reversed(task._input_provenance))
        for candidate in candidates:
            workspace_path = getattr(candidate, "workspace", None)
            if isinstance(workspace_path, str) and workspace_path:
                return Workspace(workspace_path)

        raise ValueError(
            f"{self.name}: could not find a workspace directory in the task or "
            "its input provenance; expected a WorkspaceTask (or a Task with a "
            "string 'workspace' attribute) upstream"
        )

    def get_tools(self, task: Task):
        file_tools = make_file_tools(
            self.get_workspace(task),
            read_only=self.read_only,
            max_read_chars=self.max_read_chars,
        )
        if self.tools:
            return file_tools + list(self.tools)
        return file_tools

    def get_cache_salt(self, task: Task) -> Optional[str]:
        return self._get_cache_key(task)

    def extra_cache_key(self, task: Task) -> str:
        workspace = self.get_workspace(task)
        return hash_files(workspace, self.input_globs)

    def expected_output_files(self, task: Task) -> List[str]:
        """
        Workspace-relative file paths this worker is expected to have written when
        it runs. Used to bypass a cache hit whose files are no longer present.
        Defaults to no expectations (any cache hit is honored). Subclasses that
        have the LLM write files via the file tools should override this.

        Args:
            task (Task): The input task.

        Returns:
            List[str]: Workspace-relative paths expected to exist after this worker runs.
        """
        return []

    def _cache_hit_is_valid(
        self, task: Task, cached_results: List[Tuple[str, Task]]
    ) -> bool:
        expected_files = self.expected_output_files(task)
        if not expected_files:
            return True

        try:
            workspace = self.get_workspace(task)
        except ValueError:
            # nothing we can check against; fall back to the default behavior
            return True

        for rel_path in expected_files:
            try:
                resolved = workspace.resolve(rel_path)
            except ValueError:
                return False
            if not resolved.exists():
                return False
        return True
