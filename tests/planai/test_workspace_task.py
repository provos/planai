# test_workspace_task.py

import tempfile
import unittest
from pathlib import Path
from typing import List, Type
from unittest.mock import Mock, patch

from llm_interface import LLMInterface
from llm_interface.llm_tool import Tool as LLMToolInstance
from planai.task import Task
from planai.testing.helpers import MockCache, add_input_provenance
from planai.workspace_task import WorkspaceLLMTaskWorker, WorkspaceTask


class DummyTask(Task):
    content: str


class OutputTask(Task):
    result: str


class DummyWorkspaceWorker(WorkspaceLLMTaskWorker):
    output_types: List[Type[Task]] = [OutputTask]


class WorkspaceWorkerTestCase(unittest.TestCase):
    """Shared setup for WorkspaceLLMTaskWorker tests that don't need a real cache."""

    def setUp(self):
        self.llm = LLMInterface()
        self.llm.client = Mock()
        self.workspace_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace_dir.cleanup)
        self.cache_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache_dir.cleanup)
        self.worker = DummyWorkspaceWorker(
            llm=self.llm, prompt="test prompt", cache_dir=self.cache_dir.name
        )


class TestWorkspaceLLMTaskWorkerDefaults(WorkspaceWorkerTestCase):
    def test_max_tool_rounds_overridden_to_40(self):
        self.assertEqual(self.worker.max_tool_rounds, 40)

    def test_read_only_defaults_false(self):
        self.assertFalse(self.worker.read_only)

    def test_input_globs_defaults_empty(self):
        self.assertEqual(self.worker.input_globs, [])

    def test_expected_output_files_defaults_empty(self):
        task = WorkspaceTask(workspace=self.workspace_dir.name)
        self.assertEqual(self.worker.expected_output_files(task), [])


class TestGetWorkspace(WorkspaceWorkerTestCase):
    def test_finds_workspace_from_task_itself(self):
        task = WorkspaceTask(workspace=self.workspace_dir.name)
        ws = self.worker.get_workspace(task)
        self.assertEqual(ws.root, Path(self.workspace_dir.name).resolve())

    def test_finds_workspace_from_input_provenance(self):
        task = DummyTask(content="hi")
        add_input_provenance(task, WorkspaceTask(workspace=self.workspace_dir.name))
        ws = self.worker.get_workspace(task)
        self.assertEqual(ws.root, Path(self.workspace_dir.name).resolve())

    def test_finds_nearest_workspace_when_multiple_in_chain(self):
        older_dir = tempfile.TemporaryDirectory()
        self.addCleanup(older_dir.cleanup)
        task = DummyTask(content="hi")
        add_input_provenance(task, WorkspaceTask(workspace=older_dir.name))
        add_input_provenance(task, WorkspaceTask(workspace=self.workspace_dir.name))
        ws = self.worker.get_workspace(task)
        self.assertEqual(ws.root, Path(self.workspace_dir.name).resolve())

    def test_raises_value_error_when_not_found(self):
        task = DummyTask(content="hi")
        with self.assertRaises(ValueError):
            self.worker.get_workspace(task)


class TestGetTools(WorkspaceWorkerTestCase):
    def setUp(self):
        super().setUp()
        self.task = WorkspaceTask(workspace=self.workspace_dir.name)

    def test_tools_are_bound_to_the_task_workspace(self):
        tools = {t.name: t for t in self.worker.get_tools(self.task)}
        self.assertIn("write_file", tools)
        result = tools["write_file"].execute(path="hello.txt", content="hi there")
        self.assertNotIn("Error", result)
        self.assertEqual(
            (Path(self.workspace_dir.name) / "hello.txt").read_text(), "hi there"
        )

    def test_read_only_omits_write_and_edit_tools(self):
        self.worker.read_only = True
        tools = {t.name: t for t in self.worker.get_tools(self.task)}
        self.assertNotIn("write_file", tools)
        self.assertNotIn("edit_file", tools)
        self.assertIn("read_file", tools)

    def test_static_tool_shadowing_a_file_tool_is_rejected(self):
        shadow = LLMToolInstance(
            name="read_file",
            description="not the jailed one",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "escaped",
        )
        self.worker.tools = [shadow]
        with self.assertRaises(ValueError) as ctx:
            self.worker.get_tools(self.task)
        self.assertIn("read_file", str(ctx.exception))

    def test_static_tools_are_appended(self):
        custom_tool = LLMToolInstance(
            name="custom_tool",
            description="A custom tool",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "ok",
        )
        self.worker.tools = [custom_tool]
        tools = self.worker.get_tools(self.task)
        names = {t.name for t in tools}
        self.assertIn("custom_tool", names)
        # file tools should still be present alongside the static tool
        self.assertIn("write_file", names)


class TestExtraCacheKeyAndCacheSalt(WorkspaceWorkerTestCase):
    def setUp(self):
        super().setUp()
        self.worker = DummyWorkspaceWorker(
            llm=self.llm,
            prompt="test prompt",
            cache_dir=self.cache_dir.name,
            input_globs=["*.txt"],
        )
        self.task = WorkspaceTask(workspace=self.workspace_dir.name)

    def test_cache_key_changes_when_input_file_content_changes(self):
        data_file = Path(self.workspace_dir.name) / "data.txt"
        data_file.write_text("v1")
        key1 = self.worker._get_cache_key(self.task)

        data_file.write_text("v2")
        key2 = self.worker._get_cache_key(self.task)

        self.assertNotEqual(key1, key2)

    def test_cache_key_stable_without_changes(self):
        data_file = Path(self.workspace_dir.name) / "data.txt"
        data_file.write_text("v1")
        key1 = self.worker._get_cache_key(self.task)
        key2 = self.worker._get_cache_key(self.task)
        self.assertEqual(key1, key2)

    def test_cache_salt_is_the_cache_key_for_read_only_workers(self):
        self.worker.read_only = True
        self.assertEqual(
            self.worker.get_cache_salt(self.task),
            self.worker._get_cache_key(self.task),
        )

    def test_cache_salt_is_fresh_per_execution_for_writers(self):
        key = self.worker._get_cache_key(self.task)
        salt1 = self.worker.get_cache_salt(self.task)
        salt2 = self.worker.get_cache_salt(self.task)
        self.assertTrue(salt1.startswith(key + ":"))
        self.assertNotEqual(salt1, salt2)


class TestCacheKeyIdentity(WorkspaceWorkerTestCase):
    def test_cache_key_differs_between_workspaces(self):
        other_dir = tempfile.TemporaryDirectory()
        self.addCleanup(other_dir.cleanup)
        task_a = DummyTask(content="same payload")
        add_input_provenance(task_a, WorkspaceTask(workspace=self.workspace_dir.name))
        task_b = DummyTask(content="same payload")
        add_input_provenance(task_b, WorkspaceTask(workspace=other_dir.name))

        self.assertNotEqual(
            self.worker._get_cache_key(task_a), self.worker._get_cache_key(task_b)
        )

    def test_cache_key_without_workspace_does_not_raise(self):
        task = DummyTask(content="orphan")
        self.assertEqual(self.worker.extra_cache_key(task), "")
        self.worker._get_cache_key(task)


class OutputTaskFile(Task):
    result: str


class FileWritingWorker(WorkspaceLLMTaskWorker):
    output_types: List[Type[Task]] = [OutputTaskFile]

    def expected_output_files(self, task: Task) -> List[str]:
        return ["out.txt"]


class TestCacheHitBypass(unittest.TestCase):
    def setUp(self):
        self.llm = LLMInterface()
        self.llm.client = Mock()
        self.workspace_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace_dir.cleanup)

        self.mock_cache = MockCache()
        self.cache_patcher = patch(
            "planai.cached_task.Cache", return_value=self.mock_cache
        )
        self.cache_patcher.start()
        self.addCleanup(self.cache_patcher.stop)

        self.worker = FileWritingWorker(
            llm=self.llm, prompt="test prompt", cache_dir="./unused-cache-dir"
        )
        self.task = WorkspaceTask(workspace=self.workspace_dir.name)

    def _seed_cache(self):
        cache_key = self.worker._get_cache_key(self.task)
        cached_result = [(None, OutputTaskFile(result="cached"))]
        self.mock_cache.set(cache_key, [cached_result, self.task])
        self.mock_cache.clear_stats()

    def test_cache_hit_bypassed_when_expected_file_missing(self):
        self._seed_cache()
        fresh_output = OutputTaskFile(result="fresh")
        self.llm.generate_pydantic = Mock(return_value=fresh_output)

        with patch.object(self.worker, "_publish_cached_results") as mock_publish:
            with patch("planai.llm_task.LLMTaskWorker.publish_work"):
                self.worker._pre_consume_work(self.task)
            mock_publish.assert_not_called()

        self.llm.generate_pydantic.assert_called_once()

    def test_rerun_after_invalid_hit_does_not_reuse_llm_response_cache(self):
        self._seed_cache()
        self.llm.generate_pydantic = Mock(return_value=OutputTaskFile(result="fresh"))

        with patch.object(self.worker, "_publish_cached_results"):
            with patch("planai.llm_task.LLMTaskWorker.publish_work"):
                self.worker._pre_consume_work(self.task)

        salt = self.llm.generate_pydantic.call_args.kwargs["cache_salt"]
        key = self.worker._get_cache_key(self.task)
        self.assertNotEqual(salt, key)
        self.assertTrue(salt.startswith(key + ":"))

    def test_lookup_key_is_computed_once_per_execution(self):
        self.llm.generate_pydantic = Mock(return_value=OutputTaskFile(result="fresh"))

        with patch.object(
            self.worker, "_get_cache_key", wraps=self.worker._get_cache_key
        ) as spy:
            with patch("planai.llm_task.LLMTaskWorker.publish_work"):
                self.worker._pre_consume_work(self.task)

        # once for the lookup and once, after consume_work, for the store;
        # get_cache_salt reuses the lookup key instead of computing a third
        self.assertEqual(spy.call_count, 2)

    def test_cache_hit_honored_when_expected_file_present(self):
        self._seed_cache()
        (Path(self.workspace_dir.name) / "out.txt").write_text("done")

        self.llm.generate_pydantic = Mock()

        with patch.object(self.worker, "_publish_cached_results") as mock_publish:
            self.worker._pre_consume_work(self.task)
            mock_publish.assert_called_once()

        self.llm.generate_pydantic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
