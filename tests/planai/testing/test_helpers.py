import unittest
from typing import List, Type

from planai.cached_task import CachedTaskWorker
from planai.graph import Graph
from planai.joined_task import JoinedTaskWorker
from planai.task import Task, TaskWorker
from planai.testing.helpers import (
    InvokeTaskWorker,
    MockCache,
    add_input_provenance,
    inject_mock_cache,
)


# Sample Task classes for testing
class InputTask(Task):
    data: str


class OutputTask(Task):
    result: str


class JoinedOutputTask(Task):
    results: List[str]


# Sample Worker classes for testing
class SimpleWorker(TaskWorker):
    output_types: List[Type[Task]] = [OutputTask]

    def consume_work(self, task: InputTask):
        output = OutputTask(result=f"Processed: {task.data}")
        self.publish_work(output, task)


class MultiOutputWorker(TaskWorker):
    output_types: List[Type[Task]] = [OutputTask]

    def consume_work(self, task: InputTask):
        for i in range(3):
            output = OutputTask(result=f"Batch {i}: {task.data}")
            self.publish_work(output, task)


class JoinedWorker(JoinedTaskWorker):
    join_type: Type[TaskWorker] = SimpleWorker
    output_types: List[Type[Task]] = [JoinedOutputTask]

    def consume_work_joined(
        self, tasks: List[OutputTask]
    ):  # Changed from consume_joined
        results = [task.result for task in tasks]
        output = JoinedOutputTask(results=results)
        self.publish_work(output, tasks[0])


class IncorrectWorker(TaskWorker):
    output_types: List[Type[Task]] = [OutputTask]

    def pre_consume_work(self, task: Task):
        # this is a function only avaiable in CachedTaskWorker
        pass

    def consume_work(self, task: InputTask):
        output = OutputTask(result=f"Processed: {task.data}")
        self.publish_work(output, task)


class TestMockCache(unittest.TestCase):
    def setUp(self):
        self.cache = MockCache()

    def test_set_get(self):
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("nonexistent"), None)

    def test_stats(self):
        self.cache.set("key1", "value1")
        self.cache.get("key1")
        self.cache.get("key1")

        self.assertEqual(self.cache.set_stats["key1"], 1)
        self.assertEqual(self.cache.get_stats["key1"], 2)

    def test_dont_store(self):
        cache = MockCache(dont_store=True)
        cache.set("key1", "value1")
        self.assertIsNone(cache.get("key1"))


class TestInvokeTaskWorker(unittest.TestCase):
    def test_simple_worker(self):
        worker = InvokeTaskWorker(SimpleWorker)
        input_task = InputTask(data="test")

        published = worker.invoke(input_task)

        worker.assert_published_task_count(1)
        worker.assert_published_task_types([OutputTask])
        self.assertEqual(published[0].result, "Processed: test")

    def test_multi_output_worker(self):
        worker = InvokeTaskWorker(MultiOutputWorker)
        input_task = InputTask(data="test")

        published = worker.invoke(input_task)

        worker.assert_published_task_count(3)
        worker.assert_published_task_types([OutputTask] * 3)
        self.assertEqual(len(published), 3)

    def test_joined_worker(self):
        worker = InvokeTaskWorker(JoinedWorker)
        input_tasks = [
            OutputTask(result="result1"),
            OutputTask(result="result2"),
            OutputTask(result="result3"),
        ]

        published = worker.invoke_joined(input_tasks)

        worker.assert_published_task_count(1)
        worker.assert_published_task_types([JoinedOutputTask])
        self.assertEqual(published[0].results, ["result1", "result2", "result3"])

    def test_incorrect_pre_consume_work(self):
        worker = InvokeTaskWorker(IncorrectWorker)
        input_task = InputTask(data="test")

        with self.assertRaises(RuntimeError):
            worker.invoke(input_task)

    def test_wrong_invocation_method(self):
        regular_worker = InvokeTaskWorker(SimpleWorker)
        joined_worker = InvokeTaskWorker(JoinedWorker)

        with self.assertRaises(TypeError):
            regular_worker.invoke_joined([InputTask(data="test")])

        with self.assertRaises(TypeError):
            joined_worker.invoke(InputTask(data="test"))

    def test_wrong_type_invocation(self):
        worker = InvokeTaskWorker(SimpleWorker)
        input_task = OutputTask(result="test")

        with self.assertRaises(TypeError):
            worker.invoke(input_task)


class TestHelperFunctions(unittest.TestCase):
    def test_add_input_provenance(self):
        task1 = InputTask(data="original")
        task2 = OutputTask(result="result")

        result = add_input_provenance(task2, task1)
        self.assertEqual(result._input_provenance, [task1])

    def test_inject_mock_cache(self):
        graph = Graph(name="Test Graph")

        class MyWorker(CachedTaskWorker):
            output_types: List[Type[Task]] = [InputTask]

            def consume_work(self, task: InputTask):
                pass

        worker = MyWorker()
        graph.add_workers(worker)

        mock_cache = MockCache()
        inject_mock_cache(graph, mock_cache)

        self.assertIs(worker._cache, mock_cache)


if __name__ == "__main__":
    unittest.main()
