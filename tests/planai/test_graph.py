import unittest
from typing import List, Type

from planai.graph import Graph
from planai.task import Task, TaskWorker

# Dummy task and worker classes for testing


class DummyTask(Task):
    data: str


class AnotherDummyTask(Task):
    data: str


class DummyWorker(TaskWorker):
    output_types: List[Type[Task]] = [DummyTask]

    def consume_work(self, task: DummyTask):
        self.publish_work(task.model_copy(), input_task=task)


class AnotherDummyWorker(TaskWorker):
    output_types: List[Type[Task]] = [AnotherDummyTask]

    def consume_work(self, task: AnotherDummyTask):
        pass


class TestGraph(unittest.TestCase):

    def setUp(self):
        self.graph = Graph(name="Test Graph")

    def test_add_worker(self):
        worker = DummyWorker()
        self.graph.add_worker(worker)
        self.assertIn(worker, self.graph.workers)

        # Test adding the same worker again raises ValueError
        with self.assertRaises(ValueError):
            self.graph.add_worker(worker)

    def test_set_dependency(self):
        worker_a = DummyWorker()
        worker_b = DummyWorker()  # Changed to align consumer with producer
        self.graph.add_worker(worker_a)
        self.graph.add_worker(worker_b)
        self.graph.set_dependency(worker_a, worker_b)
        self.assertIn(worker_b, self.graph.dependencies[worker_a])

    def test_set_dependency_error(self):
        worker_a = DummyWorker()
        worker_b = AnotherDummyWorker()
        self.graph.add_worker(worker_a)
        # worker_b is not added to the graph before setting dependency
        with self.assertRaises(ValueError):
            self.graph.set_dependency(worker_a, worker_b)

    def test_sink_worker(self):
        worker = DummyWorker()
        self.graph.add_worker(worker)
        self.graph.set_sink(worker, DummyTask)
        self.assertIsNotNone(self.graph._sink_worker)

        # Set the same sink again should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.graph.set_sink(worker, DummyTask)

    def test_graph_run(self):
        worker = DummyWorker()
        self.graph.add_worker(worker)
        initial_task = DummyTask(data="initial")

        class TestWorker(TaskWorker):
            output_types: list = [DummyTask]

            def consume_work(self, task: DummyTask):
                task.data += " processed"
                self.publish_work(task.model_copy(), input_task=task)

        # Register a test worker and initial task
        test_worker = TestWorker()
        self.graph.add_worker(test_worker)
        self.graph.set_dependency(worker, test_worker).sink(DummyTask)

        initial_tasks = [(worker, initial_task)]
        self.graph.run(initial_tasks, display_terminal=False)

        output = self.graph.get_output_tasks()

        # Retrieve output from graph
        self.assertEqual(output[0].data, "initial processed")

    def test_max_parallel_tasks(self):
        class ParallelTestWorker(TaskWorker):
            output_types: list = [DummyTask]

            def consume_work(self, task: DummyTask):
                pass

        worker_class = ParallelTestWorker
        self.graph.set_max_parallel_tasks(worker_class, 3)
        self.assertEqual(self.graph._max_parallel_tasks[worker_class], 3)

        # Test invalid max_parallel_tasks value
        with self.assertRaises(ValueError):
            self.graph.set_max_parallel_tasks(worker_class, 0)


if __name__ == "__main__":
    unittest.main()
