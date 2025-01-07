import unittest
from typing import Dict, List, Optional, Type

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

    def test_graph_run_with_add_work(self):
        worker = DummyWorker()
        self.graph.add_worker(worker)
        initial_task = DummyTask(data="initial")

        class TestWorker(TaskWorker):
            output_types: list = [DummyTask]

            def consume_work(self, task: DummyTask):
                task.data += " processed"
                self.publish_work(task.model_copy(), input_task=task)

        # Register a test worker and set up graph
        test_worker = TestWorker()
        self.graph.add_worker(test_worker)
        self.graph.set_dependency(worker, test_worker).sink(DummyTask)

        self.graph.prepare(display_terminal=False)
        self.graph.set_entry(worker)

        # Add metadata to be tracked with the task
        metadata = {"test_key": "test_value"}
        provenance = self.graph.add_work(worker, initial_task, metadata=metadata)

        # Run graph with empty initial tasks list
        self.graph.execute([])

        # Verify output
        output = self.graph.get_output_tasks()
        self.assertEqual(output[0].data, "initial processed")

        # Verify metadata was cleaned up
        self.assertNotIn(provenance, self.graph._provenance_tracker.task_state)

    def test_graph_run_with_sink_notify(self):
        worker = DummyWorker()
        self.graph.add_worker(worker)
        initial_task = DummyTask(data="initial")

        class TestWorker(TaskWorker):
            output_types: list = [DummyTask]

            def consume_work(self, task: DummyTask):
                task.data += " processed"
                self.publish_work(task.model_copy(), input_task=task)

        # Register a test worker and set up graph
        test_worker = TestWorker()
        self.graph.add_worker(test_worker)

        # Create a callback to receive notifications
        received_metadata = {}
        received_task = None

        def notify_callback(metadata, task):
            nonlocal received_metadata, received_task
            received_metadata = metadata
            received_task = task

        # Set up sink with notification callback
        self.graph.set_dependency(worker, test_worker)
        self.graph.set_sink(test_worker, DummyTask, notify=notify_callback)

        self.graph.prepare(display_terminal=False)
        self.graph.set_entry(worker)

        # Add metadata to be tracked with the task
        metadata = {"test_key": "test_value"}
        self.graph.add_work(worker, initial_task, metadata=metadata)

        # Run graph with empty initial tasks list
        self.graph.execute([])

        # Verify that the callback received the correct metadata and task
        self.assertEqual(received_metadata["test_key"], "test_value")
        self.assertEqual(received_task.data, "initial processed")

    def test_graph_run_with_status_callback(self):
        # Track calls to our callback
        callback_data = []

        def status_callback(
            metadata: Dict, worker: TaskWorker, task: Task, message: Optional[str]
        ):
            callback_data.append(
                {
                    "metadata": metadata,
                    "worker_name": worker.name,
                    "task_data": task.data if hasattr(task, "data") else None,
                    "message": message,
                }
            )

        # Set up workers
        worker = DummyWorker()
        self.graph.add_worker(worker)
        initial_task = DummyTask(data="initial")

        class StatusTestWorker(TaskWorker):
            output_types: list = [DummyTask]

            def consume_work(self, task: DummyTask):
                # Send some status updates
                self.notify_status(task, "Starting work")
                task.data += " processed"
                self.notify_status(task, "Work completed")
                self.publish_work(task.model_copy(), input_task=task)

        # Register test worker and set up graph
        test_worker = StatusTestWorker()
        self.graph.add_worker(test_worker)
        self.graph.set_dependency(worker, test_worker).sink(DummyTask)

        self.graph.prepare(display_terminal=False)
        self.graph.set_entry(worker)

        # Add work with metadata and status callback
        metadata = {"test_key": "test_value"}
        self.graph.add_work(
            worker, initial_task, metadata=metadata, status_callback=status_callback
        )

        # Run graph
        self.graph.execute([])

        # Verify callback data
        self.assertEqual(len(callback_data), 2)  # Should have two status updates

        # Check first status update
        self.assertEqual(callback_data[0]["metadata"]["test_key"], "test_value")
        self.assertEqual(callback_data[0]["worker_name"], "StatusTestWorker")
        self.assertEqual(callback_data[0]["task_data"], "initial")
        self.assertEqual(callback_data[0]["message"], "Starting work")

        # Check second status update
        self.assertEqual(callback_data[1]["metadata"]["test_key"], "test_value")
        self.assertEqual(callback_data[1]["worker_name"], "StatusTestWorker")
        self.assertEqual(callback_data[1]["task_data"], "initial processed")
        self.assertEqual(callback_data[1]["message"], "Work completed")


if __name__ == "__main__":
    unittest.main()
