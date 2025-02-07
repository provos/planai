import logging
import threading
import time
import unittest
from typing import Dict, List, Optional, Type

from planai.graph import Graph
from planai.provenance import ProvenanceChain
from planai.task import Task, TaskWorker

# Dummy task and worker classes for testing


class DummyTask(Task):
    data: str


class AnotherDummyTask(Task):
    data: str


class DummyWorker(TaskWorker):
    output_types: List[Type[Task]] = [DummyTask]
    should_wait: bool = False

    def consume_work(self, task: DummyTask):
        # Use copy_public() in case we're in strict mode
        if self.should_wait:
            time.sleep(0.1)
        self.publish_work(task.copy_public(), input_task=task)


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

        # Cannot set a sink that is not being produced
        with self.assertRaises(ValueError):
            self.graph.set_sink(worker, Task)

        self.graph.set_sink(worker, DummyTask)
        self.assertTrue(len(self.graph._sink_workers) > 0)

        # Set the same sink again should raise RuntimeError
        with self.assertRaises(ValueError):
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

        # Try to add work for a non-entry worker
        with self.assertRaises(ValueError):
            self.graph.add_work(test_worker, initial_task)

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
            metadata: Dict,
            prefix: ProvenanceChain,
            worker: TaskWorker,
            task: Task,
            message: Optional[str],
        ):
            callback_data.append(
                {
                    "metadata": metadata,
                    "prefix": prefix,
                    "worker_name": worker.name if worker else None,
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
        self.assertEqual(len(callback_data), 3)  # Should have two status updates

        # Check first status update
        self.assertEqual(callback_data[0]["metadata"]["test_key"], "test_value")
        self.assertEqual(callback_data[0]["prefix"], (("InitialTaskWorker", 1),))
        self.assertEqual(callback_data[0]["worker_name"], "StatusTestWorker")
        self.assertEqual(callback_data[0]["task_data"], "initial")
        self.assertEqual(callback_data[0]["message"], "Starting work")

        # Check second status update
        self.assertEqual(callback_data[1]["metadata"]["test_key"], "test_value")
        self.assertEqual(callback_data[1]["prefix"], (("InitialTaskWorker", 1),))
        self.assertEqual(callback_data[1]["worker_name"], "StatusTestWorker")
        self.assertEqual(callback_data[1]["task_data"], "initial processed")
        self.assertEqual(callback_data[1]["message"], "Work completed")

        # Check third status update
        self.assertEqual(callback_data[2]["metadata"]["test_key"], "test_value")
        self.assertEqual(callback_data[2]["prefix"], (("InitialTaskWorker", 1),))
        self.assertEqual(callback_data[2]["worker_name"], None)
        self.assertEqual(callback_data[2]["task_data"], None)
        self.assertEqual(callback_data[2]["message"], "Task removed")

    def test_strict_mode(self):
        # Create a graph with strict mode enabled
        graph = Graph(name="Test Graph", strict=True)

        success_flag = False
        failure_flag = False

        # Create a task worker that attempts to reuse tasks with provenance
        class StrictTestWorker(TaskWorker):
            output_types: list = [DummyTask]

            def consume_work(self, task: DummyTask):
                nonlocal success_flag, failure_flag
                # First try with model_copy() which should fail
                task_copy = task.model_copy()
                try:
                    self.publish_work(task_copy, input_task=task)
                except ValueError:
                    failure_flag = True

                # Now try with copy_public() which should succeed
                task_copy = task.copy_public()
                try:
                    self.publish_work(task_copy, input_task=task)
                    success_flag = True
                except ValueError:
                    pass

        # Register workers and set up graph
        worker1 = DummyWorker()
        worker2 = StrictTestWorker()
        graph.add_worker(worker1)
        graph.add_worker(worker2)
        graph.set_dependency(worker1, worker2).sink(DummyTask)

        # Run the graph
        initial_task = DummyTask(data="initial")
        initial_tasks = [(worker1, initial_task)]
        graph.run(initial_tasks, display_terminal=False)

        # Verify both conditions were met
        self.assertTrue(
            failure_flag, "publish_work() with model_copy() should have failed"
        )
        self.assertTrue(
            success_flag, "publish_work() with copy_public() should have succeeded"
        )

    def test_shutdown(self):
        """Test graph shutdown functionality."""
        # Set up graph with workers
        worker1 = DummyWorker(should_wait=True)
        worker2 = DummyWorker(should_wait=True)
        self.graph.add_workers(worker1, worker2)
        self.graph.set_dependency(worker1, worker2).sink(DummyTask)

        # Prepare graph and add some work
        self.graph.prepare(display_terminal=False, run_dashboard=False)
        self.graph.set_entry(worker1)

        # Add multiple tasks
        for i in range(5):
            self.graph.add_work(worker1, DummyTask(data=f"test-{i}"))

        # Start execution in a separate thread
        def wrapped_execute():
            self.graph.execute([])
            logging.info("Graph execution finished")

        execution_thread = threading.Thread(target=wrapped_execute)
        execution_thread.start()

        # Give some time for tasks to start
        time.sleep(0.1)

        # Test shutdown with timeout
        success = self.graph.shutdown(timeout=1.0)
        self.assertTrue(success, "Shutdown should complete within timeout")

        execution_thread.join(timeout=1.0)
        self.assertFalse(
            execution_thread.is_alive(), "Execution thread should be stopped"
        )

        # Verify cleanup
        self.assertIsNone(self.graph._dispatch_thread)
        self.assertEqual(self.graph._dispatcher._num_active_tasks, 0)

    def test_worker_state(self):
        # Define test tasks
        class Counting(Task):
            current: str

        class FinalCount(Task):
            result: str

        # Define worker that maintains state
        class Counter(TaskWorker):
            output_types: List[Type[Task]] = [Counting, FinalCount]

            def consume_work(self, task: Counting):
                # Get state for this task's prefix
                provenance = task.prefix(1)
                state = self.get_worker_state(provenance)

                # Initialize or increment count
                if "count" not in state:
                    state["count"] = 1
                else:
                    state["count"] += 1

                # Update task with current count
                current_count = state["count"]

                if current_count < 3:
                    # Continue counting - send task back to self
                    next_task = Counting(current=f"{task.current}-{current_count}")
                    self.publish_work(next_task, input_task=task)
                else:
                    # Reached final count - output final result
                    final = FinalCount(result=f"{task.current}-final:{current_count}")
                    self.publish_work(final, input_task=task)

        # Create graph
        graph = Graph(name="Counter Test")
        counter = Counter()
        graph.add_worker(counter)

        # Set up circular dependency (worker sends tasks back to itself)
        graph.set_dependency(counter, counter)
        # Set up sink for final output
        graph.set_sink(counter, FinalCount)

        # Create initial tasks
        task1 = Counting(current="A")
        task2 = Counting(current="B")
        initial_tasks = [(counter, task1), (counter, task2)]

        # Run graph
        graph.run(initial_tasks, display_terminal=False)

        # Get output tasks
        outputs = graph.get_output_tasks()
        self.assertEqual(len(outputs), 2)

        # Sort outputs by result for consistent testing
        outputs.sort(key=lambda x: x.result)

        # Verify results show proper counting sequence
        self.assertEqual(outputs[0].result, "A-1-2-final:3")
        self.assertEqual(outputs[1].result, "B-1-2-final:3")

        # Verify state was cleaned up
        self.assertEqual(len(counter._user_state), 0)


if __name__ == "__main__":
    unittest.main()
