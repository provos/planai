import logging
import threading
import time
import unittest
from typing import Dict, List, Optional, Type

from planai.graph import Graph
from planai.provenance import ProvenanceChain
from planai.task import Task, TaskWorker
from planai.user_input import UserInputRequest

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
        # XXX - we may want to track this in the graph
        # with self.assertRaises(ValueError):
        #    self.graph.set_sink(worker, DummyTask)

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
            message: Optional[str] = None,
            object: Optional[object] = None,
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
        self.assertIsNone(self.graph._dispatcher._dispatch_thread)
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

    def test_shared_dispatcher_shutdown(self):
        """Test shutdown behavior with two graphs sharing a dispatcher."""
        # Create first graph with its own dispatcher
        graph1 = Graph(name="Graph1")
        worker1 = DummyWorker(should_wait=True)
        graph1.add_worker(worker1)
        graph1.set_sink(worker1, DummyTask)
        graph1.prepare(display_terminal=False, run_dashboard=False)
        graph1.set_entry(worker1)

        # Create second graph that will share the dispatcher
        graph2 = Graph(name="Graph2")
        worker2 = DummyWorker(should_wait=True)
        graph2.add_worker(worker2)
        graph2.set_sink(worker2, DummyTask)
        graph2.prepare(display_terminal=False, run_dashboard=False)
        graph2.set_entry(worker2)

        # Get dispatcher from first graph and register it with second graph
        dispatcher = graph1.get_dispatcher()
        self.assertIsNotNone(dispatcher)
        graph2.register_dispatcher(dispatcher)

        # Add work to both graphs
        graph1.add_work(worker1, DummyTask(data="test1"))
        graph2.add_work(worker2, DummyTask(data="test2"))

        # Create threads to run both graphs
        def run_graph(graph):
            graph.execute([])

        thread1 = threading.Thread(target=run_graph, args=(graph1,))
        thread2 = threading.Thread(target=run_graph, args=(graph2,))

        thread1.start()
        thread2.start()

        # Give some time for tasks to start
        time.sleep(0.1)

        # Shutdown first graph
        success1 = graph1.shutdown(timeout=1.0)
        self.assertTrue(success1)
        thread1.join(timeout=1.0)
        self.assertFalse(thread1.is_alive())

        # Verify dispatcher is still running
        self.assertIsNone(dispatcher._dispatch_thread)

        # Shutdown second graph
        success2 = graph2.shutdown(timeout=1.0)
        self.assertTrue(success2)
        thread2.join(timeout=1.0)
        self.assertFalse(thread2.is_alive())

    def test_multiple_consumers(self):
        """Test that a worker can publish to multiple consumers of the same task type."""

        # Define a shared task type
        class SharedTask(Task):
            data: str

        # Define a parent worker that produces SharedTask
        class ParentWorker(TaskWorker):
            output_types: List[Type[Task]] = [SharedTask]
            consumer1: TaskWorker
            consumer2: TaskWorker

            def consume_work(self, task: SharedTask):
                # Publish to first child
                task1 = SharedTask(data=f"{task.data}-to-child1")
                self.publish_work(task1, input_task=task, consumer=self.consumer1)

                # Publish to second child
                task2 = SharedTask(data=f"{task.data}-to-child2")
                self.publish_work(task2, input_task=task, consumer=self.consumer2)

        # Define two child workers that both consume SharedTask
        class Child1Worker(TaskWorker):
            output_types: List[Type[Task]] = []
            received_tasks: List[str] = []

            def consume_work(self, task: SharedTask):
                Child1Worker.received_tasks.append(task.data)

        class Child2Worker(TaskWorker):
            output_types: List[Type[Task]] = []
            received_tasks: List[str] = []

            def consume_work(self, task: SharedTask):
                Child2Worker.received_tasks.append(task.data)

        # Clear class variables
        Child1Worker.received_tasks = []
        Child2Worker.received_tasks = []

        # Create graph and workers
        graph = Graph(name="Multiple Consumers Test")
        child1 = Child1Worker()
        child2 = Child2Worker()

        parent = ParentWorker(consumer1=child1, consumer2=child2)

        # Add workers to graph
        graph.add_worker(parent)
        graph.add_worker(child1)
        graph.add_worker(child2)

        # Set up dependencies - both children depend on parent
        graph.set_dependency(parent, child1)
        graph.set_dependency(parent, child2)

        # Create and run initial task
        initial_task = SharedTask(data="initial")
        initial_tasks = [(parent, initial_task)]
        graph.run(initial_tasks, display_terminal=False)

        # Verify each child received their specific task
        self.assertEqual(len(Child1Worker.received_tasks), 1)
        self.assertEqual(len(Child2Worker.received_tasks), 1)
        self.assertEqual(Child1Worker.received_tasks[0], "initial-to-child1")
        self.assertEqual(Child2Worker.received_tasks[0], "initial-to-child2")


class TestUserRequestCallback(unittest.TestCase):
    def setUp(self):
        # Create a simple graph for testing
        self.graph = Graph(name="Test User Request")

        # Create a task with metadata for testing
        self.test_task = Task()
        self.test_task._provenance = [("InitialTaskWorker", 1)]
        self.test_metadata = {"key": "value"}
        self.graph._provenance_tracker.add_state(
            self.test_task.prefix(1), self.test_metadata, None
        )

    def tearDown(self):
        if self.graph._dispatcher:
            self.graph.shutdown()

    def test_user_request_callback(self):
        # Create a test callback that puts a result in the response queue
        callback_called = False
        callback_args = []
        test_response = ("Test response data", "text/plain")

        def test_callback(metadata, request):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = [metadata, request]
            # In a real callback, this would happen after user interaction
            request._response_queue.put(test_response)

        # Set the callback
        self.graph.set_user_request_callback(test_callback)

        # Create a user input request
        test_request = UserInputRequest(
            task_id="test-id",
            instruction="Please provide input",
            provenance=self.test_task._provenance,
            accepted_mime_types=["text/plain"],
        )

        # Wait for user input
        result, mime_type = self.graph.wait_on_user_request(test_request)

        # Verify the callback was called with the right arguments
        self.assertTrue(callback_called)
        self.assertEqual(callback_args[0], self.test_metadata)
        self.assertEqual(callback_args[1], test_request)

        # Verify the result is correct
        self.assertEqual(result, test_response[0])
        self.assertEqual(mime_type, test_response[1])

    def test_user_request_from_worker(self):
        # Define task types for our test
        class InputTask(Task):
            prompt: str

        class OutputTask(Task):
            result: str

        # Define a worker that will make a user request
        class UserRequestWorker(TaskWorker):
            output_types: List[Type[Task]] = [OutputTask]

            def consume_work(self, task: InputTask):
                # Worker requests user input during processing
                user_input, mime_type = self.request_user_input(
                    task=task,
                    instruction=f"Please respond to: {task.prompt}",
                    accepted_mime_types=["text/plain"],
                )

                # Process the user input and publish result
                result_text = f"Processed: {user_input} ({mime_type})"
                self.publish_work(OutputTask(result=result_text), input_task=task)

        # Set up callback tracking
        callback_invoked = False
        received_request = None
        test_response_data = "User provided answer"
        test_mime_type = "text/plain"

        def user_request_callback(metadata, request):
            nonlocal callback_invoked, received_request
            callback_invoked = True
            received_request = request
            # Simulate user providing a response
            self.assertEqual(metadata["test_key"], "test_value")
            self.assertEqual(metadata["prompt_id"], "12345")
            request._response_queue.put((test_response_data, test_mime_type))

        # Create and configure the graph
        graph = Graph(name="User Request Test")
        worker = UserRequestWorker()
        graph.add_worker(worker)
        graph.set_entry(worker)
        graph.set_sink(worker, OutputTask)
        graph.set_user_request_callback(user_request_callback)

        # Add task metadata to be tracked
        metadata = {"test_key": "test_value", "prompt_id": "12345"}

        # Run the graph with an initial task
        initial_task = InputTask(prompt="What is the answer?")
        graph.prepare(display_terminal=False, run_dashboard=False)

        graph.add_work(worker, initial_task, metadata=metadata)

        graph.execute([])

        # Verify the callback was invoked
        self.assertTrue(callback_invoked)
        self.assertIsNotNone(received_request)
        self.assertEqual(
            received_request.instruction, "Please respond to: What is the answer?"
        )

        # Verify the output task contains the processed response
        output_tasks = graph.get_output_tasks()
        self.assertEqual(len(output_tasks), 1)
        self.assertEqual(
            output_tasks[0].result,
            f"Processed: {test_response_data} ({test_mime_type})",
        )

        # Clean up
        graph.shutdown()


if __name__ == "__main__":
    unittest.main()
