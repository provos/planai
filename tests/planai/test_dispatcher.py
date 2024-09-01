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
import concurrent.futures
import logging
import random
import threading
import time
import unittest
from collections import deque
from queue import Queue
from threading import Event
from typing import List, Type
from unittest.mock import Mock, patch

from pydantic import PrivateAttr

from planai.dispatcher import Dispatcher
from planai.graph import Graph
from planai.task import TaskWorker, TaskWorkItem


class DummyTaskWorkItem(TaskWorkItem):
    data: str


class DummyTaskWorkerSimple(TaskWorker):
    def consume_work(self, task: DummyTaskWorkItem):
        pass


class DummyTaskWorker(TaskWorker):
    output_types: List[Type[TaskWorkItem]] = [DummyTaskWorkItem]
    publish: bool = True
    _processed_count: int = PrivateAttr(0)

    def __init__(self, **data):
        super().__init__(**data)
        self._processed_count = 0

    def consume_work(self, task: DummyTaskWorkItem):
        time.sleep(random.uniform(0.001, 0.01))  # Simulate some work
        self._processed_count += 1
        if self.publish and random.random() < 0.7:  # 70% chance to produce output
            output_task = DummyTaskWorkItem(data=f"Output from {self.name}")
            logging.debug(f"Produced output: {output_task.data}")
            self.publish_work(output_task, input_task=task)


class ExceptionRaisingTaskWorker(TaskWorker):
    def consume_work(self, task: DummyTaskWorkItem):
        raise ValueError("Test exception")


class RetryTaskWorker(TaskWorker):
    fail_attempts: int
    _attempt_count: int = PrivateAttr(0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def consume_work(self, task: DummyTaskWorkItem):
        with self._lock:
            self._attempt_count += 1
            if self._attempt_count <= self.fail_attempts:
                raise ValueError(f"Simulated failure (attempt {self._attempt_count})")


class SingleThreadedExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, fn, *args, **kwargs):
        future = Mock()
        result = fn(*args, **kwargs)
        future.result.return_value = result
        self.tasks.append(future)
        return future

    def shutdown(self, wait=True):
        pass


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        self.graph = Mock(spec=Graph)
        self.graph._thread_pool = SingleThreadedExecutor()
        self.dispatcher = Dispatcher(self.graph)
        self.dispatcher.work_queue = Queue()
        self.dispatcher.stop_event = Event()

    def test_add_provenance(self):
        task = DummyTaskWorkItem(data="test")
        task._provenance = [("Task1", 1), ("Task2", 2)]
        self.dispatcher._add_provenance(task)
        self.assertEqual(
            self.dispatcher.provenance,
            {(("Task1", 1),): 1, (("Task1", 1), ("Task2", 2)): 1},
        )

    def test_remove_provenance(self):
        task = DummyTaskWorkItem(data="test")
        task._provenance = [("Task1", 1), ("Task2", 2)]
        self.dispatcher._add_provenance(task)

        with patch.object(self.dispatcher, "_notify_task_completion") as mock_notify:
            self.dispatcher._remove_provenance(task)
            self.assertEqual(self.dispatcher.provenance, {})
            mock_notify.assert_any_call((("Task1", 1),))
            mock_notify.assert_any_call((("Task1", 1), ("Task2", 2)))

    def test_notify_task_completion(self):
        notifier = Mock(spec=TaskWorker)
        self.dispatcher.notifiers = {(("Task1", 1),): [notifier]}
        self.dispatcher._notify_task_completion((("Task1", 1),))
        self.assertEqual(self.dispatcher.active_tasks, 1)
        notifier.notify.assert_called_once_with((("Task1", 1),))

    def test_watch(self):
        notifier = Mock(spec=TaskWorker)
        result = self.dispatcher.watch((DummyTaskWorkItem.__name__, 1), notifier)
        self.assertTrue(result)
        self.assertIn((DummyTaskWorkItem.__name__, 1), self.dispatcher.notifiers)
        self.assertIn(
            notifier, self.dispatcher.notifiers[(DummyTaskWorkItem.__name__, 1)]
        )

    def test_unwatch(self):
        notifier = Mock(spec=TaskWorker)
        self.dispatcher.notifiers = {(DummyTaskWorkItem.__name__, 1): [notifier]}
        result = self.dispatcher.unwatch((DummyTaskWorkItem.__name__, 1), notifier)
        self.assertTrue(result)
        self.assertNotIn((DummyTaskWorkItem.__name__, 1), self.dispatcher.notifiers)

    def test_dispatch(self):
        worker = Mock(spec=TaskWorker)
        task = DummyTaskWorkItem(data="test")
        self.dispatcher.work_queue.put((worker, task))

        # Run dispatch once
        with patch.object(self.dispatcher, "_execute_task") as mock_execute:
            self.dispatcher._dispatch_once()

            mock_execute.assert_called_once_with(worker, task)

        self.assertEqual(self.dispatcher.active_tasks, 1)

        # Simulate task completion
        future = self.graph._thread_pool.tasks[0]
        future.add_done_callback.assert_called_once()
        callback = future.add_done_callback.call_args[0][0]
        callback(future)

        self.assertEqual(self.dispatcher.active_tasks, 0)
        self.assertTrue(self.dispatcher.task_completion_event.is_set())

    def test_execute_task(self):
        worker = Mock(spec=TaskWorker)
        future = Mock()
        task = DummyTaskWorkItem(data="test")
        self.dispatcher._execute_task(worker, task)
        self.dispatcher._task_completed(worker, task, future)
        worker._pre_consume_work.assert_called_once_with(task)
        self.assertIn(task, [t[1] for t in self.dispatcher.completed_tasks])

    def test_task_to_dict(self):
        worker = DummyTaskWorkerSimple()
        task = DummyTaskWorkItem(data="test")
        task._provenance = [("Task1", 1)]
        task._input_provenance = [DummyTaskWorkItem(data="input")]
        result = self.dispatcher._task_to_dict(worker, task)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "DummyTaskWorkItem")
        self.assertEqual(result["worker"], "DummyTaskWorkerSimple")
        self.assertEqual(result["provenance"], ["Task1_1"])

    def test_get_queued_tasks(self):
        worker = DummyTaskWorkerSimple()
        task = DummyTaskWorkItem(data="test")
        self.dispatcher.work_queue.put((worker, task))
        result = self.dispatcher.get_queued_tasks()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "DummyTaskWorkItem")

    def test_get_active_tasks(self):
        worker = DummyTaskWorkerSimple()
        task = DummyTaskWorkItem(data="test")
        self.dispatcher.debug_active_tasks = {1: (worker, task)}
        result = self.dispatcher.get_active_tasks()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "DummyTaskWorkItem")

    def test_get_completed_tasks(self):
        worker = DummyTaskWorkerSimple()
        task = DummyTaskWorkItem(data="test")
        self.dispatcher.completed_tasks = deque([(worker, task)])
        result = self.dispatcher.get_completed_tasks()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "DummyTaskWorkItem")

    def test_notify_completed(self):
        future = Mock()
        self.dispatcher.active_tasks = 1
        self.dispatcher._notify_completed(future)
        self.assertEqual(self.dispatcher.active_tasks, 0)
        self.assertTrue(self.dispatcher.task_completion_event.is_set())

    def test_task_completed(self):
        worker = Mock(spec=TaskWorker)
        task = DummyTaskWorkItem(data="test")
        future = Mock()
        future.result.return_value = None

        # Set initial conditions
        self.dispatcher.active_tasks = 1
        self.dispatcher.work_queue = Queue()  # Ensure the queue is empty

        with patch.object(self.dispatcher, "_remove_provenance") as mock_remove:
            self.dispatcher._task_completed(worker, task, future)
            mock_remove.assert_called_once_with(task)

        self.assertEqual(self.dispatcher.active_tasks, 0)
        self.assertTrue(self.dispatcher.task_completion_event.is_set())

    def test_task_completed_with_remaining_tasks(self):
        worker = Mock(spec=TaskWorker)
        task = DummyTaskWorkItem(data="test")
        future = Mock()
        future.result.return_value = None

        # Set initial conditions
        self.dispatcher.active_tasks = 2
        self.dispatcher.work_queue = Queue()  # Ensure the queue is empty

        with patch.object(self.dispatcher, "_remove_provenance") as mock_remove:
            self.dispatcher._task_completed(worker, task, future)
            mock_remove.assert_called_once_with(task)

        self.assertEqual(self.dispatcher.active_tasks, 1)
        self.assertFalse(self.dispatcher.task_completion_event.is_set())

    def test_add_work(self):
        worker = Mock(spec=TaskWorker)
        task = DummyTaskWorkItem(data="test")
        with patch.object(self.dispatcher, "_add_provenance") as mock_add:
            self.dispatcher.add_work(worker, task)
            mock_add.assert_called_once_with(task)
        self.assertFalse(self.dispatcher.work_queue.empty())

    def test_stop(self):
        self.dispatcher.stop()
        self.assertTrue(self.dispatcher.stop_event.is_set())

    @patch("planai.dispatcher.is_quit_requested")
    def test_wait_for_completion(self, mock_is_quit_requested):
        mock_is_quit_requested.side_effect = [False, False, True]
        self.dispatcher.task_completion_event.set()
        self.dispatcher.wait_for_completion(wait_for_quit=True)
        self.assertEqual(mock_is_quit_requested.call_count, 3)

    @patch("planai.dispatcher.run_web_interface")
    def test_start_web_interface(self, mock_run_web_interface):
        self.dispatcher.start_web_interface()
        mock_run_web_interface.assert_called_once_with(self.dispatcher, 5000)


class TestDispatcherThreading(unittest.TestCase):
    def setUp(self):
        self.graph = Mock(spec=Graph)
        self.graph._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.dispatcher = Dispatcher(self.graph)

    def tearDown(self):
        self.graph._thread_pool.shutdown(wait=True)

    def test_concurrent_add_work(self):
        num_threads = 10
        num_tasks_per_thread = 100

        def add_work():
            for _ in range(num_tasks_per_thread):
                worker = Mock(spec=TaskWorker)
                task = DummyTaskWorkItem(data="test")
                self.dispatcher.add_work(worker, task)

        threads = [threading.Thread(target=add_work) for _ in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(
            self.dispatcher.work_queue.qsize(), num_threads * num_tasks_per_thread
        )

    def test_race_condition_provenance(self):
        num_threads = 10
        num_operations = 1000

        def modify_provenance():
            for _ in range(num_operations):
                task = DummyTaskWorkItem(data="test")
                task._provenance = [("Task1", 1)]
                self.dispatcher._add_provenance(task)
                self.dispatcher._remove_provenance(task)

        threads = [
            threading.Thread(target=modify_provenance) for _ in range(num_threads)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should cancel out, leaving the provenance empty or with zero counts
        for value in self.dispatcher.provenance.values():
            self.assertEqual(value, 0, "Provenance count should be 0 for all tasks")

    def test_stress_dispatcher(self):
        logging.basicConfig(level=logging.DEBUG)
        num_workers = 5
        num_tasks_per_worker = 1000

        workers = [Mock(spec=TaskWorker) for _ in range(num_workers)]

        def worker_task(worker):
            for i in range(num_tasks_per_worker):
                task = DummyTaskWorkItem(data=f"test-{worker.name}-{i}")
                self.dispatcher.add_work(worker, task)

        # Use a real ThreadPoolExecutor for this test
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, worker) for worker in workers]
            concurrent.futures.wait(futures)

        total_tasks = num_workers * num_tasks_per_worker
        logging.debug(f"Total tasks added: {total_tasks}")

        # Process all tasks
        processed_count = 0
        start_time = time.time()
        while (
            not self.dispatcher.work_queue.empty() or self.dispatcher.active_tasks > 0
        ) and time.time() - start_time < 60:  # 60 second timeout
            if self.dispatcher._dispatch_once():
                processed_count += 1
            time.sleep(0.001)  # Small delay to allow for task completion

        logging.debug(f"Work queue size: {self.dispatcher.work_queue.qsize()}")
        logging.debug(f"Active tasks: {self.dispatcher.active_tasks}")
        logging.debug(f"Completed tasks: {self.dispatcher.total_completed_tasks}")
        logging.debug(f"Processed count: {processed_count}")

        self.assertEqual(
            self.dispatcher.work_queue.qsize(), 0, "All tasks should be processed"
        )
        self.assertEqual(
            self.dispatcher.active_tasks, 0, "No active tasks should remain"
        )
        self.assertEqual(
            self.dispatcher.total_completed_tasks,
            total_tasks,
            f"All tasks should be completed. Expected {total_tasks}, got {self.dispatcher.total_completed_tasks}",
        )
        self.assertEqual(
            processed_count,
            total_tasks,
            f"All tasks should be processed. Expected {total_tasks}, got {processed_count}",
        )

    @patch("planai.dispatcher.logging.exception")
    def test_exception_logging(self, mock_logging_exception):
        num_tasks = 10

        # Create a worker that raises an exception
        worker = ExceptionRaisingTaskWorker()

        # Add tasks to the dispatcher
        for i in range(num_tasks):
            task = DummyTaskWorkItem(data=f"test-{i}")
            self.dispatcher.add_work(worker, task)

        # Process all tasks
        start_time = time.time()
        while (
            not self.dispatcher.work_queue.empty() or self.dispatcher.active_tasks > 0
        ) and time.time() - start_time < 10:  # 10 second timeout
            self.dispatcher._dispatch_once()
            time.sleep(0.01)  # Small delay to allow for task completion

        # Check that all tasks have been processed
        self.assertEqual(
            self.dispatcher.work_queue.qsize(), 0, "All tasks should be processed"
        )
        self.assertEqual(
            self.dispatcher.active_tasks, 0, "No active tasks should remain"
        )

        # Check that exceptions were logged
        self.assertEqual(
            mock_logging_exception.call_count,
            num_tasks,
            f"Expected {num_tasks} exceptions to be logged, but got {mock_logging_exception.call_count}",
        )

        # Check the content of the logged exceptions
        for call in mock_logging_exception.call_args_list:
            self.assertIn("Task DummyTaskWorkItem", call[0][0])
            self.assertIn("failed with exception: Test exception", call[0][0])

        # TODO: whether we should also check the number of completed tasks

    @patch("planai.dispatcher.logging.info")
    @patch("planai.dispatcher.logging.error")
    @patch("planai.dispatcher.logging.exception")
    def test_task_retry(self, mock_log_exception, mock_log_error, mock_log_info):
        worker = RetryTaskWorker(num_retries=2, fail_attempts=2)
        task = DummyTaskWorkItem(data="test-retry")

        self.dispatcher.active_tasks = 1
        self.dispatcher.work_queue = Queue()

        # Simulate task execution and failure
        future = Mock()
        future.result.side_effect = ValueError("Simulated failure")

        # First attempt
        self.dispatcher._task_completed(worker, task, future)

        # Check if task was requeued
        self.assertEqual(self.dispatcher.work_queue.qsize(), 1)
        self.assertEqual(task._retry_count, 1)
        mock_log_info.assert_any_call("Retrying task DummyTaskWorkItem for the 1 time")

        # Second attempt (should succeed)
        self.dispatcher.work_queue.get()  # Remove the task from the queue
        self.dispatcher.active_tasks += 1
        worker._attempt_count = 2  # Simulate successful attempt
        future.result.side_effect = None  # Remove the exception
        self.dispatcher._task_completed(worker, task, future)

        # Check final state
        self.assertEqual(self.dispatcher.work_queue.qsize(), 0)
        self.assertEqual(self.dispatcher.active_tasks, 0)
        self.assertEqual(self.dispatcher.total_completed_tasks, 1)

        # Check logging calls
        mock_log_exception.assert_called_once()
        mock_log_error.assert_not_called()
        mock_log_info.assert_any_call("Task DummyTaskWorkItem completed successfully")

    @patch("planai.dispatcher.logging.info")
    @patch("planai.dispatcher.logging.error")
    @patch("planai.dispatcher.logging.exception")
    def test_task_retry_exhausted(
        self, mock_log_exception, mock_log_error, mock_log_info
    ):
        worker = RetryTaskWorker(num_retries=2, fail_attempts=3)
        task = DummyTaskWorkItem(data="test-retry-exhausted")

        self.dispatcher.active_tasks = 1
        self.dispatcher.work_queue = Queue()

        # Simulate task execution and failure
        future = Mock()
        future.result.side_effect = ValueError("Simulated failure")

        # First attempt
        self.dispatcher._task_completed(worker, task, future)
        self.assertEqual(task._retry_count, 1)

        # Second attempt
        self.dispatcher.work_queue.get()  # Remove the task from the queue
        self.dispatcher.active_tasks += 1
        self.dispatcher._task_completed(worker, task, future)
        self.assertEqual(task._retry_count, 2)

        # Third attempt (should not retry anymore)
        self.dispatcher.work_queue.get()  # Remove the task from the queue
        self.dispatcher.active_tasks += 1
        self.dispatcher._task_completed(worker, task, future)

        # Check final state
        self.assertEqual(self.dispatcher.work_queue.qsize(), 0)
        self.assertEqual(self.dispatcher.active_tasks, 0)
        self.assertEqual(self.dispatcher.total_completed_tasks, 0)

        # Check logging calls
        self.assertEqual(mock_log_exception.call_count, 3)
        mock_log_error.assert_called_once_with(
            "Task DummyTaskWorkItem failed after 2 retries"
        )
        mock_log_info.assert_any_call("Retrying task DummyTaskWorkItem for the 1 time")
        mock_log_info.assert_any_call("Retrying task DummyTaskWorkItem for the 2 time")

    def test_exception_handling_end_to_end(self):
        dispatcher = Dispatcher(self.graph)
        self.graph._dispatcher = dispatcher

        # Create an ExceptionRaisingTaskWorker
        worker = ExceptionRaisingTaskWorker()
        self.graph.add_workers(worker)

        # Create a task
        task = DummyTaskWorkItem(data="test_exception")

        # Set up a thread to run the dispatcher
        dispatcher_thread = threading.Thread(target=dispatcher.dispatch)
        dispatcher_thread.start()

        # Add the work to the dispatcher
        dispatcher.add_work(worker, task)

        # Wait for the task to be processed
        dispatcher.wait_for_completion()

        # Stop the dispatcher
        dispatcher.stop()
        dispatcher_thread.join(timeout=5)

        # Assertions
        self.assertEqual(dispatcher.work_queue.qsize(), 0, "Work queue should be empty")
        self.assertEqual(dispatcher.active_tasks, 0, "No active tasks should remain")
        self.assertEqual(
            dispatcher.total_completed_tasks,
            0,
            "No tasks should be marked as completed",
        )
        self.assertEqual(
            len(dispatcher.failed_tasks),
            1,
            "One task should be in the failed tasks list",
        )

        # Check the failed task
        failed_worker, failed_task = dispatcher.failed_tasks[0]
        self.assertIsInstance(failed_worker, ExceptionRaisingTaskWorker)
        self.assertEqual(failed_task.data, "test_exception")

        # Check that the provenance was properly removed
        self.assertEqual(len(dispatcher.provenance), 0, "Provenance should be empty")

        # Verify that the task is not in the active tasks list
        self.assertEqual(
            len(dispatcher.debug_active_tasks),
            0,
            "No tasks should be in the active tasks list",
        )

        # Check that the exception was logged
        with self.assertLogs(level="ERROR") as cm:
            dispatcher._task_completed(
                worker,
                task,
                Mock(result=Mock(side_effect=ValueError("Test exception"))),
            )
        self.assertIn(
            "Task DummyTaskWorkItem failed with exception: Test exception", cm.output[0]
        )

        self.graph._thread_pool.shutdown(wait=True)


class TestDispatcherConcurrent(unittest.TestCase):
    def test_concurrent_execution(self):
        graph = Graph(name="Test Graph")
        dispatcher = Dispatcher(graph)
        graph._dispatcher = dispatcher

        # Create a chain of workers
        worker1 = DummyTaskWorker()
        worker2 = DummyTaskWorker()
        worker3 = DummyTaskWorker(publish=False)

        graph.add_workers(worker1, worker2, worker3)
        graph.set_dependency(worker1, worker2).next(worker3)

        # Create initial tasks
        initial_tasks = [DummyTaskWorkItem(data=f"Initial {i}") for i in range(100)]

        # Start the dispatcher in a separate thread
        dispatch_thread = threading.Thread(target=dispatcher.dispatch)
        dispatch_thread.start()

        # Function to add initial work
        def add_initial_work():
            for task in initial_tasks:
                dispatcher.add_work(worker1, task)

        # Start adding work in a separate thread
        add_work_thread = threading.Thread(target=add_initial_work)
        add_work_thread.start()

        # Wait for all work to be processed
        add_work_thread.join()

        # Wait for dispatcher
        dispatcher.wait_for_completion()
        dispatcher.stop()
        dispatch_thread.join()

        # Check results
        total_processed = (
            worker1._processed_count
            + worker2._processed_count
            + worker3._processed_count
        )
        assert (
            total_processed >= 100
        ), f"Expected at least 100 processed tasks, but got {total_processed}"
        assert (
            worker1._processed_count == 100
        ), f"Worker1 should have processed 100 tasks, but processed {worker1._processed_count}"
        assert (
            worker2._processed_count <= 100
        ), f"Worker2 should have processed at most 100 tasks, but processed {worker2._processed_count}"
        assert (
            worker3._processed_count <= worker2._processed_count
        ), f"Worker3 should have processed at most as many tasks as Worker2, but processed {worker3._processed_count} vs {worker2._processed_count}"

        print(f"Worker1 processed: {worker1._processed_count}")
        print(f"Worker2 processed: {worker2._processed_count}")
        print(f"Worker3 processed: {worker3._processed_count}")

        assert (
            dispatcher.total_completed_tasks == total_processed
        ), f"Completed tasks {dispatcher.total_completed_tasks} should match total processed ({total_processed})"

        graph._thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
