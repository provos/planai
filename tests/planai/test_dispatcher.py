import concurrent.futures
import logging
import random
import threading
import time
import unittest
from collections import deque
from queue import Queue
from threading import Event
from typing import Dict, List, Tuple, Type
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
        self.assertEqual(self.dispatcher.provenance, {"Task1": 1, "Task2": 1})

    def test_remove_provenance(self):
        task = DummyTaskWorkItem(data="test")
        task._provenance = [("Task1", 1), ("Task2", 2)]
        self.dispatcher.provenance = {"Task1": 1, "Task2": 1}
        with patch.object(self.dispatcher, "_notify_task_completion") as mock_notify:
            self.dispatcher._remove_provenance(task)
            self.assertEqual(self.dispatcher.provenance, {"Task1": 0, "Task2": 0})
            mock_notify.assert_any_call("Task1")
            mock_notify.assert_any_call("Task2")

    def test_notify_task_completion(self):
        notifier = Mock(spec=TaskWorker)
        self.dispatcher.notifiers = {"Task1": [notifier]}
        self.dispatcher._notify_task_completion("Task1")
        self.assertEqual(self.dispatcher.active_tasks, 1)
        notifier.notify.assert_called_once_with("Task1")

    def test_watch(self):
        notifier = Mock(spec=TaskWorker)
        result = self.dispatcher.watch(DummyTaskWorkItem, notifier)
        self.assertTrue(result)
        self.assertIn(DummyTaskWorkItem.__name__, self.dispatcher.notifiers)
        self.assertIn(notifier, self.dispatcher.notifiers[DummyTaskWorkItem.__name__])

    def test_unwatch(self):
        notifier = Mock(spec=TaskWorker)
        self.dispatcher.notifiers = {DummyTaskWorkItem.__name__: [notifier]}
        result = self.dispatcher.unwatch(DummyTaskWorkItem, notifier)
        self.assertTrue(result)
        self.assertNotIn(DummyTaskWorkItem.__name__, self.dispatcher.notifiers)

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
        task = DummyTaskWorkItem(data="test")
        self.dispatcher._execute_task(worker, task)
        worker._pre_consume_work.assert_called_once_with(task)
        self.assertIn(task, [t[2] for t in self.dispatcher.completed_tasks])

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
        self.dispatcher.completed_tasks = deque([(1, worker, task)])
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
        task = DummyTaskWorkItem(data="test")
        future = Mock()
        future.result.return_value = None

        # Set initial conditions
        self.dispatcher.active_tasks = 1
        self.dispatcher.work_queue = Queue()  # Ensure the queue is empty

        with patch.object(self.dispatcher, "_remove_provenance") as mock_remove:
            self.dispatcher._task_completed(task, future)
            mock_remove.assert_called_once_with(task)

        self.assertEqual(self.dispatcher.active_tasks, 0)
        self.assertTrue(self.dispatcher.task_completion_event.is_set())

    def test_task_completed_with_remaining_tasks(self):
        task = DummyTaskWorkItem(data="test")
        future = Mock()
        future.result.return_value = None

        # Set initial conditions
        self.dispatcher.active_tasks = 2
        self.dispatcher.work_queue = Queue()  # Ensure the queue is empty

        with patch.object(self.dispatcher, "_remove_provenance") as mock_remove:
            self.dispatcher._task_completed(task, future)
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
