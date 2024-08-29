import logging
import threading
import time
from collections import deque
from queue import Empty, Queue
from threading import Event, Lock
from typing import TYPE_CHECKING, Dict, List, Type

from .task import TaskWorker, TaskWorkItem
from .web_interface import is_quit_requested, run_web_interface

if TYPE_CHECKING:
    from .graph import Graph


class Dispatcher:
    def __init__(self, graph: "Graph", web_port=5000):
        self.graph = graph
        self.work_queue = Queue()
        self.provenance: Dict[str, int] = {}
        self.provenance_lock = Lock()
        self.notifiers: Dict[str, List[TaskWorker]] = {}
        self.notifiers_lock = Lock()
        self.stop_event = Event()
        self.active_tasks = 0
        self.task_completion_event = threading.Event()
        self.web_port = web_port
        self.debug_active_tasks: Dict[str, Tuple[TaskWorker, TaskWorkItem]] = {}
        self.completed_tasks: deque = deque(maxlen=100)  # Keep last 100 completed tasks
        self.task_id_counter = 0
        self.task_lock = threading.Lock()

    def _add_provenance(self, task: TaskWorkItem):
        with self.provenance_lock:
            for task_provenance in task._provenance:
                task_name, _ = task_provenance
                self.provenance[task_name] = self.provenance.get(task_name, 0) + 1

    def _remove_provenance(self, task: TaskWorkItem):
        to_notify = []
        with self.provenance_lock:
            for task_provenance in task._provenance:
                task_name, _ = task_provenance
                self.provenance[task_name] = self.provenance.get(task_name, 0) - 1
                if self.provenance[task_name] <= 0:
                    to_notify.append(task_name)

        for task_name in to_notify:
            self._notify_task_completion(task_name)

    def _notify_task_completion(self, task_name: str):
        to_notify = []
        with self.notifiers_lock:
            if task_name in self.notifiers:
                for notifier in self.notifiers[task_name]:
                    to_notify.append((notifier, task_name))

        for notifier, task_name in to_notify:
            self.active_tasks += 1
            future = self.graph._thread_pool.submit(notifier.notify, task_name)
            future.add_done_callback(self._notify_completed)

    def watch(self, task: Type["TaskWorkItem"], notifier: TaskWorker) -> bool:
        task_name = task.__name__
        with self.notifiers_lock:
            if task_name not in self.notifiers:
                self.notifiers[task_name] = []
            if notifier not in self.notifiers[task_name]:
                self.notifiers[task_name].append(notifier)
                return True
        return False

    def unwatch(self, task: Type["TaskWorkItem"], notifier: TaskWorker) -> bool:
        task_name = task.__name__
        with self.notifiers_lock:
            if task_name in self.notifiers:
                self.notifiers[task_name].remove(notifier)
                if len(self.notifiers[task_name]) == 0:
                    del self.notifiers[task_name]
                return True
        return False

    def dispatch(self):
        while (
            not self.stop_event.is_set()
            or not self.work_queue.empty()
            or self.active_tasks > 0
        ):
            try:
                worker, task = self.work_queue.get(timeout=1)
                self.active_tasks += 1
                future = self.graph._thread_pool.submit(
                    self._execute_task, worker, task
                )

                # Use a named function instead of a lambda to avoid closure issues
                def task_completed_callback(future, task=task):
                    self._task_completed(task, future)

                future.add_done_callback(task_completed_callback)

            except Empty:
                continue

    def _execute_task(self, worker: TaskWorker, task: TaskWorkItem):
        task_id = self._get_next_task_id()
        with self.task_lock:
            self.debug_active_tasks[task_id] = (worker, task)

        try:
            worker._pre_consume_work(task)
        finally:
            with self.task_lock:
                if task_id in self.debug_active_tasks:
                    del self.debug_active_tasks[task_id]
                    self.completed_tasks.appendleft((task_id, worker, task))

    def _get_next_task_id(self):
        with self.task_lock:
            self.task_id_counter += 1
            return self.task_id_counter

    def _task_to_dict(self, worker: TaskWorker, task: TaskWorkItem) -> Dict:
        return {
            "id": self._get_task_id(task),
            "type": type(task).__name__,
            "worker": worker.name,
            "provenance": [f"{worker}_{id}" for worker, id in task._provenance],
            "input_provenance": [
                input_task.model_dump() for input_task in task._input_provenance
            ],
        }

    def get_queued_tasks(self) -> List[Dict]:
        return [
            self._task_to_dict(worker, task) for worker, task in self.work_queue.queue
        ]

    def get_active_tasks(self) -> List[Dict]:
        with self.task_lock:
            return [
                self._task_to_dict(worker, task)
                for task_id, (worker, task) in self.debug_active_tasks.items()
            ]

    def get_completed_tasks(self) -> List[Dict]:
        with self.task_lock:
            return [
                self._task_to_dict(worker, task)
                for task_id, worker, task in self.completed_tasks
            ]

    def _get_task_id(self, task: TaskWorkItem) -> str:
        # Use the last entry in the _provenance list as the task ID
        if task._provenance:
            return f"{task._provenance[-1][0]}_{task._provenance[-1][1]}"
        else:
            # Fallback in case _provenance is empty
            return f"unknown_{id(task)}"

    def _notify_completed(self, future):
        self.active_tasks -= 1
        if self.active_tasks == 0 and self.work_queue.empty():
            self.task_completion_event.set()

    def _task_completed(self, task: TaskWorkItem, future):
        try:
            # This will raise an exception if the task failed
            result = future.result()

            # Handle successful task completion
            logging.info(f"Task {task} completed successfully")

        except Exception as e:
            # Handle task failure
            logging.exception(f"Task {task} failed with exception: {str(e)}")

            # Anything else that needs to be done when a task fails?

        finally:
            # This code will run whether the task succeeded or failed
            self._remove_provenance(task)
            self.active_tasks -= 1
            if self.active_tasks == 0 and self.work_queue.empty():
                self.task_completion_event.set()

    def add_work(self, worker: TaskWorker, task: TaskWorkItem):
        self._add_provenance(task)
        self.work_queue.put((worker, task))

    def stop(self):
        self.stop_event.set()

    def wait_for_completion(self, wait_for_quit=False):
        self.task_completion_event.wait()

        if wait_for_quit:
            while not is_quit_requested():
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.1)

    def start_web_interface(self):
        web_thread = threading.Thread(
            target=run_web_interface, args=(self, self.web_port)
        )
        web_thread.daemon = (
            True  # This ensures the web thread will exit when the main thread exits
        )
        web_thread.start()
