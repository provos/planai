import threading
from queue import Empty, Queue
from threading import Event, Lock
from typing import TYPE_CHECKING, Dict, List

from .task import TaskWorker, TaskWorkItem

if TYPE_CHECKING:
    from .graph import Graph


class Dispatcher:
    def __init__(self, graph: "Graph"):
        self.graph = graph
        self.work_queue = Queue()
        self.provenance: Dict[str, int] = {}
        self.provenance_lock = Lock()
        self.notifiers: Dict[str, List[TaskWorker]] = {}
        self.notifiers_lock = Lock()
        self.stop_event = Event()
        self.active_tasks = 0
        self.task_completion_event = threading.Event()

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
            print(f"notifying {notifier.name} for task completion: {task_name}")
            self.active_tasks += 1
            future = self.graph._thread_pool.submit(notifier.notify, task_name)
            future.add_done_callback(self._notify_completed)

    def watch(self, task_name: str, notifier: TaskWorker):
        with self.notifiers_lock:
            if task_name not in self.notifiers:
                self.notifiers[task_name] = []
            if notifier not in self.notifiers[task_name]:
                self.notifiers[task_name].append(notifier)

    def unwatch(self, task_name: str, notifier: TaskWorker):
        with self.notifiers_lock:
            if task_name in self.notifiers:
                self.notifiers[task_name].remove(notifier)
                if len(self.notifiers[task_name]) == 0:
                    del self.notifiers[task_name]

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
        worker._pre_consume_work(task)

    def _notify_completed(self, future):
        self.active_tasks -= 1
        if self.active_tasks == 0 and self.work_queue.empty():
            self.task_completion_event.set()

    def _task_completed(self, task: TaskWorkItem, future):
        self._remove_provenance(task)
        self.active_tasks -= 1
        if self.active_tasks == 0 and self.work_queue.empty():
            self.task_completion_event.set()

    def add_work(self, worker: TaskWorker, task: TaskWorkItem):
        self._add_provenance(task)
        self.work_queue.put((worker, task))

    def stop(self):
        self.stop_event.set()

    def wait_for_completion(self):
        self.task_completion_event.wait()
