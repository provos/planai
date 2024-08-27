import threading
from queue import Empty, Queue
from threading import Event
from typing import TYPE_CHECKING

from .task import TaskWorker, TaskWorkItem

if TYPE_CHECKING:
    from .dag import DAG


class Dispatcher:
    def __init__(self, dag: "DAG"):
        self.dag = dag
        self.work_queue = Queue()
        self.stop_event = Event()
        self.active_tasks = 0
        self.task_completion_event = threading.Event()

    def dispatch(self):
        while (
            not self.stop_event.is_set()
            or not self.work_queue.empty()
            or self.active_tasks > 0
        ):
            try:
                task, work_item = self.work_queue.get(timeout=1)
                self.active_tasks += 1
                future = self.dag._thread_pool.submit(
                    self._execute_task, task, work_item
                )
                future.add_done_callback(self._task_completed)
            except Empty:
                continue

    def _execute_task(self, task: TaskWorker, work_item: TaskWorkItem):
        task.consume_work(work_item)

    def _task_completed(self, future):
        self.active_tasks -= 1
        if self.active_tasks == 0 and self.work_queue.empty():
            self.task_completion_event.set()

    def add_work(self, task: TaskWorker, work_item: TaskWorkItem):
        self.work_queue.put((task, work_item))

    def stop(self):
        self.stop_event.set()

    def wait_for_completion(self):
        self.task_completion_event.wait()
