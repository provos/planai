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
import logging
import random
import statistics
import sys
import threading
import time
from collections import defaultdict, deque
from queue import Empty, Queue
from threading import Event, Lock
from typing import (
    TYPE_CHECKING,
    DefaultDict,
    Deque,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
)

from .task import Task, TaskWorker
from .web_interface import is_quit_requested, run_web_interface

if TYPE_CHECKING:
    from .graph import Graph


# Type aliases
TaskID = int
TaskName = str
ProvenanceChain = Tuple[Tuple[TaskName, TaskID], ...]


class NotificationTask(Task):
    pass


def get_inheritance_chain(cls: Type[TaskWorker]) -> List[str]:
    """
    Returns the inheritance chain of a class as a list of strings.
    """
    if len(cls.__mro__) < 3:
        raise ValueError(
            "Class must be derived from TaskWorker and have at least 3 classes in its inheritance chain"
        )
    return [c.__name__ for c in cls.__mro__[:-2]]


class Dispatcher:
    def __init__(self, graph: "Graph", web_port=5000):
        self.graph = graph
        # We have a default Queue for all tasks
        self.work_queue = Queue()

        # And configurable queues for different worker classes
        # This allows us to reduce the maximum number of parallel tasks for specific worker classes, e.g. LLMs.
        self._per_worker_queue: Dict[str, Queue] = {}
        self._per_worker_task_count: Dict[str, int] = {}
        self._per_worker_max_parallel_tasks: Dict[str, int] = {}

        # we are using the work_available Event to signal the dispatcher that there might be work
        self.work_available = threading.Event()

        self.provenance: DefaultDict[ProvenanceChain, int] = defaultdict(int)
        self.provenance_trace: Dict[ProvenanceChain, dict] = {}
        self.notifiers: DefaultDict[ProvenanceChain, List[TaskWorker]] = defaultdict(
            list
        )
        self.provenance_lock = Lock()
        self.notifiers_lock = Lock()
        self.stop_event = Event()
        self._active_tasks = 0
        self.task_completion_event = threading.Event()
        self.web_port = web_port
        self.debug_active_tasks: Dict[int, Tuple[TaskWorker, Task]] = {}
        self.completed_tasks: Deque[Tuple[TaskWorker, Task]] = deque(
            maxlen=100
        )  # Keep last 100 completed tasks
        self.failed_tasks: Deque[Tuple[TaskWorker, Task, str]] = deque(
            maxlen=100
        )  # Keep last 100 failed tasks
        self.worker_stats: Dict[str, List[float]] = defaultdict(
            list
        )  # keeps track of execution times for each worker
        self.total_completed_tasks = 0
        self.total_failed_tasks = 0
        self.task_id_counter = 0
        self.task_lock = threading.Lock()

    @property
    def active_tasks(self):
        with self.task_lock:
            return self._active_tasks

    def _generate_prefixes(self, task: Task) -> Generator[Tuple, None, None]:
        provenance = task._provenance
        for i in range(1, len(provenance) + 1):
            yield tuple(provenance[:i])

    def _add_provenance(self, task: Task):
        for prefix in self._generate_prefixes(task):
            with self.provenance_lock:
                self.provenance[prefix] = self.provenance.get(prefix, 0) + 1
                logging.debug(
                    "+Provenance for %s is now %s", prefix, self.provenance[prefix]
                )
                if prefix in self.provenance_trace:
                    trace_entry = {
                        "worker": task._provenance[-1][0],
                        "action": "adding",
                        "task": task.name,
                        "count": self.provenance[prefix],
                        "status": "",
                    }
                    self.provenance_trace[prefix].append(trace_entry)
                    logging.info(
                        "Tracing: add provenance for %s: %s", prefix, trace_entry
                    )

    def _remove_provenance(self, task: Task):
        to_notify = set()
        for prefix in self._generate_prefixes(task):
            with self.provenance_lock:
                self.provenance[prefix] -= 1
                logging.debug(
                    "-Provenance for %s is now %s", prefix, self.provenance[prefix]
                )

                effective_count = self.provenance[prefix]
                if effective_count == 0:
                    del self.provenance[prefix]
                    to_notify.add(prefix)

                if prefix in self.provenance_trace:
                    # Get the list of notifiers for this prefix
                    with self.notifiers_lock:
                        notifiers = [n.name for n in self.notifiers.get(prefix, [])]

                    status = (
                        "will notify watchers"
                        if effective_count == 0
                        else "still waiting for other tasks"
                    )
                    if effective_count == 0 and notifiers:
                        status += f" (Notifying: {', '.join(notifiers)})"

                    trace_entry = {
                        "worker": task._provenance[-1][0],
                        "action": "removing",
                        "task": task.name,
                        "count": effective_count,
                        "status": status,
                    }
                    self.provenance_trace[prefix].append(trace_entry)
                    logging.info(
                        "Tracing: remove provenance for %s: %s", prefix, trace_entry
                    )

                if effective_count < 0:
                    error_message = f"FATAL ERROR: Provenance count for prefix {prefix} became negative ({effective_count}). This indicates a serious bug in the provenance tracking system."
                    logging.critical(error_message)
                    print(error_message, file=sys.stderr)
                    sys.exit(1)

        for prefix in to_notify:
            self._notify_task_completion(prefix)

    def trace(self, prefix: ProvenanceChain):
        logging.info(f"Starting trace for {prefix}")
        with self.provenance_lock:
            if prefix not in self.provenance_trace:
                self.provenance_trace[prefix] = []

    def watch(
        self, prefix: ProvenanceChain, notifier: TaskWorker, task: Optional[Task] = None
    ) -> bool:
        """
        Watches the given prefix and notifies the specified notifier when the prefix is no longer tracked
        as part of the provenance of all tasks.

        This method sets up a watch on a specific prefix in the provenance chain. When the prefix is
        no longer part of any task's provenance, the provided notifier will be called with the prefix
        as an argument. If the prefix is already not part of any task's provenance, the notifier may
        be called immediately.

        Parameters:
        -----------
        prefix : ProvenanceChain
            The prefix to watch. Must be a tuple representing a part of a task's provenance chain.

        notifier : TaskWorker
            The object to be notified when the watched prefix is no longer in use.
            Its notify method will be called with the watched prefix as an argument.

        task : Task
            The task associated with this watch operation if it was called from consume_work.

        Returns:
        --------
        bool
            True if the notifier was successfully added to the watch list for the given prefix.
            False if the notifier was already in the watch list for this prefix.

        Raises:
        -------
        ValueError
            If the provided prefix is not a tuple.
        """
        if not isinstance(prefix, tuple):
            raise ValueError("Prefix must be a tuple")

        added = False
        with self.notifiers_lock:
            if notifier not in self.notifiers[prefix]:
                self.notifiers[prefix].append(notifier)
                added = True

        if task is not None:
            should_notify = False
            with self.provenance_lock:
                if self.provenance.get(prefix, 0) == 0:
                    should_notify = True

            if should_notify:
                self._notify_task_completion(prefix)

        return added

    def unwatch(self, prefix: ProvenanceChain, notifier: TaskWorker) -> bool:
        if not isinstance(prefix, tuple):
            raise ValueError("Prefix must be a tuple")
        with self.notifiers_lock:
            if prefix in self.notifiers and notifier in self.notifiers[prefix]:
                self.notifiers[prefix].remove(notifier)
                if not self.notifiers[prefix]:
                    del self.notifiers[prefix]
                return True
        return False

    def decrement_active_tasks(self, worker: TaskWorker) -> bool:
        """
        Decrements the count of active tasks by 1.

        Returns:
            bool: True if there are no more active tasks and the work queue is empty, False otherwise.
        """
        inherited_chain = get_inheritance_chain(worker.__class__)
        with self.task_lock:
            for cls_name in inherited_chain:
                if cls_name in self._per_worker_task_count:
                    self._per_worker_task_count[cls_name] -= 1
                    if (
                        self._per_worker_task_count[cls_name]
                        < self._per_worker_max_parallel_tasks[cls_name]
                    ):
                        self.work_available.set()
            self._active_tasks -= 1
            if (
                self._active_tasks == 0
                and self.work_queue.empty()
                and all(q.empty() for q in self._per_worker_queue.values())
            ):
                return True
            return False

    def increment_active_tasks(self, worker: TaskWorker):
        inherited_chain = get_inheritance_chain(worker.__class__)
        with self.task_lock:
            for cls_name in inherited_chain:
                if cls_name in self._per_worker_task_count:
                    self._per_worker_task_count[cls_name] += 1
            self._active_tasks += 1

    def set_max_parallel_tasks(
        self, worker_class: Type[TaskWorker], max_parallel_tasks: int
    ):
        worker_class_name = worker_class.__name__
        with self.task_lock:
            if worker_class_name not in self._per_worker_queue:
                self._per_worker_queue[worker_class_name] = Queue()
                self._per_worker_task_count[worker_class_name] = 0
            self._per_worker_max_parallel_tasks[worker_class_name] = max_parallel_tasks

    def _notify_task_completion(self, prefix: tuple):
        to_notify = []
        with self.notifiers_lock:
            for notifier in self.notifiers[prefix]:
                to_notify.append((notifier, prefix))

        for notifier, prefix in to_notify:
            logging.info(f"Notifying {notifier.name} that prefix {prefix} is complete")
            self.increment_active_tasks(notifier)

            # Use a named function instead of a lambda to avoid closure issues
            def task_completed_callback(future, worker=notifier):
                self._task_completed(worker, None, future)

            future = self.graph._thread_pool.submit(notifier.notify, prefix)
            future.add_done_callback(task_completed_callback)

    def _dispatch_once(self) -> bool:
        queues = [(None, self.work_queue)] + list(self._per_worker_queue.items())
        random.shuffle(queues)
        for name, queue in queues:
            try:
                worker, task = queue.get_nowait()
                if name is not None:
                    # Check if the worker has reached its maximum parallel tasks
                    inheritance_chain = get_inheritance_chain(worker.__class__)
                    if name in inheritance_chain:
                        with self.task_lock:
                            if (
                                self._per_worker_task_count[name]
                                >= self._per_worker_max_parallel_tasks[name]
                            ):
                                queue.put((worker, task))
                                continue
                self.increment_active_tasks(worker)
                future = self.graph._thread_pool.submit(
                    self._execute_task, worker, task
                )

                # Use a named function instead of a lambda to avoid closure issues
                def task_completed_callback(future, worker=worker, task=task):
                    self._task_completed(worker, task, future)

                future.add_done_callback(task_completed_callback)
                return True

            except Empty:
                pass  # this queue is empty, try the next one

        return False

    def dispatch(self):
        while True:
            # making sure that we can access active_tasks in a thread-safe way
            with self.task_lock:
                if (
                    self.stop_event.is_set()
                    and self.work_queue.empty()
                    and all(q.empty() for q in self._per_worker_queue.values())
                    and self._active_tasks == 0
                ):
                    break
            if self._dispatch_once():
                continue

            self.work_available.wait(timeout=0.1)
            self.work_available.clear()

    def _execute_task(self, worker: TaskWorker, task: Task):
        task_id = self._get_next_task_id()
        with self.task_lock:
            self.debug_active_tasks[task_id] = (worker, task)

        # keep track of some basic timing information
        task._start_time = time.time()
        task._end_time = None

        try:
            # since we are storing a lot of references to the task, we need to make sure
            # that we are not storing the same task object in multiple places
            worker._pre_consume_work(task.model_copy())
        except Exception:
            raise  # Re-raise the caught exception
        finally:
            with self.task_lock:
                if task_id in self.debug_active_tasks:
                    del self.debug_active_tasks[task_id]

            task._end_time = time.time()

    def _get_next_task_id(self) -> int:
        with self.task_lock:
            self.task_id_counter += 1
            return self.task_id_counter

    def _task_to_dict(self, worker: TaskWorker, task: Task, error: str = "") -> Dict:
        data = {
            "id": self._get_task_id(task),
            "type": type(task).__name__,
            "worker": worker.name,
            "start_time": task._start_time,
            "end_time": task._end_time,
            "provenance": [f"{worker}_{id}" for worker, id in task._provenance],
            "input_provenance": [
                {input_task.name: input_task.model_dump()}
                for input_task in task._input_provenance
            ],
        }
        if error:
            data["error"] = error
        return data

    def get_traces(self) -> Dict:
        with self.provenance_lock:
            return self.provenance_trace

    def get_queued_tasks(self) -> List[Dict]:
        work_items = []
        with self.task_lock:
            work_items.extend(self.work_queue.queue)
            for queue in self._per_worker_queue.values():
                work_items.extend(queue.queue)
            return [self._task_to_dict(worker, task) for worker, task in work_items]

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
                for worker, task in self.completed_tasks
            ]

    def get_failed_tasks(self) -> List[Dict]:
        with self.task_lock:
            return [
                self._task_to_dict(worker, task, error)
                for worker, task, error in self.failed_tasks
            ]

    def get_execution_statistics(self):
        stats = {}
        with self.task_lock:
            for worker, times in self.worker_stats.items():
                if times:
                    stats[worker] = {
                        "min": min(times),
                        "median": statistics.median(times),
                        "max": max(times),
                        "stdDev": statistics.stdev(times) if len(times) > 1 else 0,
                        "count": len(times),
                    }
                else:
                    stats[worker] = {
                        "min": 0,
                        "median": 0,
                        "max": 0,
                        "stdDev": 0,
                        "count": 0,
                    }
        return stats

    def _get_task_id(self, task: Task) -> str:
        # Use the last entry in the _provenance list as the task ID
        if task._provenance:
            return f"{task._provenance[-1][0]}_{task._provenance[-1][1]}"
        else:
            # Fallback in case _provenance is empty
            return f"unknown_{id(task)}"

    def _task_completed(self, worker: TaskWorker, task: Optional[Task], future):
        success: bool = False
        error_message: str = ""
        try:
            # This will raise an exception if the task failed
            _ = future.result()

            # Handle successful task completion
            if task:
                logging.info(f"Task {task.name} completed successfully")
            else:
                logging.info(
                    f"Notification for worker {worker.name} completed successfully"
                )
            success = True

            # collect execution stats
            if task and task._start_time and task._end_time:
                execution_time = task._end_time - task._start_time
                with self.task_lock:
                    self.worker_stats[worker.name].append(execution_time)

        except Exception as e:
            # Handle task failure
            error_message = str(e)
            if task:
                logging.exception(f"Task {task.name} failed with exception: {str(e)}")
            else:
                logging.exception(
                    f"Notification for worker {worker.name} failed with exception: {str(e)}"
                )

            # Anything else that needs to be done when a task fails?

        finally:
            # This code will run whether the task succeeded or failed
            if not success:
                if task is not None:
                    # Determine whether we should retry the task
                    if worker.num_retries > 0:
                        if task.retry_count < worker.num_retries:
                            task.increment_retry_count()
                            self.decrement_active_tasks(worker)
                            self._add_to_queue(worker, task)
                            logging.info(
                                f"Retrying task {task.name} for the {task.retry_count} time"
                            )
                            return

                with self.task_lock:
                    self.failed_tasks.appendleft(
                        (worker, task if task else NotificationTask(), error_message)
                    )
                    self.total_failed_tasks += 1

                if task:
                    logging.error(
                        f"Task {task.name} failed after {task.retry_count} retries"
                    )
            else:
                with self.task_lock:
                    self.completed_tasks.appendleft(
                        (worker, task if task else NotificationTask())
                    )
                    self.total_completed_tasks += 1

            if task:
                self._remove_provenance(task)

            if self.decrement_active_tasks(worker):
                self.task_completion_event.set()

    def add_work(self, worker: TaskWorker, task: Task):
        task_copy = task.model_copy()
        self._add_provenance(task_copy)
        self._add_to_queue(worker, task_copy)

    def add_multiple_work(self, work_items: List[Tuple[TaskWorker, Task]]):
        # the ordering of adding provenance first is important for join tasks to
        # work correctly. Otherwise, caching may lead to fast execution of tasks
        # before all the provenance is added.
        work_items = [(worker, task.model_copy()) for worker, task in work_items]
        for worker, task in work_items:
            self._add_provenance(task)
        for worker, task in work_items:
            self._add_to_queue(worker, task)

    def _add_to_queue(self, worker: TaskWorker, task: Task):
        inheritance_chain = get_inheritance_chain(worker.__class__)
        per_task_queue = self.work_queue
        with self.task_lock:
            for cls_name in inheritance_chain:
                if cls_name in self._per_worker_queue:
                    per_task_queue = self._per_worker_queue[cls_name]
                    break
        per_task_queue.put((worker, task))
        self.work_available.set()

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
