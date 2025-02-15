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

"""
dispatcher.py - Task Dispatching System

This module implements a sophisticated task dispatching system responsible for managing
the execution of tasks across multiple workers. It integrates with a separate provenance
tracking system to maintain a record of task lineage and dependencies.

Key Components:
- Dispatcher: The main class for handling task queues, worker assignment, and interaction
  with the provenance tracking system.
- ProvenanceTracker: External module responsible for maintaining the lineage of tasks.
- ProvenanceChain: Represents the lineage of a task as a tuple of (worker_name, task_id) pairs.
- TaskWorker: Base class for workers that process tasks.
- JoinedTaskWorker: A special type of worker that waits for multiple upstream tasks to complete.

Task Dispatching:
The Dispatcher manages tasks by assigning them to available workers, tracking their progress,
and ensuring tasks are completed in the correct order. It uses the following mechanisms:

1. Task Queues: Manages various queues for tasks, allowing for configurable parallelism limits
   on a per-worker type basis.

2. Task Lifecycle Management: Handles task state transitions from queued to active, and finally
   to completed or failed, capturing necessary statistics and logs.

3. User Input Handling: Provides facilities for tasks to request and receive user input asynchronously.

Integration with Provenance Tracking:
The Dispatcher relies on a separate ProvenanceTracker module to handle the intricacies of task
lineage and provenance. This enables:

1. Provenance Counting: Provenance count is externally managed to ensure accurate task tracking.

2. Notification Management: Ordered notifications for task dependencies are handled by the ProvenanceTracker,
   ensuring that all dependencies are satisfied before continuing to the next task.

Usage:
The Dispatcher class is the main interface for task management within PlanAI.
It coordinates with ProvenanceTracker to maintain the integrity of task dependencies while executing tasks
across various workers effectively.
"""


import logging
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event, RLock
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple, Type

from .provenance import ProvenanceChain
from .stats import WorkerStat
from .task import Task, TaskWorker
from .user_input import UserInputRequest
from .web_interface import is_quit_requested, run_web_interface, shutdown_web_interface

if TYPE_CHECKING:
    from .graph import Graph


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
    def __init__(
        self, graph: "Graph", web_port: int = 5000, start_thread_pool: bool = True
    ):
        self._thread_pool = ThreadPoolExecutor() if start_thread_pool else None

        self._graphs = [graph]
        self._web_port = web_port
        self._web_thread = None

        self.task_lock = RLock()
        # We have a default Queue for all tasks
        self.work_queue = Queue()

        # And configurable queues for different worker classes
        # This allows us to reduce the maximum number of parallel tasks for specific worker classes, e.g. LLMs.
        self._per_worker_queue: Dict[str, Queue] = {}
        self._per_worker_task_count: Dict[str, int] = {}
        self._per_worker_max_parallel_tasks: Dict[str, int] = {}

        # we are using the work_available Event to signal the dispatcher that there might be work
        self.work_available = threading.Event()

        self.stop_event = Event()
        self.task_completion_event = Event()
        self._num_active_tasks = 0
        self._active_tasks: Dict[int, Tuple[TaskWorker, Task]] = {}
        self._aborted_prefixes: List[Tuple[ProvenanceChain, "Graph"]] = []
        self.completed_tasks: Deque[Tuple[TaskWorker, Task]] = deque(
            maxlen=100
        )  # Keep last 100 completed tasks
        self.failed_tasks: Deque[Tuple[TaskWorker, Task, str]] = deque(
            maxlen=100
        )  # Keep last 100 failed tasks
        self.worker_stats: Dict[str, WorkerStat] = defaultdict(
            WorkerStat
        )  # keeps track of execution times for each worker
        self.total_completed_tasks = 0
        self.total_failed_tasks = 0
        self.task_id_counter = 0

        # managing user requests
        self.user_input_requests = Queue()
        self.user_pending_requests: Dict[str, UserInputRequest] = {}

        # Add new log management
        self._log_queue = deque(maxlen=1000)  # Keep last 1000 log messages
        self._log_lock = RLock()

        self._shutdown_initiated = False

        self._dispatch_thread = None

        logging.info(
            "Dispatcher initialized with %d threads",
            self._thread_pool._max_workers if self._thread_pool else 0,
        )

    def register_graph(self, graph: "Graph"):
        if graph in self._graphs:
            raise ValueError("Graph already registered")
        self._graphs.append(graph)

    def deregister_graph(self, graph: "Graph"):
        if graph not in self._graphs:
            raise ValueError("Graph not registered")
        # we may need to consider removing the associated tasks from the queues
        # for now, we assume that this gets called after the main graph initiated a shutdown
        # on the dispatcher
        self._graphs.remove(graph)

    @property
    def active_tasks(self):
        with self.task_lock:
            return self._num_active_tasks

    def graph(self, index: int = 0) -> "Graph":
        if index >= len(self._graphs):
            raise ValueError(f"Graph index {index} out of range")
        return self._graphs[index]

    def _check_all_tasks_completed(self):
        with self.task_lock:
            if (
                self._num_active_tasks == 0
                and self.work_queue.empty()
                and all(q.empty() for q in self._per_worker_queue.values())
            ):
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
            self.worker_stats[worker.name].decrement_active()
            for cls_name in inherited_chain:
                if cls_name in self._per_worker_task_count:
                    self._per_worker_task_count[cls_name] -= 1
                    if (
                        self._per_worker_task_count[cls_name]
                        < self._per_worker_max_parallel_tasks[cls_name]
                    ):
                        self.work_available.set()
            self._num_active_tasks -= 1
            if self._num_active_tasks < 0:
                raise ValueError("Number of active tasks is negative")
            if (
                self._num_active_tasks == 0
                and self.work_queue.empty()
                and all(q.empty() for q in self._per_worker_queue.values())
            ):
                logging.info("All tasks completed")
                return True
            return False

    def increment_active_tasks(self, worker: TaskWorker):
        inherited_chain = get_inheritance_chain(worker.__class__)
        with self.task_lock:
            self.worker_stats[worker.name].increment_active()
            for cls_name in inherited_chain:
                if cls_name in self._per_worker_task_count:
                    self._per_worker_task_count[cls_name] += 1
            self._num_active_tasks += 1

    def set_max_parallel_tasks(
        self, worker_class: Type[TaskWorker], max_parallel_tasks: int
    ):
        worker_class_name = worker_class.__name__
        with self.task_lock:
            if worker_class_name not in self._per_worker_queue:
                self._per_worker_queue[worker_class_name] = Queue()
                self._per_worker_task_count[worker_class_name] = 0
            self._per_worker_max_parallel_tasks[worker_class_name] = max_parallel_tasks

    def _dispatch_once(self) -> bool:
        # Add check for shutdown at the start
        if self._shutdown_initiated:
            return False

        # Check if there are user input requests
        self._dispatch_user_requests()

        queues = [(None, self.work_queue)] + list(self._per_worker_queue.items())
        random.shuffle(queues)
        for name, queue in queues:
            try:
                worker, task = queue.get_nowait()

                # Check if the task has been aborted
                if self._is_provenance_aborted(worker, task._provenance):
                    logging.info(
                        "Skipping task %s with %s due to aborted provenance chain",
                        task.name,
                        task._provenance,
                    )
                    worker._graph._provenance_tracker._remove_provenance(task, worker)
                    continue

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
                with self.task_lock:
                    self.worker_stats[worker.name].decrement_queued()

                # Use a named function instead of a lambda to avoid closure issues
                def task_completed_callback(future, worker=worker, task=task):
                    self._task_completed(worker, task, future)

                self.submit_work(
                    worker, [self._execute_task, worker, task], task_completed_callback
                )
                return True

            except Empty:
                pass  # this queue is empty, try the next one

        return False

    def submit_work(
        self,
        worker: TaskWorker,
        arguments: List[Any],
        task_completed_callback: callable,
    ):
        self.increment_active_tasks(worker)
        future = self._thread_pool.submit(*arguments)
        future.add_done_callback(task_completed_callback)

    def _dispatch_user_requests(self):
        """
        Handles the dispatching and processing of user input requests and their results.

        This function is responsible for the following:
        1. Checking for any pending user input requests and logging the instructions required
           for each. The requests are stored in a dictionary for tracking by their task ID.
        2. Checking for completed user input results, processing each result by matching it
           with its corresponding request, and then delivering the user's input back to the
           TaskWorker that requested it.

        The function operates in a non-blocking manner, retrieving requests and results from
        their respective queues only if they are available.
        """
        while not self.user_input_requests.empty():
            request: UserInputRequest = (
                self.user_input_requests.get_nowait()
            )  # Non-blocking retrieval
            logging.info(
                "User Input Required: Task ID %s - %s",
                request.task_id,
                request.instruction,
            )
            with self.task_lock:
                self.user_pending_requests[request.task_id] = request

    def dispatch(self):
        while True:
            # making sure that we can access active_tasks in a thread-safe way
            with self.task_lock:
                if (
                    self.stop_event.is_set()
                    and self.work_queue.empty()
                    and all(q.empty() for q in self._per_worker_queue.values())
                    and self._num_active_tasks == 0
                ) or self._shutdown_initiated:
                    logging.info("Stopping dispatcher")
                    break
            if self._dispatch_once():
                continue

            self.work_available.wait(timeout=0.1)
            self.work_available.clear()

    def _execute_task(self, worker: TaskWorker, task: Task):
        task_id = self._get_next_task_id()
        with self.task_lock:
            self._active_tasks[task_id] = (worker, task)

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
                if task_id in self._active_tasks:
                    del self._active_tasks[task_id]

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

    def get_graphs(self) -> List["Dict"]:
        return [
            {"index": index, "name": graph.name}
            for index, graph in enumerate(self._graphs)
        ]

    def get_traces(self, index: int = 0) -> Dict:
        if index >= len(self._graphs):
            return {}
        return self._graphs[index]._provenance_tracker.get_traces()

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
                for task_id, (worker, task) in self._active_tasks.items()
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
        stats = {
            worker: stat.get_statistics() for worker, stat in self.worker_stats.items()
        }
        return stats

    def get_user_input_requests(self) -> List[Dict]:
        with self.task_lock:
            return [
                {
                    "task_id": request.task_id,
                    "instruction": request.instruction,
                    "accepted_mime_types": request.accepted_mime_types,
                }
                for request in self.user_pending_requests.values()
            ]

    def set_user_input_result(
        self, task_id: str, result: Any, mime_type: Optional[str] = None
    ):
        logging.info(
            "User Input Received: Task ID %s - Result: %s (MIME Type: %s)",
            task_id,
            result[:30] if result else "<None>",
            mime_type,
        )

        # Locate the request for task_id and inform the TaskWorker
        with self.task_lock:
            if task_id in self.user_pending_requests:
                request: UserInputRequest = self.user_pending_requests.pop(task_id)
                # Provide the result to the requesting TaskWorker's queue
                request._response_queue.put(
                    (
                        result,
                        mime_type,
                    )
                )

    def _get_task_id(self, task: Task) -> str:
        # Use the last entry in the _provenance list as the task ID
        if task._provenance:
            return f"{task._provenance[-1][0]}_{task._provenance[-1][1]}"
        else:
            # Fallback in case _provenance is empty
            return f"unknown_{id(task)}"

    def create_work_hold(self):
        """Artificially increment the number of active tasks to prevent completion."""
        with self.task_lock:
            self._num_active_tasks += 1

    def release_work_hold(self):
        """Release the artificial hold we created and check whether work remains."""
        with self.task_lock:
            self._num_active_tasks -= 1
            if self._num_active_tasks == 0 and self._check_all_tasks_completed():
                self.task_completion_event.set()

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
                    self.worker_stats[worker.name].add_completion_time(execution_time)

        except Exception as e:
            # Handle task failure
            error_message = str(e)
            if task:
                logging.exception(
                    "Worker %s on Task %s failed with exception: %s",
                    worker.name,
                    task.name,
                    str(e),
                )
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
                    self.worker_stats[worker.name].increment_failed()
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
                    self.worker_stats[worker.name].increment_completed()

            if task:
                worker._graph._provenance_tracker._remove_provenance(task, worker)

            if self.decrement_active_tasks(worker):
                self.task_completion_event.set()

    def abort_work(self, graph: "Graph", ProvenanceChain: List[Tuple[str, str]]):
        with self.task_lock:
            self._aborted_prefixes.append((tuple(ProvenanceChain), graph))

    def _is_provenance_aborted(
        self, worker: TaskWorker, provenance: ProvenanceChain
    ) -> bool:
        """
        Checks if any prefix in self._aborted_prefixes matches the given provenance chain.

        Args:
            provenance: The provenance chain to check

        Returns:
            bool: True if the provenance chain has an aborted prefix
        """
        with self.task_lock:
            for aborted_prefix, graph in self._aborted_prefixes:
                if worker._graph == graph and provenance[: len(aborted_prefix)] == list(
                    aborted_prefix
                ):
                    return True
        return False

    def add_work(self, worker: TaskWorker, task: Task):
        # Skip adding the task if its provenance chain has been aborted
        if self._is_provenance_aborted(worker, task._provenance):
            logging.info("Skipping task %s due to aborted provenance chain", task.name)
            return

        task_copy = task.model_copy()
        worker._graph._provenance_tracker._add_provenance(task_copy)

        self._add_to_queue(worker, task_copy)

    def add_multiple_work(self, work_items: List[Tuple[TaskWorker, Task]]):
        for worker, task in work_items:
            self.add_work(worker, task)

    def _add_to_queue(self, worker: TaskWorker, task: Task):
        if self._shutdown_initiated:
            return

        inheritance_chain = get_inheritance_chain(worker.__class__)
        per_task_queue = self.work_queue
        with self.task_lock:
            self.worker_stats[worker.name].increment_queued()
            for cls_name in inheritance_chain:
                if cls_name in self._per_worker_queue:
                    per_task_queue = self._per_worker_queue[cls_name]
                    break
        per_task_queue.put((worker, task))
        self.work_available.set()

    def wait_for_completion(self, timeout: float = None, wait_for_quit=False):
        """
        Wait for all tasks to complete or timeout.

        Args:
            timeout (float): Maximum time to wait in seconds
            wait_for_quit (bool): Whether to wait for quit request
        """
        completed = self.task_completion_event.wait(timeout=timeout)
        if not completed:
            logging.warning("Dispatcher wait_for_completion timed out")
            wait_for_quit = False

        if wait_for_quit:
            while not is_quit_requested():
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.1)

        if self._thread_pool:
            logging.info("Shutting down thread pool")
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
            logging.info("Thread pool shut down")
            self._thread_pool = None

        if self._web_thread:
            logging.info("Stopping web thread")
            try:
                shutdown_web_interface()
            except Exception as e:
                logging.warning(f"Error shutting down web interface: {e}")
            self._web_thread.join(timeout=0.5)
            self._web_thread = None

    def has_dashboard(self):
        """
        Check if the dashboard web thread is running.

        Returns:
            bool: True if the dashboard web thread is running, False otherwise.
        """
        return self._web_thread is not None

    def start_web_interface(self):
        self._web_thread = threading.Thread(
            target=run_web_interface, args=(self, self._web_port)
        )
        self._web_thread.daemon = (
            True  # This ensures the web thread will exit when the main thread exits
        )
        self._web_thread.start()

    def request_user_input(
        self,
        task_id: str,
        instruction: str,
        accepted_mime_types: List[str] = ["text/html"],
    ) -> Any:
        request = UserInputRequest(
            task_id=task_id,
            instruction=instruction,
            accepted_mime_types=accepted_mime_types,
        )
        self.user_input_requests.put(request)
        return request._response_queue.get()  # Block until result is available

    def add_log(self, message: str) -> None:
        """Add a log message to the queue."""
        with self._log_lock:
            self._log_queue.append({"timestamp": time.time(), "message": message})

    def get_logs(self) -> List[Dict]:
        """Get all logs and clear the queue."""
        with self._log_lock:
            logs = list(self._log_queue)
            self._log_queue.clear()
            return logs

    def initiate_shutdown(self):
        """Initiates a graceful shutdown of the dispatcher."""
        self._shutdown_initiated = True
        self.task_completion_event.set()
        logging.info("Dispatcher shutdown initiated")

    def start(self):
        """Start the dispatcher thread if not already running."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor()
        if self._dispatch_thread is None:
            self._dispatch_thread = threading.Thread(target=self.dispatch)
            self._dispatch_thread.daemon = True
            self._dispatch_thread.start()
            logging.info("Dispatcher thread started")
        else:
            logging.warning("Dispatcher thread already running")

    def stop(self, timeout: float = None):
        """
        Stop the dispatcher and wait for thread completion.

        Args:
            timeout (float, optional): Maximum time to wait for thread completion in seconds
        """
        self.stop_event.set()
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=timeout)
            if self._dispatch_thread.is_alive():
                logging.warning("Dispatcher thread did not stop within timeout")
            else:
                logging.info("Dispatcher thread stopped")
            self._dispatch_thread = None
