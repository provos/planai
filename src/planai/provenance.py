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
provenance.py - Provenance Tracking System

This module provides a sophisticated provenance tracking system that manages
the lineage and dependencies of tasks executed across multiple workers.

Key Components:
- ProvenanceTracker: A class dedicated to tracking the provenance of tasks,
  managing task lineage, and coordinating notifications for task dependency completion.
- ProvenanceChain: Represents the lineage of a task as a tuple of (worker_name, task_id) pairs.

Provenance Tracking Principles:
1. Dynamic Counting: Keeps a dynamic count of tasks and notifications, tracking provenance counts in real-time.
2. Atomic Operations: Ensures atomic updates to provenance counts to prevent race conditions in multi-threaded environments.
3. Delayed Provenance Removal: Maintains task provenance until the TaskWorker that consumed it has fully processed it,
   preserving system consistency.
4. Ordered Notifications: Enforces ordered notifications to TaskWorkers upon completion of a provenance chain,
   ensuring sequential task dependencies are respected.
5. Thread-Safety: Leverages locks and atomic operations for safe concurrent operations,
   particularly important for handling complex task dependencies.

Usage:
The ProvenanceTracker class is the central entity for managing task lineage and provenance notifications.
It provides methods to add provenance, remove provenance when tasks complete, and track changes to
task lineages. Integrating this class within a task dispatching system like Dispatcher allows consistent
management of task dependencies and ordered task execution.
"""


import logging
import sys
from collections import defaultdict
from threading import Lock, RLock
from typing import (
    TYPE_CHECKING,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
)

from pydantic import BaseModel

from .task import Task, TaskStatusCallback, TaskWorker

if TYPE_CHECKING:
    from .dispatcher import Dispatcher

# Type aliases
TaskID = int
TaskName = str
ProvenanceChain = Tuple[Tuple[TaskName, TaskID], ...]


class ProvenanceTracker:
    def __init__(self, name: str = "ProvenanceTracker"):
        self.name = name
        self.provenance: DefaultDict[ProvenanceChain, int] = defaultdict(int)
        self.provenance_trace: Dict[ProvenanceChain, list] = {}  # TODO: rename to trace
        self.task_state: Dict[ProvenanceChain, Dict] = {}
        self.notifiers: DefaultDict[ProvenanceChain, List[TaskWorker]] = defaultdict(
            list
        )
        self.provenance_lock = RLock()
        self.notifiers_lock = Lock()

    def __repr__(self):
        return f"{self.name}(p:{len(self.provenance)})"

    def has_provenance(self, provenance: ProvenanceChain) -> bool:
        """Check if a provenance chain is being tracked."""
        with self.provenance_lock:
            return provenance in self.provenance

    def add_state(
        self,
        provenance: ProvenanceChain,
        metadata: Optional[Dict] = None,
        callback: Optional[TaskStatusCallback] = None,
    ):
        """Add metadata and optional callback for a task's provenance."""
        logging.info("Adding state for %s: (%s, %s)", provenance, metadata, callback)
        with self.provenance_lock:
            self.task_state[provenance] = {
                "metadata": metadata or {},
                "callback": callback,
            }

    def get_state(self, provenance: ProvenanceChain) -> Dict:
        """Get task state including metadata and callback."""
        with self.provenance_lock:
            result = self.task_state.get(provenance, {"metadata": {}, "callback": None})
        logging.info("Getting state for %s: %s", provenance, result)
        return result

    def notify_status(
        self,
        worker: TaskWorker,
        task: Task,
        message: Optional[str] = None,
        object: Optional[BaseModel] = None,
    ):
        """Execute callback if registered for this task's initial provenance."""
        logging.info("Notifying status %s on %s", worker.name, task.name)
        if not task._provenance:
            raise ValueError(f"Task {task.name} has no provenance")
        provenance_prefix = (task._provenance[0],)
        state = self.get_state(provenance_prefix)
        if state["callback"]:
            state["callback"](
                state["metadata"], provenance_prefix, worker, task, message, object
            )

    def remove_state(self, prefix: ProvenanceChain, execute_callback: bool = False):
        """Remove metadata and callback for a task's provenance.

        Removes the state information stored for a given task prefix from the internal task_state dictionary.

        Args:
            prefix (ProvenanceChain): The provenance chain prefix identifying the task
            execute_callback (bool, optional): Flag indicating whether to execute callbacks. Defaults to False.

        Note:
            This method is thread-safe as it uses the provenance_lock.
        """
        with self.provenance_lock:
            if prefix in self.task_state:
                logging.info(
                    "Removing state for %s and %sexecuting callbacks",
                    prefix,
                    "not " if not execute_callback else "",
                )
                if execute_callback and self.task_state[prefix]["callback"]:
                    logging.info(
                        "Executing callback for %s to indicate provenance removal",
                        prefix,
                    )
                    # this can be an indication to the user that the task may have failed
                    self.task_state[prefix]["callback"](
                        self.task_state[prefix]["metadata"],
                        prefix,
                        None,
                        None,
                        "Task removed",
                    )
                del self.task_state[prefix]

    def _generate_prefixes(self, task: Task) -> Generator[Tuple, None, None]:
        provenance = task._provenance
        for i in range(1, len(provenance) + 1):
            yield tuple(provenance[:i])

    def get_prefix_by_type(
        self, task: Task, worker_type: Type[TaskWorker]
    ) -> Optional[ProvenanceChain]:
        """
        Get the prefix of the provenance chain that corresponds to the first occurrence of specified worker type.

        Args:
            task (Task): The task object containing provenance information.
            worker_type (Type[TaskWorker]): The type of the worker to extract the prefix for.

        Returns:
            ProvenanceChain: A tuple containing the provenance chain elements corresponding to the specified worker type.
            None: If the worker type is not found in the task's provenance chain.
        """
        for index, entry in enumerate(task._provenance):
            if entry[0] == worker_type.__name__:
                return tuple(task._provenance[: index + 1])
        return None

    def _add_provenance(self, task: Task):
        logging.info(
            "%s: Adding provenance for %s with %s", self, task.name, task._provenance
        )
        for prefix in self._generate_prefixes(task):
            with self.provenance_lock:
                self.provenance[prefix] = self.provenance.get(prefix, 0) + 1
                logging.debug(
                    "%s: +Provenance for %s is now %s",
                    self,
                    prefix,
                    self.provenance[prefix],
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

    def _remove_provenance_check(self, task: Task, worker: TaskWorker) -> set:
        # assumed to be running under lock
        to_notify = set()
        for prefix in self._generate_prefixes(task):
            effective_count = self.provenance[prefix] - 1
            if effective_count == 0:
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
                error_message = f"FATAL ERROR in {self}: Provenance count for prefix {prefix} became negative ({effective_count}). This indicates a serious bug in the provenance tracking system."
                logging.critical(error_message)
                print(error_message, file=sys.stderr)
                sys.exit(1)

        return to_notify

    def _remove_provenance_actual(self, task: Task, worker: TaskWorker):
        for prefix in self._generate_prefixes(task):
            self.provenance[prefix] -= 1
            logging.debug(
                "%s: -Provenance for %s is now %s",
                repr(self),
                prefix,
                self.provenance[prefix],
            )

            effective_count = self.provenance[prefix]
            if effective_count == 0:
                del self.provenance[prefix]

    def _remove_provenance(self, task: Task, worker: TaskWorker):
        logging.info(
            "%s: Removing provenance for %s with %s", self, task.name, task._provenance
        )
        # we are removing provenance in two phases
        # 1. Check whether provenance removal would lead to any notifications
        # 2. If not, then actually remove the provenance
        # we don't remove the provenance when there is a notification pending
        # as the worker owns the provenance until the notification is complete
        with self.provenance_lock:
            to_notify = self._remove_provenance_check(task, worker)

            if to_notify:
                final_notify = []
                for p in to_notify:
                    for notifier in self._get_notifiers_for_prefix(p):
                        final_notify.append((notifier, p))

                if final_notify:
                    logging.info(
                        "Not committing provenance removal for %s (%s) - as we need to wait for the notification to complete before completely removing it",
                        task.name,
                        task._provenance,
                    )
                    self._notify_task_completion(final_notify, worker, task)
                    return

            self._remove_provenance_actual(task, worker)
            # delete metadata for all the prefixes that are no longer in use
            for p in to_notify:
                self.remove_state(p, execute_callback=True)

    def _get_notifiers_for_prefix(self, prefix: ProvenanceChain) -> List[TaskWorker]:
        with self.notifiers_lock:
            return self.notifiers.get(prefix, []).copy()

    def _notify_task_completion(
        self,
        to_notify: List[Tuple[TaskWorker, ProvenanceChain]],
        worker: TaskWorker,
        task: Task,
    ):
        if not to_notify:
            if task is not None:
                raise ValueError(
                    f"Task {task.name} provided without any pending notifications"
                )
            return

        # Sort the to_notify list based on the distance from the last worker in the provenance
        sorted_to_notify = sorted(
            to_notify, key=lambda x: self._get_worker_distance(x[0], worker.name)
        )
        if len(sorted_to_notify) > 1:
            logging.info(
                f"Prioritizing notifications based on distance from {worker.name}: {','.join([str(x[0].name) for x in sorted_to_notify])}"
            )

        notifier, prefix = sorted_to_notify.pop(0)

        logging.info(f"Notifying {notifier.name} that prefix {prefix} is complete")

        if len(sorted_to_notify):
            logging.info(
                f"Postponing {len(sorted_to_notify)} remaining notifications till later: {','.join([str(x[0].name) for x in sorted_to_notify])}"
            )

        # Use a named function instead of a lambda to avoid closure issues
        def task_completed_callback(
            future, worker: TaskWorker = notifier, task: Task = task
        ):
            assert worker._graph and worker._graph._dispatcher
            dispatcher: Dispatcher = worker._graph._dispatcher
            # the timing here is tricky - we might have had multiple rounds of notifications.
            # which means new work might get added to the system on the call to remove provenance
            # but the dispatcher might think that no more work is in the system.
            dispatcher.create_work_hold()
            dispatcher._task_completed(worker, None, future)
            # now that the notification is really complete, we can remove the provenance and notify the next one
            self._remove_provenance(task, worker)
            dispatcher.release_work_hold()

        assert notifier._graph and notifier._graph._dispatcher
        dispatcher: Dispatcher = notifier._graph._dispatcher
        dispatcher.submit_work(
            notifier, [notifier.notify, prefix], task_completed_callback
        )

    def _get_worker_distance(self, worker: TaskWorker, last_worker_name: str) -> float:
        if worker.name == last_worker_name:
            return 0
        if not last_worker_name:
            return float("inf")
        assert worker._graph
        return worker._graph._worker_distances.get(last_worker_name, {}).get(
            worker.name, float("inf")
        )

    def trace(self, prefix: ProvenanceChain):
        logging.info(f"Starting trace for {prefix}")
        with self.provenance_lock:
            if prefix not in self.provenance_trace:
                self.provenance_trace[prefix] = []

    def get_traces(self) -> Dict:
        with self.provenance_lock:
            return self.provenance_trace

    def watch(self, prefix: ProvenanceChain, notifier: TaskWorker) -> bool:
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
