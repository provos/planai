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
from typing import TYPE_CHECKING, DefaultDict, Dict, Generator, List, Optional, Tuple

from .task import Task, TaskWorker

if TYPE_CHECKING:
    from .dispatcher import Dispatcher

# Type aliases
TaskID = int
TaskName = str
ProvenanceChain = Tuple[Tuple[TaskName, TaskID], ...]


class ProvenanceTracker:
    def __init__(self):
        self.provenance: DefaultDict[ProvenanceChain, int] = defaultdict(int)
        self.provenance_trace: Dict[ProvenanceChain, list] = {}
        self.notifiers: DefaultDict[ProvenanceChain, List[TaskWorker]] = defaultdict(
            list
        )
        self.provenance_lock = RLock()
        self.notifiers_lock = Lock()

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

    def _remove_provenance(self, task: Task, worker: TaskWorker):
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

        if to_notify:
            final_notify = []
            for p in to_notify:
                for notifier in self._get_notifiers_for_prefix(p):
                    final_notify.append((notifier, p))

            if final_notify:
                logging.info(
                    "Re-adding provenance for %s - as we need to wait for the notification to complete before completely removing it",
                    task.name,
                )
                self._add_provenance(task)
                self._notify_task_completion(final_notify, worker, task)

    def _get_notifiers_for_prefix(self, prefix: ProvenanceChain) -> List[TaskWorker]:
        with self.notifiers_lock:
            return self.notifiers.get(prefix, []).copy()

    def _notify_task_completion(
        self,
        to_notify: List[Tuple[TaskWorker, ProvenanceChain]],
        worker: TaskWorker,
        task: Optional[Task],
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
            future, worker=notifier, to_notify=sorted_to_notify, task=task
        ):
            dispatcher: Dispatcher = worker._graph._dispatcher
            dispatcher._task_completed(worker, None, future)
            # now that the notification is really complete, we can remove the provenance and notify the next one
            self._remove_provenance(task, worker)

        dispatcher: Dispatcher = notifier._graph._dispatcher
        dispatcher.submit_work(
            notifier, [notifier.notify, prefix], task_completed_callback
        )

    def _get_worker_distance(self, worker: TaskWorker, last_worker_name: str) -> int:
        if worker.name == last_worker_name:
            return 0
        if not last_worker_name:
            return float("inf")
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
                self._notify_task_completion([(notifier, prefix)], notifier, None)

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
