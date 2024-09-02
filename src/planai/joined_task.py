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
from abc import abstractmethod
from typing import Dict, List, Set, Type

from pydantic import PrivateAttr

from .task import Task, TaskWorker


class JoinedTaskWorker(TaskWorker):
    join_type: Type[TaskWorker]
    _joined_results: Dict[tuple, List[Task]] = PrivateAttr(default_factory=dict)

    @abstractmethod
    def consume_work_joined(self, task: List[Task]):
        """
        A subclass needs to implement consume_work only for the type hint.
        It still needs to call the super() method in consume_work.
        All accumulated results will be delivered to this method.
        """
        pass

    def consume_work(self, task: Task):
        prefix = task.prefix_for_input_task(self.join_type)
        if prefix is None:
            raise ValueError(
                f"Task {task} does not have a prefix for {self.join_type.__name__} in provenance."
            )
        if prefix not in self._joined_results:
            # we will register the watch for the prefix when we see it for the first time.
            logging.info("Starting watch for %s in %s", prefix, self.name)
            self.watch(prefix)

        # we accumulate the results by the prefix described by join_type
        # we will deliver them to the sub-class when we get notified
        self._joined_results.setdefault(prefix, []).append(task)

    def notify(self, prefix: str):
        if prefix not in self._joined_results:
            raise ValueError(f"Task {prefix} does not have any results to join.")
        if not self.unwatch(prefix):
            raise ValueError(
                f"Trying to remove a Task {prefix} that is not being watched."
            )
        self.consume_work_joined(self._joined_results.pop(prefix))

    def _validate_connection(self) -> None:
        """
        Validate that the join_type is a TaskWorker that is upstream
        from the current worker in the graph.

        Raises:
            ValueError: If the join_type is not a subclass of TaskWorker,
                        or not found in the upstream path of this worker.
        """
        super()._validate_connection()

        if not issubclass(self.join_type, TaskWorker):
            raise ValueError(
                f"join_type must be a subclass of TaskWorker, got {self.join_type}"
            )

        if self._graph is None:
            raise ValueError(
                f"{self.__class__.__name__} is not associated with a Graph. Call set_graph() first."
            )

        def dfs_search(
            current_worker: TaskWorker,
            target_type: Type[TaskWorker],
            visited: Set[TaskWorker],
        ) -> bool:
            if isinstance(current_worker, target_type):
                return True

            visited.add(current_worker)

            for upstream_worker in self._graph.dependencies.keys():
                if current_worker in self._graph.dependencies[upstream_worker]:
                    if upstream_worker not in visited:
                        if dfs_search(upstream_worker, target_type, visited):
                            return True

            return False

        if not dfs_search(self, self.join_type, set()):
            raise ValueError(
                f"{self.join_type.__name__} is not found in the upstream path of {self.__class__.__name__}"
            )
