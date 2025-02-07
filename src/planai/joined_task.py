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
import inspect
import logging
import threading
from abc import abstractmethod
from operator import attrgetter
from typing import Dict, List, Set, Type, get_type_hints

from pydantic import Field, PrivateAttr

from .provenance import ProvenanceChain
from .task import Task, TaskWorker


class InitialTaskWorker(TaskWorker):
    """
    All tasks that are directly submitted to the graph will have this worker as their input provenance.
    """

    def consume_work(self, task: Task):
        pass


class JoinedTaskWorker(TaskWorker):
    """
    A JoinedTaskWorker waits for the completion of a specific set of upstream tasks
    based on the provided join_type.

    It will watch the input provenance for the worker
    specified in join_type and accumulate the results until all tasks with that input provenance
    are completed. Usually that means that the join_type worker needs to be at least two-hops
    upstream from this consumer as otherwise there won't be any results to join, i.e. there will
    ever only be one task with the input provenance of the immediate upstream worker.
    """

    join_type: Type[TaskWorker]
    enable_trace: bool = Field(
        default=False, description="Enable tracing for the join_type in the dashboard."
    )
    _joined_results: Dict[tuple, List[Task]] = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

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
                f"Task {task} does not have a prefix for {self.join_type.__name__} in provenance. "
                f"Existing provenance: {task._provenance}"
            )
        need_watch = False
        with self.lock:
            if prefix not in self._joined_results:
                need_watch = True

            # we accumulate the results by the prefix described by join_type
            # we will deliver them to the sub-class when we get notified
            self._joined_results.setdefault(prefix, []).append(task)

            if need_watch:
                # we will register the watch for the prefix when we see it for the first time.
                if self.enable_trace:
                    self.trace(prefix)
                logging.info("Starting watch for %s in %s", prefix, self.name)
                self.watch(prefix)

    def get_task_class(self):
        """
        Get the Task subclass that this worker can consume.

        This method checks consume_work_joined for the inner type of the List[] parameter.

        Returns:
            Type[Task]: The Task subclass this worker can consume.

        Raises:
            AttributeError: If the consume_work_joined method is not defined.
            TypeError: If the consume_work_joined method is not properly typed.
        """
        # First check for consume_work_joined
        consume_method = getattr(self, "consume_work_joined", None)
        if not consume_method:
            raise AttributeError(
                f"JoinedTaskWorker {self.__class__.__name__} must implement consume_work_joined method"
            )
        signature = inspect.signature(consume_method)
        parameters = signature.parameters
        if len(parameters) != 1:
            raise TypeError(
                f"Method consume_work_joined in {self.__class__.__name__} must accept one parameter"
            )
        type_hints = get_type_hints(consume_method)
        first_param_type = type_hints.get("task", None) or type_hints.get("tasks", None)
        if not first_param_type:
            raise TypeError(
                f"consume_work_joined method in {self.__class__.__name__} must have type hints"
            )
        # Extract inner type from List[Type]
        if (
            not hasattr(first_param_type, "__origin__")
            or first_param_type.__origin__ is not list
        ):
            raise TypeError(
                f"consume_work_joined parameter must be List[Type] but got {first_param_type}"
            )
        return first_param_type.__args__[0]

    def notify(self, prefix: ProvenanceChain):
        with self.lock:
            if prefix not in self._joined_results:
                raise ValueError(f"Task {prefix} does not have any results to join.")
            if not self.unwatch(prefix):
                raise ValueError(
                    f"Trying to remove a Task {prefix} that is not being watched."
                )
            logging.info("Removing watch for %s in %s", prefix, self.name)
            # Sort the tasks based on their _provenance before sending them
            # This allows for better caching and reproducibility
            sorted_tasks = sorted(
                self._joined_results[prefix], key=attrgetter("_provenance")
            )
            del self._joined_results[prefix]

        logging.info(
            "Received all (%d) results for %s in %s",
            len(sorted_tasks),
            prefix,
            self.name,
        )
        with self.work_buffer_context(sorted_tasks[0]):
            self.consume_work_joined(sorted_tasks)

        super().notify(prefix)

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

        # all tasks will have InitialTaskWorker as their very first input provenance
        if self.join_type == InitialTaskWorker:
            return True

        if self._graph is None:
            raise ValueError(
                f"{self.name} is not associated with a Graph. Call set_graph() first."
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

        visited = set()
        if not dfs_search(self, self.join_type, visited):
            raise ValueError(
                f"{self.join_type.__name__} is not found in the upstream path of {self.name}"
            )

        if len(visited) <= 1:
            raise ValueError(
                f"Joining on the immediate upstream worker {self.join_type} in {self.name} will never yield more than one result."
            )
