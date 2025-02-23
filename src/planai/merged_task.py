# Copyright 2025 Niels Provos
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

from collections import defaultdict
from typing import Dict, List, Type

from pydantic import Field, PrivateAttr

from .joined_task import JoinedTaskWorker
from .task import Task, TaskWorker


class MergedTaskWorker(JoinedTaskWorker):
    """
    A MergedTaskWorker waits for the completion of a specific set of upstream tasks
    based on the provided join_type and the merged_types. It will use the join_type
    to watch for input provenance in exactly the same way as the JoinedTaskWorker.
    The difference is that the MergedTaskWorker will merge the results of the tasks
    it receives and will provide it as a dictionary of merged results to consume_work_merged.
    The order of the keys in the dictionary will be the same as the order of the
    merged_types. It is not guaranteed that all tasks in merged_types will be present, e.g.
    potentially due to upstream failures.
    """

    merged_types: List[Type[Task]] = Field(
        default_factory=list,
        description="Set of task types that can be merged into this task.",
    )
    _type_order: Dict[Type[Task], int] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        # Create a mapping of task type to position in merged_types for sorting
        self._type_order = {
            task_type: idx for idx, task_type in enumerate(self.merged_types)
        }

    def get_task_class(self) -> Type[Task]:
        raise RuntimeError(
            "get_task_class() is not a valid method for MergedTaskWorker"
        )

    def get_task_classes(self) -> List[Type[Task]]:
        """
        Get the list of task classes that can be merged into this task.
        """
        return list(self.merged_types)

    def consume_work_joined(self, tasks: List[Task]):
        """
        A subclass needs to implement consume_work_joined. It will receive a list of tasks
        that were merged into this task.
        """
        # Sort tasks based on their type's position in merged_types
        sorted_tasks = sorted(tasks, key=lambda t: self._type_order[type(t)])

        # Create a dictionary mapping task names to tasks
        task_dict = defaultdict(list)
        for task in sorted_tasks:
            task_dict[task.name].append(task)

        # Call the consume_work_merged method with the task dictionary
        self.consume_work_merged(tasks=task_dict)

    def consume_work_merged(self, tasks: Dict[str, List[Task]]):
        """
        A subclass needs to implement consume_work_merged. It will receive a dictionary
        keyed by the task name and a list of tasks that were merged into that task.

        The order of keys will be the same as the order of the tasks in merged_types.
        It is not guaranteed that all tasks in merged_types will be present.
        """
        pass

    def _validate_post_register(self) -> None:
        """
        Validate that all the merge_types are being produced by upstream workers.

        Raises:
            ValueError: If not all merge_types are being produced by upstream workers
        """
        upstream_workers: List[TaskWorker] = []

        assert self._graph is not None
        for upstream_worker in self._graph.dependencies.keys():
            if self in self._graph.dependencies[upstream_worker]:
                upstream_workers.append(upstream_worker)

        # Check if all merge_types are in the upstream workers
        for merge_type in self.merged_types:
            if not any(
                merge_type in worker.output_types for worker in upstream_workers
            ):
                raise ValueError(
                    f"{merge_type.__name__} is not being produced by any upstream worker"
                )
