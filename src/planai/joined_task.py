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
from abc import abstractmethod
from typing import Dict, List, Type

from pydantic import PrivateAttr

from .task import TaskWorker, TaskWorkItem


class JoinedTaskWorker(TaskWorker):
    join_type: Type[TaskWorker]
    _joined_results: Dict[tuple, List[TaskWorkItem]] = PrivateAttr(default_factory=dict)

    @abstractmethod
    def consume_work_joined(self, task: List[TaskWorkItem]):
        """
        A subclass needs to implement consume_work only for the type hint.
        It still needs to call the super() method in consume_work.
        All accumulated results will be delivered to this method.
        """
        pass

    def consume_work(self, task: TaskWorkItem):
        prefix = task.prefix_for_input_task(self.join_type)
        if prefix is None:
            raise ValueError(
                f"Task {task} does not have a prefix for {self.join_type.__name__} in provenance."
            )
        if prefix not in self._joined_results:
            # we will register the watch for the prefix when we see it for the first time.
            if not self.watch(prefix):
                raise ValueError(
                    f"Task {task} does not have a watch for {self.join_type} in provenance."
                )

        # we accumulate the results by the prefix described by join_type
        # we will deliver them to the sub-class when we get notified
        self._joined_results.setdefault(prefix, []).append(task)

    def notify(self, prefix: str):
        if prefix not in self._joined_results:
            raise ValueError(f"Task {prefix} does not have any results to join.")
        self.consume_work_joined(self._joined_results.pop(prefix))
