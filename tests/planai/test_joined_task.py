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
import threading
import unittest
from typing import List, Set, Type

from pydantic import PrivateAttr

from planai.dispatcher import Dispatcher
from planai.graph import Graph
from planai.joined_task import JoinedTaskWorker
from planai.task import TaskWorker, TaskWorkItem


class Task1(TaskWorkItem):
    data: str


class Task2(TaskWorkItem):
    data: str
    source: str


class Task3(TaskWorkItem):
    data: List[str]


class TaskWorker1(TaskWorker):
    output_types: Set[Type[TaskWorkItem]] = {Task1}
    _processed_count: int = PrivateAttr(0)

    def consume_work(self, task: Task1):
        output = Task1(data=f"{task.data} start")
        self.publish_work(output, task)
        self._processed_count += 1


class TaskWorker2(TaskWorker):
    output_types: Set[Type[TaskWorkItem]] = {Task2}
    _processed_count: int = PrivateAttr(0)

    def consume_work(self, task: Task1):
        for i in range(3):
            output = Task2(data=f"{task.data}-{i}", source=self.name)
            self.publish_work(output, task)
        self._processed_count += 1


class TaskWorker3(JoinedTaskWorker):
    join_type: Type[TaskWorker] = TaskWorker1
    _processed_count: int = PrivateAttr(0)

    def consume_work(self, task: Task2):
        super().consume_work(task)

    def consume_work_joined(self, tasks: List[Task2]):
        prefixes = set([task.prefix_for_input_task(TaskWorker1) for task in tasks])
        if len(prefixes) != 1:
            raise ValueError("All tasks must have the same prefix", prefixes)
        
        self._processed_count += 1


class TestJoinedTaskWorker(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(name="Test Graph")
        self.dispatcher = Dispatcher(self.graph)
        self.graph._dispatcher = self.dispatcher

        self.worker1 = TaskWorker1()
        self.worker2 = TaskWorker2()
        self.worker3 = TaskWorker3()

        self.graph.add_workers(self.worker1, self.worker2, self.worker3)
        self.graph.set_dependency(self.worker1, self.worker2).next(self.worker3)

    def test_joined_task_worker(self):
        # Create initial tasks
        initial_tasks = [Task1(data=f"Initial {i}") for i in range(3)]

        # Start the dispatcher in a separate thread
        dispatch_thread = threading.Thread(target=self.dispatcher.dispatch)
        dispatch_thread.start()

        # Add initial work
        for task in initial_tasks:
            self.dispatcher.add_work(self.worker1, task)

        # Wait for all work to be processed
        self.dispatcher.wait_for_completion()
        self.dispatcher.stop()
        dispatch_thread.join()

        # Check results
        self.assertEqual(self.worker1._processed_count, 3)
        self.assertEqual(
            self.worker2._processed_count, 3
        )  # 3 initial tasks * 3 outputs from worker1
        self.assertEqual(self.worker3._processed_count, 3)  # 3 joined results

        # Validate that TaskWorker3 accumulated the results correctly
        self.assertEqual(
            len(self.worker3._joined_results), 0
        )  # All joined results should have been processed

        self.graph._thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
