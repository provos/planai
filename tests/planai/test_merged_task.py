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

import unittest
from typing import Dict, List, Type

from pydantic import PrivateAttr

from planai import InitialTaskWorker
from planai.graph import Graph
from planai.merged_task import MergedTaskWorker
from planai.task import Task, TaskWorker


class Distribute(Task):
    items: List[Task]


class ColorTask(Task):
    color: str


class ShapeTask(Task):
    shape: str


class SizeTask(Task):
    size: str


class MergedResult(Task):
    description: str


class Splitter(TaskWorker):
    output_types: List[Type[Task]] = [ColorTask, ShapeTask, SizeTask]

    def consume_work(self, task: Distribute):
        for item in task.items:
            self.publish_work(item, input_task=task)


class ColorWorker(TaskWorker):
    output_types: List[Type[Task]] = [ColorTask]

    def consume_work(self, task: ColorTask):
        self.publish_work(task, input_task=task)


class ShapeWorker(TaskWorker):
    output_types: List[Type[Task]] = [ShapeTask]

    def consume_work(self, task: ShapeTask):
        self.publish_work(task, input_task=task)


class SizeWorker(TaskWorker):
    output_types: List[Type[Task]] = [SizeTask]

    def consume_work(self, task: SizeTask):
        self.publish_work(task, input_task=task)


class SimpleMergedWorker(MergedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    merged_types: List[Type[Task]] = [ColorTask, ShapeTask, SizeTask]
    _results: List[Dict[str, List[Task]]] = PrivateAttr(default_factory=list)

    def consume_work_merged(self, tasks: Dict[str, List[Task]]):
        self._results.append(tasks)


class TestMergedTaskWorker(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(name="Test Merged Graph")

        self.splitter_worker = Splitter()
        self.color_worker = ColorWorker()
        self.shape_worker = ShapeWorker()
        self.size_worker = SizeWorker()
        self.merged_worker = SimpleMergedWorker()

        self.graph.add_workers(
            self.splitter_worker,
            self.color_worker,
            self.shape_worker,
            self.size_worker,
            self.merged_worker,
        )
        self.graph.set_dependency(self.splitter_worker, self.color_worker).next(
            self.merged_worker
        )
        self.graph.set_dependency(self.splitter_worker, self.shape_worker).next(
            self.merged_worker
        )
        self.graph.set_dependency(self.splitter_worker, self.size_worker).next(
            self.merged_worker
        )

    def test_merged_task_worker(self):
        # Create tasks with the same name to ensure they get merged
        color_task = ColorTask(color="red")
        shape_task = ShapeTask(shape="circle")
        size_task = SizeTask(size="large")

        # Prepare the graph and set entry points
        self.graph.prepare(display_terminal=False)
        self.graph.set_entry(self.color_worker)
        self.graph.set_entry(self.shape_worker)
        self.graph.set_entry(self.size_worker)

        # Add work in different order to test sorting
        initial_tasks = [
            (
                self.splitter_worker,
                Distribute(items=[color_task, shape_task, size_task]),
            ),
        ]

        # Execute the graph
        self.graph.execute(initial_tasks)

        # Verify results
        self.assertEqual(len(self.merged_worker._results), 1)
        merged_tasks = self.merged_worker._results[0]

        # Check that all tasks were merged under their type names
        self.assertEqual(len(merged_tasks), 3)
        self.assertIn("ColorTask", merged_tasks)
        self.assertIn("ShapeTask", merged_tasks)
        self.assertIn("SizeTask", merged_tasks)

        # Verify that tasks are ordered according to merged_types
        # and each type maps to a list containing one task
        self.assertEqual(len(merged_tasks["ColorTask"]), 1)
        self.assertEqual(len(merged_tasks["ShapeTask"]), 1)
        self.assertEqual(len(merged_tasks["SizeTask"]), 1)

        self.assertIsInstance(merged_tasks["ColorTask"][0], ColorTask)
        self.assertIsInstance(merged_tasks["ShapeTask"][0], ShapeTask)
        self.assertIsInstance(merged_tasks["SizeTask"][0], SizeTask)

    def test_incomplete_graph(self):
        # Create a new graph without the size worker
        incomplete_graph = Graph(name="Incomplete Graph")

        color_worker = ColorWorker()
        shape_worker = ShapeWorker()
        merged_worker = SimpleMergedWorker()

        incomplete_graph.add_workers(color_worker, shape_worker, merged_worker)

        # This should raise a ValueError because not all merged_types are provided
        with self.assertRaises(ValueError):
            incomplete_graph.set_dependency(color_worker, merged_worker)
            incomplete_graph.set_dependency(shape_worker, merged_worker)
            incomplete_graph.run([], display_terminal=False)


if __name__ == "__main__":
    unittest.main()
