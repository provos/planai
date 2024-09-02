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
from typing import Set, Type
from unittest.mock import Mock, patch

from planai.task import Task, TaskWorker


class TestTask(unittest.TestCase):
    def setUp(self):
        self.task = Task()

    def test_copy_provenance(self):
        self.task._provenance = [("Task1", 1), ("Task2", 2)]
        copied = self.task.copy_provenance()
        self.assertEqual(copied, [("Task1", 1), ("Task2", 2)])
        self.assertIsNot(copied, self.task._provenance)

    def test_copy_input_provenance(self):
        input_task1 = Task()
        input_task2 = Task()
        self.task._input_provenance = [input_task1, input_task2]
        copied = self.task.copy_input_provenance()
        self.assertEqual(copied, [input_task1, input_task2])
        self.assertIsNot(copied, self.task._input_provenance)

    def test_find_input_task(self):
        class Task1(Task):
            pass

        class Task2(Task):
            pass

        task1 = Task1()
        task2 = Task2()
        self.task._input_provenance = [task1, task2]

        self.assertIs(self.task.find_input_task(Task2), task2)
        self.assertIs(self.task.find_input_task(Task1), task1)
        self.assertIsNone(self.task.find_input_task(Task))


class DummyTask(Task):
    pass


class DummyWorker(TaskWorker):
    output_types: Set[Type[Task]] = {DummyTask}

    def consume_work(self, task: DummyTask):
        pass


class TestTaskWorker(unittest.TestCase):
    def setUp(self):
        self.worker = DummyWorker()

    def test_name_property(self):
        self.assertEqual(self.worker.name, "DummyWorker")

    def test_last_input_task_property(self):
        task = DummyTask()
        self.worker._last_input_task = task
        self.assertIs(self.worker.last_input_task, task)

    def test_set_graph(self):
        graph = Mock()
        self.worker.set_graph(graph)
        self.assertIs(self.worker._graph, graph)

    def test_next(self):
        graph = Mock()
        self.worker._graph = graph
        downstream = DummyWorker()
        result = self.worker.next(downstream)
        graph.set_dependency.assert_called_once_with(self.worker, downstream)
        self.assertIs(result, downstream)

    def test_next_without_graph(self):
        with self.assertRaises(ValueError):
            self.worker.next(DummyWorker())

    def test_watch(self):
        graph = Mock()
        self.worker._graph = graph
        mock_dispatcher = Mock()
        graph._dispatcher = mock_dispatcher
        task = DummyTask()
        task._provenance = [("DummyTask", 1)]
        prefix = task.prefix_for_input_task(DummyTask)
        self.assertIsNotNone(prefix)
        result = self.worker.watch(prefix)
        self.assertIsNotNone(prefix)
        mock_dispatcher.watch.assert_called_once_with(prefix, self.worker, None)
        self.assertEqual(result, mock_dispatcher.watch.return_value)

    def test_unwatch(self):
        graph = Mock()
        self.worker._graph = graph
        mock_dispatcher = Mock()
        graph._dispatcher = mock_dispatcher
        task = DummyTask()
        task._provenance = [("DummyTask", 1)]
        result = self.worker.unwatch(task.prefix_for_input_task(DummyTask))
        mock_dispatcher.unwatch.assert_called_once_with(
            task.prefix_for_input_task(DummyTask), self.worker
        )
        self.assertEqual(result, mock_dispatcher.unwatch.return_value)

    def test_pre_consume_work(self):
        task = DummyTask()
        with patch("test_task.DummyWorker.consume_work") as mock_consume:
            self.worker._pre_consume_work(task)
            self.assertIs(self.worker._last_input_task, task)
            mock_consume.assert_called_once_with(task)

    def test_init(self):
        # This method is empty in the base class, so we just ensure it doesn't raise an exception
        self.worker.init()

    def test_publish_work(self):
        graph = Mock()
        self.worker._graph = graph
        mock_dispatcher = Mock()
        graph._dispatcher = mock_dispatcher

        input_task = DummyTask()
        task = DummyTask()
        self.worker.register_consumer(DummyTask, self.worker)

        self.worker.publish_work(task, input_task)
        self.worker.flush_work_buffer()

        self.assertEqual(len(task._provenance), 1)
        self.assertEqual(task._provenance[0][0], self.worker.name)
        self.assertEqual(len(task._input_provenance), 1)
        self.assertIs(task._input_provenance[0], input_task)
        mock_dispatcher.add_multiple_work.assert_called_once_with([(self.worker, task)])

    def test_publish_work_invalid_type(self):
        class InvalidTask(Task):
            pass

        with self.assertRaises(ValueError):
            self.worker.publish_work(InvalidTask(), input_task=None)

    def test_publish_work_no_consumer(self):
        with self.assertRaises(ValueError):
            self.worker.publish_work(DummyTask(), input_task=None)

    def test_completed(self):
        # This method is empty in the base class, so we just ensure it doesn't raise an exception
        self.worker.completed()

    def test_notify(self):
        # This method is empty in the base class, so we just ensure it doesn't raise an exception
        self.worker.notify("SomeTask")

    def test_dispatch_work(self):
        consumer = Mock()
        self.worker._consumers[DummyTask] = consumer
        task = DummyTask()
        self.worker._dispatch_work(task)
        consumer.consume_work.assert_called_once_with(task)

    def test_validate_task(self):
        class ValidConsumer(TaskWorker):
            def consume_work(self, task: DummyTask):
                pass

        class InvalidConsumer(TaskWorker):
            def consume_work(self, task: Task):
                pass

        valid_consumer = ValidConsumer()
        invalid_consumer = InvalidConsumer()

        success, _ = self.worker.validate_task(DummyTask, valid_consumer)
        self.assertTrue(success)

        success, error = self.worker.validate_task(DummyTask, invalid_consumer)
        self.assertFalse(success)
        self.assertIsInstance(error, TypeError)

    def test_get_task_class(self):
        self.assertEqual(self.worker.get_task_class(), DummyTask)

    def test_register_consumer(self):
        consumer = DummyWorker()
        self.worker.register_consumer(DummyTask, consumer)
        self.assertIs(self.worker._consumers[DummyTask], consumer)

    def test_register_consumer_invalid_task(self):
        with self.assertRaises(TypeError):
            self.worker.register_consumer(int, DummyWorker())

    def test_register_consumer_invalid_consumer(self):
        class InvalidConsumer(TaskWorker):
            def consume_work(self, task: Task):
                pass

        with self.assertRaises(TypeError):
            self.worker.register_consumer(DummyTask, InvalidConsumer())

    def test_register_consumer_duplicate(self):
        consumer = DummyWorker()
        self.worker.register_consumer(DummyTask, consumer)
        with self.assertRaises(ValueError):
            self.worker.register_consumer(DummyTask, consumer)


if __name__ == "__main__":
    unittest.main()
