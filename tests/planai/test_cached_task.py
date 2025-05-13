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
from typing import List, Type
from unittest.mock import patch

from planai.cached_task import CachedTaskWorker, TaskWorker
from planai.task import Task
from planai.testing import MockCache


# Dummy Task classes
class DummyInputTask(Task):
    data: str


class DummyOutputTask(Task):
    processed_data: str


class DummyOutputTask2(Task):
    processed_data: str


# DummyCachedTaskWorker
class DummyCachedTaskWorker(CachedTaskWorker):
    output_types: List[Type[Task]] = [DummyOutputTask, DummyOutputTask2]

    def consume_work(self, task: DummyInputTask):
        processed_data = f"Processed: {task.data}"
        result = DummyOutputTask(processed_data=processed_data)
        self.publish_work(task=result, input_task=task)

    def extra_cache_key(self, task: Task) -> str:
        return "dummy_extra_key"


class SinkTaskWorker(TaskWorker):
    def consume_work(self, task: DummyOutputTask):
        pass


class SinkTaskWorker2(TaskWorker):
    def consume_work(self, task: DummyOutputTask2):
        pass


class TestCachedTaskWorker(unittest.TestCase):
    def setUp(self):
        self.mock_cache = MockCache()
        self.patcher = patch("planai.cached_task.Cache", return_value=self.mock_cache)
        self.patcher.start()
        self.worker = DummyCachedTaskWorker(
            cache_dir="./test_cache", cache_size_limit=1000000
        )
        self.sink_worker = SinkTaskWorker()
        self.worker.register_consumer(DummyOutputTask, self.sink_worker)

    def tearDown(self):
        self.patcher.stop()

    def test_init(self):
        self.assertIsInstance(self.worker._cache, MockCache)
        self.assertEqual(self.worker.cache_dir, "./test_cache")
        self.assertEqual(self.worker.cache_size_limit, 1000000)

    def test_pre_consume_work_cache_miss(self):
        task = DummyInputTask(data="test")
        cache_key = self.worker._get_cache_key(task)
        with patch(
            "test_cached_task.DummyCachedTaskWorker.consume_work"
        ) as mock_consume:
            with patch(
                "test_cached_task.DummyCachedTaskWorker.pre_consume_work"
            ) as mock_pre_consume:
                with patch(
                    "test_cached_task.DummyCachedTaskWorker.post_consume_work"
                ) as mock_post_consume:
                    self.worker._pre_consume_work(task)
                    mock_pre_consume.assert_called_once_with(task)
                    mock_consume.assert_called_once_with(task)
                    mock_post_consume.assert_called_once_with(task)
        self.assertEqual(self.mock_cache.get_stats[cache_key], 1)
        self.assertEqual(self.mock_cache.set_stats[cache_key], 1)

    def test_pre_consume_work_cache_hit(self):
        task = DummyInputTask(data="test")
        cached_result = [DummyOutputTask(processed_data="Cached: test")]
        cache_key = self.worker._get_cache_key(task)
        self.mock_cache.set(cache_key, [cached_result, task])
        self.mock_cache.clear_stats()

        with patch(
            "test_cached_task.DummyCachedTaskWorker.consume_work"
        ) as mock_consume:
            with patch.object(self.worker, "_publish_cached_results") as mock_publish:
                with patch(
                    "test_cached_task.DummyCachedTaskWorker.pre_consume_work"
                ) as mock_pre_consume:
                    with patch(
                        "test_cached_task.DummyCachedTaskWorker.post_consume_work"
                    ) as mock_post_consume:
                        self.worker._pre_consume_work(task)
                        mock_pre_consume.assert_called_once_with(task)
                        mock_consume.assert_not_called()
                        mock_publish.assert_called_once_with(cached_result, task)
                        mock_post_consume.assert_called_once_with(task)
        self.assertEqual(self.mock_cache.get_stats[cache_key], 1)
        self.assertEqual(self.mock_cache.set_stats[cache_key], 0)

    def test_publish_work(self):
        input_task = DummyInputTask(data="test")

        with self.worker.work_buffer_context(input_task):
            self.worker._pre_consume_work(input_task)

        cache_key = self.worker._get_cache_key(input_task)
        cached_results, cached_task = self.mock_cache.get(cache_key)
        self.assertIsNotNone(cached_results)
        self.assertEqual(len(cached_results), 1)
        self.assertIsNotNone(cached_results[0][0])
        self.assertEqual(cached_results[0][1].processed_data, "Processed: test")
        self.assertEqual(cached_task, input_task)
        self.assertEqual(self.mock_cache.get_stats[cache_key], 2)
        self.assertEqual(self.mock_cache.set_stats[cache_key], 1)

    def test_publish_work_exception(self):
        input_task = DummyInputTask(data="test")

        with patch.object(TaskWorker, "publish_work") as mock_super_publish:
            with patch.object(
                self.mock_cache, "set", side_effect=Exception("Cache error")
            ):
                with self.assertLogs(level="ERROR") as log:
                    with self.worker.work_buffer_context(input_task):
                        self.worker._pre_consume_work(input_task)
                    self.assertIn(
                        "ERROR:root:Error caching results for key", log.output[0]
                    )

            mock_super_publish.assert_called_once()

    def test_single_consumer_different_cached_consumer_name(self):
        task = DummyInputTask(data="test")
        another_sink_worker = SinkTaskWorker2()
        self.worker.register_consumer(DummyOutputTask2, another_sink_worker)
        cached_result = [
            ("DifferentConsumer", DummyOutputTask(processed_data="Cached: test"))
        ]
        cache_key = self.worker._get_cache_key(task)
        self.mock_cache.set(cache_key, [cached_result, task])
        self.mock_cache.clear_stats()

        with patch.object(
            self.worker, "_get_consumer", return_value=self.sink_worker
        ) as mock_get_consumer:
            self.worker._pre_consume_work(task)
            mock_get_consumer.assert_called_once_with(cached_result[0][1])

    def test_two_consumers_invalid_consumer_name(self):
        second_sink_worker = SinkTaskWorker()
        self.worker.register_consumer(DummyOutputTask, second_sink_worker)

        task = DummyInputTask(data="test")
        cached_result = [
            ("NonExistentConsumer", DummyOutputTask(processed_data="Cached: test"))
        ]
        cache_key = self.worker._get_cache_key(task)
        self.mock_cache.set(cache_key, [cached_result, task])
        self.mock_cache.clear_stats()

        with self.assertRaises(ValueError):
            self.worker._pre_consume_work(task)


if __name__ == "__main__":
    unittest.main()
