import unittest
from typing import List, Type
from unittest.mock import MagicMock, patch

from planai.cached_task import CachedTaskWorker, TaskWorker
from planai.task import TaskWorkItem


# Mock Cache
class MockCache:
    def __init__(self):
        self.store = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value


# Dummy TaskWorkItem classes
class DummyInputTask(TaskWorkItem):
    data: str


class DummyOutputTask(TaskWorkItem):
    processed_data: str


# DummyCachedTaskWorker
class DummyCachedTaskWorker(CachedTaskWorker):
    output_types: List[Type[TaskWorkItem]] = [DummyOutputTask]

    def consume_work(self, task: DummyInputTask):
        processed_data = f"Processed: {task.data}"
        result = DummyOutputTask(processed_data=processed_data)
        self.publish_work(task=result, input_task=task)

    def extra_cache_key(self) -> str:
        return "dummy_extra_key"


class SinkTaskWorker(TaskWorker):
    def consume_work(self, task: DummyOutputTask):
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

    def test_extra_cache_key(self):
        self.assertEqual(self.worker.extra_cache_key(), "dummy_extra_key")

    def test_pre_consume_work_cache_miss(self):
        task = DummyInputTask(data="test")
        with patch(
            "test_cached_task.DummyCachedTaskWorker.consume_work"
        ) as mock_consume:
            self.worker._pre_consume_work(task)
            mock_consume.assert_called_once_with(task)

    def test_pre_consume_work_cache_hit(self):
        task = DummyInputTask(data="test")
        cached_result = [DummyOutputTask(processed_data="Cached: test")]
        cache_key = self.worker._get_cache_key(task)
        self.mock_cache.set(cache_key, cached_result)

        with patch(
            "test_cached_task.DummyCachedTaskWorker.consume_work"
        ) as mock_consume:
            with patch.object(self.worker, "_publish_cached_results") as mock_publish:
                self.worker._pre_consume_work(task)
                mock_consume.assert_not_called()
                mock_publish.assert_called_once_with(cached_result, task)

    def test_publish_work(self):
        input_task = DummyInputTask(data="test")
        output_task = DummyOutputTask(processed_data="Processed: test")

        self.worker.publish_work(output_task, input_task)

        cache_key = self.worker._get_cache_key(input_task)
        cached_results = self.mock_cache.get(cache_key)
        self.assertIsNotNone(cached_results)
        self.assertEqual(len(cached_results), 1)
        self.assertEqual(cached_results[0], output_task)

    def test_publish_work_exception(self):
        input_task = DummyInputTask(data="test")
        output_task = DummyOutputTask(processed_data="Processed: test")

        with patch.object(TaskWorker, "publish_work") as mock_super_publish:
            with patch.object(
                self.mock_cache, "set", side_effect=Exception("Cache error")
            ):
                with self.assertLogs(level="ERROR") as log:
                    self.worker.publish_work(output_task, input_task)
                    self.assertIn(
                        "ERROR:root:Error caching results for key", log.output[0]
                    )

            mock_super_publish.assert_called_once_with(
                output_task, input_task=input_task
            )


if __name__ == "__main__":
    unittest.main()
