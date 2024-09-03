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
import hashlib
import logging
import threading
from typing import List

from diskcache import Cache
from pydantic import Field, PrivateAttr

from .task import Task, TaskWorker


class CachedTaskWorker(TaskWorker):
    cache_dir: str = Field("./cache", description="Directory to store the cache")
    cache_size_limit: int = Field(
        25_000_000_000, description="Cache size limit in bytes"
    )
    _cache: Cache = PrivateAttr()
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, **data):
        super().__init__(**data)
        self._cache = Cache(self.cache_dir, size_limit=self.cache_size_limit)

    def _pre_consume_work(self, task: Task):
        with self.work_buffer_context(task):
            self.pre_consume_work(task)

            cache_key = self._get_cache_key(task)
            cached_results = self._cache.get(cache_key)

            if cached_results is not None:
                logging.info("Cache hit for %s with key: %s", self.name, cache_key)
                self._publish_cached_results(cached_results, task)
            else:
                logging.info("Cache miss for %s with key: %s", self.name, cache_key)
                self.consume_work(task)

            self.post_consume_work(task)

    def pre_consume_work(self, task: Task):
        """
        This method is called before consuming the work item. It will be called even if the task has been cached.
        It can be used for state manipulation, e.g. changing state for a class specific cache key.

        Args:
            task (Task): The work item to be consumed.

        Returns:
            None
        """
        pass

    def post_consume_work(self, task: Task):
        """
        This method is called after consuming a work item in the task. It will be called even if the task has been cached.
        It can be used for state manipulation, e.g. changing state for a class specific cache key.

        Args:
            task (Task): The work item that was consumed.

        Returns:
            None
        """
        pass

    def _get_cache_key(self, task: Task) -> str:
        """Generate a unique cache key for the input task."""
        task_dict = task.model_dump()
        task_str = str(sorted(task_dict.items()))  # Ensure consistent ordering
        task_str += f" - {self.name}"  # Include the task name to avoid collisions
        for output_type in self.output_types:
            task_str += f" - {output_type.__name__}"
        if extra_key := self.extra_cache_key(task):
            task_str += f" - {extra_key}"
        return hashlib.sha1(task_str.encode()).hexdigest()

    def extra_cache_key(self, task: Task) -> str:
        """Can be implemented by subclasses to provide additional cache key information."""
        return ""

    def _publish_cached_results(self, cached_results: List[Task], input_task: Task):
        """Publish cached results."""
        for result in cached_results:
            super().publish_work(result, input_task=input_task)

    def _cache_up_call(self, input_task: Task, cached_results: List[Task]):

        cache_key = self._get_cache_key(input_task)
        try:
            # since the cache key includes the name of this task worker, we can use the lock in this class
            with self._lock:
                self._cache.set(cache_key, cached_results)
            logging.info("Task %s cached results for key: %s", self.name, cache_key)
        except Exception as e:
            logging.error("Error caching results for key %s: %s", cache_key, str(e))