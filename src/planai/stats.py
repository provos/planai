import statistics
from threading import Lock
from typing import List

from pydantic import BaseModel, Field, PrivateAttr


class WorkerStat(BaseModel):
    times: List[float] = Field(default_factory=list)
    completed_count: int = 0
    active_count: int = 0
    queued_count: int = 0
    failed_count: int = 0
    _lock: Lock = PrivateAttr(default_factory=Lock)

    def add_completion_time(self, time: float):
        with self._lock:
            self.times.append(time)

    def increment_completed(self):
        with self._lock:
            self.completed_count += 1

    def increment_active(self):
        with self._lock:
            self.active_count += 1

    def increment_queued(self):
        with self._lock:
            self.queued_count += 1

    def increment_failed(self):
        with self._lock:
            self.failed_count += 1

    def decrement_active(self):
        with self._lock:
            self.active_count = max(0, self.active_count - 1)

    def decrement_queued(self):
        with self._lock:
            self.queued_count = max(0, self.queued_count - 1)

    def get_statistics(self):
        with self._lock:
            return {
                "min": min(self.times) if self.times else 0,
                "median": statistics.median(self.times) if self.times else 0,
                "max": max(self.times) if self.times else 0,
                "stdDev": statistics.stdev(self.times) if len(self.times) > 1 else 0,
                "count": len(self.times),
                "completed": self.completed_count,
                "active": self.active_count,
                "queued": self.queued_count,
                "failed": self.failed_count,
            }
