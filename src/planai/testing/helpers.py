import logging
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Type, Union

from planai.graph import Graph
from planai.graph_task import SubGraphWorkerInternal
from planai.joined_task import JoinedTaskWorker
from planai.task import Task, TaskWorker


# Mock Cache
class MockCache:
    def __init__(self, dont_store=False):
        self.store = {}
        self._dont_store = dont_store
        self.set_stats: Dict[str, int] = defaultdict(int)
        self.get_stats: Dict[str, int] = defaultdict(int)
        self.lock = Lock()

    def get(self, key, default=None):
        logging.debug("Getting key: %s", key)
        with self.lock:
            self.get_stats[key] += 1
            return self.store.get(key, default)

    def set(self, key, value):
        if self._dont_store:
            return
        logging.debug("Setting key: %s", key)
        with self.lock:
            self.store[key] = value
            self.set_stats[key] += 1

    def clear_stats(self):
        self.set_stats.clear()
        self.get_stats.clear()


class TestTaskContext:
    """Helper class to track published work during testing."""

    def __init__(self):
        self.published_tasks: List[Task] = []
        self.current_input_task: Optional[Task] = None

    def reset(self):
        self.published_tasks = []
        self.current_input_task = None


class InvokeTaskWorker:
    """Helper class to test TaskWorker implementations.

    This class provides utilities to test TaskWorker implementations by mocking the task
    publishing functionality and providing assertions for validating published tasks.

    Parameters
    ----------
    worker_class : Type[TaskWorker]
        The TaskWorker class to test
    **kwargs
        Arguments to pass to the worker constructor

    Methods
    -------
    invoke(input_task: Task) -> List[Task]

    assert_published_task_count(expected: int)
        Assert the number of published tasks matches expected count.

    assert_published_task_types(expected_types: List[Type[Task]])
        Assert the types of published tasks match expected types.

    Example
    -------
    >>> worker = InvokeTaskWorker(MyTaskWorker)
    >>> published = worker.invoke(input_task)
    >>> worker.assert_published_task_count(1)
    >>> worker.assert_published_task_types([OutputTask])
    """

    def __init__(
        self, worker_class: Union[Type[TaskWorker], Type[JoinedTaskWorker]], **kwargs
    ):
        """
        Args:
            worker_class: The TaskWorker class to test
            **kwargs: Arguments to pass to the worker constructor
        """

        class MockProvenanceTracker:
            def notify_status(self, graph, task, status=None, object=None):
                pass

        class MockGraph:
            def __init__(self):
                self._provenance_tracker = MockProvenanceTracker()

            def watch(self, prefix, notifier):
                return True

        self.context = TestTaskContext()
        self.worker = worker_class(**kwargs)
        self.worker._graph = MockGraph()
        if hasattr(self.worker, "_cache"):
            self.worker._cache = MockCache()
        if hasattr(self.worker, "debug_mode"):
            self.worker.debug_mode = False
        self.is_joined_worker = issubclass(worker_class, JoinedTaskWorker)

    def _setup_patch(self):
        """Set up patching of publish_work and reset context."""

        def patched_publish_work(task: Task, input_task: Optional[Task]):
            self.context.published_tasks.append(task)

        original_publish_work = self.worker.publish_work
        object.__setattr__(self.worker, "publish_work", patched_publish_work)
        self.context.reset()

        return original_publish_work

    def invoke(self, input_task: Task) -> List[Task]:
        """
        Invoke the worker with an input task and return published tasks.
        Only valid for TaskWorker instances.

        Args:
            input_task: The input task to process

        Returns:
            List of tasks published during processing

        Raises:
            TypeError: If worker is a JoinedTaskWorker
        """
        if self.is_joined_worker:
            raise TypeError("Use invoke_joined() for JoinedTaskWorker instances")

        original_publish_work = self._setup_patch()
        self.context.current_input_task = input_task

        try:
            self.worker.consume_work(input_task)
        finally:
            object.__setattr__(self.worker, "publish_work", original_publish_work)

        return self.context.published_tasks

    def invoke_joined(self, input_tasks: List[Task]) -> List[Task]:
        """
        Invoke the worker with multiple input tasks and return published tasks.
        Only valid for JoinedTaskWorker instances.

        Args:
            input_tasks: The list of input tasks to process

        Returns:
            List of tasks published during processing

        Raises:
            TypeError: If worker is not a JoinedTaskWorker
        """
        if not self.is_joined_worker:
            raise TypeError("Use invoke() for regular TaskWorker instances")

        original_publish_work = self._setup_patch()

        try:
            self.worker.consume_work_joined(input_tasks)
        finally:
            object.__setattr__(self.worker, "publish_work", original_publish_work)

        return self.context.published_tasks

    def assert_published_task_count(self, expected: int):
        """Assert the number of published tasks."""
        actual = len(self.context.published_tasks)
        assert actual == expected, f"Expected {expected} published tasks, got {actual}"

    def assert_published_task_types(self, expected_types: List[Type[Task]]):
        """Assert the types of published tasks match expected types."""
        actual_types = [type(t) for t in self.context.published_tasks]
        assert (
            actual_types == expected_types
        ), f"Expected task types {expected_types}, got {actual_types}"


def add_input_provenance(input_task: Task, provenance: Task) -> Task:
    input_task._input_provenance.append(provenance)
    return input_task


def inject_mock_cache(graph: Graph, mock_cache: MockCache):
    for worker in graph.workers:
        if hasattr(worker, "_cache"):
            logging.info("Injecting mock cache to %s", worker.name)
            worker._cache = mock_cache
        elif isinstance(worker, SubGraphWorkerInternal):
            inject_mock_cache(worker.graph, mock_cache)


def unregister_output_type(worker: TaskWorker, output_type: Type[Task]):
    """
    Remove a task output type from the worker's consumers.

    Args:
        worker (TaskWorker): The worker instance from which to remove the output type.
        output_type (Type[Task]): The task type to be unregistered from the worker's consumers.

    Raises:
        KeyError: If the output_type is not registered in the worker's consumers.
    """
    worker._consumers.pop(output_type)
