from typing import List, Optional, Type

from planai.task import Task, TaskWorker


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

    def __init__(self, worker_class: Type[TaskWorker], **kwargs):
        """
        Args:
            worker_class: The TaskWorker class to test
            **kwargs: Arguments to pass to the worker constructor
        """
        self.context = TestTaskContext()
        self.worker = worker_class(**kwargs)

    def invoke(self, input_task: Task) -> List[Task]:
        """
        Invoke the worker with an input task and return published tasks.

        Args:
            input_task: The input task to process

        Returns:
            List of tasks published during processing
        """

        def patched_publish_work(task: Task, input_task: Optional[Task]):
            self.context.published_tasks.append(task)

        original_publish_work = self.worker.publish_work
        object.__setattr__(self.worker, "publish_work", patched_publish_work)

        self.context.reset()
        self.context.current_input_task = input_task
        try:
            self.worker.consume_work(input_task)
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
