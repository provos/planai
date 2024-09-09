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
import inspect
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    get_type_hints,
)

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from .dispatcher import ProvenanceChain
    from .graph import Graph


class Task(BaseModel):
    _provenance: List[Tuple[str, int]] = PrivateAttr(default_factory=list)
    _input_provenance: List["Task"] = PrivateAttr(default_factory=list)
    _retry_count: int = PrivateAttr(default=0)
    _start_time: Optional[float] = PrivateAttr(default=None)
    _end_time: Optional[float] = PrivateAttr(default=None)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def retry_count(self) -> int:
        """
        Read-only property to access the current retry count.
        """
        return self._retry_count

    def increment_retry_count(self) -> None:
        """
        Increments the retry count by 1.
        """
        self._retry_count += 1
        return True

    def copy_provenance(self) -> List[Tuple[str, int]]:
        return self._provenance.copy()

    def copy_input_provenance(self) -> List["Task"]:
        return [input.model_copy() for input in self._input_provenance]

    def find_input_task(self, task_class: Type["Task"]) -> Optional["Task"]:
        """
        Find the most recent input task of the specified class in the input provenance.

        Args:
            task_class (Type[Task]): The class of the task to find.

        Returns:
            Optional[Task]: The most recent task of the specified class,
                                    or None if no such task is found.
        """
        for task in reversed(self._input_provenance):
            if task.__class__ is task_class:
                return task
        return None

    def previous_input_task(self):
        return self._input_provenance[-1] if self._input_provenance else None

    def prefix_for_input_task(
        self, task_class: Type["Task"]
    ) -> Optional["ProvenanceChain"]:
        """
        Finds the provenance chain for the most recent input task of the specified class.

        Args:
            task_class (Type[Task]): The class of the task to find.
        Returns:
            ProvenanceChain: The provenance chain for the most recent input task of the specified class.
        """
        for i in range(len(self._provenance) - 1, -1, -1):
            if self._provenance[i][0] == task_class.__name__:
                return tuple(self._provenance[: i + 1])
        return None


class WorkBufferContext:
    def __init__(self, worker, input_task=None):
        self.worker: "TaskWorker" = worker
        self.input_task: "Task" = input_task
        self.work_buffer: List[Tuple["TaskWorker", "Task"]] = []

    def __enter__(self):
        self.worker._local.ctx = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._flush_work_buffer()
        self.worker._local.ctx = None

    def get_input_and_outputs(self):
        return self.input_task, [entry[1] for entry in self.work_buffer]

    def _flush_work_buffer(self):
        if self.worker._graph and self.worker._graph._dispatcher:
            logging.info(
                "Worker %s flushing work buffer with %d items",
                self.worker.name,
                len(self.work_buffer),
            )
            self.worker._graph._dispatcher.add_multiple_work(self.work_buffer)
        else:
            for consumer, task in self.work_buffer:
                self.worker._dispatch_work(task)
        self.work_buffer.clear()

    def add_to_buffer(self, consumer: "TaskWorker", task: "Task"):
        self.work_buffer.append((consumer, task))


class TaskWorker(BaseModel, ABC):
    """
    Base class for all task workers.

    This class is strongly typed for both input and output types. The type checking
    is performed during the registration of consumers.

    Attributes:
        output_types (Set[Type[Task]]): The types of work this task can output.
        num_retries (int): The number of retries allowed for this task. Defaults to 0.
        _id (int): A private attribute to track the task's ID.
        _consumers (Dict[Type["TaskWorker"], "TaskWorker"]): A private attribute to store registered consumers.

    Note:
        Any subclass of TaskWorker must implement consume_work.
    """

    output_types: Set[Type[Task]] = Field(
        default_factory=set,
        description="The types of work this task can output",
    )
    num_retries: int = Field(
        0, description="The number of retries allowed for this task"
    )

    _state_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _id: int = PrivateAttr(default=0)
    _consumers: Dict[Type["TaskWorker"], "TaskWorker"] = PrivateAttr(
        default_factory=dict
    )
    _graph: Optional["Graph"] = PrivateAttr(default=None)
    _last_input_task: Optional[Task] = PrivateAttr(default=None)
    _instance_id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)
    _local: threading.local = PrivateAttr(default_factory=threading.local)

    def __init__(self, **data):
        super().__init__(**data)

    def __hash__(self):
        return hash(self._instance_id)

    def __eq__(self, other):
        if isinstance(other, TaskWorker):
            return self._instance_id == other._instance_id
        return False

    def work_buffer_context(self, input_task):
        if input_task is None:
            raise ValueError("Input task cannot be None")
        return WorkBufferContext(self, input_task)

    @property
    def name(self) -> str:
        """
        Returns the name of this worker class.

        :return: The name of the class.
        :rtype: str
        """
        return self.__class__.__name__

    @property
    def lock(self) -> threading.Lock:
        """
        Returns the lock object for this worker.

        :return: The lock object.
        :rtype: threading.Lock
        """
        return self._state_lock

    @property
    def last_input_task(self) -> Optional[Task]:
        """
        Returns the last input task consumed by this worker.

        :return: The last input task as a Task object, or None if there is no last input task.
        :rtype: Optional[Task]
        """
        with self._state_lock:
            return self._last_input_task

    def set_graph(self, graph: "Graph"):
        self._graph = graph

    def next(self, downstream: "TaskWorker"):
        """
        Sets the dependency between the current task and the downstream task.

        Parameters:
            downstream (TaskWorker): The downstream task to set as a dependency.

        Returns:
            TaskWorker: The downstream task.

        Raises:
            ValueError: If the task has not been added to a Graph before setting dependencies.
        """
        if self._graph is None:
            raise ValueError(
                "Task must be added to a Graph before setting dependencies"
            )
        self._graph.set_dependency(self, downstream)
        return downstream

    def trace(self, prefix: "ProvenanceChain"):
        """
        Traces the provenance chain for a given prefix in the graph.

        This method sets up a trace on a given prefix in the provenance chain. It will be visible
        in the dispatcher dashboard.

        Parameters:
        -----------
        prefix : ProvenanceChain
            The prefix to trace. Must be a tuple representing a part of a task's provenance chain.
            This is the sequence of task identifiers leading up to (but not including) the current task.
        """
        self._graph._dispatcher.trace(prefix)

    def watch(self, prefix: "ProvenanceChain", task: Optional[Task] = None) -> bool:
        """
        Watches for the completion of a specific provenance chain prefix in the task graph.

        This method sets up a watch on a given prefix in the provenance chain. It will be notified
        in its notify method when this prefix is no longer part of any active task's provenance, indicating
        that all tasks with this prefix have been completed.

        Parameters:
        -----------
        prefix : ProvenanceChain
            The prefix to watch. Must be a tuple representing a part of a task's provenance chain.
            This is the sequence of task identifiers leading up to (but not including) the current task.

        task : Optional[Task], default=None
            The task associated with this watch operation. This parameter is optional and may be
            used for additional context or functionality in the underlying implementation.

        Returns:
        --------
        bool
            True if the watch was successfully added for the given prefix.
            False if a watch for this prefix was already present.

        Raises:
        -------
        ValueError
            If the provided prefix is not a tuple.
        """
        if not isinstance(prefix, tuple):
            raise ValueError("Prefix must be a tuple")
        return self._graph._dispatcher.watch(prefix, self, task)

    def unwatch(self, prefix: "ProvenanceChain") -> bool:
        """
        Removes the watch for this task provenance to be completed in the graph.

        Parameters:
            worker (Type["Task"]): The worker to unwatch.

        Returns:
            True if the watch was removed, False if the watch was not present.
        """
        if not isinstance(prefix, tuple):
            raise ValueError("Prefix must be a tuple")
        return self._graph._dispatcher.unwatch(prefix, self)

    def _pre_consume_work(self, task: Task):
        with self._state_lock:
            self._last_input_task = task
        with self.work_buffer_context(task):
            self.consume_work(task)

    def init(self):
        """
        Called when the graph is fully constructed and starts work.
        """
        pass

    @abstractmethod
    def consume_work(self, task: Task):
        """
        Abstract method to consume a work item.

        This method must be implemented by subclasses to define specific work consumption logic. It needs to be thread-safe
        as it may be called concurrently by multiple threads.

        Args:
            task (Task): The work item to be consumed.
        """
        pass

    def publish_work(self, task: Task, input_task: Optional[Task]):
        """
        Publish a work item.

        This method handles the publishing of work items, including provenance tracking and consumer routing.

        Args:
            task (Task): The work item to be published.
            input_task (Task): The input task that led to this work item.

        Raises:
            ValueError: If the task type is not in the output_types or if no consumer is registered for the task type.
        """
        if type(task) not in self.output_types:
            raise ValueError(
                f"Task {self.name} cannot publish work of type {type(task).__name__}"
            )

        # Copy provenance from input task if provided
        if input_task is not None:
            task._provenance = input_task.copy_provenance()
            task._input_provenance = input_task.copy_input_provenance() + [
                input_task.model_copy()
            ]
        else:
            task._provenance = []
            task._input_provenance = []

        with self._state_lock:
            self._id += 1

        task._provenance.append((self.name, self._id))
        logging.info(
            "Task %s published work with provenance %s", self.name, task._provenance
        )

        # Verify if there is a consumer for the given task class
        consumer = self._consumers.get(task.__class__)
        if consumer is None:
            raise ValueError(f"No consumer registered for {task.__class__.__name__}")

        logging.info(
            "Worker %s publishing work to consumer %s with task type %s",
            self.name,
            consumer.name,
            task.__class__.__name__,
        )

        # this requires that anything that might call publish_work is wrapped in a work_buffer_context
        self._local.ctx.add_to_buffer(consumer, task)

    def completed(self):
        """Called to let the worker know that it has finished processing all work."""
        pass

    def notify(self, task_name: str):
        """Called to notify the worker that no tasks with provenance of task_name are remaining."""
        pass

    def _dispatch_work(self, task: Task):
        consumer = self._consumers.get(task.__class__)
        consumer.consume_work(task)

    def validate_task(
        self, task_cls: Type[Task], consumer: "TaskWorker"
    ) -> Tuple[bool, Exception]:
        """
        Validate that a consumer can handle a specific Task type.

        This method checks if the consumer has a properly typed consume_work method for the given task class.

        Args:
            task_cls (Type[Task]): The Task subclass to validate.
            consumer (TaskWorker): The consumer to validate against.

        Returns:
            Tuple[bool, Exception]: A tuple containing a boolean indicating success and an exception if validation failed.
        """
        # Ensure consumer has a consume_work method taking task_cls as parameter
        consume_method = getattr(consumer, "consume_work", None)
        if not consume_method:
            return False, AttributeError(
                f"{consumer.__class__.__name__} has no method consume_work"
            )

        # Check the type of the first parameter of consume_work method
        signature = inspect.signature(consume_method)
        parameters = signature.parameters
        if len(parameters) != 1:
            return False, TypeError(
                f"Method consume_work in {consumer.__class__.__name__} must accept one parameter"
            )

        # Get the type hints for consume_work
        type_hints = get_type_hints(consume_method)
        first_param_type = type_hints.get("task", None)

        if first_param_type is not task_cls:
            return False, TypeError(
                f"TaskWorker {consumer.__class__.__name__} cannot consume tasks of type {task_cls.__name__}. It can only consume tasks of type {first_param_type.__name__}"
            )

        return True, None

    def get_task_class(self) -> Type[Task]:
        """
        Get the Task subclass that this worker can consume.

        This method inspects the consume_work method to determine the type of Task it can handle.

        Returns:
            Type[Task]: The Task subclass this worker can consume.

        Raises:
            AttributeError: If the consume_work method is not defined.
            TypeError: If the consume_work method is not properly typed.
        """
        consume_method = getattr(self, "consume_work", None)
        if not consume_method:
            raise AttributeError(
                f"{self.__class__.__name__} has no method consume_work"
            )
        signature = inspect.signature(consume_method)
        parameters = signature.parameters
        if len(parameters) != 1:
            raise TypeError(
                f"Method consume_work in {self.__class__.__name__} must accept one parameter"
            )
        type_hints = get_type_hints(consume_method)
        first_param_type = type_hints.get("task", None)
        if not first_param_type:
            raise TypeError(
                f"consume_work method in {self.__class__.__name__} must have type hints"
            )
        return first_param_type

    def register_consumer(self, task_cls: Type[Task], consumer: "TaskWorker"):
        """
        Register a consumer for a specific Task type.

        This method performs type checking to ensure that the consumer can handle the specified Task type.

        Args:
            task_cls (Type[Task]): The Task subclass to register a consumer for.
            consumer (TaskWorker): The consumer to register.

        Raises:
            TypeError: If task_cls is not a subclass of Task or if the consumer cannot handle the task type.
            ValueError: If the task type is not in the output_types or if a consumer is already registered for the task type.
        """
        # Ensure task_cls is a subclass of Task
        if not issubclass(task_cls, Task):
            raise TypeError(f"{task_cls.__name__} is not a subclass of Task")

        success, error = self.validate_task(task_cls, consumer)
        if not success:
            raise error

        if task_cls not in self.output_types:
            raise ValueError(
                f"Downstream consumer {consumer.name} only accepts work of type {task_cls.__name__} but Worker {self.name} does not produce it"
            )

        if task_cls in self._consumers:
            raise ValueError(f"Consumer for {task_cls.__name__} already registered")
        self._consumers[task_cls] = consumer

        # special cases like JoinedTask need to validate that their join_type is upstream
        consumer._validate_connection()

    def _validate_connection(self):
        pass


def main():
    class MagicTaskWork(Task):
        magic: Any = Field(..., title="Magic", description="Magic value")

    class SpecificTask(TaskWorker):
        output_types: Set[Type[Task]] = {MagicTaskWork}

        def consume_work(self, task: MagicTaskWork):
            print(
                f"SpecificTask {self.name} consuming work item with provenance: {task._provenance}: {task.magic}"
            )

    # Example usage
    task_a = SpecificTask(name="TaskA")
    work_item = MagicTaskWork(magic="something")

    # Register a consumer (for demo purposes, registering itself)
    task_a.register_consumer(MagicTaskWork, task_a)

    # Publish work
    task_a.publish_work(work_item, input_task=None)


if __name__ == "__main__":
    main()
