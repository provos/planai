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
from __future__ import annotations

import inspect
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    get_type_hints,
)

from pydantic import BaseModel, Field, PrivateAttr

from planai.utils import dict_dump_xml

if TYPE_CHECKING:
    from .graph import Graph
    from .provenance import ProvenanceChain

TaskStatusCallback = Callable[
    [Dict, "ProvenanceChain", "TaskWorker", "Task", Optional[str], Optional[BaseModel]],
    None,
]


class Task(BaseModel):
    """Base class for all tasks in the system.

    A Task represents a unit of work that can be processed by TaskWorkers. Tasks maintain
    their execution provenance and can carry both public and private state.

    Attributes:
        _provenance (List[Tuple[str, int]]): List of worker name and ID tuples tracking task history
        _input_provenance (List[Task]): List of input tasks that led to this task
        _private_state (Dict[str, Any]): Private state storage
        _retry_count (int): Number of times this task has been retried
        _start_time (Optional[float]): When task processing started
        _end_time (Optional[float]): When task processing completed
    """

    _provenance: List[Tuple[str, int]] = PrivateAttr(default_factory=list)
    _input_provenance: List[Task] = PrivateAttr(default_factory=list)
    _private_state: Dict[str, Any] = PrivateAttr(default_factory=dict)
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

    def copy_public(self, deep: bool = False) -> Task:
        """
        Creates a copy of the Task instance, excluding private attributes. This is a safer way than model_copy()
        of creating a new task from an existing one. Can be used in conjunction with enabling strict on a graph.

        Args:
            deep: Whether to perform a deep copy of the public fields.

        Returns:
            A new Task instance without the private attributes.
        """
        return self.model_validate(
            self.model_dump(
                exclude_unset=True, exclude_defaults=True, exclude_none=True
            )
        )

    def increment_retry_count(self) -> None:
        """
        Increments the retry count by 1.
        """
        self._retry_count += 1

    def copy_provenance(self) -> List[Tuple[str, int]]:
        return self._provenance.copy()

    def copy_input_provenance(self) -> List[Task]:
        return [input.copy_public() for input in self._input_provenance]

    def find_input_task(self, task_class: Type[Task]) -> Optional[Task]:
        """
        Find the most recent input task of the specified class in the input provenance.
        This is guaranteed to work only on the immediate input task and not on any tasks
        further upstream returned by this function.

        Args:
            task_class (Type[Task]): The class of the task to find.

        Returns:
            Optional[Task]: The most recent task of the specified class,
                                    or None if no such task is found.
        """
        for task in reversed(self._input_provenance):
            if type(task).__name__ == task_class.__name__:
                return task
        return None

    def find_input_tasks(self, task_class: Type[Task]) -> List[Task]:
        """
        Find all input tasks of the specified class in the input provenance.

        Args:
            task_class (Type[Task]): The class of the tasks to find.

        Returns:
            List[Task]: A list of tasks of the specified class.
        """
        return [
            task
            for task in self._input_provenance
            if type(task).__name__ == task_class.__name__
        ]

    def previous_input_task(self):
        if not self._input_provenance:
            return None
        task = self._input_provenance[-1].copy_public()
        task._provenance = self._provenance.copy()[:-1]
        task._input_provenance = self._input_provenance.copy()[:-1]
        return task

    def prefix(self, length: int) -> "ProvenanceChain":
        """
        Get a prefix of specified length from task's provenance chain.

        Args:
            task (Task): The task object containing provenance information.
            length (int): The desired length of the prefix to extract.

        Returns:
            ProvenanceChain: A tuple containing the first 'length' elements of the task's provenance chain.
        """
        return tuple(self._provenance[:length])

    def prefix_for_input_task(
        self, task_class: Type["TaskWorker"]
    ) -> Optional["ProvenanceChain"]:
        """
        Finds the provenance chain for the most recent input task of the specified class.

        Args:
            task_class (Type[TaskWorker]): The class of the task worker to find.
        Returns:
            ProvenanceChain: The provenance chain for the most recent input task of the specified class.
        """
        for i in range(len(self._provenance) - 1, -1, -1):
            if self._provenance[i][0] == task_class.__name__:
                return tuple(self._provenance[: i + 1])
        return None

    def _add_worker_provenance(self, worker: "TaskWorker") -> Task:
        provenance = worker.get_next_provenance()
        self._provenance.append(provenance)
        return self

    def _add_input_provenance(self, input_task: Optional[Task]) -> Task:
        # Copy provenance from input task if provided
        if input_task is not None:
            self._provenance = input_task.copy_provenance()
            self._input_provenance = input_task.copy_input_provenance() + [
                input_task.copy_public()
            ]
            # merge private state
            self._private_state.update(input_task._private_state)
        else:
            self._provenance = []
            self._input_provenance = []
        return self

    def inject_input(self, input_task: Task) -> Task:
        """Can be used to inject an input task into the provenance chain.

        Args:
            input_task (Task): The input task to inject.

        Returns:
            Task: The task with the input task injected into the provenance
        """
        self._input_provenance.append(input_task.copy_public())
        return self

    def add_private_state(self, key: str, value: Any) -> None:
        self._private_state[key] = value

    def get_private_state(self, key: str) -> Any:
        return self._private_state.pop(key, None)

    def model_dump_xml(self) -> str:
        """Formats the task as XML."""
        return dict_dump_xml(self.model_dump(), root=self.name)

    def is_type(self, task_class: Type[Task]) -> bool:
        """
        Check if this task is of the specified task class type.

        Args:
            task_class (Type[Task]): The task class type to check against.

        Returns:
            bool: True if the task is of the specified type, False otherwise.
        """
        return type(self).__name__ == task_class.__name__


class WorkBufferContext:
    def __init__(self, worker: TaskWorker, input_task=None):
        self.worker: TaskWorker = worker
        self.input_task: Task = input_task
        self.work_buffer: List[Tuple[TaskWorker, Task]] = []

    def __enter__(self):
        self.worker._local.ctx = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._flush_work_buffer()
        self.worker._local.ctx = None

    def get_input_and_outputs(self):
        return self.input_task, self.work_buffer

    def _flush_work_buffer(self):
        self.work_buffer.clear()

    def add_to_buffer(self, consumer: "TaskWorker", task: Task):
        self.work_buffer.append((consumer, task))


class TaskWorker(BaseModel, ABC):
    """Base class for all task workers.

    TaskWorker implements the core task processing functionality. Workers consume tasks,
    process them, and can produce new tasks for downstream workers. The system ensures
    type safety between workers and maintains execution provenance.

    Attributes:
        output_types (List[Type[Task]]): Types of tasks this worker can produce
        num_retries (int): Number of times to retry failed tasks
        _id (int): Internal worker ID counter
        _consumers (Dict[Type[Task], List[TaskWorker]]): Registered downstream consumers
        _graph (Optional[Graph]): Reference to containing workflow graph
        _instance_id (UUID): Unique worker instance identifier
        _local (threading.local): Thread-local storage
    """

    output_types: List[Type[Task]] = Field(default_factory=list)
    num_retries: int = Field(default=0)

    _state_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _id: int = PrivateAttr(default=0)
    _consumers: Dict[Type[Task], List[TaskWorker]] = PrivateAttr(default_factory=dict)
    _graph: Optional["Graph"] = PrivateAttr(default=None)
    _instance_id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)
    _local: threading.local = PrivateAttr(default_factory=threading.local)
    _strict_checking: bool = PrivateAttr(default=False)

    # allows an implementation of taskworker to associate state with a task
    # this is useful for tracking the state of a task across multiple calls to consume_work
    # for example when there are circular dependencies
    _user_state: Dict["ProvenanceChain", Dict[str, Any]] = PrivateAttr(
        default_factory=dict
    )

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
        """
        return self.__class__.__name__

    @property
    def lock(self) -> threading.RLock:
        """
        Returns the lock object for this worker.

        :return: The lock object.
        :rtype: threading.Lock
        """
        return self._state_lock

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

    def sink(
        self,
        output_type: Type[Task],
        notify: Optional[Callable[[Dict[str, Any], Task], None]] = None,
    ):
        """
        Designates the current task worker as a sink in the associated graph.

        This method marks the current task worker as a sink, which means its output
        will be collected and can be retrieved after the graph execution.

        Parameters:
            output_type (Task): The output type of the task to send to the sink.
            notify: Optional callback function to be called when the sink is executed. It will receive
              any metadata associated with the task and the task itself.

        Raises:
            ValueError: If the task worker is not associated with a graph.

        Note:
            - Only one sink can be set per graph. Attempting to set multiple sinks
              will raise a RuntimeError from the graph's set_sink method.
            - The task worker must have exactly one output type to be eligible as a sink.
            - Results from the sink can be retrieved using the graph's get_output_tasks() method
              after the graph has been executed.

        See Also:
            Graph.set_sink(): The underlying method called to set the sink.
            Graph.get_output_tasks(): Method to retrieve results from the sink after graph execution.
        """
        if self._graph is None:
            raise ValueError(
                "Task must be added to a Graph before setting a sink dependency"
            )
        self._graph.set_sink(self, output_type, notify)

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
        assert self._graph is not None
        self._graph.trace(prefix)

    def watch(self, prefix: "ProvenanceChain") -> bool:
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
        assert self._graph is not None
        return self._graph.watch(prefix, self)

    def unwatch(self, prefix: "ProvenanceChain") -> bool:
        """
        Removes the watch for this task provenance to be completed in the graph.

        Parameters:
            worker (Type[Task]): The worker to unwatch.

        Returns:
            True if the watch was removed, False if the watch was not present.
        """
        if not isinstance(prefix, tuple):
            raise ValueError("Prefix must be a tuple")
        assert self._graph is not None
        return self._graph.unwatch(prefix, self)

    def get_worker_state(self, provenance: "ProvenanceChain") -> Dict[str, Any]:
        """
        Allows a worker to store state for a specific provenance chain.

        This is helpful when we expect the worker to be called multiple times with the same provenance chain.
        For example, this can happen when there are circular dependencies in the graph. The most common case
        is when a worker needs to ask for more data from upstream workers and sends a task back to them.

        The state will be cleaned up automatically when the provenance chain is no longer active in the graph.

        Returns:
            Dict[str, Any]: The state of the task.
        """
        with self.lock:
            if provenance in self._user_state:
                return self._user_state[provenance]
            self.watch(provenance)
            self._user_state[provenance] = {}
            return self._user_state[provenance]

    def print(self, *args):
        """
        Prints a message to the console.

        Parameters:
            *args: The message to print.
        """
        assert self._graph is not None
        self._graph.print(*args)

    def get_next_provenance(self) -> Tuple[str, int]:
        """
        Gets the next provenance tuple for this worker.

        Returns:
            Tuple[str, int]: The next provenance tuple.
        """
        with self.lock:
            self._id += 1
            return tuple((self.name, self._id))

    def remove_state(self, task: Task):
        """
        Remove the state for a task.

        Args:
            task (Task): The task to remove the state for.
        """
        provenance = task.prefix(1)
        assert self._graph is not None
        self._graph._provenance_tracker.remove_state(provenance)

    def get_state(self, task: Task) -> Dict[str, Any]:
        """
        Get the state of a task.

        Parameters:
            task (Task): The task to get the state for.

        Returns:
            Dict[str, Any]: The state of the task.
        """
        provenance = task.prefix(1)
        assert self._graph is not None
        return self._graph._provenance_tracker.get_state(provenance)

    def get_metadata(self, task: Task) -> Dict[str, Any]:
        """
        Get metadata for the task.

        Returns:
            Dict[str, Any]: Metadata for the worker.

        Raises:
            RuntimeError: If the graph or ProvenanceTracker is not initialized.
        """
        if self._graph is None or self._graph._provenance_tracker is None:
            raise RuntimeError("Graph or ProvenanceTracker is not initialized.")
        result = self.get_state(task)
        return result["metadata"]

    def add_work(
        self,
        task: Task,
        metadata: Optional[Dict] = None,
        status_callback: Optional[TaskStatusCallback] = None,
    ) -> "ProvenanceChain":
        if self._graph is None:
            raise RuntimeError("Graph is not initialized.")
        return self._graph.add_work(self, task, metadata, status_callback)

    def notify_status(
        self,
        task: Task,
        message: Optional[str] = None,
        object: Optional[BaseModel] = None,
    ):
        """Notify registered callback about task status updates."""
        assert self._graph is not None
        self._graph._provenance_tracker.notify_status(self, task, message, object)

    def _pre_consume_work(self, task: Task):
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

    def publish_work(
        self,
        task: Task,
        input_task: Optional[Task],
        consumer: Optional[TaskWorker] = None,
    ):
        """
        Publish a work item.

        This method handles the publishing of work items, including provenance tracking and consumer routing.
        It is important that task is a newly created object and not a reference to an existing task. You can
        use the model_copy method to create a new object with the same data.

        Args:
            task (Task): The work item to be published.
            input_task (Task): The input task that led to this work item.
            consumer (TaskWorker): The TaskWorker to publish to if there are multiple consumers for the task type.

        Raises:
            ValueError: If the task type is not in the output_types or if no consumer is registered for the task type.
        """
        if type(task) not in self.output_types:
            raise ValueError(
                f"Task {self.name} cannot publish work of type {type(task).__name__}"
            )
        if self._strict_checking and (task._provenance or task._input_provenance):
            raise ValueError(
                "Cannot publish a task that has already been published. Use copy_public() to create a new task."
            )

        # the order of these operations is important as the first call erases the provenance
        task._add_input_provenance(input_task)
        task._add_worker_provenance(self)

        # find the consumer for this task to publish to
        consumer = self._get_consumer(task, worker=consumer)

        logging.info(
            "Worker %s publishing work to consumer %s with task type %s and provenance %s",
            self.name,
            consumer.name,
            task.__class__.__name__,
            task._provenance,
        )

        if self._graph and self._graph._dispatcher:
            logging.info(
                "Worker %s publishing work to buffer with consumer %s",
                self.name,
                consumer.name,
            )
            self._graph._dispatcher.add_work(consumer, task)
        else:
            self._dispatch_work(task)

        # this requires that anything that might call publish_work is wrapped in a work_buffer_context
        self._local.ctx.add_to_buffer(consumer, task)

    def _get_consumer_by_name(self, task: Task, worker_name: str) -> TaskWorker:
        # Verify if there is a consumer for the given task class
        consumers = self._consumers.get(task.__class__)
        if not consumers:
            logging.error(
                "%s: No consumer registered for %s, available consumers: %s",
                self.name,
                task.__class__.__name__,
                [c.name for consumers in self._consumers.values() for c in consumers],
            )
            raise ValueError(f"No consumer registered for {task.__class__.__name__}")
        for consumer in consumers:
            if consumer.name == worker_name:
                return consumer
        raise ValueError(
            f"No consumer registered for {task.__class__.__name__} with name {worker_name}"
        )

    def _get_consumer(
        self, task: Task, worker: Optional[TaskWorker] = None
    ) -> TaskWorker:
        # Verify if there is a consumer for the given task class
        consumers = self._consumers.get(task.__class__)
        if not consumers:
            logging.error(
                "%s: No consumer registered for %s, available consumers: %s",
                self.name,
                task.__class__.__name__,
                [c.name for consumers in self._consumers.values() for c in consumers],
            )
            raise ValueError(f"No consumer registered for {task.__class__.__name__}")
        if len(consumers) == 1:
            if worker and consumers[0] != worker:
                raise ValueError(
                    f"Worker {worker.name} is not a registered consumer for {task.__class__.__name__}"
                )
            return consumers[0]
        if worker is None:
            raise ValueError(
                f"Multiple consumers registered for {task.__class__.__name__}, specify worker_name"
            )

        if worker not in consumers:
            raise ValueError(
                f"Worker {worker.name} is not a registered consumer for {task.__class__.__name__}"
            )

        return worker

    def completed(self):
        """Called to let the worker know that it has finished processing all work."""
        pass

    def notify(self, prefix: "ProvenanceChain"):
        """Called to notify the worker that no tasks with this provenance prefix are remaining.

        Children implementing this method need to call the base class method to ensure that the
        state is fully removed.
        """
        with self.lock:
            if prefix in self._user_state:
                del self._user_state[prefix]
            logging.info("Removing watch for %s in %s", prefix, self.name)
            # remove the watch - which may already have been removed by the child class
            self.unwatch(prefix)

    def _dispatch_work(self, task: Task):
        consumer: Optional[TaskWorker] = self._get_consumer(task)
        assert consumer is not None
        consumer.consume_work(task)

    def validate_task(
        self, task_cls: Type[Task], consumer: TaskWorker
    ) -> Tuple[bool, Optional[BaseException]]:
        """
        Validate that a consumer can handle a specific Task type.

        This method checks if the consumer has a properly typed consume_work method for the given task class.

        Args:
            task_cls (Type[Task]): The Task subclass to validate.
            consumer (TaskWorker): The consumer to validate against.

        Returns:
            Tuple[bool, Optional[BaseException]]: A tuple containing a boolean indicating success and an exception if validation failed.
        """
        # Ensure consumer has a consume_work method taking task_cls as parameter
        first_param_type = consumer.get_task_class()

        if first_param_type is not task_cls:
            return False, TypeError(
                f"TaskWorker {consumer.__class__.__name__} cannot consume tasks of type {task_cls.__name__}. It can only consume tasks of type {first_param_type.__name__}"
            )

        return True, None

    def get_task_class(self) -> Type[Task]:
        """
        Get the Task subclass that this worker can consume.

        This method checks for the task type provided in consume_work.

        Returns:
            Type[Task]: The Task subclass this worker can consume.

        Raises:
            AttributeError: If the consume method is not defined.
            TypeError: If the consume method is not properly typed.
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
            assert error is not None
            raise error

        if task_cls not in self.output_types:
            raise ValueError(
                f"Downstream consumer {consumer.name} only accepts work of type {task_cls.__name__} but Worker {self.name} does not produce it"
            )

        if task_cls in self._consumers and consumer in self._consumers[task_cls]:
            raise ValueError(f"Consumer for {task_cls.__name__} already registered")
        if task_cls not in self._consumers:
            self._consumers[task_cls] = []
        self._consumers[task_cls].append(consumer)

        # special cases like JoinedTask need to validate that their join_type is upstream
        consumer._validate_connection()

    def _validate_connection(self):
        pass

    def request_user_input(
        self,
        task: "Task",
        instruction: str,
        accepted_mime_types: List[str] = ["text/html"],
    ) -> Tuple[Any, Optional[str]]:
        """
        Requests user input during the execution of a task with specified instructions and accepted MIME types.

        This method facilitates interaction with the user by sending a request for additional input needed
        to proceed with the task's execution. This interaction may be needed when it's not possible to get
        relevant content programmatically.

        Parameters:
            task (Task): The current task for which user input is being requested. This object must be part
              of the initialized graph and dispatcher.
            instruction (str): Instructions to the user describing the nature of the requested input.
              This string should be clear to prompt the expected response.
            accepted_mime_types (List[str], optional): A list of acceptable MIME types for the user input.
              Defaults to ["text/html"]. This parameter specifies the format expectations for input validation.

        Returns:
            Tuple[Any, Optional[str]]: A tuple where the first element is the user's input (data), and the second
              element (if available) is the MIME type of the provided data.

        Raises:
        - RuntimeError: If the graph or dispatcher is not initialized.
        """
        if self._graph is None or self._graph._dispatcher is None:
            raise RuntimeError("Graph or Dispatcher is not initialized.")

        task_id = self._graph._dispatcher._get_task_id(task)
        result, mime_type = self._graph._dispatcher.request_user_input(
            task_id, instruction, accepted_mime_types
        )
        return result, mime_type
