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
    from .graph import Graph
    from .dispatcher import ProvenanceChain


class TaskWorkItem(BaseModel):
    _provenance: List[Tuple[str, int]] = PrivateAttr(default_factory=list)
    _input_provenance: List["TaskWorkItem"] = PrivateAttr(default_factory=list)

    def copy_provenance(self) -> List[Tuple[str, int]]:
        return self._provenance.copy()

    def copy_input_provenance(self) -> List["TaskWorkItem"]:
        return self._input_provenance.copy()

    def find_input_task(
        self, task_class: Type["TaskWorkItem"]
    ) -> Optional["TaskWorkItem"]:
        """
        Find the most recent input task of the specified class in the input provenance.

        Args:
            task_class (Type[TaskWorkItem]): The class of the task to find.

        Returns:
            Optional[TaskWorkItem]: The most recent task of the specified class,
                                    or None if no such task is found.
        """
        for task in reversed(self._input_provenance):
            if task.__class__ is task_class:
                return task
        return None
    
    def previous_input_task(self):
        return self._input_provenance[-1] if self._input_provenance else None
    
    def prefix_for_input_task(
        self, task_class: Type["TaskWorker"]
    ) -> Optional['ProvenanceChain']:
        """
        Finds the provenance chain for the most recent input task of the specified class.
        
        Args:
            task_class (Type[TaskWorkItem]): The class of the task to find.
        Returns:
            ProvenanceChain: The provenance chain for the most recent input task of the specified class.
        """
        for i in range(len(self._provenance)-1, -1, -1):
            if self._provenance[i][0] == task_class.__name__:
                return tuple(self._provenance[:i+1])
        return None


class TaskWorker(BaseModel, ABC):
    """
    Base class for all task workers.

    This class is strongly typed for both input and output types. The type checking
    is performed during the registration of consumers.

    Attributes:
        name (str): The name of the task.
        output_types (Set[Type[TaskWorkItem]]): The types of work this task can output.
        _id (int): A private attribute to track the task's ID.
        _consumers (Dict[Type["TaskWorker"], "TaskWorker"]): A private attribute to store registered consumers.

    Note:
        Any subclass of TaskWorker must implement both consume_work and publish_work methods.
    """

    output_types: Set[Type[TaskWorkItem]] = Field(
        default_factory=set, description="The types of work this task can output"
    )

    _state_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _id: int = PrivateAttr(default=0)
    _consumers: Dict[Type["TaskWorker"], "TaskWorker"] = PrivateAttr(
        default_factory=dict
    )
    _graph: Optional["Graph"] = PrivateAttr(default=None)
    _last_input_task: Optional[TaskWorkItem] = PrivateAttr(default=None)
    _instance_id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)

    def __hash__(self):
        return hash(self._instance_id)

    def __eq__(self, other):
        if isinstance(other, TaskWorker):
            return self._instance_id == other._instance_id
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def last_input_task(self) -> Optional[TaskWorkItem]:
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

    def watch(self, prefix: 'ProvenanceChain') -> bool:
        """
        Watches for this task provenance to be completed in the graph.

        Parameters:
            worker (Type["TaskWorkItem"]): The worker to watch.

        Returns:
            True if the watch was added, False if the watch was already present.
        """
        if not isinstance(prefix, tuple):
            raise ValueError("Prefix must be a tuple")
        return self._graph._dispatcher.watch(prefix, self)

    def unwatch(self, prefix: 'ProvenanceChain') -> bool:
        """
        Removes the watch for this task provenance to be completed in the graph.
        
        Parameters:
            worker (Type["TaskWorkItem"]): The worker to unwatch.
        
        Returns:
            True if the watch was removed, False if the watch was not present.
        """
        if not isinstance(prefix, tuple):
            raise ValueError("Prefix must be a tuple")
        return self._graph._dispatcher.unwatch(prefix, self)

    def _pre_consume_work(self, task: TaskWorkItem):
        with self._state_lock:
            self._last_input_task = task
        self.consume_work(task)

    def init(self):
        """
        Called when the graph is fully constructed and starts work.
        """
        pass

    @abstractmethod
    def consume_work(self, task: TaskWorkItem):
        """
        Abstract method to consume a work item.

        This method must be implemented by subclasses to define specific work consumption logic. It needs to be thread-safe
        as it may be called concurrently by multiple threads.

        Args:
            task (TaskWorkItem): The work item to be consumed.
        """
        pass

    def publish_work(self, task: TaskWorkItem, input_task: Optional[TaskWorkItem]):
        """
        Publish a work item.

        This method handles the publishing of work items, including provenance tracking and consumer routing.

        Args:
            task (TaskWorkItem): The work item to be published.
            input_task (TaskWorkItem): The input task that led to this work item.

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
            task._input_provenance = input_task.copy_input_provenance() + [input_task]
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

        logging.info("Worker %s publishing work to consumer %s with task type %s", self.name, consumer.name, task.__class__.__name__)
        if self._graph and self._graph._dispatcher:
            self._graph._dispatcher.add_work(consumer, task)
        else:
            self._dispatch_work(task)

    def completed(self):
        """Called to let the worker know that it has finished processing all work."""
        pass

    def notify(self, task_name: str):
        """Called to notify the worker that no tasks with provenance of task_name are remaining."""
        pass

    def _dispatch_work(self, task: TaskWorkItem):
        consumer = self._consumers.get(task.__class__)
        consumer.consume_work(task)

    def validate_taskworkitem(
        self, task_cls: Type[TaskWorkItem], consumer: "TaskWorker"
    ) -> Tuple[bool, Exception]:
        """
        Validate that a consumer can handle a specific TaskWorkItem type.

        This method checks if the consumer has a properly typed consume_work method for the given task class.

        Args:
            task_cls (Type[TaskWorkItem]): The TaskWorkItem subclass to validate.
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

    def get_taskworkitem_class(self) -> Type[TaskWorkItem]:
        """
        Get the TaskWorkItem subclass that this worker can consume.

        This method inspects the consume_work method to determine the type of TaskWorkItem it can handle.

        Returns:
            Type[TaskWorkItem]: The TaskWorkItem subclass this worker can consume.

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

    def register_consumer(self, task_cls: Type[TaskWorkItem], consumer: "TaskWorker"):
        """
        Register a consumer for a specific TaskWorkItem type.

        This method performs type checking to ensure that the consumer can handle the specified TaskWorkItem type.

        Args:
            task_cls (Type[TaskWorkItem]): The TaskWorkItem subclass to register a consumer for.
            consumer (TaskWorker): The consumer to register.

        Raises:
            TypeError: If task_cls is not a subclass of TaskWorkItem or if the consumer cannot handle the task type.
            ValueError: If the task type is not in the output_types or if a consumer is already registered for the task type.
        """
        # Ensure task_cls is a subclass of TaskWorkItem
        if not issubclass(task_cls, TaskWorkItem):
            raise TypeError(f"{task_cls.__name__} is not a subclass of TaskWorkItem")

        success, error = self.validate_taskworkitem(task_cls, consumer)
        if not success:
            raise error

        if task_cls not in self.output_types:
            raise ValueError(
                f"Task {self.name} cannot publish work of type {task_cls.__name__}"
            )

        if task_cls in self._consumers:
            raise ValueError(f"Consumer for {task_cls.__name__} already registered")
        self._consumers[task_cls] = consumer


def main():
    class MagicTaskWork(TaskWorkItem):
        magic: Any = Field(..., title="Magic", description="Magic value")

    class SpecificTask(TaskWorker):
        output_types: Set[Type[TaskWorkItem]] = {MagicTaskWork}

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
