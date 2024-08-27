import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple, Type, get_type_hints

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from .dag import DAG


class TaskWorkItem(BaseModel):
    _provenance: Dict[str, int] = PrivateAttr(default_factory=dict)

    def copy_provenance(self) -> Dict[str, int]:
        return self._provenance.copy()


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

    name: str = Field(..., title="Task name", description="The name of the task")
    output_types: Set[Type[TaskWorkItem]] = Field(
        default_factory=set, description="The types of work this task can output"
    )

    _id: int = PrivateAttr(default=0)
    _consumers: Dict[Type["TaskWorker"], "TaskWorker"] = PrivateAttr(
        default_factory=dict
    )
    _dag: Optional["DAG"] = PrivateAttr(default=None)

    def set_dag(self, dag: "DAG"):
        self._dag = dag

    @abstractmethod
    def consume_work(self, task: TaskWorkItem):
        """
        Abstract method to consume a work item.

        This method must be implemented by subclasses to define specific work consumption logic.

        Args:
            task (TaskWorkItem): The work item to be consumed.
        """
        pass

    def publish_work(self, task: TaskWorkItem, input_task: TaskWorkItem = None):
        """
        Publish a work item.

        This method handles the publishing of work items, including provenance tracking and consumer routing.

        Args:
            task (TaskWorkItem): The work item to be published.
            input_task (TaskWorkItem, optional): The input task that led to this work item.

        Raises:
            ValueError: If the task type is not in the output_types or if no consumer is registered for the task type.
        """
        if type(task) not in self.output_types:
            raise ValueError(
                f"Task {self.name} cannot publish work of type {type(task).__name__}"
            )

        # Copy provenance from input task if provided
        if input_task:
            task._provenance = input_task.copy_provenance()

        self._id += 1
        task._provenance[self.name] = self._id
        print(f"Task {self.name} published work with provenance {task._provenance}")

        # Verify if there is a consumer for the given task class
        consumer = self._consumers.get(task.__class__)
        if consumer is None:
            raise ValueError(f"No consumer registered for {task.__class__.__name__}")

        if self._dag and self._dag._dispatcher:
            self._dag._dispatcher.add_work(consumer, task)
        else:
            self._dispatch_work(task)

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
    task_a.publish_work(work_item)


if __name__ == "__main__":
    main()
