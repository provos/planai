import inspect
from typing import Any, Dict, Set, Tuple, Type, get_type_hints

from pydantic import BaseModel, Field, PrivateAttr


class TaskWorkItem(BaseModel):
    _provenance: Dict[str, int] = PrivateAttr(default_factory=dict)

    def copy_provenance(self) -> Dict[str, int]:
        return self._provenance.copy()


class TaskWorker(BaseModel):
    name: str = Field(..., title="Task name", description="The name of the task")
    output_types: Set[Type[TaskWorkItem]] = Field(
        default_factory=set, description="The types of work this task can output"
    )

    _id: int = PrivateAttr(default=0)
    _consumers: Dict[Type["TaskWorker"], "TaskWorker"] = PrivateAttr(
        default_factory=dict
    )

    def consume_work(self, task: TaskWorkItem):
        print(f"Task {self.name} consumed work with provenance {task._provenance}")

    def publish_work(self, task: TaskWorkItem, input_task: TaskWorkItem = None):
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
        if consumer:
            consumer.consume_work(task)
        else:
            raise ValueError(f"No consumer registered for {task.__class__.__name__}")

    def validate_taskworkitem(
        self, task_cls: Type[TaskWorkItem], consumer: "TaskWorker"
    ) -> Tuple[bool, Exception]:
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
        consume_method = getattr(self, "consume_work", None)
        if not consume_method:
            raise AttributeError(f"{self.__class__.__name__} has no method consume_work")
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
