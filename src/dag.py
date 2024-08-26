from typing import Dict, List, Set, Type

from pydantic import BaseModel, Field

from src.task import TaskWorker, TaskWorkItem


class DAG(BaseModel):
    name: str
    tasks: Dict[str, TaskWorker] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)

    def add_task(self, task: TaskWorker) -> "DAG":
        """Add a task to the DAG."""
        self.tasks[task.name] = task
        self.dependencies[task.name] = []
        return self

    def set_dependency(self, upstream: str, downstream: str) -> "DAG":
        """Set a dependency between two tasks."""
        if upstream not in self.tasks or downstream not in self.tasks:
            raise ValueError(
                "Both tasks must be added to the DAG before setting dependencies."
            )

        if downstream not in self.dependencies[upstream]:
            self.dependencies[upstream].append(downstream)
            self.tasks[upstream].register_consumer(
                task_cls=self.tasks[downstream].get_taskworkitem_class(),
                consumer=self.tasks[downstream],
            )

        return self

    def get_source_tasks(self) -> Set[str]:
        """Return the set of tasks with no incoming dependencies."""
        all_tasks = set(self.tasks.keys())
        tasks_with_dependencies = set()
        for dependencies in self.dependencies.values():
            tasks_with_dependencies.update(dependencies)
        return all_tasks - tasks_with_dependencies

    def validate_dag(self) -> List[str]:
        """Return the execution order of tasks based on dependencies."""
        in_degree = {task: 0 for task in self.tasks}
        for dependencies in self.dependencies.values():
            for task in dependencies:
                in_degree[task] += 1

        queue = list(self.get_source_tasks())
        execution_order = []

        while queue:
            task = queue.pop(0)
            execution_order.append(task)
            for dependent in self.dependencies[task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(execution_order) != len(self.tasks):
            raise ValueError("Circular dependency detected in the DAG.")

        return execution_order

    def run(self, initial_work_items: List[TaskWorkItem]) -> None:
        """Execute the DAG by initiating source tasks."""
        source_tasks = self.get_source_tasks()

        # Validate initial work items
        if len(initial_work_items) != len(source_tasks):
            raise ValueError(
                "Initial work items must be provided for all and only source tasks."
            )

        accepted_work: Dict[Type["TaskWorkItem"], TaskWorker] = {}
        for task_name in source_tasks:
            task = self.tasks[task_name]
            accepted_work[task.get_taskworkitem_class()] = task

        for work_item in initial_work_items:
            task = accepted_work.get(type(work_item))
            if task:
                task.consume_work(work_item)
            else:
                raise ValueError(
                    f"Initial work item {work_item} has no corresponding task."
                )

    def __str__(self) -> str:
        return f"DAG: {self.name} with {len(self.tasks)} tasks"

    def __repr__(self) -> str:
        return self.__str__()


def main():
    # Define custom TaskWorkItem classes
    class Task1WorkItem(TaskWorkItem):
        data: str

    class Task2WorkItem(TaskWorkItem):
        processed_data: str

    class Task3WorkItem(TaskWorkItem):
        final_result: str

    # Define custom TaskWorker classes
    class Task1Worker(TaskWorker):
        output_types: Set[Type[TaskWorkItem]] = {Task2WorkItem}

        def consume_work(self, task: Task1WorkItem):
            print(f"Task1 consuming: {task.data}")
            processed = f"Processed: {task.data.upper()}"
            self.publish_work(Task2WorkItem(processed_data=processed), input_task=task)

    class Task2Worker(TaskWorker):
        output_types: Set[Type[TaskWorkItem]] = {Task3WorkItem}

        def consume_work(self, task: Task2WorkItem):
            print(f"Task2 consuming: {task.processed_data}")
            final = f"Final: {task.processed_data}!"
            self.publish_work(Task3WorkItem(final_result=final), input_task=task)

    class Task3Worker(TaskWorker):
        output_types: Set[Type[TaskWorkItem]] = set()

        def consume_work(self, task: Task3WorkItem):
            print(f"Task3 consuming: {task.final_result}")
            print("Workflow complete!")

    # Create DAG
    dag = DAG(name="Simple Workflow")

    # Create tasks
    task1 = Task1Worker(name="Task1")
    task2 = Task2Worker(name="Task2")
    task3 = Task3Worker(name="Task3")

    # Add tasks to DAG
    dag.add_task(task1).add_task(task2).add_task(task3)

    # Set dependencies
    dag.set_dependency("Task1", "Task2")
    dag.set_dependency("Task2", "Task3")

    # Validate DAG
    execution_order = dag.validate_dag()
    print(f"Execution order: {execution_order}")

    # Prepare initial work item
    initial_work = [Task1WorkItem(data="Hello, DAG!")]

    # Run the DAG
    dag.run(initial_work)


if __name__ == "__main__":
    main()
