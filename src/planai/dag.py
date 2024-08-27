from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Dict, List, Optional, Set, Type

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .dispatcher import Dispatcher
from .task import TaskWorker, TaskWorkItem


class DAG(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    workers: Dict[str, TaskWorker] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)

    _dispatcher: Optional[Dispatcher] = PrivateAttr(default=None)
    _thread_pool: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor()

    def add_worker(self, task: TaskWorker) -> "DAG":
        """Add a task to the DAG."""
        if task.name in self.workers:
            raise ValueError(f"Task with name {task.name} already exists in the DAG.")
        self.workers[task.name] = task
        self.dependencies[task.name] = []
        task.set_dag(self)
        return self

    def set_dependency(self, upstream: str, downstream: str) -> "DAG":
        """Set a dependency between two tasks."""
        if upstream not in self.workers or downstream not in self.workers:
            raise ValueError(
                "Both tasks must be added to the DAG before setting dependencies."
            )

        if downstream not in self.dependencies[upstream]:
            self.dependencies[upstream].append(downstream)
            self.workers[upstream].register_consumer(
                task_cls=self.workers[downstream].get_taskworkitem_class(),
                consumer=self.workers[downstream],
            )

        return self

    def get_source_workers(self) -> Set[str]:
        """Return the set of tasks with no incoming dependencies."""
        all_tasks = set(self.workers.keys())
        tasks_with_dependencies = set()
        for dependencies in self.dependencies.values():
            tasks_with_dependencies.update(dependencies)
        return all_tasks - tasks_with_dependencies

    def validate_dag(self) -> List[str]:
        """Return the execution order of tasks based on dependencies."""
        in_degree = {worker: 0 for worker in self.workers}
        for dependencies in self.dependencies.values():
            for worker in dependencies:
                in_degree[worker] += 1

        queue = list(self.get_source_workers())
        execution_order = []

        while queue:
            worker = queue.pop(0)
            execution_order.append(worker)
            for dependent in self.dependencies[worker]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(execution_order) != len(self.workers):
            raise ValueError("Circular dependency detected in the DAG.")

        return execution_order

    def run(self, initial_tasks: List[TaskWorkItem]) -> None:
        """Execute the DAG by initiating source tasks."""
        source_workers = self.get_source_workers()

        # Validate initial work items
        if len(initial_tasks) != len(source_workers):
            raise ValueError(
                "Initial tasks must be provided for all and only source worker."
            )

        dispatcher = Dispatcher(self)
        dispatch_thread = Thread(target=dispatcher.dispatch)
        dispatch_thread.start()
        self._dispatcher = dispatcher

        accepted_work: Dict[Type["TaskWorkItem"], TaskWorker] = {}
        for worker_name in source_workers:
            worker = self.workers[worker_name]
            accepted_work[worker.get_taskworkitem_class()] = worker

        for task in initial_tasks:
            worker = accepted_work.get(type(task))
            if worker:
                worker.consume_work(task)
            else:
                raise ValueError(
                    f"Initial task {task} has no corresponding worker."
                )

        # Wait for all tasks to complete
        dispatcher.wait_for_completion()
        dispatcher.stop()
        dispatch_thread.join()
        self._thread_pool.shutdown(wait=True)

    def __str__(self) -> str:
        return f"DAG: {self.name} with {len(self.workers)} tasks"

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
    dag.add_worker(task1).add_worker(task2).add_worker(task3)

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
