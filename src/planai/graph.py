from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Dict, List, Optional, Set, Type

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .dispatcher import Dispatcher
from .task import TaskWorker, TaskWorkItem


class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    workers: Set[TaskWorker] = Field(default_factory=set)
    dependencies: Dict[TaskWorker, List[TaskWorker]] = Field(default_factory=dict)

    _dispatcher: Optional[Dispatcher] = PrivateAttr(default=None)
    _thread_pool: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor()

    def add_worker(self, task: TaskWorker) -> "Graph":
        """Add a task to the Graph."""
        if task in self.workers:
            raise ValueError(f"Task {task} already exists in the Graph.")
        self.workers.add(task)
        self.dependencies[task] = []
        task.set_graph(self)
        return self

    def add_workers(self, *workers: TaskWorker) -> "Graph":
        """Add multiple tasks to the Graph."""
        for worker in workers:
            self.add_worker(worker)
        return self

    def set_dependency(
        self, upstream: TaskWorker, downstream: TaskWorker
    ) -> TaskWorker:
        """Set a dependency between two tasks."""
        if upstream not in self.workers or downstream not in self.workers:
            raise ValueError(
                "Both tasks must be added to the Graph before setting dependencies."
            )

        if downstream not in self.dependencies[upstream]:
            self.dependencies[upstream].append(downstream)
            upstream.register_consumer(
                task_cls=downstream.get_taskworkitem_class(),
                consumer=downstream,
            )

        return downstream

    def get_source_workers(self) -> Set[TaskWorker]:
        """Return the set of tasks with no incoming dependencies."""
        all_tasks = set(self.workers)
        tasks_with_dependencies = set()
        for dependencies in self.dependencies.values():
            tasks_with_dependencies.update(dependencies)
        return all_tasks - tasks_with_dependencies

    def validate_graph(self) -> List[TaskWorker]:
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
            raise ValueError("Circular dependency detected in the Graph.")

        return execution_order

    def run(self, initial_tasks: List[TaskWorkItem]) -> None:
        """Execute the Graph by initiating source tasks."""
        source_workers = self.get_source_workers()

        # Validate initial work items
        if len(initial_tasks) != len(source_workers):
            raise ValueError(
                "Initial tasks must be provided for all and only source worker."
            )

        dispatcher = Dispatcher(self)
        dispatch_thread = Thread(target=dispatcher.dispatch)
        dispatch_thread.start()
        dispatcher.start_web_interface()
        self._dispatcher = dispatcher

        accepted_work: Dict[Type["TaskWorkItem"], TaskWorker] = {}
        for worker in source_workers:
            accepted_work[worker.get_taskworkitem_class()] = worker

        # let the workers now that we are about to start
        for worker in self.workers:
            worker.init()

        for task in initial_tasks:
            worker = accepted_work.get(type(task))
            if worker:
                worker._pre_consume_work(task)
            else:
                raise ValueError(f"Initial task {task} has no corresponding worker.")

        # Wait for all tasks to complete
        dispatcher.wait_for_completion(wait_for_quit=True)
        dispatcher.stop()
        dispatch_thread.join()
        self._thread_pool.shutdown(wait=True)

        for worker in self.workers:
            worker.completed()

    def __str__(self) -> str:
        return f"Graph: {self.name} with {len(self.workers)} tasks"

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

    # Create Graph
    graph = Graph(name="Simple Workflow")

    # Create tasks
    task1 = Task1Worker()
    task2 = Task2Worker()
    task3 = Task3Worker()

    # Add tasks to Graph
    graph.add_worker(task1).add_worker(task2).add_worker(task3)

    # Set dependencies
    graph.set_dependency(task1, task2).next(task3)

    # Validate Graph
    execution_order = graph.validate_graph()
    print(f"Execution order: {execution_order}")

    # Prepare initial work item
    initial_work = [Task1WorkItem(data="Hello, Graph!")]

    # Run the Graph
    graph.run(initial_work)


if __name__ == "__main__":
    main()
