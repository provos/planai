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
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .dispatcher import Dispatcher
from .joined_task import InitialTaskWorker
from .task import Task, TaskType, TaskWorker


class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    workers: Set[TaskWorker] = Field(default_factory=set)
    dependencies: Dict[TaskWorker, List[TaskWorker]] = Field(default_factory=dict)

    _dispatcher: Optional[Dispatcher] = PrivateAttr(default=None)
    _thread_pool: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)
    _max_parallel_tasks: Dict[Type[TaskWorker], int] = PrivateAttr(default_factory=dict)
    _sink_tasks: List[TaskType] = PrivateAttr(default_factory=list)
    _sink_worker: Optional[TaskWorker] = PrivateAttr(default=None)

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
                f"Both workers (upstream: {upstream.__class__.__name__}) (downstream: {downstream.__class__.__name__}) must be added to the Graph before setting dependencies."
            )

        if downstream not in self.dependencies[upstream]:
            self.dependencies[upstream].append(downstream)
            upstream.register_consumer(
                task_cls=downstream.get_task_class(),
                consumer=downstream,
            )

        return downstream

    def set_sink(self, worker: TaskWorker) -> None:
        """
        Sets a worker as a data sink in the task graph.

        This method creates a special SinkWorker that consumes the output of the specified worker.
        The output from this sink can be retrieved after the graph is run using the `get_tasks()` method.

        Args:
            worker (TaskWorker): The worker whose output should be collected in the sink.

        Raises:
            ValueError: If the specified worker doesn't have exactly one output type.
            RuntimeError: If a sink worker has already been set for this graph.

        Note:
            - The sink worker is automatically added to the graph and set as a dependency of the specified worker.
            - The output type of the specified worker is used to type the tasks consumed by the sink.
            - Only workers with a single output type can be set as sinks.

        Example:
            >>> graph = Graph(name="Example Graph")
            >>> worker = SomeTaskWorker()
            >>> graph.add_worker(worker)
            >>> graph.set_sink(worker)
            >>> graph.run(initial_tasks=[(worker, SomeTask())])
            >>> results = graph.get_tasks()
        """
        if self._sink_worker is not None:
            raise RuntimeError("A sink worker has already been set for this graph.")

        if len(worker.output_types) != 1:
            raise ValueError(
                f"Worker {worker.name} must have exactly one output type to use for a sink"
            )

        output_type = worker.output_types[0]

        class SinkWorker(TaskWorker):
            def __init__(self, graph: "Graph", **data):
                super().__init__(**data)
                self._graph = graph

            def consume_work(self, task: output_type):
                with self.lock:
                    self._graph._sink_tasks.append(task)

        # Create a new class with the specific output type
        instance = SinkWorker(self)
        self.add_worker(instance)
        self.set_dependency(worker, instance)

        # Set this as the sink worker for the graph
        self._sink_worker = instance

    def get_tasks(self) -> List[TaskType]:
        """
        Retrieves all tasks that were consumed by the sink workers in the graph.

        This method returns a list of tasks that were collected by all sink workers
        after the graph has been run. Each task in the list is an instance of the
        output type specified when the corresponding sink was set.

        Returns:
            List[TaskType]: A list of tasks collected by all sink workers. The actual
            type of each task depends on the output types of the workers set as sinks.

        Note:
            - This method should be called after the graph has been run using the `run()` method.
            - If no sinks were set or if the graph hasn't been run, this method will return an empty list.
            - The order of tasks in the list corresponds to the order they were consumed by the sink workers.

        Example:
            >>> graph = Graph(name="Example Graph")
            >>> worker = SomeTaskWorker()
            >>> graph.add_worker(worker)
            >>> graph.set_sink(worker)
            >>> graph.run(initial_tasks=[(worker, SomeTask())])
            >>> results = graph.get_tasks()

        See Also:
            set_sink(): Method for setting a worker as a sink in the graph.
        """
        return self._sink_tasks

    def set_max_parallel_tasks(
        self, worker_class: Type[TaskWorker], max_parallel_tasks: int
    ) -> None:
        """
        Set the maximum number of parallel tasks for a specific worker class.

        Args:
            worker_class (Type[TaskWorker]): The class of the worker to limit.
            max_parallel_tasks (int): The maximum number of parallel tasks allowed.

        Note:
            This setting will be applied to the dispatcher when the graph is run.
            If the dispatcher is already running, it will update the limit dynamically.
        """
        if not issubclass(worker_class, TaskWorker):
            raise ValueError(f"{worker_class.__name__} is not a subclass of TaskWorker")

        if max_parallel_tasks <= 0:
            raise ValueError("max_parallel_tasks must be greater than 0")

        # Store the setting
        self._max_parallel_tasks[worker_class] = max_parallel_tasks

        # If dispatcher is already running, update it
        if self._dispatcher:
            self._dispatcher.set_max_parallel_tasks(worker_class, max_parallel_tasks)

    def validate_graph(self) -> None:
        """Return the execution order of tasks based on dependencies."""
        pass

    def run(
        self,
        initial_tasks: List[Tuple[TaskWorker, Task]],
        run_dashboard: bool = False,
    ) -> None:
        """
        Execute the Graph by initiating source tasks and managing the workflow.

        This method sets up the Dispatcher, initializes workers, and processes the initial tasks.
        It manages the entire execution flow of the graph until completion.

        Args:
            initial_tasks (List[Tuple[TaskWorker, Task]]): A list of tuples, each containing
                a TaskWorker and its corresponding Task to initiate the graph execution.
            run_dashboard (bool, optional): If True, starts a web interface for monitoring the
                graph execution. Defaults to False.

        Raises:
            ValueError: If any of the initial tasks fail validation.

        Note:
            - This method blocks until all tasks in the graph are completed.
            - It handles the initialization and cleanup of workers, dispatcher, and thread pool.
            - If run_dashboard is True, the method will wait for manual termination of the web interface.

        Example:
            graph = Graph(name="My Workflow")
            # ... (add workers and set dependencies)
            initial_work = [(task1, Task1WorkItem(data="Start"))]
            graph.run(initial_work, run_dashboard=True)
        """

        # Empty the sink tasks
        if self._sink_worker:
            self._sink_tasks = []
            if run_dashboard:
                logging.warning(
                    "The dashboard will make the graph wait for manual termination. This is usually not desired when using a sink worker."
                )

        # Start the dispatcher
        dispatcher = Dispatcher(self)
        dispatch_thread = Thread(target=dispatcher.dispatch)
        dispatch_thread.start()
        if run_dashboard:
            dispatcher.start_web_interface()
        self._dispatcher = dispatcher

        # Apply the max parallel tasks settings
        for worker_class, max_parallel_tasks in self._max_parallel_tasks.items():
            dispatcher.set_max_parallel_tasks(worker_class, max_parallel_tasks)

        # let the workers now that we are about to start
        for worker in self.workers:
            worker.init()

        # initial mock worker
        origin_worker = InitialTaskWorker()

        for worker, task in initial_tasks:
            success, error = worker.validate_task(type(task), worker)
            if not success:
                raise error

            task._provenance = [(origin_worker.name, 1)]

            dispatcher.add_work(worker, task)

        # Wait for all tasks to complete
        dispatcher.wait_for_completion(wait_for_quit=run_dashboard)
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
    # Define custom Task classes
    class Task1WorkItem(Task):
        data: str

    class Task2WorkItem(Task):
        processed_data: str

    class Task3WorkItem(Task):
        final_result: str

    # Define custom TaskWorker classes
    class Task1Worker(TaskWorker):
        output_types: List[Type[Task]] = [Task2WorkItem]

        def consume_work(self, task: Task1WorkItem):
            print(f"Task1 consuming: {task.data}")
            processed = f"Processed: {task.data.upper()}"
            self.publish_work(Task2WorkItem(processed_data=processed), input_task=task)

    class Task2Worker(TaskWorker):
        output_types: List[Type[Task]] = [Task3WorkItem]

        def consume_work(self, task: Task2WorkItem):
            print(f"Task2 consuming: {task.processed_data}")
            final = f"Final: {task.processed_data}!"
            self.publish_work(Task3WorkItem(final_result=final), input_task=task)

    class Task3Worker(TaskWorker):
        output_types: Set[Type[Task]] = set()

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
    initial_work = [(task1, Task1WorkItem(data="Hello, Graph!"))]

    # Run the Graph
    graph.run(initial_work)


if __name__ == "__main__":
    main()
