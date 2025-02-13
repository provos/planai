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
import shutil
import time
from collections import deque
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type

from colorama import Fore, Style, init
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .dispatcher import Dispatcher
from .joined_task import InitialTaskWorker
from .provenance import ProvenanceChain, ProvenanceTracker
from .task import Task, TaskStatusCallback, TaskWorker

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Provide the provenance for state tracking before work is added to the dispatcher
ProvenanceCallback = Callable[["ProvenanceChain"], None]


class Graph(BaseModel):
    """A graph for orchestrating task workers and their dependencies.

    The Graph class manages the execution flow of tasks through workers, handling dependencies,
    parallel execution, monitoring, and output collection. It supports both terminal-based
    and web dashboard monitoring of task execution.

    Attributes:
        name (str): Name identifier for the graph instance
        strict (bool): If True, the graph will enforce strict validation of tasks provided to publish_work()
        workers (Set[TaskWorker]): Set of task workers in the graph
        dependencies (Dict[TaskWorker, List[TaskWorker]]): Maps workers to their downstream dependencies

    Example:
        >>> graph = Graph(name="Data Processing Pipeline")
        >>> worker1 = DataLoader()
        >>> worker2 = DataProcessor()
        >>> graph.add_workers(worker1, worker2)
        >>> graph.set_dependency(worker1, worker2)
        >>> graph.run([(worker1, LoadTask(file="data.csv"))])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    strict: bool = False
    workers: Set[TaskWorker] = Field(default_factory=set)
    dependencies: Dict[TaskWorker, List[TaskWorker]] = Field(default_factory=dict)

    _dispatcher: Optional[Dispatcher] = PrivateAttr(default=None)
    _dispatcher_created: bool = PrivateAttr(default=False)
    _terminal_thread: Optional[Thread] = PrivateAttr(default=None)
    _provenance_tracker: ProvenanceTracker = PrivateAttr(
        default_factory=ProvenanceTracker
    )

    _max_parallel_tasks: Dict[Type[TaskWorker], int] = PrivateAttr(default_factory=dict)
    _sink_tasks: List[Type[Task]] = PrivateAttr(default_factory=list)
    _sink_workers: List[TaskWorker] = PrivateAttr(default_factory=list)
    _initial_worker: TaskWorker = PrivateAttr(default=None)
    _subgraph_workers: Set[TaskWorker] = PrivateAttr(default_factory=set)

    _worker_distances: Dict[str, Dict[str, int]] = PrivateAttr(default_factory=dict)
    _has_terminal: bool = PrivateAttr(default=False)

    def __init__(self, **data):
        super().__init__(**data)
        self._initial_worker = InitialTaskWorker()
        self._provenance_tracker = ProvenanceTracker(name=self.name)
        self.add_worker(self._initial_worker)

    def trace(self, prefix: ProvenanceChain):
        self._provenance_tracker.trace(prefix)

    def watch(self, prefix: ProvenanceChain, notifier: TaskWorker) -> bool:
        return self._provenance_tracker.watch(prefix, notifier)

    def unwatch(self, prefix: ProvenanceChain, notifier: TaskWorker) -> bool:
        return self._provenance_tracker.unwatch(prefix, notifier)

    def add_worker(self, worker: TaskWorker) -> "Graph":
        """Adds a single task worker to the graph.

        Args:
            task (TaskWorker): The worker instance to add to the graph

        Returns:
            Graph: The graph instance for method chaining

        Raises:
            ValueError: If the worker is already present in the graph

        Example:
            >>> graph = Graph(name="Pipeline")
            >>> worker = DataProcessor()
            >>> graph.add_worker(worker)
        """
        if worker in self.workers:
            raise ValueError(f"Task {worker} already exists in the Graph.")
        if self.strict:
            # This causes the worker to check during publish_work() that the provenance fields of the new
            # task are empty. This can help with a common bug patterns where a task is reused.
            worker._strict_checking = True

        self.workers.add(worker)
        # Check if any class in the MRO hierarchy is named 'SubGraphWorkerInternal'
        if any(
            cls.__name__ == "SubGraphWorkerInternal" for cls in type(worker).__mro__
        ):
            self._subgraph_workers.add(worker)
        self.dependencies[worker] = []
        worker.set_graph(self)
        return self

    def get_dispatcher(self) -> Optional[Dispatcher]:
        """Get the dispatcher instance for the graph."""
        return self._dispatcher

    def add_workers(self, *workers: TaskWorker) -> "Graph":
        """Add multiple tasks to the Graph."""
        for worker in workers:
            self.add_worker(worker)
        return self

    def get_worker_by_input_type(self, input_type: Type[Task]) -> Optional[TaskWorker]:
        """Get a worker that consumes a specific input type.

        This method searches through registered workers to find one that processes
        the specified input task type.

        Args:
            input_type (Type[Task]): The input task type class to match against workers.

        Returns:
            Optional[TaskWorker]: The matching worker if found, None otherwise.

        Example:
            worker = graph.get_worker_by_input_type(ImageTask)
        """
        for worker in self.workers:
            if input_type == worker.get_task_class():
                return worker
        return None

    def get_worker_by_output_type(
        self, output_type: Type[Task]
    ) -> Optional[TaskWorker]:
        """Get a worker that produces a specific output type.

        This method searches through registered workers to find one that produces
        the specified output task type.

        Args:
            output_type (Type[Task]): The output task type class to match against workers.

        Returns:
            Optional[TaskWorker]: The matching worker if found, None otherwise.

        Example:
            worker = graph.get_worker_by_output_type(ImageTask)
        """
        for worker in self.workers:
            if output_type in worker.output_types:
                return worker
        return None

    def set_dependency(
        self, upstream: TaskWorker, downstream: TaskWorker
    ) -> TaskWorker:
        """Set a dependency between two tasks."""
        return self._set_dependency(upstream, downstream)

    def _set_dependency(
        self, upstream: TaskWorker, downstream: TaskWorker, register: bool = True
    ) -> TaskWorker:
        if upstream not in self.workers or downstream not in self.workers:
            raise ValueError(
                f"Both workers (upstream: {upstream.__class__.__name__}) (downstream: {downstream.__class__.__name__}) must be added to the Graph before setting dependencies."
            )

        if downstream not in self.dependencies[upstream]:
            self.dependencies[upstream].append(downstream)
            if register:
                upstream.register_consumer(
                    task_cls=downstream.get_task_class(),
                    consumer=downstream,
                )

        return downstream

    def set_sink(
        self,
        worker: TaskWorker,
        output_type: Type[Task],
        notify: Optional[Callable[[Dict[str, Any], Task], None]] = None,
    ) -> None:
        """Designates a worker as a data sink for collecting specific output tasks.

        A sink worker is a special endpoint in the graph that collects and optionally processes
        output tasks of a specific type. The sink can either store tasks for later retrieval
        or forward them to a notification callback.

        Args:
            worker (TaskWorker): The worker whose output should be collected
            output_type (Type[Task]): The specific task type to collect at this sink
            notify (Callable[[Dict[str, Any], None], Task], optional): Callback function
                that receives the task's metadata and the task itself. If provided, tasks
                won't be stored in the sink's collection.

        Raises:
            ValueError: If the specified worker doesn't have the correct output type.
            ValueError: If the specifier output type has already been registered as an output

        Note:
            - The sink worker is automatically added to the graph and set as a dependency of the specified worker.
            - The output type of the specified worker is used to type the tasks consumed by the sink.
            - Only workers with a single output type can be set as sinks.

        Example:
            >>> graph = Graph(name="Example Graph")
            >>> worker = SomeTaskWorker()
            >>> graph.add_worker(worker)
            >>> graph.set_sink(worker, OutputTask)
            >>> graph.run(initial_tasks=[(worker, SomeTask())])
            >>> results = graph.get_output_tasks()
        """
        if output_type not in worker.output_types:
            raise ValueError(
                f"Worker {worker.name} does not have output type {output_type.__name__} to use for a sink"
            )

        class SinkWorker(TaskWorker):
            def __init__(self, graph: "Graph", **data):
                super().__init__(**data)
                self._graph = graph
                self._notify = notify

            def consume_work(self, task: output_type):
                if self._notify:
                    metadata = self.get_metadata(task)
                    logging.info("SinkWorker is notifying on task %s", task.name)
                    self._notify(metadata, task)
                else:
                    with self.lock:
                        self._graph._sink_tasks.append(task)
                # we should not remove metadata or callbacks here as we don't know
                # that all provenance has been removed

        # Create a new class with the specific output type
        instance = SinkWorker(self)
        self.add_worker(instance)
        self.set_dependency(worker, instance)

        # Set this as the sink worker for the graph
        self._sink_workers.append(instance)
        logging.info(
            "Sink on worker %s set for output type %s",
            worker.name,
            output_type.__name__,
        )

    def get_output_tasks(self) -> List[Type[Task]]:
        """
        Retrieves all tasks that were consumed by the sink workers in the graph.

        This method returns a list of tasks that were collected by all sink workers
        after the graph has been run. Each task in the list is an instance of the
        output type specified when the corresponding sink was set.

        Returns:
            List[Type[Task]]: A list of tasks collected by all sink workers. The actual
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
            >>> results = graph.get_output_tasks()

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

    def finalize(self):
        self.compute_worker_distances()

    def prepare(
        self,
        run_dashboard: bool = False,
        display_terminal: bool = True,
        dashboard_port: int = 5000,
    ) -> None:
        """Initializes the graph for execution by setting up monitoring and worker components.

        This method must be called before executing tasks. It sets up:
        - Task dispatcher for managing worker execution
        - Optional web dashboard for monitoring
        - Optional terminal-based status display
        - Worker parallel execution limits

        Args:
            run_dashboard (bool): If True, starts a web interface for monitoring
            display_terminal (bool): If True, shows execution progress in terminal
            dashboard_port (int): Port number for the web dashboard if enabled

        Example:
            >>> graph = Graph(name="Pipeline")
            >>> # ... add workers and dependencies ...
            >>> graph.prepare(run_dashboard=True, dashboard_port=8080)
            >>> graph.execute(initial_tasks)
        """
        # Empty the sink tasks
        if self._sink_workers:
            self._sink_tasks = []
            if run_dashboard:
                logging.warning(
                    "The dashboard will make the graph wait for manual termination. "
                    "This is usually not desired when using a sink worker."
                )

        # Allow workers to log messages
        self._log_lines = []

        # Start the dispatcher
        dispatcher = self._dispatcher
        if self._dispatcher is None:
            dispatcher = Dispatcher(self, web_port=dashboard_port)
            self._dispatcher_created = True
        elif not self._dispatcher_created:
            logging.info("Graph %s is using an existing dispatcher", self.name)
            if run_dashboard or display_terminal:
                raise RuntimeError(
                    "Dispatcher is already running. Should not start dashboard or terminal display."
                )

        dispatcher.start()
        if run_dashboard:
            dispatcher.start_web_interface()
        self._dispatcher = dispatcher

        if display_terminal:
            self._has_terminal = True
            self._start_terminal_display()
            self._terminal_thread = Thread(target=self._terminal_display_thread)
            self._terminal_thread.start()
        else:
            self._has_terminal = False

        # Apply the max parallel tasks settings
        for worker_class, max_parallel_tasks in self._max_parallel_tasks.items():
            dispatcher.set_max_parallel_tasks(worker_class, max_parallel_tasks)

    def run(
        self,
        initial_tasks: Sequence[Tuple[TaskWorker, Task]],
        run_dashboard: bool = False,
        display_terminal: bool = True,
        dashboard_port: int = 5000,
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
            display_terminal (bool, optional): If True, displays a terminal status for the graph
                execution. Defaults to True.
            dashboard_port (int, optional): The port number for the web interface. Defaults to 5000.

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
        self.prepare(
            run_dashboard=run_dashboard,
            display_terminal=display_terminal,
            dashboard_port=dashboard_port,
        )
        self.execute(initial_tasks)

    def execute(self, initial_tasks: Sequence[Tuple[TaskWorker, Task]]) -> None:
        """Executes the graph with the provided initial tasks.

        This method starts the actual task processing in the graph. It should be called
        after prepare() has been used to set up the execution environment.

        Args:
            initial_tasks (Sequence[Tuple[TaskWorker, Task]]): A sequence of worker-task
                pairs to start the graph execution

        Raises:
            Exception: If task validation fails for any worker-task pair
            RuntimeError: If prepare() hasn't been called first

        Note:
            - Blocks until all tasks complete unless a dashboard is running
            - Automatically handles worker initialization and cleanup
            - Maintains execution state for monitoring and debugging

        Example:
            >>> graph.prepare()
            >>> initial = [(worker, Task(data="start"))]
            >>> graph.execute(initial)
        """
        self.set_entry(*[x[0] for x in initial_tasks])
        # Finalize the graph (compute distances)
        self.finalize()

        # let the workers know that we are about to start
        self.init_workers()

        for worker, task in initial_tasks:
            success, error = worker.validate_task(type(task), worker)
            if not success:
                raise error

            self._add_work(worker, task)

        # Wait for all tasks to complete
        logging.info("Graph %s started - waiting for completion", self.name)
        self._dispatcher.wait_for_completion(
            wait_for_quit=self._dispatcher.has_dashboard()
        )
        logging.info("Graph %s completed", self.name)

        # we'll stop the dispatcher only if we created it
        if self._dispatcher_created:
            self._dispatcher.stop()
            logging.info("Dispatcher stopped")

        if self._has_terminal:
            self._stop_terminal_display_event.set()
            self._terminal_thread.join()
            logging.info("Terminal display stopped")

        for worker in self.workers:
            worker.completed()

        logging.info("All workers completed - execution finished")

    def init_workers(self):
        for worker in self.workers:
            worker.init()

    def _add_work(
        self,
        worker: TaskWorker,
        task: Task,
        metadata: Optional[Dict] = None,
        status_callback: Optional[TaskStatusCallback] = None,
        provenance_callback: Optional[ProvenanceCallback] = None,
    ) -> ProvenanceChain:
        provenance = self._initial_worker.get_next_provenance()
        task._provenance = [provenance] + task._provenance

        prov_chain = task.prefix(1)

        # Register state before adding work
        if metadata or status_callback:
            self._provenance_tracker.add_state(prov_chain, metadata, status_callback)

        # Allows workers to do additional state tracking based on the provenance
        if provenance_callback:
            provenance_callback(prov_chain)

        assert self._dispatcher is not None
        self._dispatcher.add_work(worker, task)
        return prov_chain

    def add_work(
        self,
        worker: TaskWorker,
        task: Task,
        metadata: Optional[Dict] = None,
        status_callback: Optional[TaskStatusCallback] = None,
    ) -> ProvenanceChain:
        if worker not in self.dependencies[self._initial_worker]:
            raise ValueError(
                f"Worker {worker.name} is not an entry point to the Graph."
            )
        return self._add_work(worker, task, metadata, status_callback)

    def abort_work(self, provenance: ProvenanceChain) -> bool:
        """
        This method attempts to abort work that is currently in progress. It first checks if the
        provenance chain exists in the provenance tracker. If found, it aborts the work through
        the dispatcher and propagates the abort request to all subgraph workers. If the provenance
        chain is not found, a warning is logged.

        Returns:
            bool: True if the work was aborted successfully, False otherwise
        """
        # We don't need to worry about race conditions here as provenance chains are unique
        # and either they exists in the provenance tracker or they have completed already
        if self._provenance_tracker.has_provenance(provenance):
            self._dispatcher.abort_work(self, provenance)
            for worker in self._subgraph_workers:
                worker.abort_work(provenance)
            return True

        logging.warning("Provenance %s not found to abort work", provenance)
        return False

    def set_entry(self, *workers: TaskWorker) -> "Graph":
        """Set the workers that are entry points to the Graph.

        This method establishes connections from the initial (root) worker to the specified
        workers, marking them as entry points in the execution graph.

        Args:
            *workers (TaskWorker): Variable number of TaskWorker instances to be set as
                entry points.

        Returns:
            Graph: The Graph instance itself for method chaining.

        Example:
            ```
            graph = Graph()
            worker1 = TaskWorker()
            worker2 = TaskWorker()
            graph.set_entry(worker1, worker2)
            ```
        """
        for worker in workers:
            self._set_dependency(self._initial_worker, worker, register=False)
        return self

    def compute_worker_distances(self):
        for worker in self.workers:
            self._worker_distances[worker.name] = self._bfs_distances(worker)

    def _bfs_distances(self, start_worker: TaskWorker) -> Dict[str, int]:
        distances = {start_worker.name: 0}
        queue = deque([(start_worker, 0)])
        visited = set()

        while queue:
            current_worker, distance = queue.popleft()
            if current_worker in visited:
                continue
            visited.add(current_worker)

            for downstream in self.dependencies.get(current_worker, []):
                if (
                    downstream.name not in distances
                    or distance + 1 < distances[downstream.name]
                ):
                    distances[downstream.name] = distance + 1
                    queue.append((downstream, distance + 1))

        return distances

    def _start_terminal_display(self):
        self._stop_terminal_display_event = Event()
        self._stop_terminal_display_event.clear()

    def _terminal_display_thread(self):
        try:
            while not self._stop_terminal_display_event.is_set():
                self.display_terminal_status()
                time.sleep(0.25)  # Update interval
        finally:
            self._clear_terminal()
            self._print_log()

    def _clear_terminal(self):
        # Clear the terminal when the thread is terminating
        print("\033[H\033[J")

    def _print_log(self, max_lines=15):
        print("\nLog:")
        if max_lines <= 0:
            for line in self._log_lines:
                print(line)
            return

        # Get terminal width and leave a margin of 10 characters
        terminal_width = max(shutil.get_terminal_size().columns - 10, 20)

        # Flatten and wrap the log lines
        flattened_lines = []
        for line in self._log_lines:
            for subline in line.splitlines():
                while len(subline) > terminal_width:
                    flattened_lines.append(subline[:terminal_width])
                    subline = subline[terminal_width:]
                flattened_lines.append(subline)

        # Print the last 'max_lines' number of lines
        for line in flattened_lines[-max_lines:]:
            print(line)

    def display_terminal_status(self):
        stats = self._dispatcher.get_execution_statistics()
        terminal_size = shutil.get_terminal_size((80, 20))
        terminal_width = terminal_size.columns
        terminal_height = terminal_size.lines

        print("\033[H\033[J")  # Clear terminal

        # Sort the workers based on their distance from InitialTaskWorker
        distances = self._worker_distances.get("InitialTaskWorker", {})
        sorted_workers = sorted(
            stats.items(), key=lambda item: distances.get(item[0], float("inf"))
        )

        for worker, data in sorted_workers:
            completed = data["completed"]
            active = data["active"]
            queued = data["queued"]
            failed = data["failed"]

            total_tasks = completed + active + queued + failed
            available_width = (
                terminal_width - 60
            ) // 2  # Adjust for worker name and separator

            if total_tasks > available_width:
                scale_factor = available_width / total_tasks
                completed_scaled = (
                    max(1, int(completed * scale_factor)) if completed > 0 else 0
                )
                active_scaled = max(1, int(active * scale_factor)) if active > 0 else 0
                queued_scaled = max(1, int(queued * scale_factor)) if queued > 0 else 0
                failed_scaled = max(1, int(failed * scale_factor)) if failed > 0 else 0
            else:
                completed_scaled, active_scaled, queued_scaled, failed_scaled = (
                    completed,
                    active,
                    queued,
                    failed,
                )

            # Create bars
            completed_bar = Fore.GREEN + "🟩" * completed_scaled
            active_bar = Fore.BLUE + "🔵" * active_scaled
            queued_bar = Fore.LIGHTYELLOW_EX + "🟠" * queued_scaled
            failed_bar = Fore.RED + "❌" * failed_scaled

            # First print: worker name and bars
            status_line = f"{worker[:25]:25} | {completed_bar}{active_bar}{queued_bar}{failed_bar}{Style.RESET_ALL}"
            print(status_line, end="")

            # Calculate and print counts at the right edge
            counts = f"C:{completed} A:{active} Q:{queued} F:{failed}"

            # Move cursor to the right edge minus the length of counts
            cursor_move = f"\033[{terminal_width - len(counts)}G"

            print(f"{cursor_move}{counts}")

        lines_to_print = terminal_height - len(sorted_workers) - 4
        self._print_log(max_lines=lines_to_print)

        # Reset the cursor to the top
        print("\033[H")

    def print(self, *args):
        message = " ".join(str(arg) for arg in args)
        logging.info("Application: %s", message)
        if self._dispatcher is not None and self._dispatcher.has_dashboard():
            self._dispatcher.add_log(message)
        if self._has_terminal:
            self._log_lines.append(message)
        else:
            print(*args)

    def __str__(self) -> str:
        return f"Graph: {self.name} with {len(self.workers)} tasks"

    def __repr__(self) -> str:
        return self.__str__()

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Gracefully shuts down the graph and all its components.

        Args:
            timeout (float): Maximum time to wait for tasks to complete in seconds

        Returns:
            bool: True if shutdown was successful, False if timeout occurred
        """
        logging.info("Initiating graph shutdown...")

        if not self._dispatcher:
            # nothing to do if the dispatcher was never created
            return True

        if self._dispatcher_created:
            # Signal dispatcher to stop accepting new work
            self._dispatcher.initiate_shutdown()

            # Wait for active tasks to complete or timeout
            self._dispatcher.wait_for_completion(timeout=timeout, wait_for_quit=False)

            # Stop the dispatcher thread
            self._dispatcher.stop(timeout=1.0)

            # Stop terminal display if active
            if self._has_terminal and self._terminal_thread:
                self._stop_terminal_display_event.set()
                self._terminal_thread.join(timeout=1.0)
        else:
            self._dispatcher.deregister_graph(self)

        logging.info("Graph shutdown completed successfully")
        return True

    def register_dispatcher(self, dispatcher: Dispatcher) -> None:
        """Register a dispatcher for this graph.

        Registers a dispatcher instance with the graph. This is useful when running multiple different graphs.

        Args:
            dispatcher (Dispatcher): The dispatcher instance to register.

        Returns:
            None

        """
        self._dispatcher = dispatcher
        self._dispatcher_created = False
        self._dispatcher.register_graph(self)


def main():  # pragma: no cover
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Simple Graph Example")
    parser.add_argument(
        "--run-dashboard", action="store_true", help="Run the web dashboard"
    )
    args = parser.parse_args()

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
            self.print(f"Task1 consuming: {task.data}")
            time.sleep(random.uniform(0.4, 1.5))
            for i in range(7):
                processed = f"Processed: {task.data.upper()} at iteration {i}"
                self.publish_work(
                    Task2WorkItem(processed_data=processed), input_task=task
                )

    class Task2Worker(TaskWorker):
        output_types: List[Type[Task]] = [Task3WorkItem]

        def consume_work(self, task: Task2WorkItem):
            self.print(f"Task2 consuming: {task.processed_data}")
            time.sleep(random.uniform(0.5, 2.5))

            if args.run_dashboard:
                # demonstrate the ability to request user input
                if random.random() < 0.15:
                    result, mime_type = self.request_user_input(
                        task=task,
                        instruction="Please provide a value",
                        accepted_mime_types=["text/html", "application/pdf"],
                    )
                    self.print(
                        f"User input: {len(result) if result else None} ({mime_type})"
                    )

            for i in range(11):
                final = f"Final: {task.processed_data} at iteration {i}!"
                self.publish_work(Task3WorkItem(final_result=final), input_task=task)

    class Task3Worker(TaskWorker):
        output_types: List[Type[Task]] = []

        def consume_work(self, task: Task3WorkItem):
            self.print(f"Task3 consuming: {task.final_result}")
            time.sleep(random.uniform(0.6, 1.2))
            self.print("Workflow complete!")

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

    # Prepare initial work item
    initial_work = [
        (task1, Task1WorkItem(data="Hello, Graph v1!")),
        (task1, Task1WorkItem(data="Hello, Graph v2!")),
    ]

    # Run the Graph
    graph.run(
        initial_work,
        run_dashboard=args.run_dashboard,
        dashboard_port=8080,
        display_terminal=not args.run_dashboard,
    )


if __name__ == "__main__":
    main()
