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
from typing import Any, Dict, List, Type

from pydantic import Field, PrivateAttr

from .graph import Graph
from .provenance import ProvenanceChain
from .task import Task, TaskWorker

PRIVATE_STATE_KEY = "_graph_task_private_state"


class SubGraphWorkerInternal(TaskWorker):
    graph: Graph = Field(
        ..., description="The graph that will be run as part of this TaskWorker"
    )
    entry_worker: TaskWorker = Field(..., description="The entry point of the graph")
    exit_worker: TaskWorker = Field(..., description="The exit point of the graph")
    _state: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _: Any):
        if len(self.exit_worker.output_types) != 1:
            raise ValueError(
                f"Exit worker must have exactly one output type, got {self.exit_worker.output_types}"
            )

        output_type = self.exit_worker.output_types[0]
        graph_task = self
        self.output_types = [output_type]

        class AdapterSinkWorker(TaskWorker):
            output_types: List[Type[Task]] = [output_type]

            def consume_work(self, task: output_type):
                # we need to move this task from the sub-graph to the main graph
                # first, we need to fix up the provenance so that it shows only the GraphTask as the parent
                old_task = task.get_private_state(PRIVATE_STATE_KEY)
                if old_task is None:
                    raise ValueError(
                        f"No task state found for {PRIVATE_STATE_KEY}: {task._private_state} "
                        f"(provenance: {task._provenance})",
                    )
                assert isinstance(task, Task)

                # remove any metadata and callbacks before changing the provenance
                self.remove_state(task)

                task._add_input_provenance(old_task)
                task._add_worker_provenance(graph_task)

                # then we can add it to the main graph using the right consumer
                assert graph_task.exit_worker._graph is not None
                assert graph_task.exit_worker._graph._dispatcher is not None
                consumer = graph_task._get_consumer(task)
                graph_task.exit_worker._graph._dispatcher.add_work(consumer, task)

        instance = AdapterSinkWorker()
        self.graph.add_workers(instance)
        self.graph.set_dependency(self.exit_worker, instance)
        self.graph.set_entry(self.entry_worker)
        self.graph.finalize()  # compute the worker distances
        self.graph.init_workers()

    def get_task_class(self) -> Type[Task]:
        # usually the entry task gets dynamically determined from consume_work but we are overriding it here
        return self.entry_worker.get_task_class()

    def init(self):
        # we need to install the graph dispatcher into the sub-graph
        assert self._graph is not None
        self.graph._dispatcher = self._graph._dispatcher

    def consume_work(self, task: Task):
        new_task = task.copy_public()

        # save the task provenance
        # xxx - we really just need to remember the provenance of the task
        old_task = task.model_copy(deep=True)
        new_task.add_private_state(PRIVATE_STATE_KEY, old_task)

        # artificially increase the provenance
        logging.debug("Adding additional provenance for %s", old_task._provenance)
        self._graph._provenance_tracker._add_provenance(old_task)

        # get any associated state and re-inject it
        state = self.get_state(task)
        metadata = state["metadata"]
        callback = state["callback"]

        def inject_state(provenance: ProvenanceChain):
            self.graph.watch(provenance, self)
            # We inject True to indicate that extra provenance is still associated with the task
            logging.debug("Injecting state for %s in %s", provenance, self.name)
            with self.lock:
                self._state[provenance] = old_task

        # and dispatch it to the sub-graph. this also sets the task provenance to InitialTaskWorker
        self.graph._add_work(
            self.entry_worker,
            new_task,
            metadata=metadata,
            status_callback=callback,
            provenance_callback=inject_state,
        )

    def notify(self, prefix: str):
        # we don't know how many tasks a subgraph will produce. we need to hold on to
        # the extra provenance as long as there is work for the initial provenance. this
        # function will get called when all the work for the prefix is completed and
        # then we can remove additional provenance here and clean up state
        self.graph.unwatch(prefix, self)
        with self.lock:
            if prefix not in self._state:
                raise ValueError(
                    f"Task {prefix} does not have any associated state: {self._state}"
                )
            task = self._state.pop(prefix)

        logging.info(
            "Subgraph completed. Removing provenance for %s in %s",
            prefix,
            self.name,
        )
        self._graph._provenance_tracker._remove_provenance(task, self)

    def abort_work(self, provenance: ProvenanceChain):
        # map the provenance to the sub-graph provenance
        need_to_abort = []
        with self.lock:
            for prefix, task in self._state.items():
                if task._provenance[: len(provenance)] == list(provenance):
                    need_to_abort.append((prefix, provenance))
        # abort the mapped provenance in our graph
        for prefix, provenance in need_to_abort:
            logging.info(
                "Aborting %s in %s (mapped from %s)", prefix, self.name, provenance
            )
            self.graph.abort_work(prefix)


def SubGraphWorker(
    *,
    graph: Graph,
    entry_worker: TaskWorker,
    exit_worker: TaskWorker,
    name: str = "SubGraphWorker",
) -> SubGraphWorkerInternal:
    """
    Factory function to create a SubGraphWorker that manages a subgraph within a larger PlanAI graph.

    Parameters
    ----------
    name : str, optional
        Custom name for the SubGraphWorker class, defaults to "SubGraphWorker"
    graph : Graph
        The graph that will be run as part of this TaskWorker
    entry_worker : TaskWorker
        The entry point worker of the graph that receives initial tasks
    exit_worker : TaskWorker
        The exit point worker of the graph that produces final outputs
        Must have exactly one output type

    Returns
    -------
    SubGraphWorkerInternal
        A new instance of SubGraphWorker with the specified configuration

    Raises
    ------
    ValueError
        If the exit_worker has more than one output type
    """
    # Create a new class with the custom name
    CustomClass = type(name, (SubGraphWorkerInternal,), {})
    return CustomClass(graph=graph, entry_worker=entry_worker, exit_worker=exit_worker)


def main():  # pragma: no cover
    import argparse
    import random
    import time
    from typing import Type

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
            time.sleep(random.uniform(0.2, 0.9))
            for i in range(7):
                processed = f"Processed: {task.data.upper()} at iteration {i}"
                self.publish_work(
                    Task2WorkItem(processed_data=processed), input_task=task
                )

    class Task2Worker(TaskWorker):
        output_types: List[Type[Task]] = [Task3WorkItem]

        def consume_work(self, task: Task2WorkItem):
            self.print(f"Task2 consuming: {task.processed_data}")
            time.sleep(random.uniform(0.3, 2.5))

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
            time.sleep(random.uniform(0.4, 1.2))
            self.print("Workflow complete!")

    # Create Graph
    sub_graph = Graph(name="Simple SubGraph")

    # Create tasks
    task1 = Task1Worker()
    task2 = Task2Worker()

    # Add tasks to Graph
    sub_graph.add_workers(task1, task2)

    # Set dependencies
    sub_graph.set_dependency(task1, task2)

    # Create the graph task
    graph_task = SubGraphWorker(
        name="SubGraph", graph=sub_graph, entry_worker=task1, exit_worker=task2
    )

    # Create the final consumer
    task3 = Task3Worker()

    graph = Graph(name="Simple Graph")
    graph.add_workers(graph_task, task3)
    graph.set_dependency(graph_task, task3)

    # Prepare initial work item
    initial_work = [
        (graph_task, Task1WorkItem(data="Hello, Graph v1!")),
        (graph_task, Task1WorkItem(data="Hello, Graph v2!")),
    ]

    # Run the Graph
    graph.run(
        initial_work,
        run_dashboard=args.run_dashboard,
        display_terminal=not args.run_dashboard,
    )


if __name__ == "__main__":
    main()
