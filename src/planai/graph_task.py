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
from typing import Any, List, Type

from pydantic import Field

from .graph import Graph
from .task import Task, TaskWorker

PRIVATE_STATE_KEY = "_graph_task_private_state"


class GraphTask(TaskWorker):
    graph: Graph = Field(
        ..., description="The graph that will be run as part of this TaskWorker"
    )
    entry_worker: TaskWorker = Field(..., description="The entry point of the graph")
    exit_worker: TaskWorker = Field(..., description="The exit point of the graph")

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
                        f"No task provenance found for {PRIVATE_STATE_KEY}"
                    )
                assert isinstance(task, Task)
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

    def get_task_class(self) -> Type[Task]:
        # usually the entry task gets dynamically determined from consume_work but we are overriding it here
        return self.entry_worker.get_task_class()

    def init(self):
        # we need to install the graph dispatcher into the sub-graph
        assert self._graph is not None
        self.graph._dispatcher = self._graph._dispatcher
        self.graph.init_workers()

    def consume_work(self, task: Task):
        # save the task provenance
        new_task = task.model_copy()
        new_task.add_private_state(PRIVATE_STATE_KEY, task)

        # and dispatch it to the sub-graph. this also sets the task provenance to InitialTaskWorker
        self.graph._add_work(self.entry_worker, new_task)


def main():
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
    graph_task = GraphTask(graph=sub_graph, entry_worker=task1, exit_worker=task2)

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
