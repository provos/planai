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
import random
import threading
import time
import unittest
from typing import List, Type

from pydantic import PrivateAttr

from planai.dispatcher import Dispatcher
from planai.graph import Graph
from planai.joined_task import JoinedTaskWorker
from planai.task import Task, TaskWorker


class Task1(Task):
    data: str


class Task2(Task):
    data: str
    source: str


class Task3(Task):
    data: List[str]


class TaskWorker1(TaskWorker):
    output_types: List[Type[Task]] = [Task1]

    _processed_count: int = PrivateAttr(0)

    def consume_work(self, task: Task1):
        output = Task1(data=f"{task.data} start")
        self.publish_work(output, task)
        self._processed_count += 1


class TaskWorker2(TaskWorker):
    output_types: List[Type[Task]] = [Task2]

    _processed_count: int = PrivateAttr(0)

    def consume_work(self, task: Task1):
        for i in range(3):
            output = Task2(data=f"{task.data}-{i}", source=self.name)
            self.publish_work(output, task)
        self._processed_count += 1


class TaskWorker3(JoinedTaskWorker):
    join_type: Type[TaskWorker] = TaskWorker1
    _processed_count: int = PrivateAttr(0)
    _processed_items: int = PrivateAttr(0)

    def consume_work_joined(self, tasks: List[Task2]):
        prefixes = set([task.prefix_for_input_task(TaskWorker1) for task in tasks])
        if len(prefixes) != 1:
            raise ValueError("All tasks must have the same prefix", prefixes)

        with self._lock:
            self._processed_count += 1
            self._processed_items += len(tasks)


class TestJoinedTaskWorker(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(name="Test Graph")
        self.dispatcher = Dispatcher(self.graph)
        self.graph._dispatcher = self.dispatcher

        self.worker1 = TaskWorker1()
        self.worker2 = TaskWorker2()
        self.worker3 = TaskWorker3()

        self.graph.add_workers(self.worker1, self.worker2, self.worker3)
        self.graph.set_dependency(self.worker1, self.worker2).next(self.worker3)

    def test_joined_task_worker(self):
        # Create initial tasks
        initial_tasks = [Task1(data=f"Initial {i}") for i in range(3)]

        # Start the dispatcher in a separate thread
        dispatch_thread = threading.Thread(target=self.dispatcher.dispatch)
        dispatch_thread.start()

        # Add initial work
        for task in initial_tasks:
            self.dispatcher.add_work(self.worker1, task)

        # Wait for all work to be processed
        self.dispatcher.wait_for_completion()
        self.dispatcher.stop()
        dispatch_thread.join()

        # Check results
        self.assertEqual(self.worker1._processed_count, 3)
        self.assertEqual(
            self.worker2._processed_count, 3
        )  # 3 initial tasks * 3 outputs from worker1
        self.assertEqual(self.worker3._processed_count, 3)  # 3 joined results

        # Validate that TaskWorker3 accumulated the results correctly
        self.assertEqual(
            len(self.worker3._joined_results), 0
        )  # All joined results should have been processed


class InitialTask(Task):
    data: str


class IntermediateTask(Task):
    data: str
    source: str


class FinalTask(Task):
    data: List[str]


class InitialTaskWorker(TaskWorker):
    output_types: List[Type[Task]] = [InitialTask]

    def consume_work(self, task: InitialTask):
        # Generate multiple tasks
        for i in range(3):
            output = InitialTask(data=f"{task.data}-{i}")
            self.publish_work(output, task)
        time.sleep(random.uniform(0.001, 0.01))  # Simulate some work


class IntermediateTaskWorker(TaskWorker):
    output_types: List[Type[Task]] = [IntermediateTask]

    def consume_work(self, task: InitialTask):
        output = IntermediateTask(data=f"Processed-{task.data}", source=self.name)
        for i in range(3):
            self.publish_work(output, task)
            time.sleep(random.uniform(0.001, 0.01))  # Simulate some work


final_task_data = []


class FinalJoinedTaskWorker(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker

    def consume_work(self, task: IntermediateTask):
        super().consume_work(task)

    def consume_work_joined(self, tasks: List[IntermediateTask]):
        with self._lock:
            final_task_data.append(FinalTask(data=[task.data for task in tasks]))


class TestJoinedTaskWorkerStress(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(name="Stress Test Graph")
        self.dispatcher = Dispatcher(self.graph)
        self.graph._dispatcher = self.dispatcher

        self.initial_worker = InitialTaskWorker()
        self.intermediate_worker = IntermediateTaskWorker()
        self.final_worker = FinalJoinedTaskWorker()

        self.graph.add_workers(
            self.initial_worker, self.intermediate_worker, self.final_worker
        )
        self.graph.set_dependency(self.initial_worker, self.intermediate_worker).next(
            self.final_worker
        )

    def test_joined_task_worker_stress(self):
        num_initial_tasks = 100
        initial_tasks = [
            InitialTask(data=f"Initial {i}") for i in range(num_initial_tasks)
        ]

        # Start the dispatcher in a separate thread
        dispatch_thread = threading.Thread(target=self.dispatcher.dispatch)
        dispatch_thread.start()

        # Function to add initial work
        def add_initial_work():
            for task in initial_tasks:
                self.dispatcher.add_work(self.initial_worker, task)

        # Start adding work in a separate thread
        add_work_thread = threading.Thread(target=add_initial_work)
        add_work_thread.start()

        # Wait for all work to be processed
        add_work_thread.join()
        self.dispatcher.wait_for_completion()
        self.dispatcher.stop()
        dispatch_thread.join()

        # Check results
        self.assertEqual(
            self.dispatcher.total_completed_tasks,
            num_initial_tasks * 3
            + num_initial_tasks * 3 * 3
            + num_initial_tasks * (3 + 1),  # 3 results plus notify
        )
        self.assertEqual(self.dispatcher.total_failed_tasks, 0)

        self.assertEqual(len(final_task_data), num_initial_tasks * 3)

        # Verify that each final task contains exactly 3 intermediate task results
        for task in final_task_data:
            self.assertEqual(len(task.data), 3)

        # Verify that the work queue is empty and there are no active tasks
        self.assertEqual(self.dispatcher.work_queue.qsize(), 0)
        self.assertEqual(self.dispatcher._num_active_tasks, 0)


# Tasks for state management test
class UpstreamTask(Task):
    iteration: int


class MiddleTask(Task):
    data: str


class CompletedTask(Task):
    result: str


# Global state to verify counter doesn't reset prematurely
state_verification = {"global_count": 0, "times_reset": 0}


class UpstreamWorker(TaskWorker):
    output_types: List[Type[Task]] = [UpstreamTask]

    def consume_work(self, task: UpstreamTask):
        # Just pass through
        output = UpstreamTask(iteration=task.iteration)
        self.publish_work(output, task)


class MiddleWorker(TaskWorker):
    output_types: List[Type[Task]] = [MiddleTask]

    def consume_work(self, task: UpstreamTask):
        # Generate multiple tasks for joining
        for i in range(3):
            output = MiddleTask(data=f"middle-{task.iteration}-{i}")
            self.publish_work(output, task)


class StatefulJoinedWorker(JoinedTaskWorker):
    join_type: Type[TaskWorker] = UpstreamWorker
    output_types: List[Type[Task]] = [UpstreamTask, CompletedTask]

    def consume_work_joined(self, tasks: List[MiddleTask]):
        # Get state for the prefix(1) - this is the UpstreamWorker provenance
        prefix = tasks[0].prefix(1)
        state = self.get_worker_state(prefix)

        # Track counter in state
        if "count" not in state:
            state["count"] = 0

        state["count"] += 1
        current_count = state["count"]

        # Update global verification state
        with self._lock:
            state_verification["global_count"] += 1
            # If count went backwards, state was reset prematurely
            if current_count < state_verification["global_count"]:
                state_verification["times_reset"] += 1
            # Protect against regressions by terminating if we detect a loop
            if state_verification["global_count"] > 10:
                raise ValueError("Loop detected in state management")

        if current_count < 3:
            # Send more work back to upstream - creates circular dependency
            next_task = UpstreamTask(iteration=current_count)
            self.publish_work(next_task, tasks[0])
        else:
            # We've reached count 3, send final task
            final = CompletedTask(result=f"completed-count:{current_count}")
            self.publish_work(final, tasks[0])


class TestJoinedTaskWorkerWithState(unittest.TestCase):
    """Test that get_worker_state() works correctly with JoinedTaskWorker."""

    def test_joined_worker_with_state_no_premature_cleanup(self):
        """
        Test that worker state is not cleaned up prematurely when using
        get_worker_state() inside a JoinedTaskWorker.

        This test verifies:
        1. The state counter increments correctly (1, 2, 3)
        2. State is not reset prematurely (no infinite loop)
        3. The final task is produced after count reaches 3
        4. No endless loop occurs
        """
        # Reset global state
        state_verification["global_count"] = 0
        state_verification["times_reset"] = 0

        # Create graph with real dispatcher
        graph = Graph(name="State Test Graph")

        upstream_worker = UpstreamWorker()
        middle_worker = MiddleWorker()
        joined_worker = StatefulJoinedWorker()

        graph.add_workers(upstream_worker, middle_worker, joined_worker)
        graph.set_dependency(upstream_worker, middle_worker).next(joined_worker)

        # Set up circular dependency: joined_worker -> upstream_worker
        graph.set_dependency(joined_worker, upstream_worker)

        # Set up sink for final task
        graph.set_sink(joined_worker, CompletedTask)

        # Run the graph
        initial_task = UpstreamTask(iteration=0)
        initial_tasks = [(upstream_worker, initial_task)]
        graph.run(initial_tasks, display_terminal=False)

        # Get output
        outputs = graph.get_output_tasks()

        # Verify we got exactly one completed task
        self.assertEqual(len(outputs), 1, "Should have exactly one completed task")
        self.assertEqual(
            outputs[0].result, "completed-count:3", "Final count should be 3"
        )

        # Verify counter incremented correctly without resetting
        self.assertEqual(
            state_verification["global_count"],
            3,
            "Counter should have reached 3",
        )
        self.assertEqual(
            state_verification["times_reset"],
            0,
            "State should never have been reset prematurely",
        )

        # Verify state was cleaned up after completion
        self.assertEqual(
            len(joined_worker._user_state),
            0,
            "Worker state should be cleaned up after completion",
        )


if __name__ == "__main__":
    unittest.main()
