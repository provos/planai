import threading
import time
import unittest
from typing import List, Type
from unittest.mock import Mock

from planai.graph import Graph, InitialTaskWorker
from planai.graph_task import SubGraphWorker
from planai.joined_task import JoinedTaskWorker
from planai.task import Task, TaskWorker


class InputTask(Task):
    data: str


class SubGraphTask(Task):
    data: str
    intermediate: bool = False  # Indicate if this task is from the subgraph


class FinalTask(Task):
    result: str


class MainWorker(TaskWorker):
    output_types: List[Type[Task]] = [SubGraphTask]

    def consume_work(self, task: InputTask):
        # Simulate processing in the main graph
        time.sleep(0.1)
        sub_task = SubGraphTask(data=task.data)
        self.publish_work(sub_task, input_task=task)


class SubGraphHandler(TaskWorker):
    output_types: List[Type[Task]] = [SubGraphTask]

    def consume_work(self, task: SubGraphTask):
        # Mark task as processed in subgraph
        task.intermediate = True
        # Simulate processing in the subgraph
        time.sleep(0.1)
        self.publish_work(task, input_task=task)


class FinalWorker(TaskWorker):
    output_types: List[Type[Task]] = []

    def consume_work(self, task: SubGraphTask):
        # Simulate final processing in the main graph
        time.sleep(0.1)
        assert task.intermediate, "Task should have been processed by subgraph"
        assert (
            len(task._provenance) == 3
        ), f"Provenance length should be 3 but got {task._provenance}"
        # The subgraph's provenance should not appear here
        expected_provenance = [
            ("InitialTaskWorker", 1),
            ("MainWorker", 1),
            ("SubGraphWorker", 1),
        ]
        self.verify_provenance(task, expected_provenance)

    def verify_provenance(self, task: Task, expected_provenance):
        actual_provenance = task._provenance
        assert actual_provenance == expected_provenance, (
            f"Provenance mismatch!\nExpected: {expected_provenance}\n"
            f"Actual: {actual_provenance}"
        )


class TestGraphTask(unittest.TestCase):
    def test_graph_task_provenance(self):
        # Create subgraph
        subgraph = Graph(name="SubGraph")
        subgraph_worker = SubGraphHandler()
        subgraph_entry = subgraph_worker  # Entry point
        subgraph_exit = subgraph_worker  # Exit point
        subgraph.add_workers(subgraph_worker)

        # Create GraphTask
        graph_task = SubGraphWorker(
            graph=subgraph, entry_worker=subgraph_entry, exit_worker=subgraph_exit
        )

        # Create main graph
        graph = Graph(name="MainGraph")
        main_worker = MainWorker()
        final_worker = FinalWorker()

        graph.add_workers(main_worker, graph_task, final_worker)
        graph.set_dependency(main_worker, graph_task).next(final_worker)

        # Prepare initial work item
        initial_task = InputTask(data="TestData")
        initial_work = [(main_worker, initial_task)]

        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Ensure that the dispatcher has completed all tasks
        dispatcher = graph._dispatcher
        self.assertIsNotNone(dispatcher)
        assert dispatcher is not None
        self.assertEqual(dispatcher.work_queue.qsize(), 0)
        self.assertEqual(dispatcher.active_tasks, 0)
        self.assertEqual(
            dispatcher.total_completed_tasks, 3 + 2 + 1
        )  # 3 tasks in the main graph, 1 in the subgraph + the AdapterWorker + Notify to clean up

        # Check that the provenance is as expected
        final_tasks = graph.get_output_tasks()
        self.assertEqual(len(final_tasks), 0)  # Since FinalWorker has no output

        # Optionally, we can check logs or other side effects if needed

    def test_graph_task_with_joined_task_worker(self):
        # Create subgraph with a JoinedTaskWorker
        subgraph = Graph(name="SubGraphWithJoin")

        class SubInputTask(Task):
            data: str

        class SubInitWorker(TaskWorker):
            output_types: List[Type[Task]] = [SubGraphTask]

            def consume_work(self, task: SubGraphTask):
                assert len(task._provenance) == 1, "Provenance length should be 3"
                # The subgraph's provenance should not appear here
                assert task._provenance == [
                    ("InitialTaskWorker", 1)
                ], "Provenance mismatch"

                sub_task = SubGraphTask(data=task.data)
                self.publish_work(sub_task, input_task=task)

        class SubWorker(TaskWorker):
            output_types: List[Type[Task]] = [SubInputTask]

            def consume_work(self, task: SubGraphTask):
                # Simulate processing in the subgraph
                time.sleep(0.1)
                # Publish multiple SubInputTasks for joining
                for i in range(3):
                    sub_task = SubInputTask(data=f"{task.data}-{i}")
                    self.publish_work(sub_task, input_task=task)

        class SubJoinWorker(JoinedTaskWorker):
            join_type: Type[TaskWorker] = SubInitWorker
            output_types: List[Type[Task]] = [SubGraphTask]

            def consume_work(self, task: SubInputTask):
                super().consume_work(task)  # Handle joining

            def consume_work_joined(self, tasks: List[SubInputTask]):
                # Simulate processing after joining tasks
                time.sleep(0.1)
                # Concatenate data from joined tasks
                joined_data = ";".join([t.data for t in tasks])
                output_task = SubGraphTask(data=joined_data, intermediate=True)
                self.publish_work(output_task, input_task=tasks[0])

        sub_init_worker = SubInitWorker()
        sub_worker = SubWorker()
        sub_join_worker = SubJoinWorker()
        subgraph_entry = sub_init_worker
        subgraph_exit = sub_join_worker

        subgraph.add_workers(sub_init_worker, sub_worker, sub_join_worker)
        subgraph.set_dependency(sub_init_worker, sub_worker).next(sub_join_worker)

        # Create GraphTask
        graph_task = SubGraphWorker(
            graph=subgraph, entry_worker=subgraph_entry, exit_worker=subgraph_exit
        )

        # Create main graph
        graph = Graph(name="MainGraphWithJoin")
        main_worker = MainWorker()
        final_worker = FinalWorker()

        graph.add_workers(main_worker, graph_task, final_worker)
        graph.set_dependency(main_worker, graph_task).next(final_worker)

        # Prepare initial work item
        initial_task = InputTask(data="TestData")
        initial_work = [(main_worker, initial_task)]

        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Ensure that the dispatcher has completed all tasks
        dispatcher = graph._dispatcher
        self.assertIsNotNone(dispatcher)
        assert dispatcher is not None
        self.assertEqual(dispatcher.work_queue.qsize(), 0)
        self.assertEqual(dispatcher.active_tasks, 0)

        # The total completed tasks should include:
        # - MainWorker: 1
        # - GraphTask: 1
        # - FinalWorker: 1
        # - SubWorker: 1 (publishes 3 SubInputTasks)
        # - SubJoinWorker: handles the 3 SubInputTasks and calls consume_work_joined once
        # - AdapterSinkWorker: 1
        total_main_graph_tasks = 3  # main_worker, graph_task, final_worker
        total_subgraph_tasks = (
            1  # sub_init_worker
            + 1  # sub_worker consumes SubGraphTask
            + 3  # sub_join_worker consumes 3 SubInputTasks
            + 1  # sub_join_worker consumes joined tasks
            + 1  # AdapterSinkWorker
            + 1  # Notify to clean up state
        )
        expected_total_tasks = total_main_graph_tasks + total_subgraph_tasks

        self.assertEqual(dispatcher.total_completed_tasks, expected_total_tasks)

        # Since FinalWorker has no output, check that it processed the task correctly
        final_tasks = graph.get_output_tasks()
        self.assertEqual(len(final_tasks), 0)

        # Optionally, you can verify that the joined data is correct
        # But since FinalWorker does not output anything, and we didn't sink any tasks, we can't retrieve outputs here

    def test_multiple_output_provenance(self):
        # Create a worker that outputs multiple tasks
        class MultiOutputWorker(TaskWorker):
            output_types: List[Type[Task]] = [SubGraphTask]

            def consume_work(self, task: SubGraphTask):
                # Output multiple tasks from the subgraph
                for i in range(3):
                    new_task = SubGraphTask(data=f"{task.data}-{i}", intermediate=True)
                    self.publish_work(new_task, input_task=task)

        # Create subgraph with multiple output worker
        subgraph = Graph(name="MultiOutputSubGraph")
        multi_worker = MultiOutputWorker()
        subgraph.add_workers(multi_worker)

        # Create GraphTask
        graph_task = SubGraphWorker(
            graph=subgraph, entry_worker=multi_worker, exit_worker=multi_worker
        )

        # Create tracking worker to verify each output
        received_tasks = []

        class TrackingWorker(TaskWorker):
            output_types: List[Type[Task]] = []

            def consume_work(self, task: SubGraphTask):
                received_tasks.append(task)
                # Verify provenance for each task
                assert (
                    len(task._provenance) == 3
                ), f"Wrong provenance length: {task._provenance}"
                # The SubGraphWorker counter will increment for each task
                provenance = task._provenance
                assert provenance[0] == (
                    "InitialTaskWorker",
                    1,
                ), f"Wrong initial provenance: {provenance[0]}"
                assert provenance[1] == (
                    "MainWorker",
                    1,
                ), f"Wrong main worker provenance: {provenance[1]}"
                assert (
                    provenance[2][0] == "SubGraphWorker"
                ), f"Wrong subgraph worker name: {provenance[2]}"
                assert (
                    1 <= provenance[2][1] <= 3
                ), f"Subgraph worker counter out of range: {provenance[2]}"

        # Create main graph
        graph = Graph(name="MainGraph")
        main_worker = MainWorker()
        tracking_worker = TrackingWorker()

        graph.add_workers(main_worker, graph_task, tracking_worker)
        graph.set_dependency(main_worker, graph_task).next(tracking_worker)

        # Run graph
        initial_task = InputTask(data="TestData")
        initial_work = [(main_worker, initial_task)]

        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Verify we received all tasks with correct provenance
        self.assertEqual(len(received_tasks), 3)
        for task in received_tasks:
            self.assertTrue(task.intermediate)
            self.assertTrue(task.data.startswith("TestData-"))

        # Verify dispatcher state
        dispatcher = graph._dispatcher
        self.assertIsNotNone(dispatcher)
        self.assertEqual(dispatcher.work_queue.qsize(), 0)
        self.assertEqual(dispatcher.active_tasks, 0)

        # Verify total task count:
        # Main graph: 1 (MainWorker) + 1 (GraphTask) + 3 (TrackingWorker)
        # Subgraph: 1 (MultiOutputWorker) + 3 (AdapterSinkWorker) + 1 (Notify to clean up)
        self.assertEqual(dispatcher.total_completed_tasks, 10)

    def test_complex_multiple_output_provenance(self):
        # Create a worker that multiples in the initial task
        class SecondWorker(TaskWorker):
            output_types: List[Type[Task]] = [SubGraphTask]

            def consume_work(self, task: SubGraphTask):
                # Output multiple tasks from the subgraph
                for i in range(2):
                    new_task = SubGraphTask(data=f"{task.data}-{i}", intermediate=True)
                    self.publish_work(new_task, input_task=task)

        # Create a worker that creates two tasks for each input
        class BranchingWorker(TaskWorker):
            output_types: List[Type[Task]] = [SubGraphTask]

            def consume_work(self, task: SubGraphTask):
                # Create two new tasks from each input
                for i in range(2):
                    new_task = SubGraphTask(
                        data=f"{task.data}-branch{i}", intermediate=True
                    )
                    self.publish_work(new_task, input_task=task)

        # Create a worker that outputs multiple tasks for each input
        class MultiOutputWorker(JoinedTaskWorker):
            output_types: List[Type[Task]] = [SubGraphTask]
            join_type: Type[TaskWorker] = InitialTaskWorker

            def consume_work_joined(self, tasks: List[SubGraphTask]):
                # Output multiple tasks from each branched task
                for i in range(3):
                    new_task = SubGraphTask(
                        data=f"{tasks[0].data}-out{i}", intermediate=True
                    )
                    self.publish_work(new_task, input_task=tasks[0])

        # Create subgraph with branching and multiple output workers
        subgraph = Graph(name="ComplexSubGraph")
        branch_worker = BranchingWorker()
        multi_worker = MultiOutputWorker()
        subgraph.add_workers(branch_worker, multi_worker)
        subgraph.set_dependency(branch_worker, multi_worker)

        # Create GraphTask
        graph_task = SubGraphWorker(
            graph=subgraph, entry_worker=branch_worker, exit_worker=multi_worker
        )

        # Track received tasks and their provenance
        received_tasks = []

        class TrackingWorker(JoinedTaskWorker):
            output_types: List[Type[Task]] = []
            join_type: Type[TaskWorker] = InitialTaskWorker

            def consume_work_joined(self, tasks: List[SubGraphTask]):
                for task in tasks:
                    received_tasks.append(task)
                    # Verify provenance for each task
                    assert (
                        len(task._provenance) == 4
                    ), f"Wrong provenance length: {task._provenance}"
                    provenance = task._provenance
                    assert provenance[0] == (
                        "InitialTaskWorker",
                        1,
                    ), f"Wrong initial provenance: {provenance[0]}"
                    assert provenance[1] == (
                        "MainWorker",
                        1,
                    ), f"Wrong main worker provenance: {provenance[1]}"
                    assert (
                        provenance[3][0] == "SubGraphWorker"
                    ), f"Wrong subgraph worker name: {provenance[2]}"
                    # Counter should be in range of total outputs (2 branches × 3 outputs)
                    assert (
                        1 <= provenance[3][1] <= 12
                    ), f"Subgraph worker counter out of range: {provenance[2]}"

        # Create main graph
        graph = Graph(name="MainGraph")
        main_worker = MainWorker()
        second_worker = SecondWorker()
        tracking_worker = TrackingWorker()

        graph.add_workers(main_worker, second_worker, graph_task, tracking_worker)
        graph.set_dependency(main_worker, second_worker).next(graph_task).next(
            tracking_worker
        )

        # Run graph with multiple initial tasks
        initial_work = [(main_worker, InputTask(data="TestData1"))]

        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Verify we received all tasks with correct provenance
        # For each initial task: 2 branches × 3 outputs = 3 final tasks
        expected_tasks = len(initial_work) * 2 * 3
        self.assertEqual(len(received_tasks), expected_tasks)

        # Verify task data patterns
        data_patterns = set()
        for task in received_tasks:
            self.assertTrue(task.intermediate)
            # Verify the task data follows the expected pattern
            self.assertTrue(task.data.startswith("TestData"))
            self.assertTrue("-branch" in task.data)
            self.assertTrue("-out" in task.data)
            data_patterns.add(task.data.split("-")[0])

        # Verify we got outputs from all initial tasks
        self.assertEqual(data_patterns, {"TestData1"})

        # Verify dispatcher state
        dispatcher = graph._dispatcher
        self.assertIsNotNone(dispatcher)
        self.assertEqual(dispatcher.work_queue.qsize(), 0)
        self.assertEqual(dispatcher.active_tasks, 0)

        # Verify total task count:
        # Main graph: 1 (MainWorker) + 2 (GraphTask) + (12 + 2) (TrackingWorker)
        # Subgraph: 2 (BranchingWorker) + 2 (MultiOutputWorker) + 6 (AdapterSinkWorker) + 2 (Notify)
        self.assertEqual(dispatcher.total_completed_tasks, 27)

    def test_abort_subgraph_propagation(self):

        class InputTask(Task):
            data: str

        class OutputTask(Task):
            data: str

        class FirstWorker(TaskWorker):
            output_types: List[Type[Task]] = [OutputTask]

            def consume_work(self, task: InputTask):
                time.sleep(0.1)  # Simulate work
                for i in range(2):
                    output = OutputTask(data=f"{task.data}-{i}")
                    self.publish_work(output, input_task=task)

        class SubGraphInnerFirst(TaskWorker):
            output_types: List[Type[Task]] = [OutputTask]

            def consume_work(self, task: OutputTask):
                time.sleep(0.5)  # Simulate work
                for i in range(2):
                    output = OutputTask(data=f"{task.data}-inner1-{i}")
                    self.publish_work(output, input_task=task)

        class SubGraphInnerSecond(TaskWorker):
            output_types: List[Type[Task]] = [OutputTask]

            def consume_work(self, task: OutputTask):
                time.sleep(0.4)  # Simulate work
                for i in range(2):
                    output = OutputTask(data=f"{task.data}-inner2-{i}")
                    self.publish_work(output, input_task=task)

        # Create subgraph
        subgraph = Graph(name="SubGraph")
        inner_first = SubGraphInnerFirst()
        inner_second = SubGraphInnerSecond()
        subgraph.add_workers(inner_first, inner_second)
        subgraph.set_dependency(inner_first, inner_second)

        # Create GraphTask
        graph_task = SubGraphWorker(
            graph=subgraph, entry_worker=inner_first, exit_worker=inner_second
        )

        # Create main graph
        graph = Graph(name="MainGraph")
        first_worker = FirstWorker()

        graph.add_workers(first_worker, graph_task)
        graph.set_dependency(first_worker, graph_task).sink(OutputTask)
        graph.set_max_parallel_tasks(TaskWorker, 3)

        # Create initial tasks
        task1 = InputTask(data="task1")
        task2 = InputTask(data="task2")

        # Prepare graph
        graph.prepare(display_terminal=False)
        graph.set_entry(first_worker)

        # Add tasks and get their provenance
        prov1 = graph.add_work(first_worker, task1)
        _ = graph.add_work(first_worker, task2)

        def abort_thread():
            time.sleep(0.4)  # Let some tasks start processing
            graph.abort_work(prov1)

        # Start a thread to abort the work
        abort_thread = threading.Thread(target=abort_thread)
        abort_thread.start()

        # Start execution and wait a bit
        graph.execute([])

        # Verify results
        dispatcher = graph._dispatcher
        self.assertIsNotNone(dispatcher)

        # Calculate potential task chain for task1:
        # - FirstWorker: 1 input -> 2 outputs
        # - SubGraphInnerFirst: 2 inputs -> 4 outputs
        # - SubGraphInnerSecond: 4 inputs -> 8 outputs
        # Total: 15 tasks (1 + 2 + 4 + 8)
        max_aborted_tasks = 5  # We expect to catch the abort early in the chain

        # Count tasks from aborted chain
        processed_aborted_count = 0
        processed_subgraph_count = 0
        for _, completed_task in dispatcher.completed_tasks:
            provenance = completed_task._provenance
            if provenance[: len(prov1)] == list(prov1):
                processed_aborted_count += 1
            if len(provenance) > 1 and provenance[1][0] == "SubGraphInnerFirst":
                processed_subgraph_count += 1

        self.assertEqual(
            processed_subgraph_count,
            12,
            f"Too many subgraph tasks were processed: {processed_subgraph_count}",
        )

        self.assertLessEqual(
            processed_aborted_count,
            max_aborted_tasks,
            f"Too many aborted tasks were processed: {processed_aborted_count}",
        )

        # Calculate tasks from task2's chain (same structure as task1)
        expected_minimum_tasks = 10  # Conservative estimate for task2 chain
        self.assertGreaterEqual(
            dispatcher.total_completed_tasks,
            expected_minimum_tasks,
            f"Expected at least {expected_minimum_tasks} completed tasks from non-aborted chain",
        )

        # Verify queue and active tasks are empty
        self.assertEqual(dispatcher.work_queue.qsize(), 0)
        self.assertEqual(dispatcher.active_tasks, 0)


class StatusNotifyingWorker(TaskWorker):
    output_types: List[Type[Task]] = [FinalTask]

    def consume_work(self, task: InputTask):
        # Notify about processing status
        self.notify_status(task, "Processing started")

        # Simulate some work
        output = FinalTask(result=f"Processed: {task.data}")

        self.notify_status(task, "Processing complete")
        self.publish_work(output, input_task=task)


class TestSubGraphMetadataAndCallbacks(unittest.TestCase):
    def setUp(self):
        # Create the subgraph
        self.subgraph = Graph(name="TestSubGraph")
        self.sub_worker = StatusNotifyingWorker()
        self.subgraph.add_workers(self.sub_worker)

        # Create the main graph with SubGraphWorker
        self.main_graph = Graph(name="MainGraph")
        self.graph_task = SubGraphWorker(
            graph=self.subgraph,
            entry_worker=self.sub_worker,
            exit_worker=self.sub_worker,
        )

        # Create a consumer for the SubGraphWorker output
        class FinalWorker(TaskWorker):
            output_types: List[Type[Task]] = []

            def consume_work(self, task: FinalTask):
                pass

        self.final_worker = FinalWorker()
        self.main_graph.add_workers(self.graph_task, self.final_worker)
        self.main_graph.set_dependency(self.graph_task, self.final_worker)

    def test_metadata_and_callback_handling(self):
        # Create a mock callback
        mock_callback = Mock()
        test_metadata = {"test_key": "test_value"}

        # Create initial task
        task = InputTask(data="test_data")

        # Add task with metadata and callback
        self.main_graph.prepare(display_terminal=False)
        self.main_graph.set_entry(self.graph_task)

        self.graph_task.add_work(
            task, metadata=test_metadata, status_callback=mock_callback
        )

        # Run the graph
        self.main_graph.execute([])

        # Verify callback was called with correct arguments
        # Should be called twice: "Processing started" and "Processing complete"
        self.assertEqual(mock_callback.call_count, 3)

        # Verify first call
        first_call = mock_callback.call_args_list[0]
        self.assertEqual(first_call.args[0], test_metadata)  # metadata
        self.assertEqual(first_call.args[1], (("InitialTaskWorker", 1),))  # prefix
        self.assertIsInstance(first_call.args[2], StatusNotifyingWorker)  # worker
        self.assertIsInstance(first_call.args[3], InputTask)  # task
        self.assertEqual(first_call.args[4], "Processing started")  # message

        # Verify second call
        second_call = mock_callback.call_args_list[1]
        self.assertEqual(second_call.args[0], test_metadata)  # metadata
        self.assertIsInstance(second_call.args[2], StatusNotifyingWorker)  # worker
        self.assertEqual(second_call.args[4], "Processing complete")  # message

        # Verify third call
        third_call = mock_callback.call_args_list[2]
        self.assertEqual(third_call.args[0], test_metadata)
        self.assertEqual(third_call.args[1], (("InitialTaskWorker", 1),))  # prefix
        self.assertIsNone(third_call.args[2])
        self.assertIsNone(third_call.args[3])

        # Verify cleanup: callback should be removed after task completion
        self.assertEqual(len(self.main_graph._provenance_tracker.task_state), 0)
        self.assertEqual(len(self.subgraph._provenance_tracker.task_state), 0)

    def test_metadata_and_callback_handling_with_failure(self):
        """Test that provenance is cleaned up when a worker in the subgraph fails."""

        class FailingStatusWorker(TaskWorker):
            output_types: List[Type[Task]] = [FinalTask]

            def consume_work(self, task: InputTask):
                # Notify about processing status
                self.notify_status(task, "Processing started")

                # Simulate a failure
                raise RuntimeError("Simulated worker failure")

        # Create the subgraph with failing worker
        self.subgraph = Graph(name="TestSubGraph")
        self.sub_worker = FailingStatusWorker()
        self.subgraph.add_workers(self.sub_worker)

        # Create the main graph with SubGraphWorker
        self.main_graph = Graph(name="MainGraph")
        self.graph_task = SubGraphWorker(
            graph=self.subgraph,
            entry_worker=self.sub_worker,
            exit_worker=self.sub_worker,
        )

        # Create a consumer for the SubGraphWorker output
        class FinalWorker(TaskWorker):
            output_types: List[Type[Task]] = []

            def consume_work(self, task: FinalTask):
                pass

        self.final_worker = FinalWorker()
        self.main_graph.add_workers(self.graph_task, self.final_worker)
        self.main_graph.set_dependency(self.graph_task, self.final_worker)

        # Create a mock callback
        mock_callback = Mock()
        test_metadata = {"test_key": "test_value"}

        # Create initial task
        task = InputTask(data="test_data")

        # Add task with metadata and callback
        self.main_graph.prepare(display_terminal=False)
        self.main_graph.set_entry(self.graph_task)

        self.graph_task.add_work(
            task, metadata=test_metadata, status_callback=mock_callback
        )

        # Run the graph, it should handle the failure gracefully
        self.main_graph.execute([])

        # Verify callback was called for both start and failure
        self.assertEqual(mock_callback.call_count, 3)

        # Verify first call (Processing started)
        first_call = mock_callback.call_args_list[0]
        self.assertEqual(first_call.args[0], test_metadata)
        self.assertIsInstance(first_call.args[2], FailingStatusWorker)
        self.assertIsInstance(first_call.args[3], InputTask)
        self.assertEqual(first_call.args[4], "Processing started")

        # Verify subgraph call
        final_call = mock_callback.call_args_list[1]
        self.assertEqual(final_call.args[0], test_metadata)
        self.assertIsNone(final_call.args[2])
        self.assertIsNone(final_call.args[3])
        self.assertEqual(final_call.args[4], "Task removed")

        # Verify final call (cleanup)
        final_call = mock_callback.call_args_list[2]
        self.assertEqual(final_call.args[0], test_metadata)
        self.assertIsNone(final_call.args[2])
        self.assertIsNone(final_call.args[3])
        self.assertEqual(final_call.args[4], "Task removed")

        # Verify all provenance is cleaned up in both graphs
        self.assertEqual(len(self.main_graph._provenance_tracker.task_state), 0)
        self.assertEqual(len(self.subgraph._provenance_tracker.task_state), 0)
        self.assertEqual(len(self.main_graph._provenance_tracker.provenance), 0)
        self.assertEqual(len(self.subgraph._provenance_tracker.provenance), 0)


if __name__ == "__main__":
    unittest.main()
