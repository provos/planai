import time
import unittest
from typing import List, Type

from planai.graph import Graph
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
            dispatcher.total_completed_tasks, 3 + 2
        )  # 3 tasks in the main graph, 1 in the subgraph + the AdapterWorker

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
        )
        expected_total_tasks = total_main_graph_tasks + total_subgraph_tasks

        self.assertEqual(dispatcher.total_completed_tasks, expected_total_tasks)

        # Since FinalWorker has no output, check that it processed the task correctly
        final_tasks = graph.get_output_tasks()
        self.assertEqual(len(final_tasks), 0)

        # Optionally, you can verify that the joined data is correct
        # But since FinalWorker does not output anything, and we didn't sink any tasks, we can't retrieve outputs here


if __name__ == "__main__":
    unittest.main()
