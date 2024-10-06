import time
import unittest
from typing import List, Type

from planai.graph import Graph
from planai.graph_task import GraphTask
from planai.task import Task, TaskWorker


class InputTask(Task):
    data: str


class SubGraphTask(Task):
    data: str
    _intermediate: bool = False  # Indicate if this task is from the subgraph


class FinalTask(Task):
    result: str


class MainWorker(TaskWorker):
    output_types: List[Type[Task]] = [SubGraphTask]

    def consume_work(self, task: InputTask):
        # Simulate processing in the main graph
        time.sleep(0.1)
        sub_task = SubGraphTask(data=task.data)
        self.publish_work(sub_task, input_task=task)


class SubGraphWorker(TaskWorker):
    output_types: List[Type[Task]] = [SubGraphTask]

    def consume_work(self, task: SubGraphTask):
        # Mark task as processed in subgraph
        task._intermediate = True
        # Simulate processing in the subgraph
        time.sleep(0.1)
        self.publish_work(task, input_task=task)


class FinalWorker(TaskWorker):
    output_types: List[Type[Task]] = []

    def consume_work(self, task: SubGraphTask):
        # Simulate final processing in the main graph
        time.sleep(0.1)
        print(task._provenance)
        assert task._intermediate, "Task should have been processed by subgraph"
        assert len(task._provenance) == 3, "Provenance length should be 3"
        # The provenance should include: MainWorker, GraphTask, FinalWorker
        # The subgraph's provenance should not appear here
        expected_provenance = [
            ("InitialTaskWorker", 1),
            ("MainWorker", 1),
            ("GraphTask", 1),
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
        subgraph_worker = SubGraphWorker()
        subgraph_entry = subgraph_worker  # Entry point
        subgraph_exit = subgraph_worker  # Exit point
        subgraph.add_workers(subgraph_worker)

        # Create GraphTask
        graph_task = GraphTask(
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
        self.assertEqual(dispatcher.work_queue.qsize(), 0)
        self.assertEqual(dispatcher.active_tasks, 0)
        self.assertEqual(
            dispatcher.total_completed_tasks, 3 + 2
        )  # 3 tasks in the main graph, 1 in the subgraph + the AdapterWorker

        # Check that the provenance is as expected
        final_tasks = graph.get_output_tasks()
        self.assertEqual(len(final_tasks), 0)  # Since FinalWorker has no output

        # Optionally, we can check logs or other side effects if needed


if __name__ == "__main__":
    unittest.main()
