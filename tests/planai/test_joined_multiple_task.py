import random
import threading
import time
import unittest
from typing import List, Tuple, Type

from planai.dispatcher import Dispatcher
from planai.graph import Graph
from planai.joined_task import InitialTaskWorker, JoinedTaskWorker
from planai.task import Task, TaskWorker


class InputTask(Task):
    data: str


class FanOutTask(Task):
    data: str


class SummaryTask(Task):
    summary: str


class FinalOutputTask(Task):
    result: int


class RepeatWorker(TaskWorker):
    output_types: List[Type[Task]] = [InputTask]

    def __init__(self, **data):
        super().__init__(**data)

    def consume_work(self, task: InputTask):
        time.sleep(random.uniform(0.1, 0.4))
        self.publish_work(task.model_copy(), task)


class FanOutWorker(TaskWorker):
    output_types: List[Type[Task]] = [FanOutTask]

    def __init__(self, **data):
        super().__init__(**data)
        self._call_count = 0

    def consume_work(self, task: InputTask):
        with self.lock:
            self._call_count += 1
        time.sleep(random.uniform(0.1, 0.4))
        for i in range(3):
            output = FanOutTask(data=f"{task.data}-{i}")
            self.publish_work(output, task)

    @property
    def call_count(self):
        with self.lock:
            return self._call_count


class JoinFanOutWorker(JoinedTaskWorker):
    join_type: Type[TaskWorker] = RepeatWorker
    output_types: List[Type[Task]] = [SummaryTask]

    def __init__(self, **data):
        super().__init__(**data)
        self._call_count = 0

    def consume_work(self, task: FanOutTask):
        super().consume_work(task)

    def consume_work_joined(self, tasks: List[FanOutTask]):
        with self.lock:
            self._call_count += 1
        summary = ", ".join([task.data for task in tasks])
        output = SummaryTask(summary=summary)
        time.sleep(random.uniform(0.1, 0.4))
        self.publish_work(output, tasks[0])

    @property
    def call_count(self):
        with self.lock:
            return self._call_count


class FinalJoinedWorker(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    output_types: List[Type[Task]] = [FinalOutputTask]

    def __init__(self, **data):
        super().__init__(**data)
        self._call_count = 0

    def consume_work(self, task: SummaryTask):
        super().consume_work(task)

    def consume_work_joined(self, tasks: List[SummaryTask]):
        if len(tasks) != 5:
            print(f"Expected 5 task, got {len(tasks)}")

        with self.lock:
            self._call_count += 1

        time.sleep(random.uniform(0.1, 0.4))
        output = FinalOutputTask(result=len(tasks))
        self.publish_work(output, tasks[0])

    @property
    def call_count(self):
        with self.lock:
            return self._call_count


class TestComplexJoinedTaskWorker(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(name="Complex Join Test Graph")
        self.dispatcher = Dispatcher(self.graph)
        self.graph._dispatcher = self.dispatcher

        self.repeat_worker = RepeatWorker()
        self.fan_out_worker = FanOutWorker()
        self.join_fan_out_worker = JoinFanOutWorker()
        self.final_joined_worker = FinalJoinedWorker()

        self.graph.add_workers(
            self.repeat_worker,
            self.fan_out_worker,
            self.join_fan_out_worker,
            self.final_joined_worker,
        )
        self.graph.set_dependency(self.repeat_worker, self.fan_out_worker).next(
            self.join_fan_out_worker
        ).next(self.final_joined_worker)

        self.final_joined_worker.sink(FinalOutputTask)

    def test_complex_joined_task_workflow(self):
        instance = InitialTaskWorker()

        initial_work: List[Tuple[TaskWorker, Task]] = []
        for i in range(5):
            initial_task = InputTask(data=f"Initial-{i}")
            instance._id = 0
            initial_task._add_worker_provenance(instance)
            initial_work.append((self.repeat_worker, initial_task))

        # the graph run method would usually do this for us
        self.graph.inject_initial_task_worker(initial_work)
        self.graph.finalize()

        # Start the dispatcher
        dispatch_thread = threading.Thread(target=self.dispatcher.dispatch)
        dispatch_thread.start()

        self.dispatcher.trace((("InitialTaskWorker", 1),))

        # Add initial work
        self.dispatcher.add_multiple_work(initial_work)

        # Wait for all work to be processed
        self.dispatcher.wait_for_completion()
        self.dispatcher.stop()
        dispatch_thread.join()

        # Check results
        self.assertEqual(self.fan_out_worker.call_count, 5)  # Initial
        self.assertEqual(
            self.join_fan_out_worker.call_count, 5
        )  # 5 sets of fan-out tasks
        self.assertEqual(self.final_joined_worker.call_count, 1)  # 1 summary tasks

        # Verify that the work queue is empty and there are no active tasks
        self.assertEqual(self.dispatcher.work_queue.qsize(), 0)
        self.assertEqual(self.dispatcher._active_tasks, 0)

        outputs = self.graph.get_output_tasks()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].result, 5)

        self.graph._thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
