import unittest
from unittest.mock import Mock, patch

from planai.provenance import ProvenanceTracker
from planai.task import Task, TaskWorker


class DummyTask(Task):
    data: str


class DummyTaskWorkerSimple(TaskWorker):
    def consume_work(self, task: DummyTask):
        pass


class TestProvenanceTracker(unittest.TestCase):
    def setUp(self):
        self.provenance_tracker = ProvenanceTracker()

    def test_add_provenance(self):
        task = DummyTask(data="test")
        task._provenance = [("Task1", 1), ("Task2", 2)]
        self.provenance_tracker._add_provenance(task)
        self.assertEqual(
            self.provenance_tracker.provenance,
            {(("Task1", 1),): 1, (("Task1", 1), ("Task2", 2)): 1},
        )

    def test_remove_provenance(self):
        task = DummyTask(data="test")
        worker = DummyTaskWorkerSimple()
        task._provenance = [("Task1", 1), ("Task2", 2)]
        self.provenance_tracker._add_provenance(task)

        with patch.object(
            self.provenance_tracker, "_notify_task_completion"
        ) as mock_notify:
            self.provenance_tracker._remove_provenance(task, worker)
            self.assertEqual(self.provenance_tracker.provenance, {})
            mock_notify.assert_not_called()

    def test_watch(self):
        notifier = Mock(spec=TaskWorker)
        result = self.provenance_tracker.watch((DummyTask.__name__, 1), notifier)
        self.assertTrue(result)
        self.assertIn((DummyTask.__name__, 1), self.provenance_tracker.notifiers)
        self.assertIn(
            notifier, self.provenance_tracker.notifiers[(DummyTask.__name__, 1)]
        )

    def test_unwatch(self):
        notifier = Mock(spec=TaskWorker)
        self.provenance_tracker.notifiers = {(DummyTask.__name__, 1): [notifier]}
        result = self.provenance_tracker.unwatch((DummyTask.__name__, 1), notifier)
        self.assertTrue(result)
        self.assertNotIn((DummyTask.__name__, 1), self.provenance_tracker.notifiers)

    def test_remove_metadata_on_empty_provenance(self):
        task = DummyTask(data="test")
        worker = DummyTaskWorkerSimple()
        task._provenance = [("Task1", 1)]

        # Add metadata for the provenance
        self.provenance_tracker.add_state((("Task1", 1),), {"some": "metadata"})
        self.provenance_tracker._add_provenance(task)

        # Verify metadata exists
        self.assertEqual(
            self.provenance_tracker.get_state((("Task1", 1),))["metadata"],
            {"some": "metadata"},
        )

        # Remove provenance
        self.provenance_tracker._remove_provenance(task, worker)

        # Verify metadata is removed
        self.assertEqual(
            self.provenance_tracker.get_state((("Task1", 1),))["metadata"],
            {},
        )
        self.assertEqual(self.provenance_tracker.task_state, {})

    def test_callback_execution_on_provenance_removal(self):
        task = DummyTask(data="test")
        task._provenance = [("Task1", 1)]

        # Create a mock callback
        mock_callback = Mock()
        mock_metadata = {"test": "data"}

        # Add metadata and callback for the provenance
        self.provenance_tracker.add_state(
            (("Task1", 1),), metadata=mock_metadata, callback=mock_callback
        )
        self.provenance_tracker._add_provenance(task)

        # Remove provenance with execute_callback=True
        self.provenance_tracker.remove_state((("Task1", 1),), execute_callback=True)

        # Verify callback was called with correct arguments
        mock_callback.assert_called_once_with(
            mock_metadata, (("Task1", 1),), None, None, "Task removed"
        )

    def test_get_prefix_by_type(self):
        # Create a task with a multi-worker provenance chain
        task = DummyTask(data="test")
        task._provenance = [("Worker1", 1), ("Worker2", 2), ("Worker3", 3)]
        task._input_provenance = []

        # Define some test worker classes
        class Worker1(TaskWorker):
            def consume_work(self, task):
                pass

        class Worker2(TaskWorker):
            def consume_work(self, task):
                pass

        class Worker3(TaskWorker):
            def consume_work(self, task):
                pass

        # Test getting prefix for different worker types
        prefix1 = self.provenance_tracker.get_prefix_by_type(task, Worker1)
        prefix2 = self.provenance_tracker.get_prefix_by_type(task, Worker2)
        prefix3 = self.provenance_tracker.get_prefix_by_type(task, Worker3)
        prefix4 = self.provenance_tracker.get_prefix_by_type(
            task, DummyTaskWorkerSimple
        )

        # Verify the correct prefixes are returned
        self.assertEqual(prefix1, (("Worker1", 1),))
        self.assertEqual(prefix2, (("Worker1", 1), ("Worker2", 2)))
        self.assertEqual(prefix3, (("Worker1", 1), ("Worker2", 2), ("Worker3", 3)))
        self.assertIsNone(prefix4)  # Should return None for worker type not in chain


if __name__ == "__main__":
    unittest.main()
