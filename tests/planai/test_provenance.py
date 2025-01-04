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


if __name__ == "__main__":
    unittest.main()
