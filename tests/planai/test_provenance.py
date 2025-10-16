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
        # Check that the notifier entry (worker, callback) is in the list
        notifier_entries = self.provenance_tracker.notifiers[(DummyTask.__name__, 1)]
        self.assertEqual(len(notifier_entries), 1)
        self.assertEqual(notifier_entries[0][0], notifier)
        self.assertIsNone(notifier_entries[0][1])  # callback should be None

    def test_unwatch(self):
        notifier = Mock(spec=TaskWorker)
        # Now notifiers store (worker, callback) tuples
        self.provenance_tracker.notifiers = {
            (DummyTask.__name__, 1): [(notifier, None)]
        }
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

    def test_notification_sort_key(self):
        """Test the notification sort key function for correct prioritization."""
        # Create mock workers with names
        worker_a = Mock(spec=TaskWorker)
        worker_a.name = "WorkerA"
        worker_a._graph = None

        worker_b = Mock(spec=TaskWorker)
        worker_b.name = "WorkerB"
        worker_b._graph = None

        source_worker = Mock(spec=TaskWorker)
        source_worker.name = "SourceWorker"

        # Mock callback functions
        callback_func = Mock()

        # Create notifier entries: (worker, callback, prefix)
        prefix1 = (("Task1", 1),)
        prefix2 = (("Task2", 2),)

        # Entry 1: WorkerA with no callback
        entry_a_none = (worker_a, None, prefix1)
        # Entry 2: WorkerA with callback
        entry_a_callback = (worker_a, callback_func, prefix2)
        # Entry 3: WorkerB with no callback
        entry_b_none = (worker_b, None, prefix1)

        # Mock the _get_worker_distance to return controlled distances
        with patch.object(
            self.provenance_tracker, "_get_worker_distance"
        ) as mock_distance:
            # Test 1: Same distance, same worker name -> None callback should come first
            mock_distance.return_value = 1.0

            key_a_none = self.provenance_tracker._notification_sort_key(
                entry_a_none, source_worker.name
            )
            key_a_callback = self.provenance_tracker._notification_sort_key(
                entry_a_callback, source_worker.name
            )

            # Both should have same distance and worker name, but different callback priority
            self.assertEqual(key_a_none[0], 1.0)  # distance
            self.assertEqual(key_a_none[1], "WorkerA")  # worker name
            self.assertEqual(key_a_none[2], 0)  # callback priority (None = 0)

            self.assertEqual(key_a_callback[0], 1.0)  # distance
            self.assertEqual(key_a_callback[1], "WorkerA")  # worker name
            self.assertEqual(key_a_callback[2], 1)  # callback priority (callback = 1)

            # Verify that entry with None callback sorts before entry with callback
            self.assertLess(key_a_none, key_a_callback)

            # Test 2: Different workers, same distance -> sorted by name
            key_b_none = self.provenance_tracker._notification_sort_key(
                entry_b_none, source_worker.name
            )

            self.assertEqual(key_b_none[0], 1.0)  # distance
            self.assertEqual(key_b_none[1], "WorkerB")  # worker name
            self.assertEqual(key_b_none[2], 0)  # callback priority

            # WorkerA should come before WorkerB (alphabetical)
            self.assertLess(key_a_none, key_b_none)

        # Test 3: Different distances -> lower distance comes first
        with patch.object(
            self.provenance_tracker, "_get_worker_distance"
        ) as mock_distance:

            def distance_side_effect(worker, source):
                if worker.name == "WorkerA":
                    return 2.0
                elif worker.name == "WorkerB":
                    return 1.0
                return float("inf")

            mock_distance.side_effect = distance_side_effect

            key_a = self.provenance_tracker._notification_sort_key(
                entry_a_none, source_worker.name
            )
            key_b = self.provenance_tracker._notification_sort_key(
                entry_b_none, source_worker.name
            )

            # WorkerB (distance 1.0) should come before WorkerA (distance 2.0)
            self.assertLess(key_b, key_a)

    def test_notification_sorting_integration(self):
        """Integration test for full notification sorting."""
        # Create mock workers
        worker_close = Mock(spec=TaskWorker)
        worker_close.name = "CloseWorker"
        worker_close._graph = None

        worker_far = Mock(spec=TaskWorker)
        worker_far.name = "FarWorker"
        worker_far._graph = None

        source_worker = Mock(spec=TaskWorker)
        source_worker.name = "SourceWorker"

        callback_func = Mock()
        prefix = (("Task1", 1),)

        # Create a list of notifier entries to sort
        to_notify = [
            (worker_far, callback_func, prefix),  # Far worker with callback
            (worker_close, callback_func, prefix),  # Close worker with callback
            (
                worker_close,
                None,
                prefix,
            ),  # Close worker with no callback (should be first)
            (worker_far, None, prefix),  # Far worker with no callback
        ]

        # Mock distances
        with patch.object(
            self.provenance_tracker, "_get_worker_distance"
        ) as mock_distance:

            def distance_side_effect(worker, source):
                if worker.name == "CloseWorker":
                    return 1.0
                elif worker.name == "FarWorker":
                    return 5.0
                return float("inf")

            mock_distance.side_effect = distance_side_effect

            # Sort using the same logic as _notify_task_completion
            sorted_notify = sorted(
                to_notify,
                key=lambda x: self.provenance_tracker._notification_sort_key(
                    x, source_worker.name
                ),
            )

            # Expected order:
            # 1. (worker_close, None) - distance 1.0, callback priority 0
            # 2. (worker_close, callback_func) - distance 1.0, callback priority 1
            # 3. (worker_far, None) - distance 5.0, callback priority 0
            # 4. (worker_far, callback_func) - distance 5.0, callback priority 1

            self.assertEqual(sorted_notify[0][0].name, "CloseWorker")
            self.assertIsNone(sorted_notify[0][1])

            self.assertEqual(sorted_notify[1][0].name, "CloseWorker")
            self.assertIsNotNone(sorted_notify[1][1])

            self.assertEqual(sorted_notify[2][0].name, "FarWorker")
            self.assertIsNone(sorted_notify[2][1])

            self.assertEqual(sorted_notify[3][0].name, "FarWorker")
            self.assertIsNotNone(sorted_notify[3][1])


if __name__ == "__main__":
    unittest.main()
