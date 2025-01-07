import unittest
import time
from concurrent.futures import ThreadPoolExecutor
from deepsearch.session import SessionManager
from deepsearch import session


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager()

    def tearDown(self):
        self.manager.stop_cleanup_worker()

    def test_session_lifecycle(self):
        # Test session creation and retrieval
        session_id = self.manager.create_session()
        self.assertIsNotNone(session_id)
        session = self.manager.get_session(session_id)
        self.assertIsNotNone(session)

        # Test session update
        test_data = {"key": "value"}
        self.manager.update_session(session_id, test_data)
        session = self.manager.get_session(session_id)
        self.assertEqual(session.get("key"), "value")

        # Test session deletion
        self.manager.delete_session(session_id)
        self.assertIsNone(self.manager.get_session(session_id))

    def test_sid_management(self):
        session_id = self.manager.create_session()
        test_sid = "test_sid"

        # Test SID registration
        self.manager.register_sid(test_sid, session_id)
        retrieved_session_id = self.manager.get_session_id_by_sid(test_sid)
        self.assertEqual(session_id, retrieved_session_id)

        # Test SID removal
        self.manager.remove_sid(test_sid)
        self.assertIsNone(self.manager.get_session_id_by_sid(test_sid))

    def test_metadata_operations(self):
        session_id = self.manager.create_session()
        metadata = self.manager.metadata(session_id)

        # Test metadata operations
        metadata["user"] = "testuser"
        self.assertTrue("user" in metadata)
        self.assertEqual(metadata.get("user"), "testuser")

        metadata.update({"role": "admin"})
        self.assertTrue("role" in metadata)

        # Verify metadata persistence
        current_metadata = self.manager.metadata(session_id)
        self.assertTrue("user" in current_metadata)
        self.assertTrue("role" in current_metadata)
        self.assertEqual(current_metadata.get("user"), "testuser")
        self.assertEqual(current_metadata.get("role"), "admin")

        # Test metadata deletion with session
        self.manager.delete_session(session_id)
        new_metadata = self.manager.metadata(session_id)
        self.assertEqual(new_metadata.to_dict(), {})

    def test_session_cleanup(self):
        # Override timeout for testing
        original_timeout = session.SESSION_TIMEOUT
        session.SESSION_TIMEOUT = 1  # 1 second timeout

        session_id = self.manager.create_session()
        metadata = self.manager.metadata(session_id)
        metadata["test"] = "value"

        # Wait for session to expire
        time.sleep(2)
        self.manager._cleanup_stale_sessions()

        # Verify session and metadata are cleaned up
        self.assertIsNone(self.manager.get_session(session_id))
        self.assertEqual(self.manager.metadata(session_id).to_dict(), {})

        # Restore original timeout
        session.SESSION_TIMEOUT = original_timeout

    def test_thread_safety(self):
        session_id = self.manager.create_session()
        metadata = self.manager.metadata(session_id)
        num_threads = 10
        iterations = 100

        def update_metadata(thread_id):
            for i in range(iterations):
                metadata[f"key_{thread_id}_{i}"] = i

        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_metadata, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        # Verify all updates are present
        final_metadata = metadata.to_dict()
        for thread_id in range(num_threads):
            for i in range(iterations):
                self.assertEqual(final_metadata[f"key_{thread_id}_{i}"], i)


if __name__ == "__main__":
    unittest.main()
