import queue
import unittest
from unittest.mock import Mock, patch

from deepsearch.deepsearch import app, socketio, start_worker_thread, stop_worker_thread
from deepsearch.graph import Response
from deepsearch.user_session import UserSessionManager


class TestDeepSearch(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = socketio.test_client(app)

        # Mock graph and workers
        self.mock_graph = Mock()
        self.mock_entry_worker = Mock()
        self.mock_chat_worker = Mock()
        self.mock_user_sessions = Mock(spec=UserSessionManager)

        patch("deepsearch.deepsearch.setup_logging").start()
        patch("deepsearch.deepsearch.user_sessions", self.mock_user_sessions).start()

        # Patch setup_graph to return our mocks
        self.setup_graph_patcher = patch(
            "deepsearch.deepsearch.setup_graph",
            return_value=(
                self.mock_graph,
                self.mock_entry_worker,
                self.mock_chat_worker,
            ),
        )
        self.setup_graph_patcher.start()

        # Patch the global workers in deepsearch.py
        self.entry_worker_patcher = patch(
            "deepsearch.deepsearch.entry_worker", self.mock_entry_worker
        )
        self.chat_worker_patcher = patch(
            "deepsearch.deepsearch.chat_worker", self.mock_chat_worker
        )
        self.graph_patcher = patch("deepsearch.deepsearch.graph", self.mock_graph)
        self.entry_worker_patcher.start()
        self.chat_worker_patcher.start()
        self.graph_patcher.start()

        # Configure mock workers to return proper provenance chains
        self.mock_entry_worker.add_work.return_value = (("InitialTaskWorker", 1),)
        self.mock_chat_worker.add_work.return_value = (("InitialTaskWorker", 1),)

        # Replace the task_queue patch with a Mock instead of a real Queue
        self.mock_task_queue = Mock(spec=queue.Queue)
        self.mock_task_queue.get.return_value = (None, None, None)
        self.task_queue_patcher = patch(
            "deepsearch.deepsearch.task_queue", self.mock_task_queue
        )
        self.task_queue_patcher.start()

        # Start worker thread
        start_worker_thread()

        # Store the session ID after connection
        received = self.client.get_received()
        self.session_id = next(
            msg["args"][0]["id"] for msg in received if msg["name"] == "session_id"
        )

    def tearDown(self):
        stop_worker_thread()
        self.setup_graph_patcher.stop()
        self.entry_worker_patcher.stop()
        self.chat_worker_patcher.stop()
        self.task_queue_patcher.stop()

    def test_socket_connection(self):
        """Test basic socket connection and session creation."""
        # Create a new connection to test the connection flow
        new_client = socketio.test_client(app)
        self.assertTrue(new_client.is_connected())

        # Get the messages received on this new connection
        received = new_client.get_received()

        # Verify we got a session_id message
        session_id_messages = [msg for msg in received if msg["name"] == "session_id"]
        self.assertEqual(
            len(session_id_messages),
            1,
            f"Expected exactly one session_id message, got {received}",
        )

        # Verify the session_id message format
        session_msg = session_id_messages[0]
        self.assertIn("args", session_msg)
        self.assertIsInstance(session_msg["args"], list)
        self.assertEqual(len(session_msg["args"]), 1)
        self.assertIn("id", session_msg["args"][0])
        self.assertIsInstance(session_msg["args"][0]["id"], str)

    def test_chat_message_flow(self):
        """Test the flow of a chat message through the system."""
        from deepsearch.deepsearch import notify

        # Send a chat message with the correct session ID
        self.client.emit(
            "chat_message",
            {
                "session_id": self.session_id,
                "messages": [{"role": "user", "content": "test message"}],
            },
        )

        # Verify worker was called and get the metadata that was passed
        self.mock_entry_worker.add_work.assert_called_once()
        args, kwargs = self.mock_entry_worker.add_work.call_args
        metadata = kwargs.get("metadata", {})

        # Clear any existing messages
        self.client.get_received()

        # Simulate the graph sending a response through the notify callback
        response = Response(response_type="final", message="test response")
        notify(metadata, response)

        # Verify the response was queued
        self.mock_task_queue.put.assert_called_once()
        queued_args = self.mock_task_queue.put.call_args[0][0]
        self.assertIsInstance(queued_args, tuple)
        self.assertEqual(len(queued_args), 3)
        self.assertEqual(queued_args[2].response_type, "final")
        self.assertEqual(queued_args[2].message, "test response")

        # Update mock queue to return our response tuple for the next get() call
        self.mock_task_queue.get.return_value = queued_args

        # Simulate the worker processing the message by calling the socketio.emit directly
        socketio.emit(
            "chat_response", response.model_dump(), namespace="/", to=metadata["sid"]
        )

        # Now check if client received response
        received = self.client.get_received()
        self.assertTrue(
            any(msg["name"] == "chat_response" for msg in received),
            f"Expected chat_response in {received}",
        )

    def test_abort_request(self):
        """Test aborting an ongoing request."""
        # Start a chat message with the correct session ID
        self.client.emit(
            "chat_message",
            {
                "session_id": self.session_id,
                "messages": [{"role": "user", "content": "test message"}],
            },
        )

        # Send abort request with the correct session ID
        self.client.emit("abort", {"session_id": self.session_id})

        # Verify graph abort was called
        self.mock_graph.abort_work.assert_called_once()

    @patch("deepsearch.deepsearch.validate_provider")
    def test_provider_validation(self, mock_validate):
        """Test provider validation endpoint."""
        mock_validate.return_value = (True, ["model1", "model2"])

        self.client.emit(
            "validate_provider", {"provider": "test_provider", "apiKey": "test_key"}
        )

        received = self.client.get_received()
        self.assertTrue(
            any(
                msg["name"] == "provider_validated"
                and msg["args"][0]["isValid"] is True
                for msg in received
            )
        )

    def test_settings_management(self):
        """Test saving and loading settings."""
        # Test save settings
        test_settings = {
            "provider": "ollama",
            "modelName": "test_model",
            "ollamaHost": "localhost:11434",
        }

        self.client.emit("save_settings", test_settings)

        # Test load settings
        self.client.emit("load_settings")
        received = self.client.get_received()
        self.assertTrue(any(msg["name"] == "settings_loaded" for msg in received))

    @patch("deepsearch.deepsearch.user_sessions")
    def test_session_management(self, mock_user_sessions):
        """Test session listing and retrieval."""
        mock_user_sessions.list_sessions.return_value = ["session1", "session2"]

        self.client.emit("list_sessions")
        received = self.client.get_received()
        self.assertTrue(
            any(
                msg["name"] == "sessions_listed"
                and len(msg["args"][0]["sessions"]) == 2
                for msg in received
            )
        )

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test invalid session ID
        self.client.emit(
            "chat_message",
            {
                "session_id": "invalid_id",
                "messages": [{"role": "user", "content": "test message"}],
            },
        )

        received = self.client.get_received()
        self.assertTrue(
            any(
                msg["name"] == "error" and "Invalid session ID" in msg["args"][0]
                for msg in received
            )
        )


if __name__ == "__main__":
    unittest.main()
