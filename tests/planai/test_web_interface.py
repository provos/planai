import io
import json
import threading
import unittest
from typing import List, Type
from unittest.mock import MagicMock, Mock, patch

from planai.graph import Graph
from planai.task import Task, TaskWorker
from planai.web_interface import MemoryStats, app, render_mermaid_graph, set_dispatcher


def create_named_worker(name: str, task_type: Type[Task] = Task) -> Type[TaskWorker]:
    """Create a TaskWorker class with a specific name."""

    class NamedWorker(TaskWorker):
        output_types: List[Type[Task]] = [task_type]

        def consume_work(self, task: task_type) -> None:
            pass

    # Give the class a unique name for better debugging
    NamedWorker.__name__ = f"TestWorker_{name}"
    return NamedWorker


class TestWebInterface(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        # Create a mock dispatcher
        self.mock_dispatcher = Mock()
        self.mock_dispatcher.graph = self._create_test_graph()
        set_dispatcher(self.mock_dispatcher)

    def _create_test_graph(self) -> Graph:
        """Create a test graph for Mermaid rendering tests."""
        graph = Graph(name="TestGraph")
        worker1 = create_named_worker("Worker1")()
        worker2 = create_named_worker("Worker2")()
        worker3 = create_named_worker("Worker3")()

        graph.add_workers(worker1, worker2, worker3)
        graph.set_dependency(worker1, worker2).next(worker3)

        return graph

    def test_index_route(self):
        """Test the index route returns the correct template."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_quit_route(self):
        """Test the quit route sets the quit event."""
        response = self.client.post("/quit")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "ok")

    def test_user_input_route_missing_task_id(self):
        """Test user input route with missing task_id."""
        response = self.client.post("/user_input")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "task_id missing")

    def test_user_input_route_missing_file(self):
        """Test user input route with missing file."""
        response = self.client.post("/user_input", data={"task_id": "123"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "File missing")

    @patch("planai.web_interface.dispatcher")
    def test_user_input_route_success(self, mock_dispatcher):
        """Test successful user input submission."""
        test_file_content = b"test content"
        response = self.client.post(
            "/user_input",
            data={
                "task_id": "123",
                "file": (io.BytesIO(test_file_content), "test.txt"),
            },
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "ok")
        mock_dispatcher.set_user_input_result.assert_called_once()

    def test_user_input_route_abort(self):
        """Test user input abort functionality."""
        response = self.client.post(
            "/user_input",
            data={"task_id": "123", "abort": "true"},
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "ok")
        self.mock_dispatcher.set_user_input_result.assert_called_with("123", None, None)

    def test_graph_route(self):
        """Test the graph route returns correct Mermaid markdown."""
        response = self.client.get("/graph")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("graph", data)
        self.assertIn("graph TD", data["graph"])
        # Verify the graph contains our test workers
        self.assertIn("TestWorker", data["graph"])

    @patch("planai.web_interface.memory_stats")
    def test_stream_route(self, mock_memory_stats):
        """Test the stream route with synthetic data."""
        # Setup mock data
        mock_memory_stats.get_stats.return_value = {
            "current": 100.0,
            "average": 95.0,
            "peak": 120.0,
        }

        mock_traces = {(1, "task1"): "trace1", (2, "task2"): "trace2"}
        self.mock_dispatcher.get_traces.return_value = mock_traces

        self.mock_dispatcher.get_queued_tasks.return_value = [
            {
                "id": "task1",
                "type": "TestTask",
                "worker": "TestWorker",
                "provenance": ["TestWorker_1"],
            }
        ]
        self.mock_dispatcher.get_active_tasks.return_value = []
        self.mock_dispatcher.get_completed_tasks.return_value = []
        self.mock_dispatcher.get_failed_tasks.return_value = []
        self.mock_dispatcher.get_execution_statistics.return_value = {
            "TestWorker": {"completed": 0, "failed": 0, "active": 0}
        }
        self.mock_dispatcher.get_user_input_requests.return_value = []
        self.mock_dispatcher.get_logs.return_value = []

        # Create an Event to signal when we've received data
        data_received = threading.Event()
        received_data = []
        stream_error = None

        def mock_stream():
            nonlocal stream_error
            try:
                with self.client.get("/stream") as response:
                    # Read the first chunk of data
                    for line in response.response:
                        try:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                json_data = json.loads(decoded_line[6:])
                                received_data.append(json_data)
                                data_received.set()
                                return  # Exit after first data received
                        except Exception as e:
                            stream_error = e
                            break
            except Exception as e:
                stream_error = e
            finally:
                data_received.set()  # Always set the event

        # Start stream in a separate thread
        stream_thread = threading.Thread(target=mock_stream)
        stream_thread.daemon = True
        stream_thread.start()

        # Wait for data with timeout
        if not data_received.wait(timeout=5.0):
            self.fail("Timeout waiting for stream data")

        # Check if there was an error in the stream thread
        if stream_error:
            self.fail(f"Error in stream thread: {stream_error}")

        # Verify that we received the expected data
        self.assertTrue(len(received_data) > 0, "No data received from stream")
        if received_data:
            data = received_data[0]
            self.assertIn("trace", data)
            expected_trace = {"1_task1": "trace1", "2_task2": "trace2"}
            self.assertEqual(data["trace"], expected_trace)

        # Verify that dispatcher methods were called
        self.mock_dispatcher.get_queued_tasks.assert_called()
        self.mock_dispatcher.get_active_tasks.assert_called()
        self.mock_dispatcher.get_completed_tasks.assert_called()
        self.mock_dispatcher.get_failed_tasks.assert_called()
        self.mock_dispatcher.get_execution_statistics.assert_called()
        self.mock_dispatcher.get_traces.assert_called()
        self.mock_dispatcher.get_user_input_requests.assert_called()


class TestMemoryStats(unittest.TestCase):
    @patch("planai.web_interface.psutil.Process")
    def setUp(self, mock_process_class):
        # Setup mock process instance that will be used during MemoryStats initialization
        mock_process_instance = mock_process_class.return_value
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=1024 * 1024 * 100
        )
        self.mock_process_class = mock_process_class
        self.memory_stats = MemoryStats(window_size=5)

    def test_memory_stats_update(self):
        """Test memory statistics updating and calculation."""
        # Setup mock process instance for update calls
        mock_process_instance = self.mock_process_class.return_value
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=1024 * 1024 * 100  # 100MB
        )

        # Update stats multiple times
        for _ in range(3):
            self.memory_stats.update()

        stats = self.memory_stats.get_stats()

        self.assertEqual(stats["current"], 100.0)
        self.assertEqual(stats["peak"], 100.0)
        self.assertEqual(stats["average"], 100.0)

    def test_memory_stats_window(self):
        """Test memory statistics window behavior."""
        mock_process_instance = self.mock_process_class.return_value
        # Simulate increasing memory usage
        for i in range(10):
            mock_process_instance.memory_info.return_value = MagicMock(
                rss=1024 * 1024 * (100 + i * 10)  # Increase by 10MB each time
            )
            self.memory_stats.update()

        stats = self.memory_stats.get_stats()

        # Should only keep last 5 samples due to window_size=5
        self.assertEqual(len(self.memory_stats.samples), 5)
        self.assertEqual(stats["current"], 190.0)
        self.assertEqual(stats["peak"], 190.0)


class TestMermaidGraphGeneration(unittest.TestCase):
    def test_simple_graph_rendering(self):
        """Test rendering of a simple graph structure."""
        graph = Graph(name="TestGraph")
        worker1 = create_named_worker("Worker1")()
        worker2 = create_named_worker("Worker2")()
        graph.add_workers(worker1, worker2)
        graph.set_dependency(worker1, worker2)

        mermaid = render_mermaid_graph(graph)

        self.assertIn("graph TD", mermaid)
        self.assertIn("Worker1", mermaid)
        self.assertIn("Worker2", mermaid)
        self.assertIn("-->", mermaid)

    def test_unique_node_ids(self):
        """Test that each node gets a unique ID."""
        graph = Graph(name="TestGraph")
        worker1 = create_named_worker("Worker1")()
        worker2 = create_named_worker("Worker2")()
        graph.add_workers(worker1, worker2)
        graph.set_dependency(worker1, worker2)

        mermaid = render_mermaid_graph(graph)

        # Each worker should appear exactly once in node definitions
        self.assertEqual(mermaid.count("Worker1"), 1)
        self.assertEqual(mermaid.count("Worker2"), 1)

    def test_subgraph_rendering(self):
        """Test rendering of a graph containing subgraphs."""

        # Create a task for the subgraph
        class SubTask(Task):
            data: str

        # Create workers for the subgraph
        class SubWorker1(TaskWorker):
            output_types: List[Type[Task]] = [SubTask]

            def consume_work(self, task: Task) -> None:
                pass

        class SubWorker2(TaskWorker):
            output_types: List[Type[Task]] = [SubTask]

            def consume_work(self, task: SubTask) -> None:
                pass

        # Create the subgraph
        subgraph = Graph(name="SubGraph")
        sub_worker1 = SubWorker1()
        sub_worker2 = SubWorker2()
        subgraph.add_workers(sub_worker1, sub_worker2)
        subgraph.set_dependency(sub_worker1, sub_worker2)

        # Create the SubGraphWorker
        from planai.graph_task import SubGraphWorker

        graph_task = SubGraphWorker(
            graph=subgraph,
            entry_worker=sub_worker1,
            exit_worker=sub_worker2,
        )
        graph_task.output_types = [SubTask]

        # Create main graph with named workers that handle SubTask
        main_graph = Graph(name="MainGraph")
        worker1 = create_named_worker(
            "MainWorker1", task_type=Task
        )()  # First worker can handle Task
        worker2 = create_named_worker(
            "MainWorker2", task_type=SubTask
        )()  # Last worker must handle SubTask

        # Add all workers and dependencies
        main_graph.add_workers(worker1, graph_task, worker2)
        main_graph.set_dependency(worker1, graph_task).next(worker2)

        # Generate and verify Mermaid markdown
        mermaid = render_mermaid_graph(main_graph)

        # Basic structure checks
        self.assertIn("graph TD", mermaid)
        self.assertIn("subgraph", mermaid)
        self.assertIn("SubGraph", mermaid)

        # Check for all workers
        self.assertIn("TestWorker_MainWorker1", mermaid)
        self.assertIn("TestWorker_MainWorker2", mermaid)
        self.assertIn("SubWorker1", mermaid)
        self.assertIn("SubWorker2", mermaid)

        # Verify graph structure
        self.assertIn(
            "[TestWorker_MainWorker1]-->", mermaid
        )  # Main worker connects to subgraph
        self.assertIn(
            "-->task_4[TestWorker_MainWorker2]", mermaid
        )  # Subgraph connects to final worker
        self.assertIn("task_6[SubWorker1]", mermaid)  # Subgraph worker 1
        self.assertIn("task_7[SubWorker2]", mermaid)  # Subgraph worker 2
        self.assertIn("task_6-->task_7", mermaid)  # Connection within subgraph

        # Verify subgraph boundaries
        self.assertIn("end", mermaid)  # Subgraph closing tag


if __name__ == "__main__":
    unittest.main()
