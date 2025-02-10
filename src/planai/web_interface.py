# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional

import psutil
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from waitress import create_server

if TYPE_CHECKING:
    from waitress.server import WSGIServer

    from .dispatcher import Dispatcher
    from .graph import Graph

app = Flask(__name__, template_folder="templates", static_folder="static")

dispatcher: "Dispatcher" = None
quit_event = threading.Event()
_server: Optional["WSGIServer"] = None


class MemoryStats:
    def __init__(self, window_size: int = 1500):
        """Initialize memory tracking with a given window size (5 minutes at 0.2s intervals)."""
        self.samples = []
        self.window_size = window_size
        self.running_sum = 0.0
        self.current_memory = 0
        self.peak_memory = 0
        self.process = psutil.Process()

    def update(self) -> None:
        """Update memory statistics."""
        self.current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.samples.append(self.current_memory)
        self.running_sum += self.current_memory

        if len(self.samples) > self.window_size:
            self.running_sum -= self.samples.pop(0)

        self.peak_memory = max(self.peak_memory, self.current_memory)

    def get_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return {
            "current": round(self.current_memory, 1),
            "average": (
                round(self.running_sum / len(self.samples), 1) if self.samples else 0
            ),
            "peak": round(self.peak_memory, 1),
        }


memory_stats = MemoryStats()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


# the selected graph - we will use this for traces and mermaid rendering
selected_graph = 0


@app.route("/stream")
def stream():
    def event_stream():
        global selected_graph
        last_data = None
        last_trace = None
        last_requests = None
        last_graphs = None
        while True:
            current_data = get_current_data()
            current_trace = get_current_trace(selected_graph)
            current_requests = dispatcher.get_user_input_requests()
            current_graphs = dispatcher.get_graphs()
            memory_stats.update()
            logs = dispatcher.get_logs()

            if (
                current_data != last_data
                or current_trace != last_trace
                or current_requests != last_requests
                or current_graphs != last_graphs
                or logs  # Always send if there are new logs
            ):
                combined_data = {
                    "tasks": current_data,
                    "trace": current_trace,
                    "stats": dispatcher.get_execution_statistics(),
                    "user_requests": current_requests,
                    "memory": memory_stats.get_stats(),
                    "graphs": current_graphs,
                    "logs": logs,
                }
                yield f"data: {json.dumps(combined_data)}\n\n"

            last_data = current_data
            last_trace = current_trace
            last_requests = current_requests
            last_graphs = current_graphs
            time.sleep(0.2)

    def get_current_data():
        return {
            "queued": dispatcher.get_queued_tasks(),
            "active": dispatcher.get_active_tasks(),
            "completed": dispatcher.get_completed_tasks(),
            "failed": dispatcher.get_failed_tasks(),
        }

    def get_current_trace(graph_id: int = 0):
        trace = dispatcher.get_traces(index=graph_id)
        return {"_".join(map(str, k)): v for k, v in trace.items()}

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/quit", methods=["POST"])
def quit():
    global quit_event
    quit_event.set()
    return jsonify({"status": "ok"})


@app.route("/user_input", methods=["POST"])
def user_input():
    # Check if 'task_id' is part of the form data
    if "task_id" not in request.form:
        return jsonify({"status": "error", "message": "task_id missing"})

    task_id = request.form["task_id"]

    # Handle abort case
    if request.form.get("abort"):
        dispatcher.set_user_input_result(task_id, None, None)
        return jsonify({"status": "ok"})

    # Check if there is a file part in the request
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "File missing"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"})

    # Get the MIME type of the uploaded file
    mime_type = file.content_type

    # Here we simply read the file content if needed
    result = file.read()

    print(
        f"Task ID: {task_id}, File Size: {len(result)} bytes - mime_type: {mime_type}"
    )  # Print file size for debugging

    # Pass the result to the dispatcher
    dispatcher.set_user_input_result(task_id, result, mime_type)

    return jsonify({"status": "ok"})


@app.route("/graph")
def get_graph():
    global selected_graph
    return jsonify({"graph": render_mermaid_graph(dispatcher.graph(selected_graph))})


@app.route("/select_graph", methods=["POST"])
def select_graph():
    global selected_graph
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400

        graph_id = int(data.get("graph_id", 0))
        selected_graph = graph_id
        return jsonify({"status": "ok", "selected_graph": graph_id})
    except (ValueError, TypeError) as e:
        return (
            jsonify({"status": "error", "message": f"Invalid graph ID: {str(e)}"}),
            400,
        )


def render_mermaid_graph(graph: "Graph"):
    # Start with graph definition
    mermaid = """graph TD\n"""

    # Track unique IDs for nodes and subgraphs
    unique_id = 0
    mapping = {}
    subgraph_workers = []

    # Helper function to get or create unique ID for a worker
    def get_worker_id(worker):
        nonlocal unique_id
        if worker not in mapping:
            unique_id += 1
            mapping[worker] = f"task_{unique_id}"
        return mapping[worker]

    # Helper function to create node definition
    def create_node(worker):
        worker_id = get_worker_id(worker)
        return f"{worker_id}[{worker.name.replace(' ', '_')}]"

    def is_subgraph(worker):
        if not hasattr(worker, "graph"):
            return False
        if not isinstance(worker.graph, object):
            return False
        return hasattr(worker.graph, "dependencies")

    # First pass: identify subgraphs and create all node mappings
    for upstream, downstream_list in graph.dependencies.items():
        get_worker_id(upstream)
        for downstream in downstream_list:
            get_worker_id(downstream)
            # Check if either worker is a SubGraphWorker
            if is_subgraph(upstream):
                if upstream not in subgraph_workers:
                    subgraph_workers.append(upstream)
            if is_subgraph(downstream):
                if downstream not in subgraph_workers:
                    subgraph_workers.append(downstream)

    # Add subgraphs first
    for worker in subgraph_workers:
        subgraph_id = get_worker_id(worker)
        mermaid += f"    subgraph {subgraph_id}[{worker.name}]\n"

        # Add nodes and edges within the subgraph
        for sub_upstream, sub_downstream_list in worker.graph.dependencies.items():
            sub_upstream_id = get_worker_id(sub_upstream)
            mermaid += f"        {create_node(sub_upstream)}\n"
            for sub_downstream in sub_downstream_list:
                sub_downstream_id = get_worker_id(sub_downstream)
                mermaid += f"        {create_node(sub_downstream)}\n"
                mermaid += f"        {sub_upstream_id}-->{sub_downstream_id}\n"

        mermaid += "    end\n"

    # Add all edges from dependencies
    for upstream, downstream_list in graph.dependencies.items():
        upstream_id = get_worker_id(upstream)
        for downstream in downstream_list:
            downstream_id = get_worker_id(downstream)
            mermaid += f"    {upstream_id}[{upstream.name}]-->{downstream_id}[{downstream.name}]\n"

    return mermaid


def set_dispatcher(disp: "Dispatcher"):
    global dispatcher
    dispatcher = disp


def run_web_interface(disp: "Dispatcher", port=5000):
    global _server, dispatcher
    set_dispatcher(disp)
    if _server is None:
        _server = create_server(app, host="0.0.0.0", port=port)
        _server.run()


def shutdown_web_interface():
    """Pretend to shutdown the Waitress web server.

    It's not possible to shutdown a Waitress web server. Instead, we are just setting the dispatcher to None.
    """
    global dispatcher
    dispatcher = None


def is_quit_requested():
    global quit_event
    return quit_event.is_set()
