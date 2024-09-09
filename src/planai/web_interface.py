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
from typing import TYPE_CHECKING, Optional

from flask import Flask, Response, jsonify, render_template, send_from_directory

if TYPE_CHECKING:
    from .dispatcher import Dispatcher

app = Flask(__name__, template_folder="templates", static_folder="static")

dispatcher: Optional["Dispatcher"] = None
quit_event = threading.Event()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


@app.route("/stream")
def stream():
    def event_stream():
        last_data = None
        last_trace = None
        while True:
            current_data = get_current_data()
            current_trace = get_current_trace()

            if current_data != last_data or current_trace != last_trace:
                combined_data = {
                    "tasks": current_data,
                    "trace": current_trace,
                    "stats": dispatcher.get_execution_statistics(),
                }
                yield f"data: {json.dumps(combined_data)}\n\n"

            last_data = current_data
            last_trace = current_trace
            time.sleep(0.2)

    def get_current_data():
        return {
            "queued": dispatcher.get_queued_tasks(),
            "active": dispatcher.get_active_tasks(),
            "completed": dispatcher.get_completed_tasks(),
            "failed": dispatcher.get_failed_tasks(),
        }

    def get_current_trace():
        trace = dispatcher.get_traces()
        return {"_".join(map(str, k)): v for k, v in trace.items()}

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/quit", methods=["POST"])
def quit():
    global quit_event
    quit_event.set()
    return jsonify({"status": "ok"})


def run_web_interface(disp: "Dispatcher", port=5000):
    global dispatcher
    dispatcher = disp
    app.run(debug=False, port=port)


def is_quit_requested():
    global quit_event
    return quit_event.is_set()
