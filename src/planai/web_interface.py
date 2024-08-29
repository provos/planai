import json
import threading
from typing import TYPE_CHECKING, Dict, List

from flask import Flask, Response, jsonify, render_template

if TYPE_CHECKING:
    from .dispatcher import Dispatcher

from .task import TaskWorker, TaskWorkItem

app = Flask(__name__, template_folder="templates")

dispatcher: "Dispatcher" = None
quit_event = threading.Event()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    def event_stream():
        while True:
            queued_tasks = dispatcher.get_queued_tasks()
            active_tasks = dispatcher.get_active_tasks()
            completed_tasks = dispatcher.get_completed_tasks()

            data = {
                "queued": queued_tasks,
                "active": active_tasks,
                "completed": completed_tasks,
            }

            yield f"data: {json.dumps(data)}\n\n"

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
