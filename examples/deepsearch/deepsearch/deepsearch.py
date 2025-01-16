import logging
import queue
import re
import threading
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from debug import DebugSaver
from flask import Flask, request
from flask_socketio import SocketIO, emit
from graph import (
    PhaseAnalyses,
    Plan,
    Request,
    Response,
    SearchQueries,
    SearchQuery,
    setup_graph,
)
from session import SessionManager

from planai import Task, TaskWorker
from planai.utils import setup_logging

app = Flask(__name__)
app.config["SECRET_KEY"] = "just_a_toy_example"
socketio = SocketIO(app, cors_allowed_origins="*")


session_manager = SessionManager()
debug_saver = None

# Create a queue to hold tasks
task_queue: queue.Queue[Tuple[str, str, Response]] = queue.Queue()

# Globals for worker thread
worker_thread = None
graph_thread = None
should_stop = False
graph = None
entry_worker = None


def format_message(message: str) -> str:
    """Format the message for display in the chat window.
    - Replaced URLs with the site name and the favicon in markdown format.
    """
    # Replace URLs with site name and favicon
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

    def get_site_info(url):
        site = urlparse(url).netloc
        return site, urlparse(url).path[:20]

    def format_url(match):
        url = match.group(0)
        site_name, path = get_site_info(url)
        return f"[{site_name}{path}...]({url})"

    message = url_pattern.sub(format_url, message)

    return message


def start_worker_thread():
    """Create and start a new worker thread."""
    global worker_thread, should_stop
    should_stop = False

    def worker():
        """Worker thread to process tasks from the queue."""
        while not should_stop:
            try:
                # message is a Response object
                sid, session_id, message = task_queue.get(
                    timeout=1.0
                )  # Get a task with timeout
                if sid is None:  # Use None as a signal to stop the worker
                    task_queue.task_done()
                    break

                try:
                    print(f"Sending response: {message} to session: {session_id}")
                    message.message = format_message(message.message)
                    response_mapping = {
                        "error": "chat_error",
                        "thinking": "thinking_update",
                        "final": "chat_response",
                    }
                    socketio.emit(
                        response_mapping.get(message.response_type, "error"),
                        message.model_dump(),
                        namespace="/",
                        to=sid,
                        callback=handle_emit_error,
                    )

                except Exception as e:
                    print(f"Error processing message in background thread: {e}")
                    socketio.emit(
                        "error",
                        str(e),
                        namespace="/",
                        to=sid,
                        callback=handle_emit_error,
                    )
                finally:
                    task_queue.task_done()
            except queue.Empty:
                continue  # Allow checking should_stop periodically

    worker_thread = threading.Thread(target=worker)
    worker_thread.daemon = True
    worker_thread.start()
    return worker_thread


@debug_saver.capture("notify") if debug_saver else lambda x: x
def notify(metadata, message: Response):
    """Callback to receive notifications from the graph."""
    global task_queue, session_manager
    session_id = metadata.get("session_id")
    sid = metadata.get("sid")
    print(f"Received response: {str(message)[:100]} for session: {session_id}")

    # this is the final response, so we can mark the session as not started
    session_metadata = session_manager.metadata(session_id)
    session_metadata["started"] = False

    task_queue.put((sid, session_id, message))


def start_graph_thread(
    provider: str = "ollama", model: str = "llama3.3:latest", ollama_port: int = 11434
):
    """Create and start a new worker thread."""
    global graph_thread, graph, entry_worker, debug_saver

    graph, entry_worker = setup_graph(
        provider=provider, model=model, ollama_port=ollama_port, notify=notify
    )

    def worker():
        """Worker thread to process tasks from the queue."""
        graph.prepare(display_terminal=False, run_dashboard=True, dashboard_port=8080)
        graph.execute([])

    graph_thread = threading.Thread(target=worker)
    graph_thread.daemon = True
    graph_thread.start()
    return worker_thread


def stop_worker_thread():
    """Stop the worker thread gracefully."""
    global should_stop, worker_thread, debug_saver
    if worker_thread and worker_thread.is_alive():
        should_stop = True
        task_queue.put((None, None, None))  # Signal to stop
        worker_thread.join(timeout=1.0)  # Reduced timeout to avoid blocking
        worker_thread = None

    if debug_saver:
        debug_saver.stop_all_replays()


def handle_emit_error(e=None):
    if e:
        print(f"Emit error: {e}")


def setup_web_interface(port=5050):
    global worker_thread

    start_worker_thread()

    try:
        socketio.run(app, port=port, debug=False, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        stop_worker_thread()
    finally:
        session_manager.stop_cleanup_worker()


@socketio.on("connect")
def handle_connect():
    # Check if client is attempting to restore a session
    session_id = request.args.get("session_id")

    if session_id and session_id in session_manager._sessions:
        # Restore existing session
        print(f"Restoring session: {session_id} for SID: {request.sid}")
        session_manager.register_sid(request.sid, session_id)
        session_manager.update_session_timestamp(session_id)
    else:
        # Create new session
        session_id = session_manager.create_session()
        print(f"New connection: {session_id} with SID: {request.sid}")
        session_manager.register_sid(request.sid, session_id)

    emit("session_id", {"id": session_id})


@socketio.on("disconnect")
def handle_disconnect():
    # Retrieve the session_id using the disconnected SID from SessionManager
    session_id = session_manager.remove_sid(request.sid)

    if session_id:
        session_manager.delete_session(session_id)
        print(f"Disconnected: {session_id} (SID: {request.sid})")
    else:
        print(f"Disconnected unknown session (SID: {request.sid})")


@socketio.on("abort")
def handle_abort(data):
    session_id = data.get("session_id")
    if not session_id or session_manager.get_session(session_id) is None:
        print(f"Invalid session ID: {session_id}")
        session_manager.debug_dump()
        emit("error", "Invalid session ID")
        return

    # Check if the session_id is associated with the current SID
    expected_session_id = session_manager.get_session_id_by_sid(request.sid)
    if session_id != expected_session_id:
        emit("error", "Session ID does not match current connection")
        return

    print(f"Aborting session: {session_id}")
    session_metadata = session_manager.metadata(session_id)

    # If in replay mode, abort the replay
    global debug_saver
    if debug_saver and debug_saver.mode == "replay":
        debug_saver.abort_replay(session_id)
    else:
        global graph
        graph.abort_work(session_metadata.get("provenance"))

    # Update session timestamp on activity
    session_manager.update_session_timestamp(session_id)


@socketio.on("chat_message")
def handle_message(data):
    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id or session_manager.get_session(session_id) is None:
        print(f"Invalid session ID: {session_id}")
        session_manager.debug_dump()
        emit("error", "Invalid session ID")
        return

    # Check if the session_id is associated with the current SID
    expected_session_id = session_manager.get_session_id_by_sid(request.sid)
    if session_id != expected_session_id:
        emit("error", "Session ID does not match current connection")
        return

    print(f'Received message: "{message}" from session: {session_id}')

    # Capture the request.sid (SocketIO SID)
    sid = request.sid
    current_metadata = {"session_id": session_id, "sid": sid}

    # record that we started
    session_metadata = session_manager.metadata(session_id)
    session_metadata["started"] = True

    # If in replay mode, trigger replay with current session
    global debug_saver
    if debug_saver and debug_saver.mode == "replay":
        debug_saver.start_replay_session(message.strip(), current_metadata)
        return

    # Update session timestamp on activity
    session_manager.update_session_timestamp(session_id)

    @debug_saver.capture("notify_planai") if debug_saver else lambda x: x
    def wrapped_notify_planai(*args, **kwargs):
        return notify_planai(*args, **kwargs)

    # Add the task to the graph
    global graph, entry_worker
    user_request = Request(user_input=message)
    provenance = entry_worker.add_work(
        user_request,
        metadata={"session_id": session_id, "sid": sid},
        status_callback=wrapped_notify_planai,
    )
    # Remember the provenance for this session, so that we can abort it if needed
    session_metadata["provenance"] = provenance


@debug_saver.capture("notify_planai") if debug_saver else lambda x: x
def notify_planai(
    metadata: Dict[str, Any], worker: TaskWorker, task: Task, message: str
):
    """Callback to receive notifications from the graph."""
    session_id = metadata.get("session_id")
    sid = metadata.get("sid")

    # we failed the task
    if worker is None:
        global session_manager
        # get the metadata for this session
        session_metadata = session_manager.metadata(session_id)
        if session_metadata.get("started"):
            # this indicates that we failed the task
            task_queue.put(
                (
                    sid,
                    session_id,
                    Response(
                        response_type="error", message=f"Unknown error: {message}"
                    ),
                )
            )
        return

    logging.info(
        "Got notification from %s and task: %s with message: %s",
        worker.name,
        task.name,
        message,
    )

    # get the metadata for this session
    session_metadata = session_manager.metadata(session_id)

    # try to determine which phase of the plan we are in
    phase = "unknown"
    if isinstance(task, Plan):
        phase = "plan"
    elif isinstance(task, SearchQueries):
        phase = "search"
    elif isinstance(task, SearchQuery):
        phase = task.metadata
    elif isinstance(task, PhaseAnalyses):
        phase = "plan"
    else:
        search_query: SearchQuery = task.find_input_task(SearchQuery)
        if search_query:
            phase = search_query.metadata

    print(f"Received response: {str(message)[:100]} for session: {session_id}")
    task_queue.put(
        (
            sid,
            session_id,
            Response(response_type="thinking", phase=phase, message=message),
        )
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--provider", type=str, default="ollama")
    parser.add_argument("--model", type=str, default="llama3.3:latest")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--replay-delay",
        type=float,
        default=0.2,
        help="Delay between replay events in seconds",
    )
    args = parser.parse_args()

    if args.debug:
        global debug_saver
        debug_saver = DebugSaver(
            "debug_output",
            mode="replay" if args.replay else "capture",
            replay_delay=args.replay_delay,
        )

        if args.replay:
            # Register the original functions for replay
            debug_saver.register_replay_handler("notify", notify)
            debug_saver.register_replay_handler("notify_planai", notify_planai)
            # Load replay data
            debug_saver.load_replays()

    setup_logging(level=logging.DEBUG)
    start_graph_thread(args.provider, args.model, args.ollama_port)
    setup_web_interface(port=args.port)


if __name__ == "__main__":
    main()
