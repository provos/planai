import queue
import threading
import time

from flask import Flask, request
from flask_socketio import SocketIO, emit
from session import SessionManager

from planai import (  # noqa
    CachedLLMTaskWorker,
    CachedTaskWorker,
    LLMInterface,
    Task,
    llm_from_config,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "just_a_toy_example"
socketio = SocketIO(app, cors_allowed_origins="*")
llm = None

session_manager = SessionManager()

# Create a queue to hold tasks
task_queue = queue.Queue()

# Globals for worker thread
worker_thread = None
should_stop = False


def start_worker_thread():
    """Create and start a new worker thread."""
    global worker_thread, should_stop
    should_stop = False

    def worker():
        """Worker thread to process tasks from the queue."""
        while not should_stop:
            try:
                sid, session_id, message = task_queue.get(
                    timeout=1.0
                )  # Get a task with timeout
                if sid is None:  # Use None as a signal to stop the worker
                    task_queue.task_done()
                    break
                try:
                    # Simulate a multi-step thinking process
                    steps = [
                        "**Processing** your request...",
                        "1. *Analyzing* input\n2. *Retrieving* relevant context",
                        "3. *Generating* response\n4. *Evaluating* response",
                        "**Finalizing** response...",
                    ]

                    for step in steps:
                        socketio.emit(
                            "thinking_update",
                            {"message": step},
                            namespace="/",
                            to=sid,
                            callback=handle_emit_error,
                        )
                        time.sleep(1)  # Simulate processing time

                    response = (
                        "Here is the final response with some `code` and *formatting*"
                    )
                    print(f"Sending response: {response} to session: {session_id}")
                    socketio.emit(
                        "chat_response",
                        {"message": response},
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


def stop_worker_thread():
    """Stop the worker thread gracefully."""
    global should_stop, worker_thread
    if worker_thread and worker_thread.is_alive():
        should_stop = True
        task_queue.put((None, None, None))  # Signal to stop
        worker_thread.join(timeout=1.0)  # Reduced timeout to avoid blocking
        worker_thread = None


def handle_emit_error(e=None):
    if e:
        print(f"Emit error: {e}")


def setup_web_interface(port=5050):
    global worker_thread

    start_worker_thread()

    try:
        socketio.run(app, port=port, debug=True)
    except (KeyboardInterrupt, SystemExit):
        stop_worker_thread()
    finally:
        session_manager.stop_cleanup_worker()


def setup_graph(provider: str, model: str):
    global llm
    llm = llm_from_config(provider, model)


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
    if llm is None:
        emit("error", "LLM not initialized")
        return

    # Capture the request.sid (SocketIO SID)
    sid = request.sid

    # Update session timestamp on activity
    session_manager.update_session_timestamp(session_id)

    # Add the task to the queue
    task_queue.put((sid, session_id, message))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    setup_graph(args.provider, args.model)
    setup_web_interface(port=args.port)


if __name__ == "__main__":
    main()
