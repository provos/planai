import json
import queue
import threading
import time
import uuid

from flask import Flask, request
from flask_socketio import SocketIO, emit

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

# Add constants for session management
SESSION_TIMEOUT = 30 * 60  # 30 minutes in seconds
CLEANUP_INTERVAL = 5 * 60  # Check for stale sessions every 5 minutes


class SessionManager:
    def __init__(self):
        self._sessions = {}  # In-memory storage for session data
        self._sid_to_session_id = {}  # Mapping between SIDs and session IDs
        self._session_timestamps = {}  # Track last activity
        self._cleanup_thread = None
        self._running = False
        self.start_cleanup_worker()

    def start_cleanup_worker(self):
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker)
        self._cleanup_thread.daemon = True
        self._cleanup_thread.start()

    def stop_cleanup_worker(self):
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join()

    def _cleanup_worker(self):
        while self._running:
            self._cleanup_stale_sessions()
            time.sleep(CLEANUP_INTERVAL)

    def _cleanup_stale_sessions(self):
        current_time = time.time()
        stale_sessions = []

        for session_id, last_active in self._session_timestamps.items():
            if current_time - last_active > SESSION_TIMEOUT:
                stale_sessions.append(session_id)

        for session_id in stale_sessions:
            print(f"Cleaning up stale session: {session_id}")
            self.delete_session(session_id)
            # Find and remove corresponding SID
            for sid, sess_id in list(self._sid_to_session_id.items()):
                if sess_id == session_id:
                    self.remove_sid(sid)

    def debug_dump(self):
        print("Sessions:")
        print(json.dumps(self._sessions, indent=2))
        print("SID to session ID:")
        print(json.dumps(self._sid_to_session_id, indent=2))

    def create_session(self, session_id=None):
        if not session_id:
            session_id = str(uuid.uuid4())
        self._sessions[session_id] = {}  # Initialize data as needed
        self._session_timestamps[session_id] = time.time()
        return session_id

    def get_session(self, session_id):
        return self._sessions.get(session_id)

    def update_session(self, session_id, data):
        if session_id in self._sessions:
            self._sessions[session_id].update(data)

    def update_session_timestamp(self, session_id):
        if session_id in self._sessions:
            self._session_timestamps[session_id] = time.time()

    def delete_session(self, session_id):
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._session_timestamps.pop(session_id, None)

    def get_all_session_ids(self):
        return list(self._sessions.keys())

    def register_sid(self, sid, session_id):
        self._sid_to_session_id[sid] = session_id

    def get_session_id_by_sid(self, sid):
        return self._sid_to_session_id.get(sid)

    def remove_sid(self, sid):
        return self._sid_to_session_id.pop(sid, None)


session_manager = SessionManager()

# Create a queue to hold tasks
task_queue = queue.Queue()


def background_worker():
    """Worker thread to process tasks from the queue."""
    while True:
        sid, session_id, message = task_queue.get()  # Get a task from the queue
        print(f'Processing message: "{message}" from session: {session_id}')
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

            response = "Here is the final response with some `code` and *formatting*"
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
                "error", str(e), namespace="/", to=sid, callback=handle_emit_error
            )
        finally:
            task_queue.task_done()


def handle_emit_error(e=None):
    if e:
        print(f"Emit error: {e}")


def setup_web_interface(port=5050):
    worker_thread = threading.Thread(target=background_worker)
    worker_thread.daemon = (
        True  # Allow the app to exit when only the worker thread is running
    )
    worker_thread.start()

    try:
        socketio.run(app, port=port, debug=True)
    finally:
        # Cleanup on server shutdown
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
