import json
import threading
import time
import uuid

# Constants for session management
SESSION_TIMEOUT = 30 * 60  # 30 minutes in seconds
CLEANUP_INTERVAL = 5 * 60  # Check for stale sessions every 5 minutes


class SessionManager:
    def __init__(self):
        self._sessions = {}  # In-memory storage for session data
        self._sid_to_session_id = {}  # Mapping between SIDs and session IDs
        self._session_timestamps = {}  # Track last activity
        self._cleanup_thread = None
        self._running = False
        self._cleanup_event = threading.Event()
        self.start_cleanup_worker()

    def start_cleanup_worker(self):
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker)
        self._cleanup_thread.daemon = True
        self._cleanup_thread.start()

    def stop_cleanup_worker(self):
        self._running = False
        if self._cleanup_thread:
            self._cleanup_event.set()
            if self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=1.0)

    def _cleanup_worker(self):
        while not self._cleanup_event.wait(timeout=60):  # Check every minute
            self._cleanup_stale_sessions()

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
