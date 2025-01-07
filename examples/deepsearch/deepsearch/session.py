import json
import threading
import time
import uuid

# Constants for session management
SESSION_TIMEOUT = 30 * 60  # 30 minutes in seconds
CLEANUP_INTERVAL = 5 * 60  # Check for stale sessions every 5 minutes


class ThreadSafeDict:
    """A thread-safe dictionary wrapper that can be used directly."""

    def __init__(self, lock, initial_dict=None):
        self._dict = initial_dict if initial_dict is not None else {}
        self._lock = lock

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def update(self, other):
        with self._lock:
            self._dict.update(other)

    def clear(self):
        with self._lock:
            self._dict.clear()

    def to_dict(self):
        """Return a copy of the underlying dictionary"""
        with self._lock:
            return dict(self._dict)


class SessionManager:
    def __init__(self):
        self._sessions = {}  # In-memory storage for session data
        self._sid_to_session_id = {}  # Mapping between SIDs and session IDs
        self._session_timestamps = {}  # Track last activity
        self._session_metadata = {}  # New dictionary for metadata
        self._lock = threading.RLock()  # Reentrant lock for thread safety
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

        with self._lock:
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
        with self._lock:
            print("Sessions:")
            print(json.dumps(self._sessions, indent=2))
            print("SID to session ID:")
            print(json.dumps(self._sid_to_session_id, indent=2))
            print("Session metadata:")
            print(json.dumps(self._session_metadata, indent=2))

    def create_session(self, session_id=None):
        with self._lock:
            if not session_id:
                session_id = str(uuid.uuid4())
            self._sessions[session_id] = {}  # Initialize data as needed
            self._session_timestamps[session_id] = time.time()
            self._session_metadata[session_id] = {}
            return session_id

    def get_session(self, session_id):
        with self._lock:
            return self._sessions.get(session_id)

    def update_session(self, session_id, data):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].update(data)

    def update_session_timestamp(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                self._session_timestamps[session_id] = time.time()

    def delete_session(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._session_timestamps.pop(session_id, None)
                self._session_metadata.pop(session_id, None)

    def get_all_session_ids(self):
        return list(self._sessions.keys())

    def register_sid(self, sid, session_id):
        with self._lock:
            self._sid_to_session_id[sid] = session_id

    def get_session_id_by_sid(self, sid):
        with self._lock:
            return self._sid_to_session_id.get(sid)

    def remove_sid(self, sid):
        with self._lock:
            return self._sid_to_session_id.pop(sid, None)

    def metadata(self, session_id):
        """Return a thread-safe metadata dictionary for the session."""
        with self._lock:
            if session_id not in self._session_metadata:
                self._session_metadata[session_id] = {}
            return ThreadSafeDict(self._lock, self._session_metadata[session_id])
