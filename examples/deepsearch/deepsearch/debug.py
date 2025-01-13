import pickle
import threading
import time
import logging
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Tuple

from pydantic import BaseModel

from planai import TaskWorker


class DummyTaskWorker(BaseModel):
    fake_name: str

    def consume_work(self, task: Any):
        pass

    @property
    def name(self):
        return self.fake_name


def replace_task_worker(arg: Any) -> Any:
    """Replace TaskWorker instances with DummyTaskWorker."""
    if isinstance(arg, TaskWorker) and not isinstance(arg, DummyTaskWorker):
        dummy = DummyTaskWorker(fake_name=arg.name)
        return dummy
    elif isinstance(arg, (list, tuple)):
        return type(arg)(replace_task_worker(x) for x in arg)
    elif isinstance(arg, dict):
        return {k: replace_task_worker(v) for k, v in arg.items()}
    return arg


class DebugSaver:
    def __init__(
        self,
        output_dir: str,
        mode: Literal["capture", "replay"] = "capture",
        replay_delay: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.replay_delay = replay_delay
        self.fp = None
        self._replay_data = {}
        self._function_registry = {}
        self._active_replays = {}
        self._replay_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def capture(self, func_name: str = None):
        # In replay mode, return a no-op decorator
        if self.mode == "replay":
            return lambda f: f

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get session_id from metadata in either args or kwargs
                metadata = next(
                    (
                        arg
                        for arg in args
                        if isinstance(arg, dict) and "session_id" in arg
                    ),
                    kwargs.get("metadata", {}),
                )
                session_id = metadata.get("session_id", "default")

                # Ensure we have a file open for this session
                if not self.fp:
                    self.fp = open(self.output_dir / f"{session_id}.log", "wb")

                # Replace TaskWorker instances with DummyTaskWorker
                safe_args = tuple(replace_task_worker(arg) for arg in args)
                safe_kwargs = {k: replace_task_worker(v) for k, v in kwargs.items()}

                # Capture the call
                call_data = {
                    "func_name": func_name or func.__name__,
                    "args": pickle.dumps(safe_args),
                    "kwargs": pickle.dumps(safe_kwargs),
                    "timestamp": datetime.now().isoformat(),
                }

                # Write the pickled call data
                pickle.dump(call_data, self.fp)
                self.fp.flush()

                return func(*args, **kwargs)

            return wrapper

        return decorator

    def register_replay_handler(self, func_name: str, func: Callable):
        """Register original function for replay."""
        if self.mode != "replay":
            raise RuntimeError("Handlers can only be registered in replay mode")
        self._function_registry[func_name] = func

    def patch_metadata(
        self, args: list, kwargs: dict, current_metadata: dict
    ) -> Tuple[list, dict]:
        """Replace session metadata in args/kwargs with current session info."""
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, dict) and "session_id" in arg:
                new_args[i] = current_metadata
                break

        new_kwargs = kwargs.copy()
        if "metadata" in new_kwargs:
            new_kwargs["metadata"] = current_metadata

        return new_args, new_kwargs

    def _replay_session_thread(self, prompt: str, session_id: str):
        """Thread function to replay events with delays."""
        with self._lock:
            replay_state = self._active_replays.get(session_id)
            if not replay_state:
                logging.debug(f"No replay state found for session {session_id}")
                return

        replay_data = self._replay_data.get(prompt, None)
        if replay_data is None:
            # pick the first entry
            replay_data = self._replay_data[list(self._replay_data.keys())[0]]

        logging.debug(
            f"Starting replay for session {session_id} with {len(replay_data)} events"
        )

        while replay_state["index"] < len(replay_data):
            if replay_state.get("abort", False):
                logging.debug(f"Abort detected for session {session_id}")
                break

            call_data = replay_data[replay_state["index"]]
            func_name = call_data["func_name"]
            args = list(pickle.loads(call_data["args"]))
            kwargs = pickle.loads(call_data["kwargs"])

            func = self._function_registry.get(func_name)
            if func:
                patched_args, patched_kwargs = self.patch_metadata(
                    args, kwargs, replay_state["metadata"]
                )
                func(*patched_args, **patched_kwargs)

            replay_state["index"] += 1
            if replay_state["index"] < len(replay_data) and not replay_state.get(
                "abort", False
            ):
                time.sleep(self.replay_delay)

        logging.debug(f"Replay finished or aborted for session {session_id}")
        # Cleanup when done or aborted, indicate we're calling from within thread
        self._cleanup_replay_session(session_id, from_thread=True)

    def _cleanup_replay_session(self, session_id: str, from_thread: bool = False):
        """Clean up replay session resources."""
        thread = None
        with self._lock:
            if session_id in self._replay_threads:
                thread = self._replay_threads[session_id]
            # Only attempt to join if we're not in the replay thread

        if thread:
            if thread.is_alive() and not from_thread:
                logging.debug(f"Waiting for replay thread {session_id} to finish")
                thread.join(timeout=1.0)
            with self._lock:
                if session_id in self._replay_threads:
                    del self._replay_threads[session_id]

        with self._lock:
            if session_id in self._active_replays:
                del self._active_replays[session_id]
        logging.debug(f"Cleaned up replay session {session_id}")

    def abort_replay(self, session_id: str):
        """Abort a specific replay session."""
        logging.debug(f"Attempting to abort replay for session {session_id}")
        if session_id in self._active_replays:
            logging.debug(f"Setting abort flag for session {session_id}")
            self._active_replays[session_id]["abort"] = True

            # Give the thread a moment to notice the abort flag
            time.sleep(0.1)
            # Clean up from outside the thread
            self._cleanup_replay_session(session_id, from_thread=False)

    def stop_all_replays(self):
        """Stop all active replay threads."""
        logging.debug("Stopping all replay sessions")
        session_ids = list(self._active_replays.keys())
        for session_id in session_ids:
            self.abort_replay(session_id)

    def start_replay_session(self, prompt: str, current_metadata: dict):
        """Start a new replay session with given metadata in a separate thread."""
        session_id = current_metadata["session_id"]

        # Stop existing replay if any
        self.abort_replay(session_id)

        # Reset or create replay state
        self._active_replays[session_id] = {
            "metadata": current_metadata,
            "index": 0,
            "abort": False,
        }

        # Start new replay thread
        thread = threading.Thread(
            target=self._replay_session_thread, args=(prompt, session_id), daemon=True
        )
        self._replay_threads[session_id] = thread
        thread.start()

    def load_replay_session(self, session_id: str):
        """Load replay data into memory."""
        if self.mode != "replay":
            raise RuntimeError("Loading replay data only allowed in replay mode")

        replay_session_prompt = (
            (self.output_dir / f"{session_id}.prompt").read_text().strip()
        )
        replay_session_file = self.output_dir / f"{session_id}.log"
        self._replay_data[replay_session_prompt] = []

        with open(replay_session_file, "rb") as fp:
            while True:
                try:
                    call_data = pickle.load(fp)
                    self._replay_data[replay_session_prompt].append(call_data)
                except EOFError:
                    break

    def load_replays(self):
        """Load all replay data into memory."""
        for session_file in self.output_dir.glob("*.prompt"):
            session_id = session_file.stem
            self.load_replay_session(session_id)
