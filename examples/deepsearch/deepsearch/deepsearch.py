# Copyright (c) 2024 Niels Provos
#
# This example is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/
# or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# This example is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License for more details.

import logging
import os
import queue
import re
import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from flask import Flask, request
from flask_socketio import SocketIO, emit
from llm_interface import ListResponse, llm_from_config

from planai import ChatMessage, ChatTask, ProvenanceChain, Task, TaskWorker
from planai.utils import setup_logging

from .debug import DebugSaver
from .graph import (
    PhaseAnalyses,
    Plan,
    Request,
    Response,
    SearchQueries,
    SearchQuery,
    setup_graph,
)
from .session import SessionManager
from .user_session import UserSessionManager

app = Flask(__name__)
app.config["SECRET_KEY"] = "just_a_toy_example"
socketio = SocketIO(app, cors_allowed_origins="*")


session_manager = SessionManager()
user_sessions = UserSessionManager("saved_sessions")
debug_saver = None

# Create a queue to hold tasks
task_queue: queue.Queue[Tuple[str, str, Response]] = queue.Queue()

# Globals for worker thread
worker_thread = None
graph_thread = None
should_stop = False
graph = None
entry_worker: TaskWorker = None  # Executes the whole plan
chat_worker: TaskWorker = None  # Allows the user to just chat with the AI assistant

# Add new global settings variable
current_settings = {
    "provider": "ollama",
    "model": "llama2",
    "serperApiKey": os.environ.get("SERPER_API_KEY", ""),
    "openAiApiKey": os.environ.get("OPENAI_API_KEY", ""),
    "anthropicApiKey": os.environ.get("ANTHROPIC_API_KEY", ""),
    "ollamaHost": "localhost:11434",  # Add default Ollama host
}


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
                    if message and message.response_type == "final":
                        global user_sessions
                        user_sessions.add_message(
                            session_id,
                            ChatMessage(role="assistant", content=message.message),
                        )
                        user_sessions.save_session(session_id)

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


def start_graph_thread(current_settings: Dict[str, str]):
    """Modified to use current settings."""
    global graph_thread, graph, entry_worker, chat_worker, debug_saver

    provider = current_settings["provider"]
    model = current_settings["model"]
    host = current_settings["ollamaHost"]

    graph, entry_worker, chat_worker = setup_graph(
        provider=provider, model=model, host=host, notify=notify
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
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )
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
    messages = data.get("messages", [])  # Get full message history

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

    print(f'Received messages: "{messages}" from session: {session_id}')

    # Capture the request.sid (SocketIO SID)
    sid = request.sid
    current_metadata = {"session_id": session_id, "sid": sid}

    # Update session timestamp on activity
    session_manager.update_session_timestamp(session_id)

    # record that we started
    session_metadata = session_manager.metadata(session_id)
    session_metadata["started"] = True

    # get the last message:
    message = messages[-1].get("content", "")

    global user_sessions
    user_sessions.add_message(session_id, ChatMessage(role="user", content=message))
    user_sessions.save_session(session_id)

    # If in replay mode, trigger replay with current session
    global debug_saver
    if debug_saver:
        if debug_saver.mode == "replay":
            if debug_saver.start_replay_session(
                message.strip(), current_metadata, allow_unknown=True
            ):
                return
            # Fall through and allow the session to proceed
        else:
            debug_saver.save_prompt(session_id, message.strip())

    # XXX - this is a hack to create the chat interaction capability
    if len(messages) > 1:
        global chat_worker
        task = ChatTask(messages=messages)
        provenance = chat_worker.add_work(
            task,
            metadata={"session_id": session_id, "sid": sid},
        )
        session_metadata["provenance"] = provenance
        return

    # Add the task to the graph
    global graph, entry_worker
    user_request = Request(user_input=message)
    provenance = entry_worker.add_work(
        user_request,
        metadata={"session_id": session_id, "sid": sid},
        status_callback=notify_planai,
    )
    # Remember the provenance for this session, so that we can abort it if needed
    session_metadata["provenance"] = provenance


def notify_planai(
    metadata: Dict[str, Any],
    prefix: ProvenanceChain,
    worker: TaskWorker,
    task: Task,
    message: str,
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


def patch_notify_functions():
    """Patch the notify functions with debug_saver decorators."""
    global notify, notify_planai, debug_saver
    if debug_saver:
        notify = debug_saver.capture("notify")(notify)
        notify_planai = debug_saver.capture("notify_planai")(notify_planai)


def stop_graph_thread(timeout: float = 5.0) -> bool:
    """Stop the graph thread gracefully.

    Args:
        timeout (float): Maximum time to wait for tasks to complete in seconds

    Returns:
        bool: True if shutdown was successful, False if timeout occurred
    """
    global graph, graph_thread

    if graph and graph_thread and graph_thread.is_alive():
        success = graph.shutdown(timeout=timeout)
        if success:
            graph_thread.join(timeout=1.0)
            graph_thread = None
            graph = None
            return True
        return False
    return True


def stop_threads():
    """Stop all background threads gracefully."""
    stop_worker_thread()
    stop_graph_thread()


def guess_model(settings: Dict[str, str]):
    """Guess a model based on the current configuration."""
    if settings["provider"] == "ollama":
        interface = llm_from_config(
            provider=settings["provider"], host=settings["ollamaHost"]
        )
        # try to see whether we can list the models from ollama - our preferred case
        try:
            models = interface.list()
            if settings["model"] not in [model.model for model in models.models]:
                # choose either a llama3 or phi4 model
                for model in models.models:
                    if model.model.startswith("llama3") or model.model.startswith(
                        "phi4"
                    ):
                        settings["model"] = model.model
                        return
        except Exception:
            pass

    if os.environ.get("OPENAI_API_KEY"):
        settings["provider"] = "openai"
        settings["model"] = "gpt-4o-mini"
        return

    if os.environ.get("ANTHROPIC_API_KEY"):
        settings["provider"] = "anthropic"
        settings["model"] = "claude-3-haiku-20240307"

    # we leave the model as is
    return


def validate_provider(provider: str, api_key: str = None) -> Tuple[bool, List[str]]:
    """Validate provider and return available models.

    For Ollama, api_key parameter is used to pass the host address.
    For other providers, it's used as the API key.
    """
    global current_settings
    try:
        kwargs = {}
        if provider == "ollama":
            # For Ollama, use api_key as host if provided
            kwargs["host"] = api_key or current_settings["ollamaHost"]
        else:
            # For other providers, set API key in environment
            if api_key:
                os.environ[f"{provider.upper()}_API_KEY"] = api_key

        interface = llm_from_config(provider=provider, **kwargs)
        response = interface.list()
        if not isinstance(response, ListResponse):
            return False, []

        # for anthropic, only return models that just with claude
        # for openai, only return models that start with gpt or o
        if provider == "anthropic":
            response.models = [
                model for model in response.models if model.model.startswith("claude")
            ]
        elif provider == "openai":
            response.models = [
                model
                for model in response.models
                if model.model.startswith(("gpt", "o"))
            ]

        # If we successfully validated a new Ollama host, store it
        if provider == "ollama" and api_key:
            current_settings["ollamaHost"] = api_key

        return True, [model.model for model in response.models]
    except Exception as e:
        if provider == "ollama":
            print(f"Error validating provider {provider} with host {api_key}: {e}")
        else:
            print(f"Error validating provider {provider}: {e}")
        return False, []


@socketio.on("validate_provider")
def handle_validate_provider(data):
    """Validate provider and API key/host, return available models."""
    provider = data.get("provider", "").lower()
    value = data.get("apiKey")  # This could be either an API key or Ollama host

    is_valid, models = validate_provider(provider, api_key=value)

    emit(
        "provider_validated",
        {"provider": provider, "isValid": is_valid, "availableModels": models},
    )


@socketio.on("load_settings")
def handle_load_settings():
    """Load current settings and validate all providers."""
    global current_settings

    settings = {
        "provider": current_settings["provider"],
        "modelName": current_settings["model"],
        "serperApiKey": bool(current_settings["serperApiKey"]),
        "ollamaHost": current_settings["ollamaHost"],  # Add Ollama host to settings
        "providers": {},
    }

    # Check Ollama
    ollama_valid, ollama_models = validate_provider("ollama")
    settings["providers"]["ollama"] = {
        "available": ollama_valid,
        "models": ollama_models,
    }

    # Check OpenAI
    openai_valid, openai_models = validate_provider("openai")
    settings["providers"]["openai"] = {
        "available": openai_valid,
        "models": openai_models,
        "hasKey": bool(current_settings["openAiApiKey"]),
    }

    # Check Anthropic
    anthropic_valid, anthropic_models = validate_provider("anthropic")
    settings["providers"]["anthropic"] = {
        "available": anthropic_valid,
        "models": anthropic_models,
        "hasKey": bool(current_settings["anthropicApiKey"]),
    }

    emit("settings_loaded", settings)


@socketio.on("save_settings")
def handle_save_settings(data):
    """Save settings and update environment variables."""
    logging.info(f"Received settings: {data.keys()}")
    global current_settings
    try:
        # Update current settings with lowercase provider
        current_settings["provider"] = data.get("provider", "ollama").lower()
        current_settings["model"] = data.get("modelName", "llama2")
        current_settings["ollamaHost"] = data.get(
            "ollamaHost", "localhost:11434"
        )  # Save Ollama host

        # Update environment variables if new keys provided
        if data.get("serperApiKey"):
            os.environ["SERPER_API_KEY"] = data["serperApiKey"]
            current_settings["serperApiKey"] = data["serperApiKey"]

        if data.get("openAiApiKey"):
            os.environ["OPENAI_API_KEY"] = data["openAiApiKey"]
            current_settings["openAiApiKey"] = data["openAiApiKey"]

        if data.get("anthropicApiKey"):
            os.environ["ANTHROPIC_API_KEY"] = data["anthropicApiKey"]
            current_settings["anthropicApiKey"] = data["anthropicApiKey"]

        # Restart graph with new settings if provider or model changed
        global graph, entry_worker
        if graph:
            stop_graph_thread()
            start_graph_thread(current_settings)

        emit("settings_saved", {"status": "success"})
    except Exception as e:
        emit("error", str(e))


@socketio.on("list_sessions")
def handle_list_sessions():
    """List all past user sessions."""
    logging.info("Listing user sessions")
    global user_sessions
    sessions = user_sessions.list_sessions()
    emit("sessions_listed", {"sessions": sessions})


@socketio.on("get_session")
def handle_get_session(data):
    """Retrieve a specific user session."""
    logging.info("Retrieving user session: %s", data)
    session_id = data.get("session_id")
    if not session_id:
        emit("error", "Invalid session ID")
        return

    global user_sessions
    session = user_sessions.get_session(session_id)
    if not session:
        emit("error", "Session not found")
        return

    emit("session_retrieved", session.model_dump())


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--provider", type=str, default="ollama")
    parser.add_argument("--model", type=str, default="llama3.3:latest")
    parser.add_argument("--ollama-host", type=str, default="localhost:11434")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--replay-delay",
        type=float,
        default=0.2,
        help="Delay between replay events in seconds",
    )
    args = parser.parse_args()

    # Initialize current settings from args
    global current_settings
    current_settings["provider"] = args.provider
    current_settings["model"] = args.model
    current_settings["ollamaHost"] = args.ollama_host
    guess_model(current_settings)
    print(
        f"Starting with settings: {current_settings['provider']} {current_settings['model']}"
    )

    if args.debug:
        global debug_saver
        debug_saver = DebugSaver(
            "debug_output",
            mode="replay" if args.replay else "capture",
            replay_delay=args.replay_delay,
        )

        # Patch the notify functions after debug_saver is initialized
        patch_notify_functions()

        if args.replay:
            # Register the original functions for replay
            debug_saver.register_replay_handler("notify", notify)
            debug_saver.register_replay_handler("notify_planai", notify_planai)
            # Load replay data
            debug_saver.load_replays()

    setup_logging(level=logging.DEBUG if args.debug else logging.ERROR)
    start_graph_thread(current_settings=current_settings)
    setup_web_interface(port=args.port)


if __name__ == "__main__":
    main()
