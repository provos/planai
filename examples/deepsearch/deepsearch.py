from flask import Flask
from flask_socketio import SocketIO

from planai import (  # noqa
    CachedLLMTaskWorker,
    CachedTaskWorker,
    LLMInterface,
    Task,
    llm_from_config,
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
llm = None


def setup_web_interface(port=5050):
    socketio.run(app, port=port, debug=True)


def setup_graph(provider: str, model: str):
    global llm
    llm = llm_from_config(provider, model)


@socketio.on("search")
def handle_search(query):
    print(f'Searching for "{query}"')
    if llm is None:
        socketio.emit("search_results", {"error": "LLM not initialized"})
        return

    try:
        # This is a placeholder response
        results = [
            {"title": f'Deep search result 1 for "{query}"', "url": "#"},
            {"title": f'Deep search result 2 for "{query}"', "url": "#"},
        ]
        print(f"Sending results: {results}")
        socketio.emit("search_results", results)
    except Exception as e:
        print(f"Error processing search: {e}")
        socketio.emit("search_error", str(e))


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
