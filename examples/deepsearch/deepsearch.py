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


@socketio.on("chat_message")
def handle_message(message):
    print(f'Received message: "{message}"')
    if llm is None:
        socketio.emit("error", "LLM not initialized")
        return

    try:
        # Use the LLM to generate a response
        response = "Here is a placeholder response"
        print(f"Sending response: {response}")
        socketio.emit("chat_response", {"message": response})
    except Exception as e:
        print(f"Error processing message: {e}")
        socketio.emit("error", str(e))


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
