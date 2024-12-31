from planai import (  # noqa
    CachedLLMTaskWorker,
    CachedTaskWorker,
    LLMInterface,
    Task,
    llm_from_config,
)


def setup_web_interface(port=5050):
    pass


def setup_graph(provider: str, model: str):
    llm = llm_from_config(provider, model)  # noqa


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    setup_web_interface(port=args.port)
    setup_graph(args.provider, args.model)


if __name__ == "__main__":
    main()
