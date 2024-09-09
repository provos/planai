import os
from typing import Literal, Optional

from .anthropic import AnthropicWrapper
from .llm_interface import LLMInterface
from .openai import OpenAIWrapper
from .remote_ollama import RemoteOllama
from .ssh import SSHConnection


def llm_from_config(
    provider: Literal["ollama", "remote_ollama", "openai"] = "ollama",
    model_name: str = "llama3",
    max_tokens: int = 4096,
    host: Optional[str] = None,
    hostname: Optional[str] = None,
    username: Optional[str] = None,
    log_dir: str = "logs",
) -> LLMInterface:
    match provider:
        case "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            wrapper = OpenAIWrapper(api_key=api_key, max_tokens=max_tokens)
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=wrapper,
            )
        case "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            wrapper = AnthropicWrapper(api_key=api_key, max_tokens=max_tokens)
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=wrapper,
                support_json_mode=False,
            )
        case "remote_ollama":
            ssh = SSHConnection(
                hostname=hostname,
                username=username,
            )
            client = RemoteOllama(ssh_connection=ssh, model_name=model_name)
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=client,
            )
        case "ollama":
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                host=host,
            )

    raise ValueError(f"Invalid LLM provider in config: {provider}")
