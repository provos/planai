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
    use_cache: bool = True,
) -> LLMInterface:
    """
    Creates and configures a language model interface based on specified provider and parameters.

    This function initializes a LLMInterface instance with the appropriate wrapper/client
    based on the selected provider (ollama, remote_ollama, openai, or anthropic).

    Args:
        provider (Literal["ollama", "remote_ollama", "openai"]): The LLM provider to use.
            Defaults to "ollama".
        model_name (str): Name of the model to use. Defaults to "llama3".
        max_tokens (int): Maximum number of tokens for model responses. Defaults to 4096.
        host (Optional[str]): Host address for local ollama instance. Only used with "ollama" provider.
        hostname (Optional[str]): Remote hostname for SSH connection. Required for "remote_ollama".
        username (Optional[str]): Username for SSH connection. Required for "remote_ollama".
        log_dir (str): Directory for storing logs. Defaults to "logs".
        use_cache (bool): Whether to cache model responses. Defaults to True.

    Returns:
        LLMInterface: Configured interface for interacting with the specified LLM.

    Raises:
        ValueError: If required API keys are not found in environment variables,
            or if an invalid provider is specified.

    Examples:
        >>> # Create an OpenAI interface
        >>> llm = llm_from_config(provider="openai", model_name="gpt-4")

        >>> # Create a local Ollama interface
        >>> llm = llm_from_config(provider="ollama", model_name="llama2")

        >>> # Create a remote Ollama interface
        >>> llm = llm_from_config(
        ...     provider="remote_ollama",
        ...     hostname="example.com",
        ...     username="user"
        ... )
    """
    match provider:
        case "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            wrapper = OpenAIWrapper(api_key=api_key, max_tokens=max_tokens)
            # add gpt-4o once the switch is made
            support_structured_outputs = model_name in [
                "gpt-4o-mini",
                "gpt-4o-mini-2024-07-18",
                "gpt-4o-2024-08-06",
                "gpt-4o-2024-11-20",
                "gpt-4o",
            ]
            support_json_mode = model_name not in ["o1-mini", "o1-preview"]
            support_system_prompt = model_name not in ["o1-mini", "o1-preview"]
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=wrapper,
                support_json_mode=support_json_mode,
                support_structured_outputs=support_structured_outputs,
                support_system_prompt=support_system_prompt,
                use_cache=use_cache,
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
                use_cache=use_cache,
            )
        case "ollama" | "remote_ollama":
            # Enable structured outputs for Llama 3+ models
            supports_structured = True
            if provider == "remote_ollama":
                ssh = SSHConnection(
                    hostname=hostname,
                    username=username,
                )
                client = RemoteOllama(ssh_connection=ssh, model_name=model_name)
            else:
                client = None
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=client,
                host=host,
                support_json_mode=True,
                support_structured_outputs=supports_structured,
                use_cache=use_cache,
            )

    raise ValueError(f"Invalid LLM provider in config: {provider}")
