import hashlib
import inspect
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import diskcache
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from ollama import Client
from utils import setup_logging

from openai import OpenAIWrapper
from remote_ollama import RemoteOllama
from ssh import SSHConnection

# Load environment variables from .env.local file
load_dotenv(".env.local")


class LLMInterface:
    def __init__(
        self,
        model_name: str = "llama2",
        log_dir: str = "logs",
        client: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.client = client if client else Client()

        self.logger = setup_logging(
            logs_dir=log_dir, logs_prefix="llm_interface", logger_name=__name__
        )

        # Initialize disk cache for caching responses
        self.disk_cache = diskcache.Cache(
            directory=".response_cache", eviction_policy="least-recently-used"
        )

    def _cached_generate(self, prompt: str, system: str = "", format: str = "") -> str:
        # Hash the prompt to use as the cache key
        prompt_hash = self._generate_hash(
            self.model_name + "\n" + system + "\n" + prompt
        )

        # Check if prompt response is in cache
        response = self.disk_cache.get(prompt_hash)

        if response is None:
            # If not in cache, make request to client
            response = self.client.generate(
                model=self.model_name, prompt=prompt, system=system, format=format
            )

            # Cache the response with hashed prompt as key
            self.disk_cache.set(prompt_hash, response)

        return response

    def generate(self, prompt: str, system: str = "") -> str:
        self.logger.info("Generating text with prompt: %s...", prompt[:850])
        response = self._cached_generate(prompt=prompt, system=system)
        self.logger.info("Generated text: %s...", response["response"][:850])
        return response["response"].strip()

    def _strip_text_from_json_response(self, response: str) -> str:
        pattern = r"^[^{\[]*([{\[].*[}\]])[^}\]]*$"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1)
        else:
            return response  # Return original response if no JSON block is found

    def generate_pydantic(
        self,
        prompt_template: str,
        output_schema: BaseModel,
        system: str = "",
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> Optional[BaseModel]:
        """
        Generate a python object based on the given prompt template and output schema.

        Args:
            prompt_template (str): The prompt template with placeholders for variables.
            output_schema (BaseModel): A Pydantic model defining the expected output structure.
            system (str): The system prompt to use for generation.
            **kwargs: Keyword arguments to fill in the prompt template.

        Returns:
            Dict[str, Any]: The formatted prompt and generated output in JSON format.
        """
        parser = PydanticOutputParser(pydantic_object=output_schema)

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=list(kwargs.keys()),
        )

        formatted_prompt = prompt.format(**kwargs)
        self.logger.info("Generated prompt: %s", formatted_prompt)
        if logger:
            logger.info("Generated prompt: %s", formatted_prompt)

        iteration = 0
        while iteration < 3:
            iteration += 1

            raw_response = self._cached_generate(
                prompt=formatted_prompt, system=system, format="json"
            )["response"].strip()

            error_message, response = self._parse_response(raw_response, parser)

            if response is not None:
                break

            formatted_prompt = (
                f"Your previous response did not follow the JSON format instructions. Here is the error message {error_message}\n\n"
                f"Try again and closly follow these instructions:\n{formatted_prompt}"
            )

        return response

    def _generate_hash(self, prompt: str) -> str:
        hash_object = hashlib.sha256(prompt.encode())
        return hash_object.hexdigest()

    def _parse_response(
        self, response: str, parser: PydanticOutputParser
    ) -> Tuple[str, Dict[str, Any]]:
        self.logger.info("Parsing JSON response: %s", response)
        error_message = None
        try:
            response = parser.parse(response)
        except Exception as e:
            self.logger.error("Error parsing response: %s", e)
            error_message = str(e)
            response = None
        return error_message, response


def llm_from_config(config: SystemConfig) -> LLMInterface:
    config = config.config
    llm_config = config.get("LLM", {})
    # switch on config.config['provider']
    match llm_config.get("provider", "ollama"):
        case "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            wrapper = OpenAIWrapper(api_key=api_key)
            return LLMInterface(
                model_name=llm_config.get("model_name", "gpt-3.5-turbo"),
                log_dir=llm_config.get("log_dir", "logs"),
                client=wrapper,
            )
        case "remote_ollama":
            ssh = SSHConnection(
                hostname=llm_config.get("hostname"),
                username=llm_config.get("username"),
            )
            client = RemoteOllama(
                ssh_connection=ssh, model_name=llm_config.get("model_name", "llama3")
            )
            return LLMInterface(
                model_name=llm_config.get("model_name", "llama3"),
                log_dir=llm_config.get("log_dir", "logs"),
                client=client,
            )
        case "ollama":
            return LLMInterface(
                model_name=llm_config.get("model_name", "llama3"),
                log_dir=llm_config.get("log_dir", "logs"),
            )

    raise ValueError("Invalid LLM provider in config: %s", llm_config.get("provider"))
