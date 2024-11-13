# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import diskcache
from dotenv import load_dotenv
from ollama import Client
from pydantic import BaseModel

from .pydantic_output_parser import MinimalPydanticOutputParser
from .utils import setup_logging

# Load environment variables from .env.local file
load_dotenv(".env.local")


class LLMInterface:
    def __init__(
        self,
        model_name: str = "llama2",
        log_dir: str = "logs",
        client: Optional[Any] = None,
        host: Optional[str] = None,
        support_json_mode: bool = True,
        support_structured_outputs: bool = False,
        support_system_prompt: bool = True,
    ):
        self.model_name = model_name
        self.client = client if client else Client(host=host)
        self.support_json_mode = support_json_mode
        self.support_structured_outputs = support_structured_outputs
        self.support_system_prompt = support_system_prompt

        self.logger = setup_logging(
            logs_dir=log_dir, logs_prefix="llm_interface", logger_name=__name__
        )

        # Initialize disk cache for caching responses
        self.disk_cache = diskcache.Cache(
            directory=".response_cache", eviction_policy="least-recently-used"
        )

    def _cached_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        # Concatenate all messages to use as the cache key
        message_content = "".join([msg["role"] + msg["content"] for msg in messages])
        prompt_hash = self._generate_hash(
            self.model_name + f"-{temperature}" if temperature else "" + message_content
        )

        # Check if prompt response is in cache
        response = self.disk_cache.get(prompt_hash)

        if response is None:
            kwargs = {}

            # some models can generate structured outputs
            if self.support_structured_outputs and response_schema:
                kwargs["response_schema"] = response_schema
            elif self.support_json_mode:
                kwargs["format"] = "json"

            # ollama expects temperature to be passed as an option
            options = {}
            if temperature:
                options["temperature"] = temperature
                kwargs["options"] = options

            # If not in cache, make request to client using chat interface
            response = self.client.chat(
                model=self.model_name, messages=messages, **kwargs
            )

            # Cache the response with hashed prompt as key
            self.disk_cache.set(prompt_hash, response)

        if "error" in response:
            self.logger.error("Error in chat response: %s", response["error"])
        elif "refusal" in response:
            self.logger.error("Model refused the request: %s", response["refusal"])

        return response["message"]["content"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        self.logger.info("Chatting with messages: %s", messages)
        response = self._cached_chat(
            messages=messages, temperature=temperature, response_schema=response_schema
        )
        self.logger.info(
            "Received chat response: %s...",
            response[:850] if isinstance(response, str) else response,
        )
        return response.strip() if isinstance(response, str) else response

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

    def generate_full_prompt(
        self, prompt_template: str, system: str = "", **kwargs
    ) -> str:
        """
        Generate a full prompt with input variables filled in.

        Args:
            prompt_template (str): The prompt template with placeholders for variables.
            system (str): The system prompt to use for generation.
            **kwargs: Keyword arguments to fill in the prompt template.

        Returns:
            str: The formatted prompt
        """
        formatted_prompt = prompt_template.format(**kwargs)
        return formatted_prompt

    def generate_pydantic(
        self,
        prompt_template: str,
        output_schema: Type[BaseModel],
        system: str = "",
        logger: Optional[logging.Logger] = None,
        debug_saver: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
        extra_validation: Optional[Callable[[BaseModel], Optional[str]]] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Optional[BaseModel]:
        """
        Generates a Pydantic model instance based on a specified prompt template and output schema.

        This function uses a prompt template with variable placeholders to generate a full prompt. It utilizes
        this prompt in combination with a specified system prompt to interact with a chat-based interface,
        aiming to produce a structured output conforming to a given Pydantic schema. The function attempts up to
        three iterations to obtain a valid response, applying parsing, validation, and optional extra validation
        functions. If all iterations fail, None is returned.

        Args:
            prompt_template (str): The template containing placeholders for formatting the prompt.
            output_schema (Type[BaseModel]): A Pydantic model that defines the expected schema of the output data.
            system (str): An optional system prompt used during the generation process.
            logger (Optional[logging.Logger]): An optional logger for recording the generated prompt and events.
            debug_saver (Optional[Callable[[str, Dict[str, Any], str], None]]): An optional callback for saving debugging information,
                which receives the prompt and the response.
            extra_validation (Optional[Callable[[BaseModel], str]]): An optional function for additional validation of
                the generated output. It should return an error message if validation fails, otherwise None.
            **kwargs: Additional keyword arguments for populating the prompt template.

        Returns:
            Optional[BaseModel]: An instance of the specified Pydantic model with generated data if successful,
            or None if all attempts at generation fail or the response is invalid.
        """
        parser = MinimalPydanticOutputParser(pydantic_object=output_schema)

        formatted_prompt = self.generate_full_prompt(
            prompt_template=prompt_template, system=system, **kwargs
        )

        self.logger.info("Generated prompt: %s", formatted_prompt)
        if logger:
            logger.info("Generated prompt: %s", formatted_prompt)

        messages = []
        if self.support_system_prompt:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": formatted_prompt})

        iteration = 0
        while iteration < 3:
            iteration += 1

            raw_response = self.chat(
                messages=messages,
                temperature=temperature,
                response_schema=output_schema,
            )
            if not self.support_json_mode:
                raw_response = self._strip_text_from_json_response(raw_response)

            if self.support_structured_outputs:
                # If the model supports structured outputs, the response is already a pydantic object
                response = raw_response
                error_message = (
                    "The model refused the request" if response is None else None
                )
            else:
                error_message, response = self._parse_response(raw_response, parser)

            if response is None:
                messages.extend(
                    [
                        {"role": "assistant", "content": raw_response},
                        {
                            "role": "user",
                            "content": f"Try again. Your previous response was invalid and led to this error message: {error_message}",
                        },
                    ]
                )
                continue

            if extra_validation:
                extra_error_message = extra_validation(response)
                if extra_error_message:
                    if self.support_structured_outputs:
                        # the raw response was a pydantic object, so we need to dump it to a string
                        raw_response = raw_response.model_dump_json()
                    elif not isinstance(raw_response, str):
                        raise ValueError(
                            "The response should be a string if the model does not support structured outputs."
                        )
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw_response},
                            {
                                "role": "user",
                                "content": f"Try again. Your previous response was invalid and led to this error message: {extra_error_message}",
                            },
                        ]
                    )
                    continue
            break

        if debug_saver is not None:
            debug_saver(formatted_prompt, kwargs, response)

        return response

    def _generate_hash(self, prompt: str) -> str:
        hash_object = hashlib.sha256(prompt.encode())
        return hash_object.hexdigest()

    def _parse_response(
        self, response: str, parser: MinimalPydanticOutputParser
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

    @staticmethod
    def get_format_instructions(pydantic_object: Type[BaseModel]) -> str:
        """
        Generate format instructions for a Pydantic model's JSON output.

        This function creates a string of instructions on how to format JSON output
        based on the schema of a given Pydantic model. It's compatible with both
        Pydantic v1 and v2.

        Args:
            pydantic_object (Type[BaseModel]): The Pydantic model class to generate instructions for.

        Returns:
            str: A string containing the format instructions.

        Note:
            This function is adapted from the LangChain framework.
            Original source: https://github.com/langchain-ai/langchain
            License: MIT (https://github.com/langchain-ai/langchain/blob/master/LICENSE)
        """
        _PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```
"""
        schema = pydantic_object.model_json_schema().copy()

        schema.pop("title", None)
        schema.pop("type", None)

        schema_str = json.dumps(schema, ensure_ascii=False)

        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)
