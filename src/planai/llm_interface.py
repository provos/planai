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
import logging
import re
from typing import Any, Dict, Optional, Tuple, Callable

import diskcache
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from ollama import Client

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
    ):
        self.model_name = model_name
        self.client = client if client else Client(host=host)
        self.support_json_mode = support_json_mode

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
        debug_saver: Optional[Callable[[str, str], None]] = None,
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
            if not self.support_json_mode:
                raw_response = self._strip_text_from_json_response(raw_response)

            error_message, response = self._parse_response(raw_response, parser)

            if response is not None:
                break

            formatted_prompt = (
                f"Your previous response did not follow the JSON format instructions. Here is the error message {error_message}\n\n"
                f"Try again and closly follow these instructions:\n{formatted_prompt}"
            )

        if debug_saver is not None:
            debug_saver(prompt=formatted_prompt, response=response)

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
