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
from typing import Any, Dict, List, Literal, Mapping

from openai import OpenAI


class OpenAIWrapper:
    def __init__(self, api_key: str, max_tokens: int = 4096):
        self.client = OpenAI(api_key=api_key)
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system: str = "",
        format: Literal["", "json"] = "",
        model: str = "gpt-3.5-turbo",
    ) -> Mapping[str, Any]:
        """
        Create a response using the requested OpenAI model.

        Args:
            prompt (str): The main input prompt for the model.
            system (str, optional): The system message to set the behavior of the assistant. Defaults to ''.
            format (Literal['', 'json'], optional): If set to 'json', the response will be in JSON format. Defaults to ''.
            model (str, optional): The OpenAI model to use. Defaults to 'gpt-3.5-turbo'.

        Raises:
            Exception: For any API errors.

        Returns:
            Mapping[str, Any]: A dictionary containing the response and completion status.
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        elif format == "json":
            messages.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds in JSON format.",
                }
            )

        messages.append({"role": "user", "content": prompt})

        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        if format == "json":
            api_params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**api_params)
            content = response.choices[0].message.content

            return {"response": content, "done": True}

        except Exception as e:
            raise e

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Conduct a chat conversation using the OpenAI API, with optional structured output.

        This method interfaces with the OpenAI API to provide conversational capabilities.
        It supports structured outputs using Pydantic models to ensure responses adhere
        to specific JSON schemas if a `response_schema` is provided.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries representing the conversation
                                             history, where each dictionary contains 'role' (e.g., 'system',
                                             'user', 'assistant') and 'content' (the message text).
            **kwargs: Additional parameters that can include:
                - model (str): The OpenAI model to be used. Defaults to 'gpt-3.5-turbo' if not specified.
                - max_tokens (int): Maximum number of tokens for the API call. Defaults to the instance's max_tokens.
                - temperature (float): The temperature setting for the response generation.
                - response_schema (Type[BaseModel]): An optional Pydantic model for structured output.

        Returns:
            Dict[str, Any]: If a `response_schema` is specified, returns a dictionary containing parsed content
                            according to the schema. If the model refuses the request, it returns a refusal message.
                            Without a `response_schema`, it returns the OpenAI response formatted with the message content.

        Raises:
            Exception: Propagates any exceptions raised during the API interaction.
        """
        api_params = {
            "model": kwargs.get("model", "gpt-3.5-turbo"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if "options" in kwargs:
            if "temperature" in kwargs["options"]:
                api_params["temperature"] = kwargs["options"]["temperature"]

        try:
            if "response_schema" in kwargs:
                response = self.client.beta.chat.completions.parse(
                    response_format=kwargs.get("response_schema"),
                    **api_params,
                )
                message = response.choices[0].message

                # Check for refusal
                if "refusal" in message:
                    return {"refusal": message.refusal, "content": None, "done": False}

                content = message.parsed
            else:
                response = self.client.chat.completions.create(**api_params)
                content = response.choices[0].message.content

            return {"message": {"content": content}}

        except Exception as e:
            raise e
