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
from typing import Any, Literal, Mapping

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

    def chat(self, messages: list[Mapping[str, str]], **kwargs):
        """
        Conduct a chat conversation using the OpenAI API.

        Args:
            messages (list[Mapping[str, str]]): A list of message dictionaries, each containing 'role' and 'content'.
            **kwargs: Additional arguments to pass to the generate function.

        Returns:
            The response from the generate function.
        """
        system_message = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_messages = [m["content"] for m in messages if m["role"] == "user"]

        prompt = "\n".join(user_messages)

        return self.generate(prompt=prompt, system=system_message, **kwargs)
