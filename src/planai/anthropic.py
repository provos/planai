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

from anthropic import Anthropic, APIError


class AnthropicWrapper:
    def __init__(self, api_key: str, max_tokens: int = 4096):
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system: str = "",
        format: Literal["", "json"] = "",
        model: str = "claude-3-5-sonnet-20240620",
    ) -> Mapping[str, Any]:
        """
        Create a response using the requested Anthropic model.

        Args:
            prompt (str): The main input prompt for the model.
            system (str, optional): The system message to set the behavior of the assistant. Defaults to ''.
            format (Literal['', 'json'], optional): If set to 'json', the response will be in JSON format. Defaults to ''.
            model (str, optional): The Anthropic model to use. Defaults to 'claude-3-5-sonnet-20240620'.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 4096.

        Raises:
            Exception: For any API errors.

        Returns:
            Mapping[str, Any]: A dictionary containing the response and completion status.
        """
        messages = []
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.messages.create(
                max_tokens=self.max_tokens,
                messages=messages,
                model=model,
                system=system,
            )

            # Extract the text from content blocks
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )

            return {"response": content, "done": True}

        except APIError as e:
            raise e

    def chat(self, messages: list[Mapping[str, str]], **kwargs):
        """
        Conduct a chat conversation using the Anthropic API.

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
