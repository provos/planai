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
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, APIError

from .llm_tool import Tool


class AnthropicWrapper:
    def __init__(self, api_key: str, max_tokens: int = 4096):
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Conduct a chat conversation using the Anthropic API.

        Args:
            messages (list[Mapping[str, str]]): A list of message dictionaries, each containing 'role' and 'content'.
            **kwargs: Additional arguments to pass to the generate function.

        Returns:
            A dictionary containing the Anthropic response formatted to match Ollama's expected output.
        """
        # Extract the system message from the messages and prepare it as a separate argument
        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )

        # Filter out the system message to prevent duplication if it's not needed in the messages parameter
        filtered_messages = [msg for msg in messages if msg["role"] != "system"]

        try:
            # Common parameters
            params = {
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "messages": filtered_messages,
                "model": kwargs.get("model", "claude-3-5-sonnet-20240620"),
                "system": system_message,  # Pass the system message here
            }

            # Conditionally add temperature if it exists in kwargs
            if "options" in kwargs:
                if "temperature" in kwargs["options"]:
                    params["temperature"] = kwargs["options"]["temperature"]

            # Call the function with the constructed parameters
            response = self.client.messages.create(**params)

            # Extract content blocks as text and simulate Ollama-like response
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )

            return {"message": {"content": content}}

        except APIError as e:
            raise e
