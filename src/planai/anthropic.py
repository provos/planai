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


def translate_tools_for_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translate a list of tools from Ollama/API format to Anthropic format.

    Args:
        tools (List[Tool]): List of tool objects from the Ollama/API.

    Returns:
        List[Dict[str, Any]]: Translated tools ready for Anthropic API consumption.
    """
    anthropic_tools = []

    for tool in tools:
        # Extract the function from the tool
        function = tool["function"]

        # Assuming Tool objects have keys 'name', 'description', and 'parameters' which is a dict
        translated_tool = {
            "name": function["name"],
            "description": function["description"],
            "input_schema": {
                "type": "object",
                "properties": function["parameters"]["properties"],
                "required": function["parameters"]["required"],
            },
        }
        anthropic_tools.append(translated_tool)

    return anthropic_tools


def translate_messages_for_anthropic(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Translate messages from Ollama/API format to Anthropic format.

    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries in Ollama format

    Returns:
        List[Dict[str, Any]]: Translated messages in Anthropic format
    """
    translated_messages = []

    for msg in messages:
        if msg["role"] == "user":
            # Regular user messages pass through unchanged
            translated_messages.append({"role": "user", "content": msg["content"]})

        elif msg["role"] == "assistant" and "tool_calls" in msg:
            # Convert assistant tool calls to Anthropic format
            tool_call = msg["tool_calls"][0]  # Assume single tool call for now
            translated_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<thinking>I need to use {tool_call['function']['name']} to help answer this question.</thinking>",
                        },
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": tool_call["function"]["arguments"],
                        },
                    ],
                }
            )

        elif msg["role"] == "tool":
            # Convert tool response to Anthropic's tool_result format
            translated_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"],
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"Unknown message role: {msg['role']}")

    return translated_messages


class AnthropicWrapper:
    def __init__(self, api_key: str, max_tokens: int = 4096):
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
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

        # Translate messages into Anthropic format
        if any(msg["role"] == "tool" for msg in filtered_messages):
            filtered_messages = translate_messages_for_anthropic(filtered_messages)

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

        if tools:
            # Translate tools into Anthropic format
            anthropic_tools = translate_tools_for_anthropic(tools)
            params["tools"] = anthropic_tools

        try:
            # Call the function with the constructed parameters
            response = self.client.messages.create(**params)

            # Handle tool calls if present
            if any(block.type == "tool_use" for block in response.content):
                # Find all tool use blocks
                tool_use_blocks = [
                    block for block in response.content if block.type == "tool_use"
                ]

                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_block.id,
                                "name": tool_block.name,
                                "arguments": tool_block.input,  # Anthropic uses 'input' instead of 'arguments'
                            }
                            for tool_block in tool_use_blocks
                        ],
                    }
                }

            # Extract content blocks as text and simulate Ollama-like response
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )

            return {"message": {"content": content}}

        except APIError as e:
            raise e
