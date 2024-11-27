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
import json
import logging
from typing import Any, Dict, List, Optional

from openai import ContentFilterFinishReasonError, LengthFinishReasonError, OpenAI


def translate_tools_for_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = messages.copy()
    for message in messages:
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if (
                    "function" not in tool_call
                    or "arguments" not in tool_call["function"]
                ):
                    continue
                if isinstance(tool_call["function"]["arguments"], dict):
                    tool_call["function"]["arguments"] = json.dumps(
                        tool_call["function"]["arguments"]
                    )

    return messages


class OpenAIWrapper:
    def __init__(self, api_key: str, max_tokens: int = 4096):
        self.client = OpenAI(api_key=api_key)
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
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

        # if there are tool_calls, convert them to OpenAI format
        messages = translate_tools_for_openai(messages)

        api_params = {
            "model": kwargs.get("model", "gpt-3.5-turbo"),
            "messages": messages,
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if tools:
            # Convert tools to OpenAI function format which at the moment is identical to the tool format
            api_params["tools"] = tools

        if "options" in kwargs:
            if "temperature" in kwargs["options"]:
                api_params["temperature"] = kwargs["options"]["temperature"]

        logging.debug("API parameters: %s", api_params)

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
                if "format" in kwargs and kwargs["format"] == "json":
                    api_params["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**api_params)
                message = response.choices[0].message
                content = message.content

            # Log the usage details
            usage = response.usage
            logging.info(
                "Usage details - Prompt tokens: %d, Completion tokens: %d, Total tokens: %d, Cached tokens: %d, Reasoning tokens: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                (
                    usage.prompt_tokens_details.cached_tokens
                    if usage.prompt_tokens_details
                    else 0
                ),
                (
                    usage.completion_tokens_details.reasoning_tokens
                    if usage.completion_tokens_details
                    else 0
                ),
            )

            return_message = {"message": {"content": content}}
            # Check for tool calls
            if message.tool_calls:
                return_message["message"]["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                    for tool_call in message.tool_calls
                ]

            return return_message
        except LengthFinishReasonError:
            # Handle the length error
            return {
                "error": "Response exceeded the maximum allowed length.",
                "content": None,
                "done": False,
            }
        except ContentFilterFinishReasonError:
            # Handle the content filter error
            return {
                "error": "Content was rejected by the content filter.",
                "content": None,
                "done": False,
            }
        except Exception as e:
            raise e
