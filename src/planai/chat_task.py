"""Provides chat-based interaction capabilities for PlanAI workflows.

This module enables the integration of traditional chat-based interactions with PlanAI's
complex graph-based workflows. The ChatTaskWorker allows developers to create interactive
sessions where users can:
- Engage with results from complex graph executions
- Trigger new workflow branches based on their inputs
- Maintain contextual conversations while leveraging PlanAI's advanced features

The chat functionality serves as a bridge between automated workflows and user-driven
processes, making it ideal for applications that require human oversight or
interactive decision-making within larger automated systems.
"""

# Copyright 2025 Niels Provos
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
from datetime import datetime
from typing import List, Type

from .llm_task import BaseLLMTaskWorker
from .task import Task


class ChatMessage(Task):
    role: str
    content: str


class ChatTask(Task):
    messages: List[ChatMessage]


class ChatTaskWorker(BaseLLMTaskWorker):
    output_types: List[Type[Task]] = [ChatMessage]
    system_prompt: str = "You are a helpful AI assistant. Today is {date}."

    def _format_messages(self, messages: List[ChatMessage]):
        formatted_messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(
                    date=datetime.today().strftime("%Y-%m-%d")
                ),
            },
        ]
        formatted_messages.extend(
            [{"role": message.role, "content": message.content} for message in messages]
        )
        return formatted_messages

    def consume_work(self, task: ChatTask):
        response = self.llm.chat(self._format_messages(task.messages))
        self.publish_work(ChatMessage(role="assistant", content=response), task)
