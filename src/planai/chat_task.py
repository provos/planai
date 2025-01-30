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
