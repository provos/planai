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
from textwrap import dedent
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ConfigDict, Field

from .cached_task import CachedTaskWorker
from .llm_interface import LLMInterface
from .task import TaskWorker, TaskWorkItem


class LLMTaskWorker(TaskWorker):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLMInterface = Field(
        ..., title="LLM", description="The LLM to use for the task"
    )
    prompt: str = Field(
        ..., title="Prompt", description="The prompt to use for the task"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if len(self.output_types) != 1:
            raise ValueError("LLMTask must have exactly one output type")

    def consume_work(self, task: TaskWorkItem):
        return self._invoke_llm(task)

    def _output_type(self):
        return list(self.output_types)[0]

    def _invoke_llm(self, task: TaskWorkItem) -> TaskWorkItem:
        prompt = dedent(
            """
        Here is your input data:
        {task}
        
        Here are your instructions:
        {instructions}
        
        {format_instructions}
        """
        ).strip()

        parser = PydanticOutputParser(pydantic_object=self._output_type())

        # allow subclasses to pre-process the task and present it more clearly to the LLM
        processed_task = self.pre_process(task)

        response = self.llm.generate_pydantic(
            prompt_template=prompt,
            output_schema=self._output_type(),
            system="You are a helpful AI assistant. Please help the user with the following task and produce output in JSON.",
            task=processed_task.model_dump_json(indent=2),
            instructions=self.prompt,
            format_instructions=parser.get_format_instructions(),
        )

        self.post_process(response=response, input_task=task)

    def pre_process(self, task: TaskWorkItem) -> TaskWorkItem:
        """
        Pre-processes the input task before sending it to the LLM. Subclasses can override this method to do additional
        processing or filtering.

        Args:
            task (TaskWorkItem): The input task.

        Returns:
            TaskWorkItem: The pre-processed task.
        """
        return task

    def post_process(self, response: Optional[TaskWorkItem], input_task: TaskWorkItem):
        """
        Post-processes the response from the LLM and publishes the work. Subclasses can override this method to do
        additional processing or filtering. They should call super().post_process() after their custom logic.

        Args:
            response (Optional[TaskWorkItem]): The response from LLM.
            input_task (TaskWorkItem): The input task.
        """
        if response is not None:
            self.publish_work(task=response, input_task=input_task)
        else:
            logging.error(
                "LLM did not return a valid response for task %s with provenance %s",
                input_task.__class__.__name__,
                input_task._provenance,
            )


class CachedLLMTaskWorker(CachedTaskWorker, LLMTaskWorker):
    def _get_cache_key(self, task: TaskWorkItem) -> str:
        """Generate a unique cache key for the input task including the prompt template and model name."""
        upstream_cache_key = super()._get_cache_key(task)

        upstream_cache_key += f" - {self.prompt} - {self.llm.model_name}"
        return hashlib.sha1(upstream_cache_key.encode()).hexdigest()
