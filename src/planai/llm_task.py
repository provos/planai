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
import json
import logging
from pathlib import Path
from textwrap import dedent
from typing import Optional, Type

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ConfigDict, Field

from .cached_task import CachedTaskWorker
from .llm_interface import LLMInterface
from .task import Task, TaskWorker


class LLMTaskWorker(TaskWorker):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm_output_type: Optional[Type[Task]] = Field(
        None,
        description="The output type of the LLM if it differs from the task output type",
    )

    llm: LLMInterface = Field(
        ..., title="LLM", description="The LLM to use for the task"
    )
    prompt: str = Field(
        ..., title="Prompt", description="The prompt to use for the task"
    )
    system_prompt: str = Field(
        "You are a helpful AI assistant. Please help the user with the following task and produce output in JSON.",
        description="The system prompt to use for the task",
    )
    debug_mode: bool = Field(
        False,
        description="Whether to run the LLM to save prompts and responses in json for debugging",
    )
    debug_dir: str = Field("debug", description="The directory to save debug output in")

    def __init__(self, **data):
        super().__init__(**data)
        if self.llm_output_type is None and len(self.output_types) != 1:
            raise ValueError(
                "LLMTask must either have llm_output_type or exactly one output_type"
            )

    def consume_work(self, task: Task):
        return self._invoke_llm(task)

    def _output_type(self):
        if self.llm_output_type is not None:
            return self.llm_output_type
        # the convention is that we pick the first output type if llm_output_type is not set
        return list(self.output_types)[0]

    def _invoke_llm(self, task: Task) -> Task:
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

        # allow subclasses to customize the prompt based on the input task
        task_prompt = self.format_prompt(task)

        # allow subclasses to pre-process the task and present it more clearly to the LLM
        processed_task = self.pre_process(task)

        response = self.llm.generate_pydantic(
            prompt_template=prompt,
            output_schema=self._output_type(),
            system=self.system_prompt,
            task=processed_task.model_dump_json(indent=2),
            instructions=task_prompt,
            format_instructions=parser.get_format_instructions(),
            debug_saver=self._save_debug_output if self.debug_mode else None,
        )

        self.post_process(response=response, input_task=task)

    def format_prompt(self, task: Task) -> str:
        """
        Formats the prompt for the LLM based on the input task. Can be customized by subclasses.

        Args:
            task (Task): The input task.

        Returns:
            str: The formatted prompt.
        """
        return self.prompt

    def pre_process(self, task: Task) -> Task:
        """
        Pre-processes the input task before sending it to the LLM. Subclasses can override this method to do additional
        processing or filtering.

        Args:
            task (Task): The input task.

        Returns:
            Task: The pre-processed task.
        """
        return task

    def post_process(self, response: Optional[Task], input_task: Task):
        """
        Post-processes the response from the LLM and publishes the work. Subclasses can override this method to do
        additional processing or filtering. They should call super().post_process() if they want the task to be published
        for downstream processing.

        Args:
            response (Optional[Task]): The response from LLM.
            input_task (Task): The input task.
        """
        if response is not None:
            self.publish_work(task=response, input_task=input_task)
        else:
            logging.error(
                "LLM did not return a valid response for task %s with provenance %s",
                input_task.name,
                input_task._provenance,
            )

    def _save_debug_output(self, prompt: str, response: Optional[Task]):
        """
        Save the prompt and response in JSON format for debugging purposes.

        Args:
            prompt (str): The prompt used for the LLM.
            response (Optional[Task]): The response from the LLM.
            task_prompt (str): The prompt based on the input task.
        """
        if response is None:
            return

        output = Path(self.debug_dir) / f"{self.name}.json"
        if not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)

        output_dict = {
            "prompt_template": prompt,
            "response": response.model_dump() if response is not None else None,
        }

        with open(output, "a", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2)


class CachedLLMTaskWorker(CachedTaskWorker, LLMTaskWorker):
    def _get_cache_key(self, task: Task) -> str:
        """Generate a unique cache key for the input task including the prompt template and model name."""
        upstream_cache_key = super()._get_cache_key(task)

        upstream_cache_key += f" - {self.prompt} - {self.llm.model_name}"
        return hashlib.sha1(upstream_cache_key.encode()).hexdigest()
