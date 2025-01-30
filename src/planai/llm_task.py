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
from typing import Any, Dict, Optional, Type

from llm_interface import LLMInterface
from pydantic import ConfigDict, Field

from .cached_task import CachedTaskWorker
from .task import Task, TaskWorker

PROMPT_TEMPLATE = dedent(
    """
    Here is your input data:
    {task}

    Here are your instructions:
    {instructions}
    """
).strip()

PROMPT_FORMAT_INSTRUCTIONS = "\n\n{format_instructions}"


class BaseLLMTaskWorker(TaskWorker):
    """Base class for all TaskWorkers that use LLMs. This can be used to limit the amount of parallel LLM calls. It is not meant to be used directly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: LLMInterface = Field(
        ..., title="LLM", description="The LLM to use for the task"
    )
    system_prompt: str = Field(
        "You are a helpful AI assistant. Please help the user with the following task and produce output in JSON.",
        description="The system prompt to use for the task",
    )


class LLMTaskWorker(BaseLLMTaskWorker):
    llm_output_type: Optional[Type[Task]] = Field(
        None,
        description="The output type of the LLM if it differs from the task output type",
    )
    llm_input_type: Optional[Type[Task]] = Field(
        None,
        description="The input type of the LLM can be provided here instead of consume_work",
    )

    prompt: str = Field(
        ..., title="Prompt", description="The prompt to use for the task"
    )
    debug_mode: bool = Field(
        False,
        description="Whether to run the LLM to save prompts and responses in json for debugging",
    )
    debug_dir: str = Field("debug", description="The directory to save debug output in")
    temperature: Optional[float] = Field(
        None,
        description="The temperature to use for the LLM. If not set, the LLM default is used",
        le=1.0,
        ge=0.0,
    )
    use_xml: bool = Field(
        False, description="Whether to use XML format for the data input to the LLM"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.llm_output_type is None and len(self.output_types) != 1:
            raise ValueError(
                "LLMTask must either have llm_output_type or exactly one output_type"
            )

    def consume_work(self, task: Task):
        return self._invoke_llm(task)

    def get_task_class(self) -> Type[Task]:
        """Get the Task class type used for this task.

        This method provides a convenience way to specify the task class via llm_input_type
        instead of having to override consume_work(). If llm_input_type is not set, it falls
        back to the parent class implementation.

            Type[Task]: The Task class type to be used for this task. Either the value of
                        llm_input_type if set, or the parent class's task type.
        """
        return self.llm_input_type or super().get_task_class()

    def _output_type(self):
        if self.llm_output_type is not None:
            return self.llm_output_type
        # the convention is that we pick the first output type if llm_output_type is not set
        return list(self.output_types)[0]

    def _format_task(self, task: Task) -> str:
        if task is None:
            return ""

        return (
            task.model_dump_json(indent=2)
            if not self.use_xml
            else task.model_dump_xml()
        )

    def _invoke_llm(self, task: Task) -> Task:
        # allow subclasses to customize the prompt based on the input task
        task_prompt = self.format_prompt(task)

        # allow subclasses to pre-process the task and present it more clearly to the LLM
        processed_task = self.pre_process(task)

        def save_debug_with_task(
            prompt: str, kwargs: Dict[str, Any], response: Optional[Task]
        ):
            self._save_debug_output(task=task, prompt=prompt, response=response)

        # allow subclasses to do extra validation on the response
        def extra_validation_with_task(response: Task):
            return self.extra_validation(response, task)

        response = self.llm.generate_pydantic(
            prompt_template=(
                PROMPT_TEMPLATE
                if processed_task is not None
                else "{instructions}"
                + (
                    PROMPT_FORMAT_INSTRUCTIONS
                    if not self.llm.support_structured_outputs
                    else ""
                )
            ),
            output_schema=self._output_type(),
            system=self.system_prompt,
            task=self._format_task(processed_task),
            temperature=self.temperature,
            instructions=task_prompt,
            format_instructions=LLMInterface.get_format_instructions(
                self._output_type()
            ),
            debug_saver=save_debug_with_task if self.debug_mode else None,
            extra_validation=extra_validation_with_task,
        )

        self.post_process(response=response, input_task=task)

    def get_full_prompt(self, task: Task) -> str:
        task_prompt = self.format_prompt(task)

        processed_task = self.pre_process(task)

        return self.llm.generate_full_prompt(
            prompt_template=(
                PROMPT_TEMPLATE
                if processed_task is not None
                else "{instructions}"
                + (
                    PROMPT_FORMAT_INSTRUCTIONS
                    if not self.llm.support_structured_outputs
                    else ""
                )
            ),
            system=self.system_prompt,
            task=self._format_task(processed_task),
            instructions=task_prompt,
            format_instructions=LLMInterface.get_format_instructions(
                self._output_type()
            ),
        )

    def extra_validation(self, response: Task, input_task: Task) -> Optional[str]:
        """
        Validates the response from the LLM. Subclasses can override this method to do additional validation.

        Args:
            response (Task): The response from the LLM.
            input_task (Task): The input task.

        Returns:
            Optional[str]: An error message if the response is invalid, None otherwise.
        """
        return None

    def format_prompt(self, task: Task) -> str:
        """
        Formats the prompt for the LLM based on the input task. Can be customized by subclasses.

        Args:
            task (Task): The input task.

        Returns:
            str: The formatted prompt.
        """
        return self.prompt

    def pre_process(self, task: Task) -> Optional[Task]:
        """
        Pre-processes the input task before sending it to the LLM. Subclasses can override this method to do additional
        processing or filtering.

        Args:
            task (Task): The input task.

        Returns:
            Task: The pre-processed task or None if all data will be provided in the prompt.
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

    def _save_debug_output(self, task: Task, prompt: str, response: Optional[Task]):
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

        # for debugging, we want to save the input task completely including the input provenance
        task_dict = task.model_dump()
        task_dict["_input_provenance"] = [
            t.model_dump() for t in task._input_provenance
        ]
        task_dict["_input_provenance_classes"] = [
            t.__class__.__name__ for t in task._input_provenance
        ]

        output_dict = {
            "input_task": task_dict,
            "prompt_template": prompt,
            "response": response.model_dump(),
        }

        with open(output, "a", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2)


class CachedLLMTaskWorker(CachedTaskWorker, LLMTaskWorker):
    def _get_cache_key(self, task: Task) -> str:
        """Generate a unique cache key for the input task including the prompt template and model name."""
        upstream_cache_key = super()._get_cache_key(task)

        upstream_cache_key += (
            f" - {self.system_prompt} - {self.prompt} - {self.llm.model_name}"
        )
        return hashlib.sha1(upstream_cache_key.encode()).hexdigest()
