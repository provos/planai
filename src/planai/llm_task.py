import logging
from textwrap import dedent
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field

from llm_interface import LLMInterface
from task import TaskWorker, TaskWorkItem


class LLMTask(TaskWorker):
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
        {{task}}
        
        Here are your instructions:
        {{instructions}}
        
        {{format_instructions}}
        """
        ).strip()

        parser = PydanticOutputParser(pydantic_object=self._output_type())

        response = self.llm.generate_pydantic(
            prompt_template=prompt,
            output_schema=self._output_type(),
            system="You are a helpful AI assistant. Please help the user with the following task and produce output in JSON.",
            task=task,
            instructions=prompt,
            format_instructions=parser.format_instructions(),
        )

        self.post_process(response=response, input_task=task)

    def post_process(self, response: Optional[TaskWorkItem], input_task: TaskWorkItem):
        if response is not None:
            self.publish_work(response)
        else:
            logging.error(
                "LLM did not return a valid response for task %s with provenance %s",
                input_task.__class__.__name__,
                input_task._provenance,
            )
