# test_llm_task.py

import unittest
from typing import List, Optional, Type
from unittest.mock import MagicMock, Mock, patch

from planai.llm_interface import LLMInterface
from planai.llm_task import LLMTaskWorker
from planai.task import Task


class DummyTask(Task):
    content: str


class DummyOutputTask(Task):
    result: str


class TestLLMTaskWorker(unittest.TestCase):
    def setUp(self):
        self.llm = LLMInterface()
        self.mock_client = Mock()
        self.llm.client = self.mock_client
        self.worker = LLMTaskWorker(
            llm=self.llm, prompt="Test prompt", output_types=[DummyOutputTask]
        )

    def test_initialization(self):
        self.assertEqual(self.worker.prompt, "Test prompt")
        self.assertEqual(self.worker.output_types, [DummyOutputTask])
        self.assertFalse(self.worker.debug_mode)
        self.assertEqual(self.worker.debug_dir, "debug")

    @patch("planai.llm_task.Path")
    @patch("builtins.open")
    @patch("planai.llm_task.json.dump")
    def test_debug_mode(self, mock_json_dump, mock_open, mock_path):
        # Enable debug mode
        self.worker.debug_mode = True

        mock_open.return_value = MagicMock()

        # Prepare test data
        input_task = DummyTask(content="Test input")
        output_task = DummyOutputTask(result="Test output")
        valid_json_response = output_task.model_dump_json()
        self.llm._cached_chat = Mock(return_value=valid_json_response)

        # Call the method that triggers debug output
        with patch("planai.llm_task.LLMTaskWorker.publish_work") as mock_publish_work:
            with self.worker.work_buffer_context(input_task):
                self.worker._invoke_llm(input_task)

        mock_publish_work.assert_called_once()

        # Assert that the debug output was written
        mock_json_dump.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        self.assertIn("input_task", args[0])
        self.assertIn("prompt_template", args[0])
        self.assertIn("response", args[0])

    def test_format_prompt(self):
        task = DummyTask(content="Test content")
        formatted_prompt = self.worker.format_prompt(task)
        self.assertEqual(formatted_prompt, self.worker.prompt)

    def test_pre_process(self):
        task = DummyTask(content="Test content")
        processed_task = self.worker.pre_process(task)
        self.assertEqual(processed_task, task)

    @patch("planai.llm_task.LLMTaskWorker.publish_work")
    def test_post_process(self, mock_publish_work):
        input_task = DummyTask(content="Test input")
        output_task = DummyOutputTask(result="Test output")

        self.worker.post_process(output_task, input_task)

        mock_publish_work.assert_called_once_with(
            task=output_task, input_task=input_task
        )

    @patch("planai.llm_task.LLMTaskWorker.publish_work")
    def test_post_process_with_none_response(self, mock_publish_work):
        input_task = DummyTask(content="Test input")

        with self.assertLogs(level="ERROR") as log:
            self.worker.post_process(None, input_task)

        self.assertIn("LLM did not return a valid response", log.output[0])
        mock_publish_work.assert_not_called()

    def test_invoke_llm(self):
        input_task = DummyTask(content="Test input")
        output_task = DummyOutputTask(result="Test output")
        self.llm.generate_pydantic = Mock(return_value=output_task)

        with patch("planai.llm_task.LLMTaskWorker.publish_work") as mock_publish_work:
            self.worker._invoke_llm(input_task)

        self.llm.generate_pydantic.assert_called_once()
        mock_publish_work.assert_called_once_with(
            task=output_task, input_task=input_task
        )

    def test_extra_validation(self):
        task = DummyOutputTask(result="Test output")
        input_task = DummyTask(content="Test input")
        result = self.worker.extra_validation(task, input_task)
        self.assertIsNone(result)

    def test_get_full_prompt(self):
        task = DummyTask(content="Test content")
        full_prompt = self.worker.get_full_prompt(task)
        self.assertIn(self.worker.prompt, full_prompt)


class CustomLLMTaskWorker(LLMTaskWorker):
    def pre_process(self, task: Task) -> Optional[Task]:
        return None


class TestLLMTaskWorkerPromptTemplate(unittest.TestCase):
    def setUp(self):
        self.llm = LLMInterface()
        self.mock_client = Mock()
        self.llm.client = self.mock_client
        self.worker = CustomLLMTaskWorker(
            llm=self.llm, prompt="Test instruction", output_types=[DummyOutputTask]
        )

    def test_prompt_template_omission(self):
        # Create input task
        input_task = DummyTask(content="Test content")

        # Mock LLM's generate_full_prompt method
        self.llm.generate_full_prompt = Mock(return_value="test prompt")

        # Get the full prompt
        prompt = self.worker.get_full_prompt(input_task)

        # Verify that generate_full_prompt was called with the correct template
        self.llm.generate_full_prompt.assert_called_once()
        template_arg = self.llm.generate_full_prompt.call_args[1]["prompt_template"]

        # Check that PROMPT_TEMPLATE is not used (no {task} parameter)
        self.assertNotIn("{task}", template_arg)
        self.assertIn("{instructions}", template_arg)
        self.assertIn("{format_instructions}", template_arg)


class InputTask(Task):
    data: str


class OutputTask(Task):
    result: str


class DeclarativeInputWorker(LLMTaskWorker):
    llm_input_type: Type[Task] = InputTask
    output_types: List[Type[Task]] = [OutputTask]


class ExplicitInputWorker(LLMTaskWorker):
    output_types: List[Type[Task]] = [OutputTask]

    def consume_work(self, task: InputTask):
        return super().consume_work(task)


class TestLLMTaskWorkerInputType(unittest.TestCase):
    def setUp(self):
        self.llm = LLMInterface()
        self.mock_client = Mock()
        self.llm.client = self.mock_client

        self.declarative_worker = DeclarativeInputWorker(
            llm=self.llm, prompt="Test prompt"
        )
        self.explicit_worker = ExplicitInputWorker(llm=self.llm, prompt="Test prompt")

    def test_input_type_specification(self):
        # Both workers should accept InputTask
        self.assertEqual(self.declarative_worker.get_task_class(), InputTask)
        self.assertEqual(self.explicit_worker.get_task_class(), InputTask)

        # Test that both workers can process InputTask
        input_task = InputTask(data="test data")
        output_task = OutputTask(result="test result")
        self.llm.generate_pydantic = Mock(return_value=output_task)

        # Test declarative worker
        with patch("planai.llm_task.LLMTaskWorker.publish_work") as mock_publish:
            self.declarative_worker._invoke_llm(input_task)
            mock_publish.assert_called_once_with(
                task=output_task, input_task=input_task
            )

        # Test explicit worker
        with patch("planai.llm_task.LLMTaskWorker.publish_work") as mock_publish:
            self.explicit_worker._invoke_llm(input_task)
            mock_publish.assert_called_once_with(
                task=output_task, input_task=input_task
            )


if __name__ == "__main__":
    unittest.main()
