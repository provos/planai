# test_llm_task.py

import unittest
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


if __name__ == "__main__":
    unittest.main()
