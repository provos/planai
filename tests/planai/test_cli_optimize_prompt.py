import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field

from planai.cli_optimize_prompt import (
    ImprovedPrompt,
    PromptPerformanceOutput,
    create_input_task,
    load_debug_log,
    optimize_prompt,
    sanitize_prompt,
)
from planai.task import Task
from planai.testing import MockLLM, MockLLMResponse


# Example test classes to simulate a real LLMTaskWorker's input/output
class FakeInputTask(Task):
    query: str = Field(description="Test query")


class FakeOutputTask(Task):
    answer: str = Field(description="Test answer")


class TestCliOptimizePrompt(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, tmp_path):
        self.temp_dir = tmp_path

    def create_tmp_data(self):
        # Create temporary files for tests
        self.python_file = Path(self.temp_dir) / "test_worker.py"
        self.debug_log = Path(self.temp_dir) / "debug.json"

        # Create a mock Python file with a test LLMTaskWorker
        self.python_file.write_text(
            """
from typing import Type
from planai import LLMTaskWorker, Task
from test_cli_optimize_prompt import FakeInputTask, FakeOutputTask

class TestWorker(LLMTaskWorker):
    output_types: Type[Task] = [FakeOutputTask]
    llm_input_type: Type[Task] = FakeInputTask
    prompt: str = "Original prompt with {query}"

    def format_prompt(self, task: FakeInputTask
) -> str:
        return self.prompt.format(query=task.query)
"""
        )

        # Create a mock debug log with proper structure
        debug_data = {
            "input_task": {
                "query": "test query",
                "_input_provenance": [],
                "_input_provenance_classes": [],
            }
        }
        # Write as single line to match real debug log format
        self.debug_log.write_text(json.dumps(debug_data))

    def test_sanitize_prompt(self):
        original = "Test prompt with {query} and {another_param}"
        new = "New prompt with {query} and some {{literal}} braces"
        result = sanitize_prompt(original, new)
        self.assertEqual(
            result, "New prompt with {query} and some {{{literal}}} braces"
        )

    def test_create_input_task(self):
        data = {
            "input_task": {
                "query": "test query",
                "_input_provenance": [],
                "_input_provenance_classes": [],
            }
        }
        module = MagicMock()
        module.__name__ = "test_module"

        # Mock get_class_from_module to return our FakeInputTask

        with patch(
            "planai.cli_optimize_prompt.get_class_from_module"
        ) as mock_get_class:
            mock_get_class.return_value = FakeInputTask

            task = create_input_task(module, "FakeInputTask", data)

            self.assertIsInstance(task, FakeInputTask)
            self.assertEqual(task.query, "test query")

    def test_load_debug_log(self):
        self.create_tmp_data()
        result = load_debug_log(str(self.debug_log))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["input_task"]["query"], "test query")

    def test_optimize_prompt(self):
        # Create temporary files for tests
        self.create_tmp_data()
        # Create mock LLMs with pre-defined responses for different steps
        mock_fast_llm = MockLLM(
            responses=[
                # For TestWorker (the real LLMTaskWorker)
                MockLLMResponse(
                    pattern=".*with test query$",  # Match any prompt ending with 'with test query'
                    response=FakeOutputTask(answer="test response"),
                ),
                MockLLMResponse(
                    pattern=".*Analyze how well.*",
                    response=PromptPerformanceOutput(
                        critique="Good improvement in clarity and structure",
                        score=0.8,
                    ),
                ),
            ]
        )

        mock_reason_llm = MockLLM(
            responses=[
                # For PromptGenerationWorker
                MockLLMResponse(
                    pattern=".*Analyze the provided prompt_template.*",
                    response=ImprovedPrompt(
                        prompt_template="Better prompt with {query}",
                        comment="Even better improvement",
                    ),
                ),
                # For PromptImprovementWorker
                MockLLMResponse(
                    pattern=".*multiple prompt_templates.*",
                    response=ImprovedPrompt(
                        prompt_template="Final improved prompt with {query}",
                        comment="Final improvements applied",
                    ),
                ),
            ]
        )

        # Create mock arguments using our real test files
        args = MagicMock()
        args.python_file = str(self.python_file)
        args.class_name = "TestWorker"
        args.debug_log = str(self.debug_log)
        args.goal_prompt = "Make the prompt better"
        args.search_path = self.temp_dir
        args.num_iterations = 1
        args.llm_opt_model = None
        args.llm_opt_provider = None
        args.config = None
        args.output_config = None
        args.output_dir = self.temp_dir

        # Run optimize_prompt with our mock LLMs
        optimize_prompt(mock_fast_llm, mock_reason_llm, args, debug=True)

        # Check that output files were created
        output_files = list(Path(self.temp_dir).glob("TestWorker_prompt_*.txt"))
        self.assertEqual(len(output_files), 2)

        # Verify content of first output file
        content = output_files[0].read_text()
        self.assertIn("Score:", content)
        self.assertIn("prompt", content.lower())


if __name__ == "__main__":
    unittest.main()
