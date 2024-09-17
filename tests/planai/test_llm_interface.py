import json
import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel

from planai.llm_interface import LLMInterface


# Simple In-Memory Cache to Mock diskcache.Cache for Testing
class InMemoryCache:
    def __init__(self):
        self.store = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value


class DummyPydanticModel(BaseModel):
    field1: str
    field2: int


class TestLLMInterface(unittest.TestCase):
    def setUp(self):
        # Mock the Client
        self.mock_client = Mock()

        # Initialize the LLMInterface with InMemoryCache
        self.llm_interface = LLMInterface(client=self.mock_client)
        self.llm_interface.disk_cache = InMemoryCache()

        self.prompt = "What is the capital of France?"
        self.system = "test_system"
        self.response_content = "Paris"
        self.response_data = {"message": {"content": self.response_content}}

    def test_generate_with_cache_miss(self):
        self.mock_client.generate.return_value = {"response": self.response_content}

        # Call generate
        response = self.llm_interface.generate(prompt=self.prompt, system=self.system)

        self.mock_client.generate.assert_called_once_with(
            model=self.llm_interface.model_name,
            prompt=self.prompt,
            system=self.system,
            format="",
        )
        # Since we changed to use self.response_content directly
        self.assertEqual(response, self.response_content)

    def test_generate_with_cache_hit(self):
        prompt_hash = self.llm_interface._generate_hash(
            self.llm_interface.model_name + "\n" + self.system + "\n" + self.prompt
        )
        self.llm_interface.disk_cache.set(
            prompt_hash, {"response": self.response_content}
        )

        # Call generate
        response = self.llm_interface.generate(prompt=self.prompt, system=self.system)

        # Since it's a cache hit, no chat call should happen
        self.mock_client.generate.assert_not_called()

        # Confirming expected parsing
        self.assertEqual(response, self.response_content)

    def test_generate_invalid_json_response(self):
        # Simulate invalid JSON response
        invalid_json_response = {"response": "Not a JSON {...."}
        self.mock_client.generate.return_value = invalid_json_response

        with patch("planai.llm_interface.logging.Logger") as mock_logger:
            self.llm_interface.logger = mock_logger
            response = self.llm_interface.generate(
                prompt=self.prompt, system=self.system
            )

            # Expecting the invalid content since there's no parsing
            self.assertEqual(response, "Not a JSON {....")

    def test_generate_pydantic_valid_response(self):
        output_model = DummyPydanticModel(field1="test", field2=42)
        valid_json_response = '{"field1": "test", "field2": 42}'
        self.mock_client.chat.return_value = {
            "message": {"content": valid_json_response}
        }

        response = self.llm_interface.generate_pydantic(
            prompt_template=self.prompt,
            output_schema=DummyPydanticModel,
            system=self.system,
        )

        self.assertEqual(response, output_model)

    def test_generate_pydantic_invalid_response(self):
        # Simulate an invalid JSON that cannot be parsed into DummyPydanticModel
        invalid_json_response = '{"field1": "test"}'
        self.mock_client.chat.return_value = {
            "message": {"content": invalid_json_response}
        }

        response = self.llm_interface.generate_pydantic(
            prompt_template=self.prompt,
            output_schema=DummyPydanticModel,
            system=self.system,
        )

        self.assertIsNone(response)  # Expecting None due to parsing error

    def test_cached_generate_caching_mechanism(self):
        # First call should miss cache and make client call
        self.mock_client.generate.return_value = self.response_data
        response = self.llm_interface._cached_generate(self.prompt, self.system)
        self.assertEqual(response, self.response_data)

        # Second call should hit cache, no additional client call
        response = self.llm_interface._cached_generate(self.prompt, self.system)
        self.mock_client.generate.assert_called_once()  # Still called only once
        self.assertEqual(response, self.response_data)

    def test_generate_pydantic_with_retry_logic_and_prompt_check(self):
        # Simulate an invalid JSON response that fails to parse initially
        invalid_content = '{"field1": "test"}'
        valid_content = '{"field1": "valid", "field2": 42}'

        self.mock_client.chat.side_effect = [
            {"message": {"content": invalid_content}},
            {"message": {"content": valid_content}},
        ]

        output_model = DummyPydanticModel(field1="valid", field2=42)

        with patch("planai.llm_interface.logging.Logger") as mock_logger:
            self.llm_interface.logger = mock_logger

            response = self.llm_interface.generate_pydantic(
                prompt_template=self.prompt,
                output_schema=DummyPydanticModel,
                system=self.system,
            )

            # Assert the method eventually returns the correct output after retry
            self.assertEqual(response, output_model)
            self.assertEqual(
                self.mock_client.chat.call_count, 2
            )  # Should be called twice due to retry

            # Check the messages passed to the chat function of the client
            first_call_messages = self.mock_client.chat.call_args_list[0][1]["messages"]
            second_call_messages = self.mock_client.chat.call_args_list[1][1][
                "messages"
            ]

            # Assert the second message includes instruction for correcting the format
            self.assertIn("field2", second_call_messages[3]["content"])
            self.assertNotIn(
                "field2", first_call_messages[1]["content"]
            )  # Ensure initial message was clean

    def test_get_format_instructions(self):
        instructions = LLMInterface.get_format_instructions(DummyPydanticModel)

        # Check if the instructions contain the expected elements
        self.assertIn("The output should be formatted as a JSON instance", instructions)
        self.assertIn("Here is the output schema:", instructions)

        # Check if the schema contains the fields from DummyPydanticModel
        self.assertIn('"field1":', instructions)
        self.assertIn('"type": "string"', instructions)
        self.assertIn('"field2":', instructions)
        self.assertIn('"type": "integer"', instructions)

        # Verify that the schema is valid JSON
        try:
            schema_start = instructions.index("```\n") + 4
            schema_end = instructions.rindex("\n```")
            json_schema = instructions[schema_start:schema_end]
            parsed_schema = json.loads(json_schema)

            # Check if parsed schema has the expected structure
            self.assertIn("properties", parsed_schema)
            self.assertIn("field1", parsed_schema["properties"])
            self.assertIn("field2", parsed_schema["properties"])
        except json.JSONDecodeError:
            self.fail("The schema in the instructions is not valid JSON")

    def test_generate_pydantic_with_debug_saver(self):
        output_model = DummyPydanticModel(field1="test", field2=42)
        valid_json_response = '{"field1": "test", "field2": 42}'
        self.mock_client.chat.return_value = {
            "message": {"content": valid_json_response}
        }

        # Mock debug_saver function
        mock_debug_saver = Mock()

        # Additional kwargs for the prompt template
        additional_kwargs = {"extra_param": "value"}

        response = self.llm_interface.generate_pydantic(
            prompt_template="Test prompt with {extra_param}",
            output_schema=DummyPydanticModel,
            system=self.system,
            debug_saver=mock_debug_saver,
            **additional_kwargs
        )

        self.assertEqual(response, output_model)

        # Assert that debug_saver was called with the correct arguments
        mock_debug_saver.assert_called_once()
        call_args = mock_debug_saver.call_args[0]

        # Check the prompt
        self.assertEqual(call_args[0], "Test prompt with value")

        # Check the kwargs
        self.assertEqual(call_args[1], additional_kwargs)

        # Check the response
        self.assertEqual(call_args[2], output_model)


if __name__ == "__main__":
    unittest.main()
