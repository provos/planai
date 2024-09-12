import unittest
from unittest.mock import Mock, patch

from langchain_core.pydantic_v1 import BaseModel

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
        self.response_data = {"response": "Paris"}

    def test_generate_with_cache_miss(self):
        self.mock_client.generate.return_value = self.response_data

        # Call generate
        response = self.llm_interface.generate(prompt=self.prompt, system=self.system)

        # Assert calls and response
        self.mock_client.generate.assert_called_once_with(
            model=self.llm_interface.model_name,
            prompt=self.prompt,
            system=self.system,
            format="",
        )
        self.assertEqual(response, "Paris")

    def test_generate_with_cache_hit(self):
        prompt_hash = self.llm_interface._generate_hash(
            self.llm_interface.model_name + "\n" + self.system + "\n" + self.prompt
        )
        self.llm_interface.disk_cache.set(prompt_hash, self.response_data)

        # Call generate
        response = self.llm_interface.generate(prompt=self.prompt, system=self.system)

        # Since it's a cache hit, mock_client should not be called
        self.mock_client.generate.assert_not_called()

        # Assert response
        self.assertEqual(response, "Paris")

    def test_generate_invalid_json_response(self):
        # Simulate invalid JSON response
        self.mock_client.generate.return_value = {"response": "Not a JSON {...."}

        with patch("planai.llm_interface.logging.Logger") as mock_logger:
            self.llm_interface.logger = mock_logger
            response = self.llm_interface.generate(
                prompt=self.prompt, system=self.system
            )

            # Expecting the invalid response directly as is
            self.assertEqual(response, "Not a JSON {....")

    def test_generate_pydantic_valid_response(self):
        output_model = DummyPydanticModel(field1="test", field2=42)
        valid_json_response = '{"field1": "test", "field2": 42}'
        self.mock_client.generate.return_value = {"response": valid_json_response}

        response = self.llm_interface.generate_pydantic(
            prompt_template=self.prompt,
            output_schema=DummyPydanticModel,
            system=self.system,
        )

        self.assertEqual(response, output_model)

    def test_generate_pydantic_invalid_response(self):
        # Simulate an invalid JSON that cannot be parsed into DummyPydanticModel
        invalid_json_response = '{"field1": "test"}'
        self.mock_client.generate.return_value = {"response": invalid_json_response}

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


if __name__ == "__main__":
    unittest.main()
