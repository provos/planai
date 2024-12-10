import unittest
from typing import List, Optional
from pydantic import BaseModel

from planai.testing.mock_llm import MockLLM, MockLLMResponse


# Test Pydantic models
class Person(BaseModel):
    name: str
    age: int
    hobbies: Optional[List[str]] = None


class Address(BaseModel):
    street: str
    city: str
    country: str


class TestMockLLM(unittest.TestCase):
    def setUp(self):
        # Create some sample mock responses
        self.responses = [
            MockLLMResponse(
                pattern=r"Get person named Alice",
                response=Person(name="Alice", age=30, hobbies=["reading", "hiking"]),
            ),
            MockLLMResponse(
                pattern=r"Get person named Bob",
                response=Person(name="Bob", age=25, hobbies=["gaming"]),
            ),
            MockLLMResponse(
                pattern=r"Get invalid person",
                raise_exception=True,
                exception=ValueError("Invalid person request"),
            ),
            MockLLMResponse(pattern=r"Get none result", response=None),
            MockLLMResponse(
                pattern=r"Get address.*",
                response=Address(
                    street="123 Main St", city="Springfield", country="USA"
                ),
            ),
        ]
        self.mock_llm = MockLLM(responses=self.responses)

    def test_successful_response(self):
        """Test successful response matching."""
        result = self.mock_llm.generate_pydantic(
            prompt_template="Get person named Alice",
            output_schema=Person,
            system="Test system prompt",
        )

        self.assertIsInstance(result, Person)
        self.assertEqual(result.name, "Alice")
        self.assertEqual(result.age, 30)
        self.assertEqual(result.hobbies, ["reading", "hiking"])

    def test_multiple_patterns(self):
        """Test multiple different patterns work correctly."""
        result1 = self.mock_llm.generate_pydantic(
            prompt_template="Get person named Alice", output_schema=Person
        )
        result2 = self.mock_llm.generate_pydantic(
            prompt_template="Get person named Bob", output_schema=Person
        )

        self.assertEqual(result1.name, "Alice")
        self.assertEqual(result2.name, "Bob")

    def test_exception_raising(self):
        """Test that exceptions are raised correctly."""
        with self.assertRaises(ValueError) as context:
            self.mock_llm.generate_pydantic(
                prompt_template="Get invalid person", output_schema=Person
            )

        self.assertEqual(str(context.exception), "Invalid person request")

    def test_none_response(self):
        """Test that None responses are handled correctly."""
        result = self.mock_llm.generate_pydantic(
            prompt_template="Get none result", output_schema=Person
        )

        self.assertIsNone(result)

    def test_no_matching_pattern(self):
        """Test behavior when no pattern matches."""
        with self.assertRaises(ValueError) as context:
            self.mock_llm.generate_pydantic(
                prompt_template="This won't match any pattern", output_schema=Person
            )

        self.assertIn("No matching mock response found", str(context.exception))

    def test_regex_pattern_matching(self):
        """Test that regex patterns work correctly."""
        result = self.mock_llm.generate_pydantic(
            prompt_template="Get address for John Doe", output_schema=Address
        )

        self.assertIsInstance(result, Address)
        self.assertEqual(result.street, "123 Main St")
        self.assertEqual(result.city, "Springfield")

    def test_different_output_schemas(self):
        """Test that different output schemas work correctly."""
        person_result = self.mock_llm.generate_pydantic(
            prompt_template="Get person named Alice", output_schema=Person
        )
        address_result = self.mock_llm.generate_pydantic(
            prompt_template="Get address for someone", output_schema=Address
        )

        self.assertIsInstance(person_result, Person)
        self.assertIsInstance(address_result, Address)

    def test_with_system_prompt(self):
        """Test that system prompts are included in pattern matching."""
        result = self.mock_llm.generate_pydantic(
            prompt_template="Get person named Alice",
            system="This is a test system prompt",
            output_schema=Person,
        )

        self.assertIsInstance(result, Person)
        self.assertEqual(result.name, "Alice")

    def test_json_mode(self):
        """Test MockLLM with JSON mode (non-structured outputs)."""
        mock_llm = MockLLM(responses=self.responses, support_structured_outputs=False)

        result = mock_llm.generate_pydantic(
            prompt_template="Get person named Alice", output_schema=Person
        )

        self.assertIsInstance(result, Person)
        self.assertEqual(result.name, "Alice")


if __name__ == "__main__":
    unittest.main()
