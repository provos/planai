"""
Minimal Pydantic Output Parser

This module provides a minimal implementation of a Pydantic output parser,
inspired by and adapted from the LangChain framework. It allows parsing
of text output into Pydantic models without external dependencies beyond
Pydantic itself.

The main class, MinimalPydanticOutputParser, can be used to parse text
(typically from language models) into specified Pydantic models, providing
a simple way to structure and validate outputs.

Note:
    This implementation is adapted from the LangChain framework.
    Original source: https://github.com/langchain-ai/langchain
    License: MIT (https://github.com/langchain-ai/langchain/blob/master/LICENSE)

Example:
    from pydantic import BaseModel

    class MyModel(BaseModel):
        name: str
        age: int

    parser = MinimalPydanticOutputParser(MyModel)
    result = parser.parse('{"name": "Alice", "age": 30}')
"""

import json
import re
from typing import Generic, Type, TypeVar

from pydantic import BaseModel, ValidationError


class OutputParserException(Exception):
    """Exception raised when the output parsing fails."""

    def __init__(self, message: str, llm_output: str):
        super().__init__(message)
        self.llm_output = llm_output


T = TypeVar("T", bound=BaseModel)


class MinimalPydanticOutputParser(Generic[T]):
    """A minimal implementation of PydanticOutputParser."""

    def __init__(self, pydantic_object: Type[T]):
        self.pydantic_object = pydantic_object

    def parse_json_markdown(self, text: str) -> str:
        """Remove markdown code blocks from the text."""
        # Remove markdown code blocks if present
        match = re.match(r"^```json\n(.*)\n```$", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def parse(self, text: str) -> T:
        """Parse the output of an LLM call to a pydantic object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed pydantic object.

        Raises:
            OutputParserException: If the output is not valid JSON or fails Pydantic validation.
        """
        try:
            # Strip markdown formatting if present
            cleaned_text = self.parse_json_markdown(text)
            json_object = json.loads(cleaned_text)
            return self._model_validate(json_object)
        except json.JSONDecodeError as e:
            raise OutputParserException(f"Invalid JSON: {e}", llm_output=text)
        except OutputParserException as e:
            raise e
        except Exception as e:
            raise OutputParserException(f"Failed to parse output: {e}", llm_output=text)

    def _model_validate(self, obj: dict) -> T:
        """Convert a dictionary to a Pydantic object."""
        try:
            return self.pydantic_object.model_validate(obj)
        except ValidationError as e:
            raise self._create_parser_exception(e, obj)

    def _create_parser_exception(
        self, e: Exception, obj: dict
    ) -> OutputParserException:
        """Create an OutputParserException."""
        json_string = json.dumps(obj)
        name = self.pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {json_string}. Got: {e}"
        return OutputParserException(msg, llm_output=json_string)
