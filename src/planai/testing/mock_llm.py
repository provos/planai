import re
from typing import Dict, List, Optional, Pattern, Type, Union

from pydantic import BaseModel

from ..llm_interface import LLMInterface, NoCache


class MockLLMResponse:
    """Configuration for a mock response."""

    def __init__(
        self,
        pattern: Union[str, Pattern],
        response: Optional[BaseModel] = None,
        raise_exception: bool = False,
        exception: Optional[Exception] = None,
    ):
        """
        Args:
            pattern: Regex pattern to match against the full prompt
            response: Pydantic object to return when pattern matches
            raise_exception: If True, raise an exception instead of returning response
            exception: Specific exception to raise (defaults to ValueError if not specified)
        """
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.response = response
        self.raise_exception = raise_exception
        self.exception = exception or ValueError("Mock LLM error")


class MockLLM(LLMInterface):
    """A mock LLM implementation for testing purposes."""

    def __init__(
        self,
        responses: List[MockLLMResponse],
        support_structured_outputs: bool = True,
    ):
        """
        Args:
            responses: List of MockLLMResponse objects defining pattern-response pairs
            support_structured_outputs: Whether to return structured outputs directly
        """
        super().__init__(
            model_name="mock",
            client=None,
            support_structured_outputs=support_structured_outputs,
            use_cache=False,
        )
        self.responses = responses
        self.disk_cache = NoCache()

    def _cached_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        # Reconstruct the full prompt from messages
        full_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                full_prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                full_prompt += f"User: {msg['content']}\n"

        # Try to match the prompt against our patterns
        for mock_response in self.responses:
            if mock_response.pattern.search(full_prompt):
                if mock_response.raise_exception:
                    raise mock_response.exception

                if mock_response.response is None:
                    return None

                if self.support_structured_outputs:
                    return mock_response.response.model_copy()
                else:
                    # Convert Pydantic object to JSON string
                    return mock_response.response.model_dump_json()

        raise ValueError(f"No matching mock response found for prompt: {full_prompt}")
