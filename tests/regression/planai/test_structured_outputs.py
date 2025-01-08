from textwrap import dedent
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from planai.llm_config import llm_from_config


class Person(BaseModel):
    name: str = Field(..., description="The person's full name")
    age: int = Field(..., description="The person's age in years")
    occupation: str = Field(..., description="The person's current job or role")
    hobbies: List[str] = Field(..., description="List of the person's hobbies")
    address: Optional[str] = Field(
        None, description="The person's address (if available)"
    )


class Summary(BaseModel):
    main_points: List[str] = Field(..., description="List of main points from the text")
    sentiment: str = Field(
        ..., description="Overall sentiment (positive/negative/neutral)"
    )
    word_count: int = Field(..., description="Number of words in the text")
    key_topics: List[str] = Field(..., description="Key topics discussed in the text")


@pytest.fixture(scope="module")
def llm_client(request):
    provider = request.config.getoption("--provider", default="ollama")
    model_name = request.config.getoption("--model", default="llama3.2")
    host = request.config.getoption("--host", default=None)
    hostname = request.config.getoption("--hostname", default=None)
    username = request.config.getoption("--username", default=None)

    client = llm_from_config(
        provider=provider,
        model_name=model_name,
        host=host,
        hostname=hostname,
        username=username,
        log_dir="logs",
        use_cache=False,
    )

    return client


@pytest.mark.regression
class TestStructuredOutputs:
    """End-to-end regression tests for structured output generation."""

    def test_basic_structured_output(self, llm_client):
        """Test generating a simple structured output about a person."""
        prompt = """Generate a structured profile for John Smith. He is a 35-year-old
        software engineer who loves hiking, photography, and reading. He lives in
        San Francisco."""

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=Person,
            system="You are a helpful assistant that creates structured profiles.",
        )

        # Verify the structured response
        assert isinstance(response, Person)
        assert response.name == "John Smith"
        assert response.age == 35
        assert "software engineer" in response.occupation.lower()
        assert len(response.hobbies) >= 3
        assert all(
            hobby.lower() in ["hiking", "photography", "reading"]
            for hobby in response.hobbies
        )
        assert "san francisco" in response.address.lower()

    def test_text_analysis(self, llm_client):
        """Test generating structured analysis of a text passage."""
        text = """The new renewable energy project has exceeded all expectations.
        Initial estimates suggested a 15% reduction in carbon emissions, but actual
        measurements show a 25% decrease. This success has led to increased funding
        for similar projects across the region. However, some concerns remain about
        the long-term maintenance costs."""

        prompt = dedent(
            f"""
            Analyze the following text and provide a structured summary:
            {text}

            The summary should include a sentiment that is either positive, negative, or neutral.
            """
        ).strip()

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=Summary,
            system=(
                "You are a text analysis assistant. "
                "Extract key information and provide structured summaries."
            ),
        )

        # Verify the structured analysis
        assert isinstance(response, Summary)
        assert len(response.main_points) >= 2
        assert response.sentiment.lower() in ["positive", "negative", "neutral"]
        assert response.word_count > 0
        topics = [topic.lower() for topic in response.key_topics]
        assert any("renewable energy" in topic for topic in topics)
        assert len(response.key_topics) >= 2

    @pytest.mark.parametrize("temperature", [0.0, 0.7])
    def test_temperature_impact(self, llm_client, temperature):
        """Test how temperature affects structured output generation."""
        prompt = """Generate a structured profile for a fictional person who works
        in the technology industry."""

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=Person,
            system="You are a creative assistant that generates fictional profiles.",
            temperature=temperature,
        )

        # Core structure should be maintained regardless of temperature
        assert isinstance(response, Person)
        assert response.name != ""
        assert response.age > 0
        assert response.occupation != ""
        assert len(response.hobbies) > 0

    def test_structured_output_validation(self, llm_client):
        """Test structured output with custom validation rules."""

        def validate_age(person: Person) -> Optional[str]:
            if person.age < 0 or person.age > 120:
                return "Age must be between 0 and 120"
            return None

        prompt = """Generate a structured profile for Sarah Johnson, who is
        28 years old and works as a data scientist."""

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=Person,
            system="You are a profile generation assistant.",
            extra_validation=validate_age,
        )

        assert isinstance(response, Person)
        assert 0 <= response.age <= 120
        assert "data scientist" in response.occupation.lower()

    def test_structured_output_with_minimal_input(self, llm_client):
        """Test generating structured output from minimal input."""
        prompt = "Generate a profile for someone named Alex."

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=Person,
            system="You are a creative assistant. Generate reasonable defaults for missing information.",
        )

        assert isinstance(response, Person)
        assert "Alex" in response.name
        assert response.age > 0
        assert response.occupation != ""
        assert len(response.hobbies) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
