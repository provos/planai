import json
from typing import Optional
from unittest.mock import create_autospec

import pytest
from pydantic import BaseModel, Field

from planai.llm_config import llm_from_config
from planai.llm_tool import tool


class FlightInfo(BaseModel):
    departure_city: str = Field(..., description="The departure city airport code")
    arrival_city: str = Field(..., description="The arrival city airport code")
    departure_time: str = Field(..., description="The departure time")
    arrival_time: str = Field(..., description="The arrival time")
    duration: str = Field(..., description="The flight duration")


class AIHistoryFact(BaseModel):
    year: int = Field(..., description="The year of the historical event")
    event: str = Field(..., description="Description of the historical event")
    significance: str = Field(
        ..., description="The significance of this event in AI history"
    )


@pytest.fixture
def tracked_tools():
    def create_tracked_tool(original_tool):
        # Create a spy on the original tool's execute method
        original_execute = original_tool.execute
        execution_tracker = create_autospec(original_execute)

        # Make sure we call the original function and get its return value
        def execute_with_tracking(**kwargs):
            return original_execute(**kwargs)

        execution_tracker.side_effect = execute_with_tracking

        # Replace the execute method but maintain all other properties
        original_tool.execute = execution_tracker
        return original_tool, execution_tracker

    return create_tracked_tool


@tool(name="get_flight_times")
def get_flight_times(departure: str, arrival: str) -> str:
    """Get the flight times between two cities.

    Args:
        departure: The departure city (airport code)
        arrival: The arrival city (airport code)
    """
    flights = {
        "JFK-LAX": {
            "departure_time": "08:00 AM",
            "arrival_time": "11:30 AM",
            "duration": "5h 30m",
        },
        "LAX-JFK": {
            "departure_time": "02:00 PM",
            "arrival_time": "10:30 PM",
            "duration": "5h 30m",
        },
    }

    key = f"{departure}-{arrival}".upper()

    return json.dumps(flights.get(key, {"error": "Flight not found"}))


@tool(name="search_data_in_vector_db")
def search_data_in_vector_db(query: str) -> str:
    """Search about Artificial Intelligence data in a vector database.

    Args:
        query: The search query
    """
    # Mock response that would come from Milvus
    mock_results = [
        {
            "text": "Artificial intelligence was founded as an academic discipline in 1956.",
            "subject": "history",
            "distance": 0.57,
        },
        {
            "text": "Alan Turing was the first person to conduct substantial research in AI.",
            "subject": "history",
            "distance": 0.41,
        },
    ]
    return json.dumps(mock_results)


# Update the fixture to use a configurable LLM
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
class TestFunctionCalling:
    """Regression tests for function calling using a real LLM client."""

    def test_flight_time_query(self, llm_client, tracked_tools):
        """Test that the LLM correctly handles flight time queries using function calling."""
        tracked_tool, execute_tracker = tracked_tools(get_flight_times)

        llm_client.support_json_mode = False
        question = "What is the flight time from JFK to LAX?"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can look up flight times. Always use the get_flight_times function when asked about flights.",
            },
            {"role": "user", "content": question},
        ]

        response = llm_client.chat(messages=messages, tools=[tracked_tool])

        # Log the full response for debugging
        print(f"\nFlight query response: {response}")

        # Verify the tool was called with correct parameters
        execute_tracker.assert_called_once()
        call_kwargs = execute_tracker.call_args[1]
        assert call_kwargs["departure"].upper() == "JFK"
        assert call_kwargs["arrival"].upper() == "LAX"

        # Basic assertions about the response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Check if the response contains relevant flight information
        # Note: The actual response format might vary depending on the model
        assert any(
            term in response.lower() for term in ["jfk", "lax", "flight", "time"]
        )

    def test_ai_history_query(self, llm_client, tracked_tools):
        """Test that the LLM correctly handles AI history queries using vector search."""
        tracked_tool, execute_tracker = tracked_tools(search_data_in_vector_db)

        llm_client.support_json_mode = False
        question = "When was Artificial Intelligence founded?"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can search a knowledge base about AI history. Use the search_data_in_vector_db function to find relevant information.",
            },
            {"role": "user", "content": question},
        ]

        response = llm_client.chat(messages=messages, tools=[tracked_tool])

        # Log the full response for debugging
        print(f"\nAI history query response: {response}")

        # Verify the tool was called with correct parameters
        execute_tracker.assert_called_once()
        call_kwargs = execute_tracker.call_args[1]
        assert "query" in call_kwargs

        # Basic assertions about the response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Check if the response contains relevant AI history information
        assert any(
            term in response.lower()
            for term in ["1956", "artificial intelligence", "founded"]
        )

    @pytest.mark.parametrize(
        "query,expected_tool",
        [
            ("What time does the flight from JFK to LAX depart?", "get_flight_times"),
            (
                "Tell me about the founding of artificial intelligence.",
                "search_data_in_vector_db",
            ),
        ],
    )
    def test_tool_selection(self, llm_client, query, expected_tool):
        """Test that the LLM correctly selects the appropriate tool based on the query."""
        llm_client.support_json_mode = False
        tools = [get_flight_times, search_data_in_vector_db]

        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that can:
                1. Look up flight times using get_flight_times
                2. Search AI history using search_data_in_vector_db
                Use the appropriate function based on the user's question and integrate the tool response into your answer.""",
            },
            {"role": "user", "content": query},
        ]

        response = llm_client.chat(messages=messages, tools=tools)

        # Log the full response for debugging
        print(f"\nTool selection query: {query}")
        print(f"Response: {response}")

        # Basic assertions about the response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # The response should contain information relevant to the tool that was used
        if expected_tool == "get_flight_times":
            assert any(term in response.lower() for term in ["flight", "jfk", "lax"])
        else:
            assert any(
                term in response.lower()
                for term in ["artificial intelligence", "founded"]
            )


@pytest.mark.regression
class TestStructuredOutput:
    """Regression tests for structured output generation using generate_pydantic."""

    def test_flight_info_structured(self, llm_client, tracked_tools):
        """Test generating structured flight information."""
        tracked_tool, execute_tracker = tracked_tools(get_flight_times)

        prompt = """Generate structured information about the flight from JFK to LAX.
        Use the get_flight_times function to get accurate flight details."""

        system = """You are a helpful assistant that provides flight information.
        Always use the get_flight_times function to look up accurate flight details.
        Structure the response according to the specified schema."""

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=FlightInfo,
            system=system,
            tools=[tracked_tool],
        )

        # Verify the tool was called with correct parameters
        execute_tracker.assert_called_once()
        call_kwargs = execute_tracker.call_args[1]
        assert call_kwargs["departure"].upper() == "JFK"
        assert call_kwargs["arrival"].upper() == "LAX"

        # Verify the structured response
        assert isinstance(response, FlightInfo)
        assert response.departure_city == "JFK"
        assert response.arrival_city == "LAX"
        assert "AM" in response.departure_time
        assert "AM" in response.arrival_time
        assert "5h" in response.duration

    def test_ai_history_structured(self, llm_client):
        """Test generating structured AI history information."""
        prompt = """Generate structured information about when AI was founded as an academic discipline.
        Use the search_data_in_vector_db function to get accurate historical information."""

        system = """You are a helpful assistant that provides information about AI history.
        Always use the search_data_in_vector_db function to look up accurate historical data.
        Structure the response according to the specified schema."""

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=AIHistoryFact,
            system=system,
            tools=[search_data_in_vector_db],
        )

        # Verify the structured response
        assert isinstance(response, AIHistoryFact)
        assert response.year == 1956
        assert "academic discipline" in response.event.lower()
        assert response.significance != ""

    @pytest.mark.parametrize("temp", [0.0, 0.7])
    def test_temperature_effect(self, llm_client, temp):
        """Test generating structured output with different temperature settings."""
        prompt = """Generate structured information about the flight from LAX to JFK.
        Use the get_flight_times function to get accurate flight details."""

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=FlightInfo,
            system="You are a flight information assistant.",
            tools=[get_flight_times],
            temperature=temp,
        )

        # Core facts should remain consistent regardless of temperature
        assert isinstance(response, FlightInfo)
        assert response.departure_city == "LAX"
        assert response.arrival_city == "JFK"
        assert "PM" in response.departure_time
        assert "PM" in response.arrival_time

    def test_extra_validation(self, llm_client):
        """Test generating structured output with additional validation."""

        def validate_duration(model: FlightInfo) -> Optional[str]:
            if not any(unit in model.duration for unit in ["h", "hr", "hour"]):
                return "Duration must include hours"
            return None

        prompt = """Generate structured information about the flight from JFK to LAX.
        Use the get_flight_times function to get accurate flight details."""

        response = llm_client.generate_pydantic(
            prompt_template=prompt,
            output_schema=FlightInfo,
            system="You are a flight information assistant.",
            tools=[get_flight_times],
            extra_validation=validate_duration,
        )

        assert isinstance(response, FlightInfo)
        assert any(unit in response.duration for unit in ["h", "hr", "hour"])


# Add main execution to run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
