import json
import os
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from planai.llm_config import llm_from_config
from planai.llm_interface import LLMInterface
from planai.llm_tool import Tool, tool


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

    # a hack to disable json mode as that makes testing easier
    client.support_json_mode = False

    return client


@pytest.mark.regression
class TestFunctionCalling:
    """Regression tests for function calling using a real LLM client."""

    def test_flight_time_query(self, llm_client):
        """Test that the LLM correctly handles flight time queries using function calling."""
        tools = [get_flight_times]
        question = "What is the flight time from JFK to LAX?"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can look up flight times. Always use the get_flight_times function when asked about flights.",
            },
            {"role": "user", "content": question},
        ]

        response = llm_client.chat(messages=messages, tools=tools)

        # Log the full response for debugging
        print(f"\nFlight query response: {response}")

        # Basic assertions about the response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Check if the response contains relevant flight information
        # Note: The actual response format might vary depending on the model
        assert any(
            term in response.lower() for term in ["jfk", "lax", "flight", "time"]
        )

    def test_ai_history_query(self, llm_client):
        """Test that the LLM correctly handles AI history queries using vector search."""
        tools = [search_data_in_vector_db]
        question = "When was Artificial Intelligence founded?"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can search a knowledge base about AI history. Use the search_data_in_vector_db function to find relevant information.",
            },
            {"role": "user", "content": question},
        ]

        response = llm_client.chat(messages=messages, tools=tools)

        # Log the full response for debugging
        print(f"\nAI history query response: {response}")

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


# Add main execution to run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
