from typing import List, Type

import pytest
from llm_interface import LLMInterface, llm_from_config

# Removed incorrect LLMConfigArgs import
from llm_interface.llm_tool import tool
from pydantic import Field

from planai import Graph, LLMTaskWorker, Task


# 1. Define a simple tool
@tool(name="get_capital_city_test_tool")
def get_capital_city_for_test(country: str) -> str:
    """Returns the capital city of a given country for testing."""
    if country.lower() == "testland":
        return "Testopolis"
    return "Unknown"


# 2. Define Pydantic Task models
class CountryQueryTaskForToolsTest(Task):
    country_name: str = Field(description="The name of the country")


class CapitalResponseTaskForToolsTest(Task):
    capital_city: str = Field(description="The capital city")


@pytest.mark.regression
class TestLLMTaskWorkerWithToolsRegression:
    def test_llm_task_worker_with_real_llm_and_tool_execution(self, request):
        # 1. Configure LLM from pytest options
        provider = request.config.getoption("--provider")
        model_name = request.config.getoption("--model")
        host = request.config.getoption("--host")
        ssh_hostname = request.config.getoption("--hostname")
        ssh_username = request.config.getoption("--username")

        # Construct arguments for llm_from_config, filtering out None values for optional params
        llm_args = {
            "provider": provider,
            "model_name": model_name,
        }
        if host:
            llm_args["host"] = host
        if ssh_hostname:
            llm_args["hostname"] = ssh_hostname
        if ssh_username:
            llm_args["username"] = ssh_username

        real_llm: LLMInterface = llm_from_config(**llm_args)

        # 2. Define LLMTaskWorker subclass
        class CapitalFinderTestWorker(LLMTaskWorker):
            llm_input_type: Type[Task] = CountryQueryTaskForToolsTest
            output_types: List[Type[Task]] = [CapitalResponseTaskForToolsTest]
            llm_output_type: Type[Task] = CapitalResponseTaskForToolsTest

            prompt: str = (
                "You are an AI assistant. You have a tool available named 'get_capital_city_test_tool' "
                "which can find the capital city of a given country. "
                "The country is specified in the input task. "
                "Please use this tool to find the capital of the provided country."
                "Respond ONLY with the Pydantic object containing the capital city. Do not add any other text."
            )
            temperature: float = 0.0

        # 3. Instantiate worker with the real LLM and the tool
        worker = CapitalFinderTestWorker(
            llm=real_llm, tools=[get_capital_city_for_test]
        )

        # 4. Set up and run a minimal PlanAI graph
        graph = Graph(name="TestToolGraphWithRealLLM")
        graph.add_workers(worker)
        graph.set_entry(worker)

        final_results_capture_list = []

        graph.set_sink(worker, CapitalResponseTaskForToolsTest)

        initial_task_payload = CountryQueryTaskForToolsTest(country_name="Testland")

        graph.run(
            initial_tasks=[(worker, initial_task_payload)], display_terminal=False
        )
        final_results_capture_list = graph.get_output_tasks()

        # 5. Assertions
        assert len(final_results_capture_list) == 1, (
            f"Should have received one result task, but got {len(final_results_capture_list)}. "
            f"Content: {final_results_capture_list}"
        )

        result_task = final_results_capture_list[0]
        assert isinstance(
            result_task, CapitalResponseTaskForToolsTest
        ), f"Output task is not of type CapitalResponseTaskForToolsTest. Got: {type(result_task)}"
        # Ensure the LLM actually used the tool and didn't hallucinate
        assert (
            result_task.capital_city == "Testopolis"
        ), f"Output capital city mismatch. Expected 'Testopolis' (from tool), got '{result_task.capital_city}'"
