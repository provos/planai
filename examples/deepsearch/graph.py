from textwrap import dedent
from typing import Any, Callable, Dict, List, Tuple, Type

from pydantic import Field

from planai import CachedLLMTaskWorker, Graph, Task, TaskWorker, llm_from_config


class Request(Task):
    user_input: str = Field(..., description="User input for the LLM")


class Plan(Task):
    response: str = Field(..., description="A detailed plan for the task")


class PlanWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [Plan]
    llm_input_type: Type[Task] = Request
    use_xml: bool = True
    prompt: str = dedent(
        """
        Please provide a detailed plan for the task.

        Your response should be provided as markdown in the response field.
        """
    )


def setup_graph(
    provider: str = "ollama",
    model: str = "llama3.3:latest",
    notify: Callable[Dict[str, Any], None] = None,
) -> Tuple[Graph, TaskWorker]:
    llm = llm_from_config(provider=provider, model_name=model)

    graph = Graph(name="Plan Graph")
    plan_worker = PlanWorker(llm=llm)
    graph.add_worker(plan_worker)
    graph.set_entry(plan_worker)
    graph.set_sink(plan_worker, Plan, notify=notify)
    return graph, plan_worker
