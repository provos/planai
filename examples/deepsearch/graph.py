from textwrap import dedent
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

from pydantic import Field

from planai import CachedLLMTaskWorker, Graph, Task, TaskWorker, llm_from_config


class Request(Task):
    user_input: str = Field(..., description="User input for the LLM")


class Plan(Task):
    response: str = Field(..., description="A detailed plan for the task")


class SearchQuery(Task):
    query: str = Field(..., description="A search query")
    phase: str = Field(
        ..., description="The phase of the plan including description this query is for"
    )


class SearchQueries(Task):
    queries: List[SearchQuery] = Field(..., description="A list of search queries")


class Response(Task):
    response_type: Literal["final", "thinking"] = Field(
        ..., description="The type of response"
    )
    message: str = Field(..., description="The response to the user")


class PlanWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [Plan, Response]
    llm_output_type: Type[Task] = Plan
    llm_input_type: Type[Task] = Request
    use_xml: bool = True
    system_prompt: str = dedent(
        """You are an expert internet researcher and comply with all requests by the user."""
    ).strip()
    prompt: str = dedent(
        """
        Please provide a plan for the task assuming that it will require web searches
        on different topics. You are preparing the plan to be processed by an LLM further down.

        A good plan should be one paragraph of text and involve multiple phases of research.
        Where each phase might result in a different search query and retrieval of information.
        Provide a single line of text for each phase.

        Your response should be provided as markdown in the response field.
        """
    )

    def post_process(self, response: Plan, input_task: Request):
        self.publish_work(task=response, input_task=input_task)
        self.publish_work(
            Response(response_type="thinking", message=response.response), input_task
        )


class SearchCreator(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [Response]
    llm_output_type: Type[Task] = SearchQueries
    llm_input_type: Type[Task] = Plan
    use_xml: bool = True
    system_prompt: str = dedent(
        """You are an expert internet researcher and comply with all requests by the user."""
    ).strip()
    prompt: str = dedent(
        """
        Please provide a search query for each phase of the plan with the ulitmate goal of
        helping the user with their request:

        <user request>
        {request}
        </user request>

        Your response should be provided as a list of search queries and phase descriptions. The
        phase description should contain the question that we ultimately want to answer with the
        search results.
        """
    )

    def format_prompt(self, input_task: Plan) -> str:
        request: Request = input_task.find_input_task(Request)
        if request is None:
            raise ValueError("The input task is missing a Request task")
        return self.prompt.format(request=request.user_input)

    def post_process(self, response: SearchQueries, input_task: Plan):
        self.publish_work(
            Response(response_type="final", message=str(response.queries)), input_task
        )


class ResponsePublisher(TaskWorker):
    """Re-iterates the response to the user, so that we can use a sink to notify the user on thinking updates"""

    output_types: List[Type[Task]] = [Response]

    def consume_work(self, task: Response):
        self.publish_work(task, input_task=task)


def setup_graph(
    provider: str = "ollama",
    model: str = "llama3.3:latest",
    notify: Callable[Dict[str, Any], None] = None,
) -> Tuple[Graph, TaskWorker]:
    llm = llm_from_config(provider=provider, model_name=model, use_cache=False)

    graph = Graph(name="Plan Graph")
    plan_worker = PlanWorker(llm=llm)
    search_worker = SearchCreator(llm=llm)
    response_publisher = ResponsePublisher()
    graph.add_workers(plan_worker, search_worker, response_publisher)
    graph.set_dependency(plan_worker, search_worker).next(response_publisher)
    graph.set_dependency(plan_worker, response_publisher)
    graph.set_entry(plan_worker)
    graph.set_sink(response_publisher, Response, notify=notify)
    return graph, plan_worker
