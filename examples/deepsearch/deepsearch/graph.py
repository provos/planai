from textwrap import dedent
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type

from pydantic import Field

from planai import (
    CachedLLMTaskWorker,
    Graph,
    InitialTaskWorker,
    JoinedTaskWorker,
    Task,
    TaskWorker,
    llm_from_config,
)
from planai.patterns import ConsolidatedPages, SearchQuery, create_search_fetch_worker


class Request(Task):
    user_input: str = Field(..., description="User input for the LLM")


class Plan(Task):
    response: str = Field(..., description="A detailed plan for the task")


class SearchQueryWithPhase(Task):
    query: str = Field(..., description="A search query")
    phase: str = Field(
        ...,
        description="The phase of the plan including a description about the purpose of the query",
    )


class SearchQueries(Task):
    queries: List[SearchQueryWithPhase] = Field(
        ..., description="A list of search queries"
    )


class Response(Task):
    response_type: Literal["final", "thinking"] = Field(
        ..., description="The type of response"
    )
    message: str = Field(..., description="The response to the user")
    phase: Optional[str] = Field(None, description="The phase of the plan")


class PhaseAnalysisInterim(Task):
    summary: str = Field(..., description="Summary of the analysis")


class PhaseAnalysis(Task):
    phase: str = Field(..., description="The phase of the plan")
    summary: str = Field(..., description="Summary of the analysis")


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
    output_types: List[Type[Task]] = [SearchQueries]
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

    def pre_process(self, task: Plan):
        self.notify_status(task, "Creating search queries for each phase")
        return task


class SearchSplitter(TaskWorker):
    output_types: List[Type[Task]] = [SearchQuery]

    def consume_work(self, task: SearchQueries):
        # we need to adopt the queries to the search-fetch pattern
        for query in task.queries:
            self.publish_work(
                SearchQuery(query=query.query, metadata=query.phase), input_task=task
            )


class SearchSummarizer(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [PhaseAnalysis]
    llm_input_type: Type[Task] = ConsolidatedPages
    llm_output_type: Type[Task] = PhaseAnalysisInterim
    use_xml: bool = True
    system_prompt: str = dedent(
        """You are an expert internet data analyst and comply with all requests by the user."""
    ).strip()
    prompt: str = dedent(
        """
        Summarize the content of the search results as they pertain to the overall plan and phase:

        <overall plan>
        {plan}
        </overall plan>

        <current phase>
        {phase}
        </current phase>

        Your response should contain a detailed summary of the page contents in order to fill in the
        necessary information for this phase of the plan.
        """
    )

    def format_prompt(self, input_task: ConsolidatedPages) -> str:
        plan: Plan = input_task.find_input_task(Plan)
        if plan is None:
            raise ValueError("The input task is missing a Plan task")
        query: SearchQuery = input_task.find_input_task(SearchQuery)
        if query is None:
            raise ValueError("The input task is missing a SearchQuery task")
        return self.prompt.format(plan=plan.response, phase=query.metadata)

    def post_process(
        self, response: PhaseAnalysisInterim, input_task: ConsolidatedPages
    ):
        query: SearchQuery = input_task.find_input_task(SearchQuery)
        if query is None:
            raise ValueError("The input task is missing a SearchQuery task")
        return self.publish_work(
            PhaseAnalysis(phase=query.metadata, summary=response.summary), input_task
        )


class AnalysisJoiner(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    output_types: List[Type[Task]] = [Response]

    def consume_work_joined(self, tasks: List[PhaseAnalysis]):
        summary = "\n".join([f"**{task.phase}**: {task.summary}" for task in tasks])
        self.publish_work(Response(response_type="final", message=summary), tasks[0])


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
    split_worker = SearchSplitter()
    search_fetch_worker = create_search_fetch_worker(llm=llm)
    analysis_worker = SearchSummarizer(llm=llm)
    analysis_joiner = AnalysisJoiner()
    response_publisher = ResponsePublisher()
    graph.add_workers(
        plan_worker,
        search_worker,
        split_worker,
        response_publisher,
        search_fetch_worker,
        analysis_worker,
        analysis_joiner,
    )
    graph.set_dependency(plan_worker, response_publisher)
    graph.set_dependency(plan_worker, search_worker).next(split_worker).next(
        search_fetch_worker
    ).next(analysis_worker).next(analysis_joiner).next(response_publisher)
    graph.set_entry(plan_worker)
    graph.set_sink(response_publisher, Response, notify=notify)
    return graph, plan_worker
