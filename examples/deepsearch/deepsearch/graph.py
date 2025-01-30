# Copyright (c) 2024 Niels Provos
#
# This example is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/
# or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# This example is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License for more details.

from textwrap import dedent
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type

from pydantic import Field

from planai import (
    CachedLLMTaskWorker,
    ChatMessage,
    ChatTaskWorker,
    Graph,
    InitialTaskWorker,
    JoinedTaskWorker,
    LLMTaskWorker,
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
    response_type: Literal["final", "thinking", "error"] = Field(
        ..., description="The type of response"
    )
    message: str = Field(..., description="The response to the user")
    phase: Optional[str] = Field(None, description="The phase of the plan")


class PhaseAnalysisInterim(Task):
    extraction: str = Field(
        ..., description="The extracted information for this phase of the plan"
    )


class PhaseAnalysis(Task):
    phase: str = Field(..., description="The phase of the plan")
    extraction: str = Field(
        ..., description="The extracted information for this phase of the plan"
    )


class PhaseAnalyses(Task):
    analyses: List[PhaseAnalysis] = Field(
        ..., description="A list of extracted information for each phase of the plan"
    )


class FinalWriteup(Task):
    writeup: str = Field(..., description="The final writeup in markdown format")


class PlanWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [Plan, Response]
    llm_output_type: Type[Task] = Plan
    llm_input_type: Type[Task] = Request
    use_xml: bool = True
    system_prompt: str = dedent(
        """You are an expert research agent and will create detailed multi-step plans to answer user's queries.
        You have access to a web search tool to gather information.
        You should think of different ways to gather information from the web to answer the user query.
        Your plan should be ambitious and utilize the web search tool as much as possible to create a comprehensive
        and well-informed response to the user's query."""
    ).strip()
    prompt: str = dedent(
        """
        Please provide a detailed step-by-step plan for researching this query. The plan should involve multiple phases of research,
        where each phase utilizes web searches to gather different types of information.

        Each phase should have a clear objective and contribute to a comprehensive understanding of the topic.

        Structure your plan in the following format:

        Phase 1: [Objective of the phase]
        Phase 2: [Objective of the phase]
        ...

        Provide a concise description for each phase, outlining what information you aim to gather and how it fits into the overall research strategy.

        Your plan should be presented as markdown in the response field.
        """
    )

    def pre_consume_work(self, task):
        self.notify_status(task, "Creating a research plan")

    def post_process(self, response: Plan, input_task: Request):
        self.publish_work(task=response, input_task=input_task)
        self.publish_work(
            Response(response_type="thinking", phase="plan", message=response.response),
            input_task,
        )


class SearchCreator(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [SearchQueries]
    llm_input_type: Type[Task] = Plan
    use_xml: bool = True
    system_prompt: str = dedent(
        """You are an expert search query generator. Given a research plan, you will generate targeted search queries
        for each phase of the plan. Your goal is to create queries that will retrieve the most relevant information
        from the web to fulfill the objectives of each research phase."""
    ).strip()
    prompt: str = dedent(
        """
        Given the provided research plan and the original user request:

        <user request>
        {request}
        </user request>

        Please generate a specific search query for each phase of the plan. Each query should be designed to
        gather information relevant to the objective of that phase.

        For each phase, provide:
        1. The phase description, including the specific question or objective that the search query aims to address.
        2. The search query itself, formulated to retrieve the most relevant and useful information for that phase.

        Structure your response as follows:

        Phase: [Phase description]
        Query: [Generated search query]

        Repeat this structure for each phase of the plan.
        """
    )

    def format_prompt(self, input_task: Plan) -> str:
        request: Request = input_task.find_input_task(Request)
        if request is None:
            raise ValueError("The input task is missing a Request task")
        return self.prompt.format(request=request.user_input)

    def pre_consume_work(self, task: Plan):
        self.notify_status(task, "Creating search queries for each phase")


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
        """You are a master research scientist, adept at synthesizing complex information from multiple sources into clear, concise, and engaging extractions.
        You will be provided with research results related to a specific phase of a larger research plan.
        Your task is to distill the core findings and insights from these materials, focusing on answering the research question posed in that phase.
        Craft a narrative that seamlessly integrates the information, prioritizing a natural flow of knowledge over explicitly referencing individual sources.
        Assume the reader is intelligent but may not be familiar with the specific details of the research. Explain complex concepts clearly and avoid jargon where possible.
        The goal is to produce a comprehensive and insightful extraction that stands alone as a valuable piece of knowledge, directly contributing to a broader understanding of the overarching research topic."""
    ).strip()
    prompt: str = dedent(
        """
        These documents were retrieved based on the following phase of a research plan:

        <current phase>
        {phase}
        </current phase>

        And within the context of the overall research plan:

        <overall plan>
        {plan}
        </overall plan>

        Your task is to synthesize the information from the provided research results (web pages) into a comprehensive extraction that directly addresses the objective of this research phase.

        Focus on distilling the key findings, insights, and concepts from the research.

        **Do not refer to the documents themselves in your extraction (e.g., avoid phrases like "This document discusses..." or "The provided sources indicate...").**

        Instead, integrate the information into a cohesive narrative that stands on its own, as if you are explaining the findings to a colleague.

        Explain the findings in a way that is both accurate and accessible to someone who may not be deeply familiar with the topic.

        Your extraction should be a well-structured, informative, and engaging piece of text that contributes directly to a broader understanding of the overall research topic as outlined in the plan.

        Provide your extraction in markdown format.
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
            PhaseAnalysis(phase=query.metadata, extraction=response.extraction),
            input_task,
        )


class AnalysisJoiner(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    output_types: List[Type[Task]] = [PhaseAnalyses]

    def consume_work_joined(self, tasks: List[PhaseAnalysis]):
        self.publish_work(PhaseAnalyses(analyses=tasks), tasks[0])


class FinalNarrativeWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [Response]
    llm_input_type: Type[Task] = PhaseAnalyses
    llm_output_type: Type[Task] = FinalWriteup
    use_xml: bool = True
    system_prompt: str = dedent(
        """You are a master science communicator, adept at explaining complex scientific concepts to a curious and intelligent audience.
        You will be provided with a user's original query and a series of detailed research extractions related to that query.
        Your task is to synthesize these extractions into a single, comprehensive, and engaging narrative that directly and thoroughly answers the user's query.
        Maintain scientific accuracy while presenting the information in a narrative style, as if guiding the reader through a fascinating scientific journey.
        Use Markdown formatting to enhance the readability and presentation of your narrative."""
    ).strip()
    prompt: str = dedent(
        """
        Original User Query:
        <user_query>
        {user_query}
        </user_query>

        Research Plan:
        <plan>
        {plan}
        </plan>

        Your task is to synthesize the provided research extractions into a single, cohesive, and detailed narrative that directly addresses the user's original query.

        **Depth and Detail:** Go beyond a basic overview. Provide a more in-depth explanation of the core concepts, findings, and their implications. Incorporate relevant scientific terminology and concepts where appropriate, defining them clearly for a non-expert audience. Use specific examples where appropriate.

        **Narrative Structure:** Structure your response as a continuous narrative, not as a collection of separate summaries. Build upon each extraction to create a comprehensive and engaging story of discovery and understanding. Connect the concepts in a logical flow, building upon previous ideas to create a cohesive understanding.

        **Markdown for Clarity:** Use Markdown formatting to enhance clarity and presentation:

        *   Use **bold** and *italics* to emphasize key terms and concepts.
        *   Employ bullet points or numbered lists to break down complex information into digestible parts.
        *   If relevant, use `code blocks` to present equations or formulas in a clear format, but keep it simple and explain them thoroughly.
        *   You can use links, but only if you think they will greatly enhance the answer. Keep it minimal.

        **Integrate, Don't Repeat:** Do not simply restate the extractions. Instead, distill their essence and weave them into a unified narrative that provides a deeper understanding of the topic.

        **Audience Awareness:** Assume your audience is intelligent and eager to learn but may not have a strong background in the specific scientific field. Explain complex ideas clearly and accessibly, defining jargon and providing context where needed.

        **Address the Query Completely:** Ensure that your narrative directly and completely answers the user's original query, leaving no major aspects unaddressed.

        **Maintain Accuracy:** While maintaining a narrative style, prioritize scientific accuracy. Ensure your explanations are consistent with the provided research.

        The final narrative should be a well-written, informative, accurate, and engaging piece of scientific writing that stands alone as a satisfying answer to the user's original question.
        """
    ).strip()

    def pre_consume_work(self, task):
        self.notify_status(task, "Creating the final writeup")

    def format_prompt(self, input_task: ConsolidatedPages) -> str:
        plan: Plan = input_task.find_input_task(Plan)
        if plan is None:
            raise ValueError("The input task is missing a Plan task")
        request: Request = input_task.find_input_task(Request)
        if request is None:
            raise ValueError("The input task is missing a Request task")
        return self.prompt.format(plan=plan.response, user_query=request.user_input)

    def post_process(self, response: FinalWriteup, input_task: Response):
        return self.publish_work(
            Response(response_type="final", message=response.writeup), input_task
        )


class UserChat(ChatTaskWorker):
    pass


class ChatAdapter(TaskWorker):
    output_types: List[Type[Task]] = [Response]

    def consume_work(self, task: ChatMessage):
        self.publish_work(Response(response_type="final", message=task.content), task)


class ResponsePublisher(TaskWorker):
    """Re-iterates the response to the user, so that we can use a sink to notify the user on thinking updates"""

    output_types: List[Type[Task]] = [Response]

    def consume_work(self, task: Response):
        self.publish_work(task, input_task=task)


def setup_graph(
    provider: Literal["ollama", "remote_ollama", "openai"] = "ollama",
    model: str = "llama3.3:latest",
    host: str = "localhost:11434",
    notify: Optional[Callable[Dict[str, Any], None]] = None,
) -> Tuple[Graph, TaskWorker, TaskWorker]:
    llm = llm_from_config(
        provider=provider,
        model_name=model,
        host=host,
        use_cache=False,
    )

    llm_chat = llm_from_config(
        provider=provider,
        model_name=model,
        host=host,
        use_cache=False,
    )
    llm_chat.support_json_mode = False
    llm_chat.support_structured_outputs = False

    graph = Graph(name="Plan Graph")
    plan_worker = PlanWorker(llm=llm)
    search_worker = SearchCreator(llm=llm)
    split_worker = SearchSplitter()
    search_fetch_worker = create_search_fetch_worker(llm=llm)
    analysis_worker = SearchSummarizer(llm=llm)
    analysis_joiner = AnalysisJoiner()
    final_narrative_worker = FinalNarrativeWorker(llm=llm)

    chat_worker = UserChat(llm=llm_chat)
    chat_adapter = ChatAdapter()

    response_publisher = ResponsePublisher()
    graph.add_workers(
        plan_worker,
        search_worker,
        split_worker,
        response_publisher,
        search_fetch_worker,
        analysis_worker,
        analysis_joiner,
        final_narrative_worker,
        chat_worker,
        chat_adapter,
    )
    graph.set_dependency(plan_worker, response_publisher)
    graph.set_dependency(plan_worker, search_worker).next(split_worker).next(
        search_fetch_worker
    ).next(analysis_worker).next(analysis_joiner).next(final_narrative_worker).next(
        response_publisher
    )
    graph.set_dependency(chat_worker, chat_adapter).next(response_publisher)
    graph.set_entry(plan_worker)
    graph.set_entry(chat_worker)
    graph.set_sink(response_publisher, Response, notify=notify)

    # limit the amount of LLM calls we will do in parallel
    graph.set_max_parallel_tasks(LLMTaskWorker, 2 if provider == "ollama" else 6)
    return graph, plan_worker, chat_worker
