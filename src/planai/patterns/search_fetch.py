import re
import tempfile
from textwrap import dedent
from typing import Callable, List, Optional, Type

from pydantic import Field

from ..cached_task import CachedTaskWorker
from ..graph import Graph
from ..graph_task import SubGraphWorker
from ..integrations import SerperGoogleSearchTool, WebBrowser
from ..joined_task import JoinedTaskWorker
from ..llm_interface import LLMInterface
from ..llm_task import CachedLLMTaskWorker
from ..task import Task, TaskWorker


class SearchQuery(Task):
    query: str = Field(..., description="The search query to execute")
    metadata: Optional[str] = Field(
        None, description="Metadata to pass along with the query", example="phase1"
    )


class SearchResult(Task):
    title: str
    link: str
    snippet: str


class SearchResults(Task):
    results: List[SearchResult]


class PageResult(Task):
    url: str
    title: str
    content: Optional[str]


class PageAnalysis(Task):
    is_relevant: bool = Field(
        ..., description="Whether the page contains relevant content"
    )
    summary: str = Field(..., description="Summary of the analysis")


class ConsolidatedPages(Task):
    pages: List[PageResult]


class SearchExecutor(CachedTaskWorker):
    output_types: List[Type[Task]] = [SearchResults]
    max_results: int = Field(10, description="Maximum number of results per query")

    def consume_work(self, task: SearchQuery):
        self.print(f"Executing search for: {task.query}")
        self.notify_status(task, f"Searching for: {task.query}")
        results = SerperGoogleSearchTool.search_internet(
            task.query, num_results=self.max_results, print_func=self.print
        )
        self.publish_work(
            task=SearchResults(
                results=[
                    SearchResult(title=r["title"], link=r["link"], snippet=r["snippet"])
                    for r in results
                ]
            ),
            input_task=task,
        )


class SearchResultSplitter(TaskWorker):
    output_types: List[Type[Task]] = [SearchResult]

    def consume_work(self, task: SearchResults):
        for result in task.results:
            self.publish_work(task=result, input_task=task)


class PageFetcher(CachedTaskWorker):
    output_types: List[Type[Task]] = [PageResult]
    extract_pdf_func: Optional[Callable] = None
    support_user_input: bool = False

    def __init__(self, extract_pdf_func: Optional[Callable] = None):
        super().__init__()
        self.extract_pdf_func = extract_pdf_func

    def consume_work(self, task: SearchResult):
        self.print(f"Fetching content from: {task.link}")
        self.notify_status(task, f"Fetching content from: {task.link}")
        content = WebBrowser.get_markdown_from_page(
            task.link,
            extract_markdown_from_pdf=self.extract_pdf_func,
            print_func=self.print,
        )

        if content is None and self.support_user_input:
            self.print(f"Failed to fetch content for {task.link}, requesting from user")
            data, mime_type = self.request_user_input(
                task=task,
                instruction=f'Please provide the content for: <a href="{task.link}" target="_blank">{task.link}</a>',
                accepted_mime_types=["text/html", "application/pdf"],
            )

            if data and mime_type:
                if mime_type == "application/pdf" and self.extract_pdf_func:
                    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                        temp_file.write(data)
                        temp_file.flush()
                        content = self.extract_pdf_func(temp_file.name, self.print)
                else:  # text/html
                    content = WebBrowser.extract_markdown(data)

        if content:
            # Remove markdown links while preserving the link text
            content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

        result = PageResult(url=task.link, title=task.title, content=content)
        self.publish_work(task=result, input_task=task)


class PageRelevanceFilter(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [PageResult]
    llm_output_type: Type[Task] = PageAnalysis
    llm_input_type: Type[Task] = PageResult
    use_xml: bool = True
    prompt: str = dedent(
        """
        Analyze if this content contains information that is factual, credible and useful:

        Guidelines:
        - Content should be from a credible source
        - Information should be factual and verifiable
        - Avoid content that is purely promotional or opinion-based
        - Content should offer substantive information
        """
    ).strip()

    def pre_process(self, task):
        self.notify_status(task, f"Analyzing relevance of: {task.url}")
        return task

    def post_process(self, response: PageAnalysis, input_task: PageResult):
        if response.is_relevant:
            self.publish_work(task=input_task.model_copy(), input_task=input_task)


class PageConsolidator(JoinedTaskWorker):
    output_types: List[Type[Task]] = [ConsolidatedPages]
    join_type: Type[TaskWorker] = SearchExecutor

    def consume_work_joined(self, tasks: List[PageResult]):
        self.publish_work(
            task=ConsolidatedPages(pages=[t for t in tasks if t.content]),
            input_task=tasks[0],
        )


def create_search_fetch_worker(
    *,
    llm: LLMInterface,
    name: str = "SearchFetchWorker",
    extract_pdf_func: Optional[Callable] = None,
) -> TaskWorker:
    """Creates a SubGraphWorker that searches and fetches web content.

    This worker creates a subgraph that processes a search query through multiple stages:
    1. Executes web search
    2. Fetches content from result pages
    3. Filters for relevance using LLM
    4. Consolidates relevant pages

    Args:
        llm: LLM interface for content analysis
        name: Name for the worker
        extract_pdf_func: Optional function to extract text from PDFs

    Input Task:
        SearchQuery: A task containing a single 'query' string to search for

    Output Task:
        ConsolidatedPages: A task containing a list of PageResult objects, each with:
            - url: The page URL
            - title: Page title
            - content: Extracted page content (if successfully fetched)

    Returns:
        A SubGraphWorker that implements the search and fetch pattern
    """
    graph = Graph(name=f"{name}Graph")

    search = SearchExecutor()
    splitter = SearchResultSplitter()
    fetcher = PageFetcher(extract_pdf_func=extract_pdf_func)
    relevance = PageRelevanceFilter(llm=llm)
    consolidator = PageConsolidator()

    graph.add_workers(search, splitter, fetcher, relevance, consolidator)
    graph.set_dependency(search, splitter).next(fetcher).next(relevance).next(
        consolidator
    )

    return SubGraphWorker(
        name=name,
        graph=graph,
        entry_worker=search,
        exit_worker=consolidator,
    )
