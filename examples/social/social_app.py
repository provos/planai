import argparse
from datetime import datetime
from textwrap import dedent
from typing import List, Optional, Type

import requests
import yaml
from browse import WebBrowser
from pydantic import Field

from planai import (
    CachedLLMTaskWorker,
    CachedTaskWorker,
    Graph,
    InitialTaskWorker,
    JoinedTaskWorker,
    Task,
    llm_from_config,
)
from planai.integrations import SerperGoogleSearchTool
from planai.utils import setup_logging


# Task Definitions
class SearchQuery(Task):
    query: str  # Represents a search query


class NewsResult(Task):
    title: str
    link: str
    snippet: str


class SelectedNews(Task):
    results: List[NewsResult]


class PageResult(Task):
    title: str
    link: str
    content: str


class SelectedPages(Task):
    pages: List[PageResult]


class SocialMediaPost(Task):
    post1: str = Field(description="The first social media post")
    post2: str = Field(description="The second social media post")
    post3: str = Field(description="The third social media post")


# Task Workers
class SearchNewsWorker(CachedTaskWorker):
    output_types: List[Type[Task]] = [SelectedNews]

    def extra_cache_key(self, task: SearchQuery) -> str:
        # let's get different news each day
        return datetime.now().strftime("%Y-%m-%d")

    def consume_work(self, task: SearchQuery):
        # Use SerperGoogleSearchTool to search news
        news_results = SerperGoogleSearchTool.search_internet(
            query=task.query, search_type="news"
        )
        if news_results is None:
            self.print(f"No results found for query: {task.query}")
            return

        output: List[NewsResult] = []
        for result in news_results:
            output.append(
                NewsResult(
                    title=result["title"],
                    link=result["link"],
                    snippet=result["snippet"],
                )
            )

        self.print(f"Got {len(news_results)} search results for: '{task.query}'")
        self.publish_work(SelectedNews(results=output), input_task=task)


class SelectInterestingNews(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [NewsResult]
    llm_output_type: Type[Task] = SelectedNews
    prompt: str = dedent(
        """
You are tasked with evaluating news articles to determine their relevance for a specific social media profile.
The profile is focused on certain interests and thematics which you will keep in mind when making your selections.

Profile Description:
{profile_description}

Your task is to select news articles that would be engaging and beneficial for this profile to share, considering its audience and thematic focus.

Criteria for selection:
1. Relevance: The news article should directly relate to the interests described in the profile. Look for keywords and concepts that align with those interests.
2. Engagement: The topic should have potential for generating discussion or sparking curiosity among the profile's followers.
3. Novelty: Prefer articles that provide new insights or information, especially those that might not be widely known yet.
        """
    ).strip()
    profile_description: str

    def format_prompt(self, task: Task) -> str:
        return self.prompt.format(profile_description=self.profile_description)

    def consume_work(self, task: SelectedNews):
        super().consume_work(task)

    def post_process(self, response: Optional[SelectedNews], input_task: SelectedNews):
        if response is None:
            self.print("No news articles selected.")
            return

        for news in response.results:
            self.publish_work(news, input_task=input_task)


class PageFetcher(CachedTaskWorker):
    output_types: List[Type[Task]] = [PageResult]

    def consume_work(self, task: NewsResult):
        # Fetch the HTML content of the page
        try:
            response = WebBrowser.get_markdown_from_page(
                task.link, print_func=self.print
            )
            if response is None:
                self.print(f"Failed to fetch page for link: {task.link}")
                return

            # Create a PageResult with title, link, and content
            page_result = PageResult(title=task.title, link=task.link, content=response)

            # Publish the result
            self.publish_work(page_result, input_task=task)
            self.print(f"Successfully fetched page: {task.title}")

        except requests.RequestException as e:
            self.print(f"Failed to fetch page for link: {task.link}. Error: {str(e)}")


class PageCleaner(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [PageResult]
    llm_output_type: Type[Task] = PageResult
    prompt: str = dedent(
        """
Your task is to meticulously clean the provided Markdown content by identifying and removing irrelevant elements originating from the original web page. The goal is to ensure that the main, informative content remains intact and clearly presented.

Guidelines:
1. Focus on preserving all essential and meaningful parts of the content that contribute directly to the understanding of the main topic.
2. Carefully identify sections that add no informational value, such as navigation links, sidebars, advertisements, and repetitive headers or footers, and remove them.
3. Maintain the logical flow and coherence of the content to ensure it remains complete and easy to read.
4. Retain any contextual or supporting information that enhances comprehension and provides value to the main content.
5. Ensure that the cleaned Markdown reflects a concise, unobstructed view of the core content, free from unrelated or distracting elements.

Adhere to these principles to produce a final document that is streamlined, relevant, and focused solely on conveying the primary information effectively.
        """
    ).strip()

    def pre_process(self, task: PageResult) -> PageResult:
        task = task.model_copy()
        task.content = task.content[:45000]
        return task

    def consume_work(self, task: PageResult):
        return super().consume_work(task)


class CombineResults(JoinedTaskWorker):
    join_type: Type[Task] = InitialTaskWorker
    output_types: List[Type[Task]] = [SelectedPages]

    def consume_work(self, task: PageResult):
        super().consume_work(task)

    def consume_work_joined(self, tasks: List[PageResult]):
        tasks = [
            PageResult(
                title=task.title,
                link=task.link,
                content=task.content[:9000],
            )
            for task in tasks
        ]
        output_task = SelectedPages(pages=tasks)
        self.publish_work(output_task, input_task=tasks[0])


class CreatePostWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [SocialMediaPost]
    debug_mode: bool = True
    system_message: str = (
        f"As of {datetime.now().strftime('%a %b %d %Y')}, I am an AI assistant who thrives as a social media content creator and philosopher. I blend creativity with thoughtful insights to engage audiences and provoke meaningful discussions across platforms."
    )
    prompt: str = dedent(
        """
Your task is to create a series of engaging and thought-provoking social media posts for the following profile; drawing inspiration from the list of selected news articles provided above. Each post should captivate curiosity and spark discussion among followers by focusing on a single theme or insight derived from the news titles and contents.

Profile Description:
{profile_description}

Guidelines for creating the posts:
1. Begin with a captivating opening which can be partially inspired by the profile's core interests but should mainly be based on interesting news.
2. Draw inspiration from specific content in the news articles, offering intriguing insights related to that single focus.
3. Avoid forcefully mixing disparate topics; instead, allow each post to explore one theme deeply or provide a fresh perspective.
4. Foster engagement by posing a thought-provoking, open-ended question or inviting followers to share their perspectives.
5. The call for action should be subtle and not overaly generic, encouraging followers to engage with the post in a meaningful way.
6. Write so that it looks like it's coming from a human, not a robot.

Ensure each social media post is concise yet impactful, adhering to the character limits of platforms like Twitter (280 characters) and Threads, while maintaining engagement and clarity. Each response should be a single, coherent post ready for publication. Don't use any other emojis or special characters. You are allowed to use smileys like :) or :(.

Example Posts:
"AI's new frontier is reshaping our creativity. With fusion models on the rise, how might this redefine our artistic expressions? I will explore this in my next project. Curious what anyone else can report?"
"Quantum challenges could soon defeat traditional crypot. lattice encryption will likely be standardized but it's so expensive. When is the right moment to switch over to it? Thoughts?"

Now, craft three potential posts that each relate to a specific insight or theme from the selected news articles, allowing for depth and relevance:
    """
    ).strip()

    profile_description: str

    def format_prompt(self, task: Task) -> str:
        return self.prompt.format(profile_description=self.profile_description)

    def consume_work(self, task: SelectedPages):
        super().consume_work(task)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a social media post from search queries and a profile description."
    )
    parser.add_argument(
        "--yaml", type=str, required=True, help="Path to the YAML input file."
    )

    # Parse arguments
    args = parser.parse_args()

    # Load data from YAML
    with open(args.yaml, "r") as file:
        input_data = yaml.safe_load(file)

    profile_description = input_data["profile_description"]

    # Graph Setup
    graph = Graph(name="Social Media Post Creation")

    search_worker = SearchNewsWorker()

    llm = llm_from_config(provider="openai", model_name="gpt-4o-mini")

    select_worker = SelectInterestingNews(
        llm=llm, profile_description=profile_description
    )
    fetch_worker = PageFetcher()
    cleaner_worker = PageCleaner(llm=llm)
    combine_worker = CombineResults()

    llm_reasoning = llm_from_config(provider="openai", model_name="gpt-4o-2024-08-06")
    post_worker = CreatePostWorker(
        llm=llm_reasoning, profile_description=profile_description
    )

    graph.add_workers(
        search_worker,
        select_worker,
        fetch_worker,
        cleaner_worker,
        combine_worker,
        post_worker,
    )

    # Define dependencies
    graph.set_dependency(search_worker, select_worker).next(fetch_worker).next(
        cleaner_worker
    ).next(combine_worker).next(post_worker).sink(SocialMediaPost)

    # Create initial tasks for search queries
    input_tasks = []
    for query in input_data["search_queries"]:
        input_tasks.append((search_worker, SearchQuery(query=query)))

    # Write all logs to a file
    setup_logging()

    # Run the graph
    graph.run(initial_tasks=input_tasks)

    outputs = graph.get_output_tasks()
    for output in outputs:
        print(f"Post 1: {output.post1}")
        print(f"Post 2: {output.post2}")
        print(f"Post 3: {output.post3}")
        print()


if __name__ == "__main__":
    main()
