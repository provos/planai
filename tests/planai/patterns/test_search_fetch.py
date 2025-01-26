import unittest
from unittest.mock import patch

from planai.graph import Graph
from planai.patterns.search_fetch import (
    ConsolidatedPages,
    PageAnalysis,
    PageResult,
    SearchQuery,
    create_search_fetch_graph,
    create_search_fetch_worker,
)
from planai.testing import MockCache, MockLLM, MockLLMResponse, inject_mock_cache


class TestSearchFetch(unittest.TestCase):
    def setUp(self):
        # Set up mock cache
        self.mock_cache = MockCache(dont_store=True)

        # Set up mock LLM
        self.mock_llm = MockLLM(
            responses=[
                MockLLMResponse(
                    pattern=".*",  # Match any prompt
                    response=PageAnalysis(
                        is_relevant=True, summary="This is a test summary"
                    ),
                )
            ]
        )

        # Mock search results
        self.mock_search_results = [
            {
                "title": "Test Result 1",
                "link": "https://example.com/1",
                "snippet": "Test snippet 1",
            },
            {
                "title": "Test Result 2",
                "link": "https://example.com/2",
                "snippet": "Test snippet 2",
            },
        ]

        # Mock page content
        self.mock_page_content = "# Test Content\nThis is test content."

        # Create patches
        self.search_patch = patch(
            "planai.patterns.search_fetch.SerperGoogleSearchTool.search_internet"
        )
        self.browser_patch = patch("planai.patterns.search_fetch.WebBrowser")

        # Start patches
        self.mock_search = self.search_patch.start()
        self.mock_browser = self.browser_patch.start()

        # Configure mocks
        self.mock_search.return_value = self.mock_search_results  # Changed this line
        self.mock_browser.get_markdown_from_page.return_value = self.mock_page_content

    def tearDown(self):
        self.search_patch.stop()
        self.browser_patch.stop()

    def test_search_fetch_workflow(self):
        # Create main graph
        graph = Graph(name="TestGraph")

        # Create search fetch worker with mock cache injection
        search_fetch = create_search_fetch_worker(
            llm=self.mock_llm, name="TestSearchFetch"
        )

        graph.add_workers(search_fetch)
        search_fetch.sink(ConsolidatedPages)

        # Inject mock cache into all cached workers
        inject_mock_cache(graph, self.mock_cache)

        # Create initial query
        query = SearchQuery(query="test query")
        initial_work = [(search_fetch, query)] * 10

        # Run the graph
        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Verify search was called 10 times
        self.assertEqual(self.mock_search.call_count, 10)
        self.mock_search.assert_has_calls(
            [
                unittest.mock.call(
                    "test query", num_results=10, print_func=unittest.mock.ANY
                )
            ]
            * 10
        )

        # Verify web browser calls
        self.assertEqual(
            self.mock_browser.get_markdown_from_page.call_count,
            len(self.mock_search_results) * 10,
            "Expected number of page fetches",
        )

        # Get output tasks
        output_tasks = graph.get_output_tasks()

        # Should have one consolidated output
        self.assertEqual(len(output_tasks), 10)

        consolidated = output_tasks[0]
        self.assertIsInstance(consolidated, ConsolidatedPages)

        # Verify pages were processed
        self.assertEqual(len(consolidated.pages), len(self.mock_search_results))

        # Verify content of pages
        for page in consolidated.pages:
            self.assertIsInstance(page, PageResult)
            self.assertTrue(page.url.startswith("https://example.com/"))
            self.assertTrue(page.title.startswith("Test Result"))
            self.assertEqual(page.content, self.mock_page_content)

    def test_search_fetch_with_failed_fetches(self):
        # Configure browser mock to fail for second URL
        def mock_get_markdown(url, **kwargs):
            if url == "https://example.com/2":
                return None
            return self.mock_page_content

        self.mock_browser.get_markdown_from_page.side_effect = mock_get_markdown

        # Create and run graph with mock cache injection
        graph = Graph(name="TestGraph")
        search_fetch = create_search_fetch_worker(
            llm=self.mock_llm, name="TestSearchFetch"
        )

        graph.add_workers(search_fetch)
        search_fetch.sink(ConsolidatedPages)

        # Inject mock cache into all cached workers
        inject_mock_cache(graph, self.mock_cache)

        query = SearchQuery(query="test query")
        initial_work = [(search_fetch, query)]

        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Get output tasks
        output_tasks = graph.get_output_tasks()
        consolidated = output_tasks[0]

        # Should only have one successful page
        self.assertEqual(len(consolidated.pages), 1)
        self.assertEqual(consolidated.pages[0].url, "https://example.com/1")

    def test_search_fetch_graph_workflow(self):
        # Create main graph using the plain graph version
        graph, _, exit_worker = create_search_fetch_graph(
            llm=self.mock_llm, name="TestSearchFetch"
        )
        graph.set_sink(exit_worker, ConsolidatedPages)

        # Find the search executor (entry point)
        search_executor = next(
            w for w in graph.workers if w.__class__.__name__ == "SearchExecutor"
        )

        # Inject mock cache into all cached workers
        inject_mock_cache(graph, self.mock_cache)

        # Create initial query
        query = SearchQuery(query="test query")
        initial_work = [(search_executor, query)] * 10

        # Run the graph
        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Verify search was called 10 times
        self.assertEqual(self.mock_search.call_count, 10)
        self.mock_search.assert_has_calls(
            [
                unittest.mock.call(
                    "test query", num_results=10, print_func=unittest.mock.ANY
                )
            ]
            * 10
        )

        # Verify web browser calls
        self.assertEqual(
            self.mock_browser.get_markdown_from_page.call_count,
            len(self.mock_search_results) * 10,
            "Expected number of page fetches",
        )

        # Get output tasks
        output_tasks = graph.get_output_tasks()

        # Should have one consolidated output
        self.assertEqual(len(output_tasks), 10)

        consolidated = output_tasks[0]
        self.assertIsInstance(consolidated, ConsolidatedPages)

        # Verify pages were processed
        self.assertEqual(len(consolidated.pages), len(self.mock_search_results))

        # Verify content of pages
        for page in consolidated.pages:
            self.assertIsInstance(page, PageResult)
            self.assertTrue(page.url.startswith("https://example.com/"))
            self.assertTrue(page.title.startswith("Test Result"))
            self.assertEqual(page.content, self.mock_page_content)

    def test_search_fetch_worker_distances(self):
        # Create graph and get references to workers
        graph, search_executor, exit_worker = create_search_fetch_graph(
            llm=self.mock_llm, name="TestSearchFetch"
        )

        # Compute distances
        graph.set_entry(search_executor)
        graph.finalize()

        # Validate specific distances from InitialTaskWorker
        initial_distances = graph._worker_distances["InitialTaskWorker"]
        expected_order = {
            "SearchExecutor": 1,
            "SearchResultSplitter": 2,
            "PageFetcher": 3,
            "PageRelevanceFilter": 4,
            "PageAnalysisConsumer": 4,
            "PageConsolidator": 5,
        }
        for worker_name, expected_distance in expected_order.items():
            self.assertEqual(
                initial_distances[worker_name],
                expected_distance,
                f"Expected {worker_name} to be at distance {expected_distance}, "
                f"got {initial_distances[worker_name]}",
            )

        # Validate specific distance between PageAnalysisConsumer and PageConsolidator
        analysis_consumer_distances = graph._worker_distances["PageAnalysisConsumer"]
        self.assertEqual(
            analysis_consumer_distances["PageConsolidator"],
            1,
            "Expected PageConsolidator to be at distance 1 from PageAnalysisConsumer",
        )


if __name__ == "__main__":
    unittest.main()
