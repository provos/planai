import os
import unittest
from unittest.mock import MagicMock, patch

from planai.integrations.search import SerperGoogleSearchTool


class TestSerperGoogleSearchTool(unittest.TestCase):
    def setUp(self):
        # Setup any necessary environment variables
        os.environ["SERPER_API_KEY"] = "dummy_api_key"

    @patch("planai.integrations.search.requests.post")
    def test_search_internet_successful(self, mock_post):
        # Mock response from the requests.post call
        mocked_response = MagicMock()
        mocked_response.raise_for_status = MagicMock()
        mocked_response.json.return_value = {
            "organic": [
                {
                    "title": "Test Title",
                    "link": "http://example.com",
                    "snippet": "A test snippet",
                }
            ]
        }
        mock_post.return_value = mocked_response

        # Call the method
        results = SerperGoogleSearchTool.search_internet(
            query="test query", num_results=1
        )

        # Assertions
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Test Title")
        self.assertEqual(results[0]["link"], "http://example.com")
        self.assertEqual(results[0]["snippet"], "A test snippet")

    @patch("planai.integrations.search.requests.post")
    def test_search_internet_no_results(self, mock_post):
        # Mock response with no results for the search
        mocked_response = MagicMock()
        mocked_response.raise_for_status = MagicMock()
        mocked_response.json.return_value = {"organic": []}
        mock_post.return_value = mocked_response

        # Call the method
        results = SerperGoogleSearchTool.search_internet(
            query="test query", num_results=1
        )

        # Assertions
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 0)

    @patch("planai.integrations.search.requests.post")
    def test_search_internet_exception(self, mock_post):
        # Simulate an error in the request
        mock_post.side_effect = Exception("Network error")

        # Call the method
        results = SerperGoogleSearchTool.search_internet(
            query="test query", num_results=1
        )

        # Assertions
        self.assertIsNone(results)

    @patch("planai.integrations.search.requests.post")
    def test_check_valid_api_key(self, mock_post):
        # Mock successful response
        mocked_response = MagicMock()
        mocked_response.raise_for_status = MagicMock()
        mock_post.return_value = mocked_response

        # Test the check method
        result = SerperGoogleSearchTool.check()

        # Assertions
        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch("planai.integrations.search.requests.post")
    def test_check_invalid_api_key(self, mock_post):
        # Simulate an API error
        mock_post.side_effect = Exception("Invalid API key")

        # Test the check method
        result = SerperGoogleSearchTool.check()

        # Assertions
        self.assertFalse(result)
        mock_post.assert_called_once()

    def tearDown(self):
        # Clean up the environment variables, if necessary
        del os.environ["SERPER_API_KEY"]


if __name__ == "__main__":
    unittest.main()
