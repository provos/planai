from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from planai.integrations.browse import WebBrowser


@pytest.fixture
def browser_setup():
    test_url = "https://example.com"
    mock_content = "<html><body><p>Test content</p></body></html>"
    return test_url, mock_content


@patch("planai.integrations.browse.sync_playwright")
def test_get_page_content_html(mock_playwright, browser_setup, tmp_path):
    test_url, mock_content = browser_setup
    # Setup playwright mocks
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_response = MagicMock()

    # Configure mock chain
    mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value = (
        mock_browser
    )
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page
    mock_page.goto.return_value = mock_response
    mock_page.content.return_value = mock_content

    # Configure response
    mock_response.ok = True
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}

    # Test the method
    content_type, content = WebBrowser.get_page_content(
        test_url, download_path=str(tmp_path)
    )

    assert content_type == "text/html"
    assert content == mock_content


@patch("planai.integrations.browse.sync_playwright")
def test_get_page_content_pdf(mock_playwright, browser_setup, tmp_path):
    test_url, _ = browser_setup
    # Setup playwright mocks
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()

    # Configure mock chain
    mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value = (
        mock_browser
    )
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    # Simulate PDF download
    def mock_route_handler(pattern, handler):
        mock_route = MagicMock()
        mock_request = MagicMock()

        mock_request.resource_type = "document"
        mock_request.url = "https://example.com/test.pdf"

        mock_route.fetch.return_value.body.return_value = b"fake pdf content"

        handler(mock_route, mock_request)

    mock_page.route.side_effect = mock_route_handler

    # Test the method with tmp_path
    content_type, file_path = WebBrowser.get_page_content(
        test_url, download_path=str(tmp_path)
    )

    assert content_type == "application/pdf"
    assert Path(file_path).parent == tmp_path


def test_extract_markdown():
    html_content = """
    <html>
        <head>Skip this</head>
        <body>
            <main>
                <h1>Test Title</h1>
                <p>Test paragraph</p>
            </main>
            <footer>Skip this too</footer>
        </body>
    </html>
    """
    markdown = WebBrowser.extract_markdown(html_content)
    assert "Test Title" in markdown
    assert "Test paragraph" in markdown
    assert "Skip this" not in markdown


@patch.object(WebBrowser, "get_page_content")
def test_get_markdown_from_page_html(mock_get_content, browser_setup):
    test_url, mock_content = browser_setup
    mock_get_content.return_value = ("text/html", mock_content)
    result = WebBrowser.get_markdown_from_page(test_url)
    assert result is not None
    assert "Test content" in result


@patch.object(WebBrowser, "get_page_content")
def test_get_markdown_from_page_error(mock_get_content, browser_setup):
    test_url, _ = browser_setup
    mock_get_content.return_value = (None, None)
    result = WebBrowser.get_markdown_from_page(test_url)
    assert result is None
