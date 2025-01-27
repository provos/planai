import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse

import html2text
from bs4 import BeautifulSoup
from playwright.sync_api import Error, Request, Route, sync_playwright


class WebBrowser:

    @staticmethod
    def get_page_content(
        url: str, download_path: str = None, print_func: callable = print
    ) -> tuple:
        """
        Retrieves the content of a web page specified by the given URL.

        Args:
            url (str): The URL of the web page to retrieve.
            download_path (str, optional): The path to save downloaded files. Defaults to None.
                                         Caller is responsible for cleanup of this directory.
            print_func (callable, optional): Function for logging/printing messages. Defaults to print.

        Returns:
            tuple: A tuple containing the content type and the page content.
                - The content type is a string indicating the type of the content.
                - The page content is a string representing the HTML content of the page.

                If the page contains a PDF file, the content type will be "application/pdf"
                and the page content will be the file path of the downloaded PDF.

                If the page is successfully loaded and does not contain a PDF file, the
                content type will be determined based on the response headers and the
                page content will be the HTML content of the page.

                If the page fails to load or an error occurs during the process, both the
                content type and the page content will be None.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--disable-gpu"])
            context = browser.new_context()
            page = context.new_page()

            download_path = (
                Path(download_path) if download_path else Path.cwd() / "downloads"
            )
            download_path.mkdir(parents=True, exist_ok=True)
            page.set_extra_http_headers(
                {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
                }
            )

            pdf_file_path = None

            def handle_route(route: Route, request: Request):
                nonlocal pdf_file_path
                if request.resource_type == "document" and request.url.lower().endswith(
                    ".pdf"
                ):
                    try:
                        file_name = Path(urlparse(request.url).path).name
                        pdf_file_path = str(download_path / file_name)
                        logging.debug("Downloading PDF: %s", file_name)
                        with open(pdf_file_path, "wb") as f:
                            f.write(route.fetch().body())
                        return route.fulfill(
                            headers={
                                "Content-Type": "application/pdf",
                                "Content-Disposition": f"attachment; filename={file_name}",
                            }
                        )
                    except Error as e:
                        print_func(f"Error downloading PDF: {e}")
                        route.abort()
                        return  # Do not continue processing the route
                return route.continue_()

            page.route("**/*", handle_route)

            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if pdf_file_path:
                    logging.debug("PDF downloaded to: %s", pdf_file_path)
                    return "application/pdf", pdf_file_path
                elif response.ok:
                    content_type = response.headers.get("content-type", "").split(";")[
                        0
                    ]
                    return content_type, page.content()
                else:
                    logging.error(
                        "Failed to load page. Status code: %s", str(response.status)
                    )
                    return None, None
            except Exception as e:
                if pdf_file_path:
                    logging.debug("PDF download completed despite error: %s", str(e))
                    return "application/pdf", pdf_file_path
                else:
                    print_func(f"An error occurred: {e}")
                    return None, None
            finally:
                browser.close()

    @staticmethod
    def extract_markdown(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")

        # we remove these following tags automatically as they are unlikely to contain the main content
        for elem in soup(["head", "nav", "footer", "script", "style", "aside"]):

            elem.decompose()

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0

        markdown = h.handle(str(soup))
        return markdown

    @staticmethod
    def get_markdown_from_page(
        url: str,
        extract_markdown_from_pdf: callable = None,
        print_func: callable = print,
    ) -> str:
        """Retrieves and converts webpage content to markdown format.

        This method fetches content from a URL and converts it to markdown. For PDF files,
        a conversion function must be supplied.

        Args:
            url (str): The URL of the webpage to process
            extract_markdown_from_pdf (callable, optional): Function to convert PDF content to markdown.
                Must accept binary PDF content as first argument and print_func as keyword argument.
                Required for processing PDF files.
            print_func (callable, optional): Function for logging/printing messages. Defaults to print.

        Returns:
            str: The webpage content converted to markdown format.
                Returns None if content type is unsupported and content cannot be extracted.

        Raises:
            Any exceptions from URL fetching or content processing are propagated.
        """
        mime_type, content = WebBrowser.get_page_content(url)
        if mime_type == "application/pdf" and extract_markdown_from_pdf:
            return extract_markdown_from_pdf(content, print_func=print_func)
        elif mime_type and mime_type.startswith("text/html"):
            return WebBrowser.extract_markdown(content)
        else:
            print_func(
                f"Unsupported mime type: {mime_type} - {len(content) if content else 0} bytes"
            )
            return WebBrowser.extract_markdown(content) if content else None


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Extract main content from a webpage or PDF"
    )
    parser.add_argument(
        "url", help="URL of the webpage or PDF", default="https://example.com"
    )
    args = parser.parse_args()

    mime_type, content = WebBrowser.get_page_content(args.url)
    if mime_type and mime_type.startswith("text/html"):
        main_content = WebBrowser.extract_markdown(content)
    else:
        print(f"Unsupported mime type: {mime_type}")
        main_content = None

    print(
        f"Content extracted from {args.url}: {mime_type} {len(content) if content else 0} bytes"
    )
    print(f"Markdown content: {len(main_content) if main_content else 0} characters")

    if main_content:
        print(main_content)
    else:
        print("Failed to extract content")


if __name__ == "__main__":
    main()
