import argparse
import logging

import html2text
from bs4 import BeautifulSoup
from playwright.sync_api import Request, sync_playwright


class WebBrowser:

    @staticmethod
    def get_page_content(url: str, print_func: callable = print) -> tuple:
        """
        Retrieves the content of a web page specified by the given URL.

        Args:
            url (str): The URL of the web page to retrieve.

        Returns:
            tuple: A tuple containing the content type and the page content.
                - The content type is a string indicating the type of the content.
                - The page content is a string representing the HTML content of the page.

                If the page is successfully loaded, the content type will be determined
                based on the response headers and the page content will be the HTML
                content of the page.

                If the page fails to load or an error occurs during the process, both the
                content type and the page content will be None.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context()
            page = context.new_page()

            page.set_extra_http_headers(
                {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
                }
            )

            def handle_route(route, request: Request):
                return route.continue_()

            page.route("**/*", handle_route)

            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if response.ok:
                    content_type = response.headers.get("content-type", "").split(";")[
                        0
                    ]
                    return content_type, page.content()
                else:
                    logging.error(
                        f"Failed to load page. Status code: {response.status}"
                    )
                    return None, None
            except Exception as e:
                print_func(f"An error occurred: {e}")
                return None, None
            finally:
                browser.close()

    @staticmethod
    def extract_markdown(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")

        for elem in soup(["header", "nav", "footer", "script", "style", "aside"]):
            elem.decompose()

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0

        markdown = h.handle(str(soup))
        return markdown

    @staticmethod
    def get_markdown_from_page(url: str, print_func: callable = print) -> str:
        mime_type, content = WebBrowser.get_page_content(url)
        if mime_type and mime_type.startswith("text/html"):
            return WebBrowser.extract_markdown(content)
        else:
            print_func(
                f"Unsupported mime type: {mime_type} - {len(content) if content else 0} bytes"
            )
            return WebBrowser.extract_markdown(content) if content else None


def main():
    parser = argparse.ArgumentParser(description="Extract main content from a webpage")
    parser.add_argument("url", help="URL of the webpage", default="https://example.com")
    args = parser.parse_args()

    mime_type, content = WebBrowser.get_page_content(args.url)
    if mime_type and mime_type.startswith("text/html"):
        main_content = WebBrowser.extract_markdown(content)
    else:
        print(f"Unsupported mime type: {mime_type}")
        main_content = None

    if main_content:
        print(main_content)
    else:
        print("Failed to extract content")


if __name__ == "__main__":
    main()
