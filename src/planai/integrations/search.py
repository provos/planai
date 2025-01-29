# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import List, Literal, Optional

import requests


class SerperGoogleSearchTool:

    @staticmethod
    def search_internet(
        query: str,
        num_results: int = 10,
        start_index: int = 1,
        search_type: Literal["search", "news"] = "search",
        print_func: callable = print,
    ) -> Optional[List[dict]]:
        """
        Searches the internet for a given query using the Serper API and returns relevant results.

        Args:
            query (str): The search query.
            num_results (int, optional): The number of results to return. Defaults to 10.
            start_index (int, optional): The starting index for the search results. Defaults to 1.
            search_type (Literal["search", "news"], optional): The type of search to perform, either "search" or "news". Defaults to "search".
            print_func (callable, optional): A function to print messages. Defaults to print.

        Returns:
            List[dict]: A list of dictionaries containing the search results with keys 'title', 'link', and 'snippet'.

        Raises:
            AssertionError: If the "SERPER_API_KEY" is not found in the environment variables.
            ValueError: If an invalid search type is provided.
            Exception: If an error occurs during the search process.
        """
        if "SERPER_API_KEY" not in os.environ:
            raise EnvironmentError("SERPER_API_KEY not found in environment variables")

        print_func(f"Executing Google search via Serper for query: {query}")
        try:
            url = f"https://google.serper.dev/{search_type}"
            payload = {
                "q": query,
                "num": num_results,
                "page": start_index // num_results + 1,
            }
            headers = {
                "X-API-KEY": os.environ["SERPER_API_KEY"],
                "Content-Type": "application/json",
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            match search_type:
                case "search":
                    results = result.get("organic", [])
                case "news":
                    results = result.get("news", [])
                case _:
                    raise ValueError(f"Invalid search type: {search_type}")

            items = []
            for res in results:
                item = {
                    "title": res.get("title"),
                    "link": res.get("link"),
                    "snippet": res.get("snippet"),
                }
                items.append(item)

            logging.debug(
                f'Google search via Serper "{query}" completed. Found {len(items)} results.'
            )
            print_func(
                f'Google search via Serper "{query}" completed. Found {len(items)} results.'
            )
            return items
        except Exception as e:
            logging.error(
                'An error occurred during Google search via Serper "%s": %s',
                query,
                str(e),
            )
            return None

    @staticmethod
    def check() -> bool:
        """
        Checks if the Serper API key is valid by performing a test search.

        Returns:
            bool: True if the API key is valid and working, False otherwise.
        """
        if "SERPER_API_KEY" not in os.environ:
            return False

        try:
            url = "https://google.serper.dev/search"
            payload = {"q": "test", "num": 1}
            headers = {
                "X-API-KEY": os.environ["SERPER_API_KEY"],
                "Content-Type": "application/json",
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error("Serper API key validation failed: %s", str(e))
            return False


def main():  # pragma: no cover
    import argparse
    import json

    from dotenv import load_dotenv

    load_dotenv(".env.local")

    parser = argparse.ArgumentParser(
        description="Perform a Google search using Serper API"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the Serper API key is valid",
    )
    parser.add_argument("query", type=str, help="The search query", nargs="?")
    parser.add_argument(
        "-n",
        "--num_results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=1,
        help="Start index for search results (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    serper_search = SerperGoogleSearchTool()

    if args.check:
        if serper_search.check():
            print("Serper API key is valid and working.")
            return 0
        print("Serper API key is invalid or not working.")
        return 1

    if not args.query:
        parser.error("query is required when not using --check")

    results = serper_search.search_internet(
        args.query, num_results=args.num_results, start_index=args.start_index
    )

    if results:
        print(f"Search results for: '{args.query}'")
        if args.output == "json":
            print(json.dumps(results, indent=2))
        else:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   URL: {result['link']}")
                print(f"   Snippet: {result['snippet']}")
    else:
        print("No results found or an error occurred.")


if __name__ == "__main__":
    import sys

    sys.exit(main())
