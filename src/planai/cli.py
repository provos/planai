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

"""
PlanAI Command Line Interface

This module provides a command-line interface for PlanAI, focusing on automated prompt optimization.

The main functionality includes:
1. Optimizing prompts based on debug logs
2. Configuring LLM providers and models
3. Processing input from Python files, debug logs, and goal prompts
4. Outputting optimized configurations

Usage:
    python -m planai.cli --llm-provider <provider> --llm-model <model> --llm-reason-model <reason_model> optimize-prompt [options]
"""

import argparse
import sys
from typing import List

from planai import llm_from_config

from .cli_cache import handle_cache_subcommand
from .cli_optimize_prompt import optimize_prompt


def parse_comma_separated_list(arg: str) -> List[str]:
    # Split the argument by commas and strip any extra whitespace
    return [item.strip() for item in arg.split(",")]


def create_parser():
    parser = argparse.ArgumentParser(description="planai command line interface")

    # Global arguments
    parser.add_argument(
        "--llm-provider", type=str, required=False, help="LLM provider name"
    )
    parser.add_argument(
        "--llm-model", type=str, required=False, help="LLM model name for generation"
    )
    parser.add_argument(
        "--llm-reason-model",
        type=str,
        required=False,
        help="LLM model name for reasoning",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Subcommand optimize-prompt
    optimize_parser = subparsers.add_parser(
        "optimize-prompt", help="Optimize prompt based on debug logs"
    )
    optimize_parser.add_argument(
        "--python-file", type=str, required=False, help="Path to the Python file"
    )
    optimize_parser.add_argument(
        "--class-name", type=str, required=False, help="Class name in the Python file"
    )
    optimize_parser.add_argument(
        "--debug-log", type=str, required=False, help="Path to the JSON debug log file"
    )
    optimize_parser.add_argument(
        "--goal-prompt", type=str, required=False, help="Goal prompt for optimization"
    )
    optimize_parser.add_argument(
        "--output-config", type=str, help="Output a configuration file"
    )
    optimize_parser.add_argument(
        "--search-path", type=str, help="Optional path to include in module search path"
    )
    optimize_parser.add_argument(
        "--config", type=str, help="Path to a configuration file"
    )
    optimize_parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of optimization iterations",
    )
    optimize_parser.add_argument(
        "--llm-opt-provider",
        type=str,
        required=False,
        help="LLM provider name to be used for the prompt that is being optimized. This should be the same LLM as being used in production.",
    )
    optimize_parser.add_argument(
        "--llm-opt-model",
        type=str,
        required=False,
        help="LLM model name for generation to be used for the prompt that is being optimized. This should be the same LLM as being used in production.",
    )
    optimize_parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write output files",
    )

    # Subcommand cache
    cache_parser = subparsers.add_parser("cache", help="Inspect and manipulate cache")
    cache_parser.add_argument(
        "cache_dir", type=str, help="Directory of the diskcache to operate on"
    )
    cache_parser.add_argument(
        "--output-task-filter",
        type=str,
        help="Filter for output task type",
        default=None,
    )
    cache_parser.add_argument(
        "--delete", type=str, help="Delete a specific cache key", default=None
    )
    cache_parser.add_argument("--clear", action="store_true", help="Clear the cache")
    cache_parser.add_argument(
        "--search-dirs",
        type=parse_comma_separated_list,
        help="Comma-separated list of directories to search for Python modules",
        default=None,
    )

    return parser


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "cache":
        handle_cache_subcommand(parsed_args)
        return

    llm_fast = llm_from_config(
        provider=parsed_args.llm_provider, model_name=parsed_args.llm_model
    )
    llm_reason = llm_from_config(
        provider=parsed_args.llm_provider, model_name=parsed_args.llm_reason_model
    )

    if parsed_args.command == "optimize-prompt":
        optimize_prompt(llm_fast, llm_reason, parsed_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
