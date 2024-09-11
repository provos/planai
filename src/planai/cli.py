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
import argparse
import sys

from planai import llm_from_config

from .cli_optimize_prompt import optimize_prompt


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="planai command line interface")

    # Global arguments
    parser.add_argument(
        "--llm-provider", type=str, required=True, help="LLM provider name"
    )
    parser.add_argument(
        "--llm-model", type=str, required=True, help="LLM model name for generation"
    )
    parser.add_argument(
        "--llm-reason-model",
        type=str,
        required=True,
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

    parsed_args = parser.parse_args(args)

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
