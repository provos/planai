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
import importlib.util
import json
import os
import sys
from typing import Optional

from planai import llm_from_config
from planai.llm_interface import LLMInterface


def load_class_from_file(
    python_file: str, class_name: str, search_path: Optional[str] = None
):
    """
    Dynamically load a class from a given Python file with an optional search path.

    :param python_file: The path to the Python file.
    :param class_name: The name of the class to be loaded.
    :param search_path: An optional path to include in the module search path.
    :return: The class object if found, else None.
    """
    python_file = os.path.abspath(python_file)
    if search_path:
        search_path = os.path.abspath(search_path)
        if search_path not in sys.path:
            sys.path.insert(0, search_path)

    try:
        # Derive the module's name and package structure
        relative_path = os.path.relpath(python_file, start=search_path)
        module_parts = os.path.splitext(relative_path.replace(os.path.sep, "."))[
            0
        ].split(".")
        package = ".".join(module_parts[:-1])
        module_name = module_parts[-1]

        # Load the target module
        spec = importlib.util.spec_from_file_location(
            f"{package}.{module_name}", python_file
        )
        if spec is None:
            print(f"Could not load spec for file {python_file}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package}.{module_name}"] = module
        spec.loader.exec_module(module)

        # Retrieve the class from the module
        cls = getattr(module, class_name, None)
        if cls is None:
            print(f"Class '{class_name}' not found in module '{module_name}'")
        return cls

    except Exception as e:
        print(f"Error loading class '{class_name}' from '{python_file}': {e}")
        return None

    finally:
        # Clean up the path entry
        if search_path and search_path in sys.path:
            sys.path.remove(search_path)

    return None


def get_class(
    python_file: str, class_name: str, search_path: Optional[str] = None
) -> Optional[object]:
    """
    Dynamically load a class from a Python file and retrieve the 'prompt' attribute
    from an instance of the specified class, with optional module search path.

    :param python_file: Path to the Python file.
    :param class_name: The name of the class to instantiate.
    :param search_path: An optional path to include in the module search path.
    :return: The prompt attribute value or None if retrieval fails.
    """
    try:
        # Load the class from the file
        cls = load_class_from_file(python_file, class_name, search_path)

        if cls is None:
            print(f"Class '{class_name}' not found in Python file {python_file}.")
            return None

        # Create an instance of the class, passing the required 'llm' parameter
        instance = cls(llm=LLMInterface())
        return instance

    except Exception as e:
        print(f"Error instantiating class '{class_name}' from '{python_file}': {e}")
        return None


def optimize_prompt(llm: LLMInterface, args: argparse.Namespace):
    if args.config:
        # Read from configuration file
        try:
            with open(args.config, "r") as config_file:
                config = json.load(config_file)
                python_file = config.get("python_file")
                class_name = config.get("class_name")
                debug_log = config.get("debug_log")
                goal_prompt = config.get("goal_prompt")
                search_path = config.get("search_path")

                missing_fields = []
                if not python_file:
                    missing_fields.append("python_file")
                if not class_name:
                    missing_fields.append("class_name")
                if not debug_log:
                    missing_fields.append("debug_log")
                if not goal_prompt:
                    missing_fields.append("goal_prompt")
                if not search_path:
                    missing_fields.append("search_path")

                if missing_fields:
                    print(
                        f"Configuration file is missing the following fields: {', '.join(missing_fields)}."
                    )
                    sys.exit(1)
        except FileNotFoundError:
            print(f"Configuration file {args.config} not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print("Error decoding JSON from the configuration file.")
            sys.exit(1)
    else:
        # Assign directly from command-line arguments
        python_file = args.python_file
        class_name = args.class_name
        debug_log = args.debug_log
        goal_prompt = args.goal_prompt
        search_path = args.search_path

    # Write out configuration if requested
    if args.output_config:
        config_data = {
            "python_file": python_file,
            "class_name": class_name,
            "debug_log": debug_log,
            "goal_prompt": goal_prompt,
            "search_path": search_path,
        }
        with open(args.output_config, "w") as config_file:
            json.dump(config_data, config_file, indent=4)
        print(f"Configuration written to {args.output_config}")
        exit(0)

    print(
        f"Optimizing prompt for class '{class_name}' in {python_file} using debug log from {debug_log}. Goal: {goal_prompt}"
    )

    llm_class = get_class(python_file, class_name, search_path)
    # Attempt to retrieve the 'prompt' attribute
    if hasattr(llm_class, "prompt"):
        prompt = getattr(llm_class, "prompt")
    else:
        print(f"'prompt' attribute not found in class '{class_name}'.")
        sys.exit(1)

    print(f"Current prompt: {prompt}")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="planai command line interface")

    # Global arguments
    parser.add_argument(
        "--llm-provider", type=str, required=True, help="LLM provider name"
    )
    parser.add_argument("--llm-model", type=str, required=True, help="LLM model name")

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

    llm = llm_from_config(
        provider=parsed_args.llm_provider, model_name=parsed_args.llm_model
    )

    if parsed_args.command == "optimize-prompt":
        optimize_prompt(llm, parsed_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
