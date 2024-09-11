import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from planai import LLMTaskWorker, Task
from planai.llm_interface import LLMInterface

from .cli_utils import (
    get_class_from_module,
    instantiate_class_from_module,
    load_module_from_file,
)


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
        sys.exit(0)

    print(
        f"Optimizing prompt for class '{class_name}' in {python_file} using debug log from {debug_log}. Goal: {goal_prompt}"
    )

    # First, load the module
    module = load_module_from_file(python_file, search_path)
    if module is not None:
        # Then, load and instantiate the class
        llm_class: LLMTaskWorker = instantiate_class_from_module(module, class_name)
    else:
        print(f"Failed to load module from {python_file}")
        sys.exit(1)

    # Attempt to retrieve the 'prompt' attribute
    if hasattr(llm_class, "prompt"):
        prompt = getattr(llm_class, "prompt")
        print(f"Current prompt: {prompt}")
    else:
        print(f"'prompt' attribute not found in class '{class_name}'.")
        sys.exit(1)

    # Load the debug log
    data = load_debug_log(debug_log)
    print(f"Loaded {len(data)} prompts and responses from {debug_log}")

    task_name = llm_class.get_task_class().__name__

    task = create_input_task(module, task_name, data[0])
    print(f"Prompt example:\n{llm_class.format_prompt(task)[:100]}...")
    print(f"Response example:\n{data[0]['response']}")


def create_input_task(module: Any, class_name: str, data: Dict[str, Any]) -> Task:
    """
    Create an input task object.

    Args:
        module (Any): The module containing the class definition.
        class_name (str): The name of the class.
        data (Dict[str, Any]): The data for the input task.

    Returns:
        Task: The created input task object derived from PlanAI's Task class.

    Raises:
        ValueError: If the class is not found in the module.
    """
    cls: Optional[Task] = get_class_from_module(module, class_name)
    if cls is None:
        raise ValueError(
            f"Class '{class_name}' not found in module '{module.__name__}'"
        )

    data = data["input_task"]
    task = cls.model_validate(data)

    input_provenance = []
    for data, cls_name in zip(
        data["_input_provenance"], data["_input_provenance_classes"]
    ):
        cls = get_class_from_module(module, cls_name)
        tmp_obj = cls.model_validate(data)
        input_provenance.append(tmp_obj)

    task._input_provenance = input_provenance
    return task


def load_debug_log(debug_log: str) -> List[Dict[str, Any]]:
    """
    Load a debug log file containing prompts and responses.

    :param debug_log: The path to the debug log file.
    :return: A list of dictionaries containing prompts and responses.
    """

    # read the debug log line by line
    with open(debug_log, "r") as file:
        lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        if "}{" in lines[i]:
            lines[i] = lines[i].replace("}{", "},\n{")

    fixed_json = "[\n" + "".join(lines) + "\n]"
    return json.loads(fixed_json)
