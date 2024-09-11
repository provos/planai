import argparse
import hashlib
import json
import random
import sys
from textwrap import dedent
from typing import Any, Dict, List, Optional, Type

from pydantic import Field, PrivateAttr

from planai import (
    CachedLLMTaskWorker,
    Graph,
    InitialTaskWorker,
    JoinedTaskWorker,
    LLMTaskWorker,
    Task,
    TaskWorker,
)
from planai.llm_interface import LLMInterface
from planai.utils import setup_logging

from .cli_utils import (
    get_class_from_module,
    instantiate_llm_class_from_module,
    load_module_from_file,
)

# models


class PromptInput(Task):
    optimization_goal: str = Field(description="The goal of the optimization")
    prompt_template: str = Field(description="The prompt template to optimize")


class PromptPerformanceInput(Task):
    optimization_goal: str = Field(description="The goal of the optimization")
    prompt_template: str = Field(description="The prompt template to optimize")
    prompt_full: str = Field(description="The full prompt with all data filled in")
    response: Dict[str, Any] = Field(
        description="The response from the LLM to the full prompt"
    )


class PromptPerformanceOutput(Task):
    critique: str = Field(
        description="The critique of the prompt in terms of the optimization goal"
    )
    score: float = Field(
        description="A score from 0 to 1 indicating the quality of the prompt in terms of the response meeting the goal"
    )


class CombinedPromptCritique(Task):
    critique: List[str] = Field(
        description="The critiques of the prompt in terms of the optimization goal"
    )
    score: float = Field(
        description="A score from 0 to 1 indicating the quality of the prompt in terms of the response meeting the goal"
    )


class ImprovedPrompt(Task):
    prompt_template: str = Field(description="The improved prompt template")
    comment: str = Field(description="A comment on the improvement")


# optmization worker classes


class PromptGenerationWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [ImprovedPrompt]
    prompt: str = dedent(
        """
        Improve the provided prompt_template to better meet the optimization_goal.
        """
    ).strip()

    def consume_work(self, task: PromptInput):
        return super().consume_work(task)


class PrepareInputWorker(TaskWorker):
    _reference_data: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _random_seed: int = PrivateAttr()
    _module: Any = PrivateAttr()

    def __init__(
        self,
        module: Any,
        task_name: str,
        reference_data: List[Dict[str, Any]],
        random_seed: int = 42,
        **data,
    ):
        super().__init__(**data)
        output_class: Optional[Type[Task]] = get_class_from_module(module, task_name)
        if output_class is None:
            raise ValueError(
                f"Class '{task_name}' not found in module '{module.__name__}'"
            )
        self.output_types = [output_class]
        self._reference_data = reference_data
        self._random_seed = random_seed
        self._module = module

    def consume_work(self, task: ImprovedPrompt):
        # Hash the improved_prompt field
        prompt_hash = hashlib.md5(task.prompt_template.encode()).hexdigest()

        # Convert the hex hash to an integer and combine with the random seed
        task_seed = int(prompt_hash, 16) + self._random_seed + self._id

        # Set the random seed
        random.seed(task_seed)

        # Pick a deterministically random element of the reference data
        selected_reference = random.choice(self._reference_data)

        # Create and return a new task with the selected reference data
        output_task = create_input_task(
            self._module, self.output_types[0].__name__, selected_reference
        )

        # make a new input task from scratch
        task = task.model_copy(deep=True)
        # we have to splice in the input provenance so that the llm class can get all the data it needs
        task._provenance = task._provenance + output_task._provenance
        task._input_provenance = task._input_provenance + output_task._input_provenance

        self.publish_work(output_task, input_task=task)


class PromptPerformanceWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [PromptPerformanceOutput]
    prompt: str = dedent(
        """
        Analyze how well the provided prompt_template allows an LLM generate a response that meets the optimization_goal.
        Provide a critique of the prompt_template in terms of meeting the optimization_goal.
        Also, provide a floating point score from 0 to 1 indicating the quality of the prompt_template.

        Note: prompt_full is the prompt_template with all data filled in.
        """
    ).strip()

    def consume_work(self, task: PromptPerformanceInput):
        return super().consume_work(task)

    def post_process(
        self,
        response: Optional[PromptPerformanceOutput],
        input_task: PromptPerformanceInput,
    ):
        print(f"Prompt critique: {response.critique}")
        print(f"Prompt score: {response.score}")
        super().post_process(response, input_task)


class JoinPromptPerformanceOutput(JoinedTaskWorker):
    output_types: List[Type[Task]] = [CombinedPromptCritique]
    join_type: Type[CachedLLMTaskWorker] = InitialTaskWorker

    def consume_work(self, task: PromptPerformanceOutput):
        return super().consume_work(task)

    def consume_work_joined(self, tasks: List[PromptPerformanceOutput]):
        print(f"Received {len(tasks)} prompt performance outputs")
        self.publish_work(
            CombinedPromptCritique(
                critique=[task.critique for task in tasks],
                score=sum(task.score for task in tasks) / len(tasks),
            ),
            input_task=tasks[0],
        )


class PromptImprovementWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [ImprovedPrompt]
    prompt: str = dedent(
        """
        Improve the provided prompt_template based on the attached critique to better meet the optimization_goal.

        Prompt template:
        {prompt_template}

        Provide the improved prompt_template and a comment on the improvement.
        """
    ).strip()

    def consume_work(self, task: CombinedPromptCritique):
        return super().consume_work(task)

    def format_prompt(self, task: CombinedPromptCritique) -> str:
        prompt_input: Optional[PromptPerformanceInput] = task.find_input_task(
            PromptPerformanceInput
        )
        if prompt_input is None:
            raise ValueError("No input task found")

        return self.prompt.format(prompt_template=prompt_input.prompt_template)


class PromptPrinter(TaskWorker):

    def consume_work(self, task: ImprovedPrompt):
        print(f"Improved prompt: {task.prompt_template}")
        print(f"Improvement comment: {task.comment}")


# optimization driver code


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

    if not python_file or not class_name or not debug_log or not goal_prompt:
        print(
            "Missing required arguments. Please provide --python-file, --class-name, --debug-log, and --goal-prompt."
        )
        sys.exit(1)

    # First, load the module
    module = load_module_from_file(python_file, search_path)
    if module is not None:
        # Then, load and instantiate the class
        llm_class: Optional[LLMTaskWorker] = instantiate_llm_class_from_module(
            module=module, class_name=class_name, llm=llm
        )
        if llm_class is None:
            print(f"Failed to load class '{class_name}' from {python_file}")
            sys.exit(1)
        # we don't want to log debug output since we don't want to accidentally overwrite the debug log we are using
        # for prompt optimization
        llm_class.debug_mode = False
    else:
        print(f"Failed to load module from {python_file}")
        sys.exit(1)

    # Attempt to retrieve the 'prompt' attribute
    if not hasattr(llm_class, "prompt"):
        print(f"'prompt' attribute not found in class '{class_name}'.")
        sys.exit(1)

    # Load the debug log
    data = load_debug_log(debug_log)
    print(
        f"Optimizing prompt for class '{class_name}' in {python_file} using debug log from {debug_log}: {len(data)} examples.\n"
        f"Goal: {goal_prompt}"
    )

    # Infer the input class name from the loaded class
    task_name = llm_class.get_task_class().__name__

    setup_logging()

    graph = Graph(name="Prompt Optimization Graph")
    generation = PromptGenerationWorker(llm=llm)

    prepare_input = PrepareInputWorker(
        module=module, task_name=task_name, reference_data=data
    )

    prompt_analysis = PromptPerformanceWorker(llm=llm)
    joined_worker = JoinPromptPerformanceOutput()
    improvement_worker = PromptImprovementWorker(
        llm=llm
    )  # need a more powerful LLM here
    printer = PromptPrinter()
    graph.add_workers(
        generation,
        prepare_input,
        llm_class,
        prompt_analysis,
        joined_worker,
        improvement_worker,
        printer,
    )

    # we will inject the llm_class into the graph
    graph.set_dependency(generation, prepare_input).next(llm_class).sink()

    # create two new prompts
    input_tasks = []
    for example in data[:2]:
        task = create_input_task(module, task_name, example)
        prompt_template = llm_class.prompt
        input_tasks.append(
            (
                generation,
                PromptInput(
                    optimization_goal=goal_prompt, prompt_template=prompt_template
                ),
            )
        )

    graph.run(initial_tasks=input_tasks, run_dashboard=False)

    output = graph.get_tasks()
    for task in output:
        print(task.model_dump_json(indent=2))


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
