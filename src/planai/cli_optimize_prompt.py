import argparse
import hashlib
import json
import random
import re
import sys
from operator import attrgetter
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Type

from pydantic import Field, PrivateAttr

from planai import (
    CachedLLMTaskWorker,
    Graph,
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
    id: Optional[int] = Field(0, description="The id of the input task")


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


class PromptCritique(Task):
    prompt_template: str = Field(description="The prompt template")
    critique: List[str] = Field(
        description="The critiques of the prompt in terms of the optimization goal"
    )
    score: float = Field(
        description="A score from 0 to 1 indicating the quality of the prompt in terms of the response meeting the goal"
    )


class MultipleCombinedPromptCritique(Task):
    critiques: List[PromptCritique] = Field(
        description="The critiques of the prompt in terms of the optimization goal"
    )


class ImprovedPrompt(Task):
    prompt_template: str = Field(description="The improved prompt template")
    comment: str = Field(description="A comment on the improvement")


# optmization worker classes


class PromptGenerationWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [ImprovedPrompt]
    prompt: str = dedent(
        """
Analyze the provided prompt_template and suggest an improvement to better meet the optimization_goal. Your response should be structured as follows:

1. prompt_template: Provide a single, cohesive improved version of the prompt_template. This should be a complete, ready-to-use prompt that incorporates your suggested enhancements.

2. comment: Write a brief comment (2-3 sentences) explaining the key improvements made and how they better meet the optimization_goal.

When crafting your improvement:

- Maintain any existing {{keywords}} that may be present for .format() expansion. Do not remove or modify these.
- If you need to include literal curly braces in the prompt text, use double braces: {{{{ for a literal opening brace and }}}} for a literal closing brace.
- Focus on the structure and approach of the prompt rather than any specific subject matter.
- Consider clarity, effectiveness, and adaptability of the prompt.
- Aim to enhance the prompt's ability to generate responses that meet the optimization_goal.
- Ensure your improved prompt_template is a complete, standalone prompt that can be used as-is.

Your output should be formatted as a valid JSON object with two fields: "prompt_template" and "comment".
        """
    ).strip()

    def consume_work(self, task: PromptInput):
        return super().consume_work(task)

    def post_process(self, response: ImprovedPrompt, input_task: PromptInput):
        # sanitize the prompt to keep the keywords
        response.prompt_template = sanitize_prompt(
            input_task.prompt_template, response.prompt_template
        )
        return super().post_process(response, input_task)


class PrepareInputWorker(TaskWorker):
    random_seed: int = Field(
        42, description="The random seed to use for selecting reference data"
    )
    num_examples: int = Field(5, description="The number of examples to produce")
    _reference_data: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _random_seed: int = PrivateAttr()
    _module: Any = PrivateAttr()

    def __init__(
        self,
        module: Any,
        task_name: str,
        reference_data: List[Dict[str, Any]],
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
        self._module = module

    def consume_work(self, task: ImprovedPrompt):
        # Hash the improved_prompt field
        prompt_hash = hashlib.md5(task.prompt_template.encode()).hexdigest()

        # Convert the hex hash to an integer and combine with the random seed
        task_seed = int(prompt_hash, 16) + self.random_seed + self._id

        # Set the random seed
        random.seed(task_seed)

        for _ in range(self.num_examples):
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
            task._input_provenance = (
                task._input_provenance + output_task._input_provenance
            )

            self.publish_work(output_task, input_task=task)


class PromptPerformanceWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [PromptPerformanceOutput]
    prompt: str = dedent(
        """
Analyze how well the provided prompt_template allows an LLM to generate a response that meets the optimization_goal. Your assessment should include:

1. Comprehensiveness: How well does the prompt encourage complete answers derived from the text? (Score 0-1)
2. Accuracy and Factual Integrity: How effectively does the prompt ensure responses are accurate and factual? (Score 0-1)
3. Clarity and Structure: How well does the prompt guide the creation of clear, well-structured responses? (Score 0-1)
4. Variance in Approaches: How effectively does the prompt encourage significantly different approaches or perspectives in the two answers? (Score 0-1)
5. Depth of Analysis: How well does the prompt promote in-depth understanding and analysis of the text? (Score 0-1)

For each criterion:
- Provide a brief explanation of your score integrated into the final critique field
- Give a specific example from the prompt_template that supports your assessment
- Suggest one way to improve the prompt for this criterion

Additionally:
- Identify any potential unintended consequences of the current prompt_template
- Propose one overall improvement to better meet the optimization_goal

Calculate the final score as the average of the five criterion scores.
Your final output should have only two fields: critique and score.

Note: prompt_full is the prompt_template with all data filled in.
        """
    ).strip()

    def consume_work(self, task: PromptPerformanceInput):
        return super().consume_work(task)


class JoinPromptPerformanceOutput(JoinedTaskWorker):
    output_types: List[Type[Task]] = [CombinedPromptCritique]
    join_type: Type[TaskWorker] = PromptGenerationWorker

    def consume_work(self, task: PromptPerformanceOutput):
        return super().consume_work(task)

    def consume_work_joined(self, tasks: List[PromptPerformanceOutput]):
        self.print(f"Received {len(tasks)} prompt performance outputs")
        self.publish_work(
            CombinedPromptCritique(
                critique=[task.critique for task in tasks],
                score=sum(task.score for task in tasks) / len(tasks),
            ),
            input_task=tasks[0],
        )


class AccumulateCritiqueOutput(TaskWorker):
    iterations: int = Field(3, description="The number of iterations to run")
    output_types: List[Type[Task]] = [MultipleCombinedPromptCritique, PromptCritique]
    _state: Dict[str, PromptCritique] = PrivateAttr(default_factory=dict)
    _count: int = PrivateAttr(default=0)

    def consume_work(self, task: CombinedPromptCritique):
        input_task = task.find_input_task(ImprovedPrompt)
        if input_task is None:
            raise ValueError("No input task found")

        critique = PromptCritique(
            prompt_template=input_task.prompt_template,
            critique=task.critique,
            score=task.score,
        )

        with self.lock:
            self._state[input_task.prompt_template] = critique

            if len(self._state) < 2:
                self.print(
                    f"Only {len(self._state)} critiques so far - waiting for more"
                )
                # we need at least two critiques to compare
                return

            # keep the top three scores
            top_three = sorted(
                self._state.values(), key=attrgetter("score"), reverse=True
            )[:3]

            self._state = {critique.prompt_template: critique for critique in top_three}
            self.print(
                f"Top three scores: {[critique.score for critique in top_three]}"
            )

            final_output = MultipleCombinedPromptCritique(
                critiques=list(self._state.values())
            )

            self._count += 1
            if self._count >= self.iterations:
                # send the top three critiques to the final sink
                for critique in self._state.values():
                    self.publish_work(critique, input_task=task)
                return

            self.publish_work(final_output, input_task=task)


class PromptImprovementWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [ImprovedPrompt]
    prompt: str = dedent(
        """
You are being provided with multiple prompt_templates, their respective critique and how they scored
in terms of meeting the stated optimization_goal. Your task is to create an improved prompt_template
based on the attached prompts and critiques. The improved prompt_template should be better at meeting the optimization_goal.

Optmization Goal: {optimization_goal}

When crafting your improvement:
- Focus on the structure and approach of the prompt rather than any specific subject matter.
- Consider clarity, effectiveness, and adaptability of the prompt.
- Aim to enhance the prompt's ability to generate responses that meet the optimization_goal.
- Ensure your improved prompt_template is a complete, standalone prompt that can be used as-is.

For the improved prompt you are generating, it is extremley important:
- To maintain any existing {{keywords}} that may be present for .format() expansion. Do not remove or modify these.
- If you need to include literal curly braces in the prompt text, use double braces: {{{{ for a literal opening brace and }}}} for a literal closing brace.

Provide the improved prompt_template and a comment on the improvement.
        """
    ).strip()
    _llm_class: LLMTaskWorker = PrivateAttr()

    def __init__(self, llm_class: LLMTaskWorker, **data):
        super().__init__(**data)
        self._llm_class = llm_class

    def consume_work(self, task: MultipleCombinedPromptCritique):
        return super().consume_work(task)

    def format_prompt(self, task: CombinedPromptCritique) -> str:
        goal_input: Optional[PromptInput] = task.find_input_task(PromptInput)
        if goal_input is None:
            raise ValueError("No input task found for PromptInput")

        return self.prompt.format(
            optimization_goal=goal_input.optimization_goal,
        )

    def extra_validation(
        self, response: ImprovedPrompt, input_task: Task
    ) -> Optional[str]:
        input_class = self._llm_class.get_task_class()
        llm_input = input_task.find_input_task(input_class)
        if llm_input is None:
            raise ValueError(f"No input task found for {input_class.__name__}")

        # we will try whether the new prompt actually works for the llm class
        # any erros will be fed back to the LLM to try again

        # let's inject the prompt brute-force
        with self._llm_class.lock:
            old_prompt = self._llm_class.prompt
            self._llm_class.prompt = response.prompt_template

            error = None
            try:
                self._llm_class.get_full_prompt(llm_input)
            except Exception as e:
                error = str(e)
            finally:
                self._llm_class.prompt = old_prompt

        return error


# optimization driver code


def optimize_prompt(
    llm_fast: LLMInterface, llm_reason: LLMInterface, args: argparse.Namespace
):
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
                num_iterations = config.get("num_iterations", 3)

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
        num_iterations = args.num_iterations

    # Write out configuration if requested
    if args.output_config:
        config_data = {
            "python_file": python_file,
            "class_name": class_name,
            "debug_log": debug_log,
            "goal_prompt": goal_prompt,
            "search_path": search_path,
            "num_iterations": num_iterations,
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
    if module is None:
        print(f"Failed to load module from {python_file}")
        sys.exit(1)

    # Then, load and instantiate the class
    llm_class: Optional[LLMTaskWorker] = instantiate_llm_class_from_module(
        module=module, class_name=class_name, llm=llm_fast
    )
    if llm_class is None:
        print(f"Failed to load class '{class_name}' from {python_file}")
        sys.exit(1)
    # we don't want to log debug output since we don't want to accidentally overwrite the debug log we are using
    # for prompt optimization
    llm_class.debug_mode = False

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

    # The output type of the LLM class is the input type for the PromptPerformanceWorker
    # We may need to change this if the LLM class has multiple output types
    llm_output_type: Type[Task] = llm_class.output_types[0]

    # We need to create this class dynamically because it depends on the output type of the LLM class
    class OutputAdapter(TaskWorker):
        output_types: List[Type[Task]] = [PromptPerformanceInput]

        def consume_work(self, task: llm_output_type):
            input_task = task.previous_input_task()
            full_prompt = llm_class.get_full_prompt(input_task)

            prompt_task = task.find_input_task(ImprovedPrompt)
            if prompt_task is None:
                raise ValueError("No input task found for ImprovedPrompt")

            output = PromptPerformanceInput(
                optimization_goal=goal_prompt,
                prompt_template=llm_class.prompt,
                prompt_full=full_prompt,
                response=task.model_dump(),
            )
            self.publish_work(output, input_task=task)

    setup_logging()

    graph = Graph(name="Prompt Optimization Graph")
    generation = PromptGenerationWorker(llm=llm_reason)

    prepare_input = PrepareInputWorker(
        module=module, task_name=task_name, reference_data=data
    )
    # we are injecting llm_class between these two workers
    adapt_output = OutputAdapter()

    prompt_analysis = PromptPerformanceWorker(llm=llm_fast)
    joined_worker = JoinPromptPerformanceOutput()

    accumulate_critique = AccumulateCritiqueOutput(iterations=num_iterations)

    improvement_worker = PromptImprovementWorker(
        llm=llm_reason,
        llm_class=llm_class,
    )  # need a more powerful LLM here

    graph.add_workers(
        generation,
        prepare_input,
        llm_class,
        adapt_output,
        prompt_analysis,
        joined_worker,
        accumulate_critique,
        improvement_worker,
    )

    # we will inject the llm_class into the graph
    graph.set_dependency(generation, prepare_input).next(llm_class).next(
        adapt_output
    ).next(prompt_analysis).next(joined_worker).next(accumulate_critique).next(
        improvement_worker
    ).next(
        prepare_input
    )
    graph.set_sink(accumulate_critique, PromptCritique)

    # create two new prompts
    input_tasks = []
    for i, example in enumerate(data[:2]):
        # test whether we can create the input task before we spin up the whole graph
        if i == 0:
            try:
                task = create_input_task(module, task_name, example)
                _ = llm_class.get_full_prompt(task)
            except Exception as e:
                print(
                    f"Error creating input task from {example} - did you provide the right debug log: {e}"
                )
                exit(1)

        prompt_template = llm_class.prompt
        input_tasks.append(
            (
                generation,
                PromptInput(
                    optimization_goal=goal_prompt,
                    prompt_template=prompt_template,
                    id=i,
                ),
            )
        )

    # we are creating one special task for the original prompt
    # this requires that we fake the provenance so that JoinPromptPerformanceOutput can find it
    special_task = ImprovedPrompt(
        prompt_template=llm_class.prompt, comment="Original prompt"
    )
    # the order here matters
    special_task._add_input_provenance(
        PromptInput(
            optimization_goal=goal_prompt, prompt_template=llm_class.prompt, id=0
        )
    )
    special_task._add_worker_provenance(generation)

    input_tasks.append((prepare_input, special_task))

    # Make sure to pick the prompt from the upstream workers and reflect it in the cache key
    inject_prompt_awareness(llm_class)

    graph.run(initial_tasks=input_tasks, run_dashboard=False)

    output = graph.get_output_tasks()
    write_results(llm_class.name, output)


def write_results(class_name: str, output: List[PromptCritique]):
    """
    Writes the results from prompt optimization to a text file and a JSON file.

    Parameters:
        class_name (str): The name of the worker class for which the prompt was optimized.
        output (List[Task]): A list of Task objects containing the prompt data and scores.
    """

    def get_available_filename(base_name, ext):
        """
        Get the next available file name by checking existing files,
        incrementing version number if necessary.
        """
        version = 1
        file_path = Path(f"{base_name}.v{version}.{ext}")
        while file_path.exists():
            version += 1
            file_path = Path(f"{base_name}.v{version}.{ext}")
        return file_path

    for index, task in enumerate(output, start=1):
        # Create the base file name prefixed with the class name and prompt number.
        base_filename = f"{class_name}_prompt_{index}"

        # Create the text file containing the prompt and score.
        text_filename = get_available_filename(base_filename, "txt")
        text_filename.write_text(f"Score: {task.score}\n{task.prompt_template}")

        print(f"Prompt {index} and score written to {text_filename}")

        # Create the JSON file dumping the whole content.
        json_filename = get_available_filename(base_filename, "json")
        json_filename.write_text(json.dumps(task.model_dump(), indent=2))


def sanitize_prompt(original_template: str, prompt_template: str) -> str:
    # Extract all {keywords} from the original template to preserve them
    keywords_pattern = r"{(\S+?)}"
    keywords = set(re.findall(keywords_pattern, original_template))

    # Function to replace single braces with double braces if they are not keywords
    def brace_replacer(match):
        interior = match.group(1)
        if interior in keywords:
            return f"{{{interior}}}"  # Keep single braces for keywords
        return f"{{{{{interior}}}}}"  # Use double braces for literals

    # Replace any standalone {} with double braces except for keywords
    return re.sub(r"{(.*?)}", brace_replacer, prompt_template)


def inject_prompt_awareness(llm_class: LLMTaskWorker):
    original_format_prompt = llm_class.format_prompt
    original_extra_cache_key = llm_class.extra_cache_key

    def new_format_prompt(task: Task) -> str:
        input_prompt: Optional[ImprovedPrompt] = task.find_input_task(ImprovedPrompt)
        if input_prompt is None:
            raise ValueError("No input task found for ImprovedPrompt")
        with llm_class.lock:
            llm_class.prompt = input_prompt.prompt_template
            return original_format_prompt(task)

    def new_extra_cache_key(task: Task) -> str:
        input_prompt: Optional[ImprovedPrompt] = task.find_input_task(ImprovedPrompt)
        if input_prompt is None:
            raise ValueError("No input task found")
        return original_extra_cache_key(task) + input_prompt.prompt_template

    # we need to use brute-force because of pydantic's checks
    llm_class.__dict__["format_prompt"] = new_format_prompt
    llm_class.__dict__["extra_cache_key"] = new_extra_cache_key


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
