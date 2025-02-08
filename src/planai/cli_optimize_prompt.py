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
PlanAI Prompt Optimization Module

This module implements the 'optimize-prompt' subcommand for the PlanAI tool, which automates the process of refining
prompts for Large Language Models (LLMs). It leverages more advanced LLMs to improve prompt effectiveness through
iterative optimization.

Key Features:
1. Automated Iteration: Runs multiple optimization cycles to progressively improve the prompt.
2. Real Data Integration: Utilizes debug logs with actual input-output pairs from production runs.
3. Dynamic Class Loading: Leverages PlanAI's use of Pydantic to dynamically load and use real production classes.
4. Scoring Mechanism: Employs an LLM with a scoring prompt to evaluate the accuracy and effectiveness of each iteration.
5. Adaptability: Designed to be agnostic to specific use cases, applicable to various LLM tasks.

The module includes several worker classes that form a graph-based optimization pipeline:
- PromptGenerationWorker: Generates improved prompts based on the optimization goal.
- PrepareInputWorker: Prepares input data for optimization from reference data.
- PromptPerformanceWorker: Analyzes the performance of prompts against the optimization goal.
- JoinPromptPerformanceOutput: Combines multiple performance outputs.
- AccumulateCritiqueOutput: Accumulates and ranks prompt critiques over multiple iterations.
- PromptImprovementWorker: Creates an improved prompt based on accumulated critiques.

Usage:
This module is typically invoked through the PlanAI CLI using the 'optimize-prompt' subcommand.
It requires specifying the target Python file, class name, debug log, and optimization goal.

Example:
    PYTHONPATH=. planai optimize-prompt --python-file your_app.py --class-name YourLLMTaskWorker \
        --debug-log debug/YourLLMTaskWorker.json --goal-prompt "Your optimization goal here"

The module outputs optimized prompts as text files and corresponding metadata as JSON files.

Note: This tool requires a comprehensive debug log with diverse examples for effective optimization.
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from operator import attrgetter
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Type

from llm_interface import LLMInterface
from pydantic import Field, PrivateAttr

from planai import (
    CachedLLMTaskWorker,
    Graph,
    InitialTaskWorker,
    JoinedTaskWorker,
    LLMTaskWorker,
    Task,
    TaskWorker,
    llm_from_config,
)
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


class PromptInputs(Task):
    inputs: List[PromptInput] = Field(description="The list of prompt inputs")


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


class PromptDistributor(TaskWorker):
    generator: TaskWorker = Field(description="The worker that generates the prompts")
    goal_prompt: str = Field(description="The goal of the optimization")
    original_prompt: str = Field(description="The original prompt to optimize")
    output_types: List[Type[Task]] = [PromptInput, ImprovedPrompt]

    def consume_work(self, task: PromptInputs):
        for input_task in task.inputs:
            self.publish_work(input_task, input_task=task)

        # we are creating one special task for the original prompt
        # this requires that we fake the provenance so that JoinPromptPerformanceOutput can find it
        special_task = ImprovedPrompt(
            prompt_template=self.original_prompt, comment="Original prompt"
        )

        fake_input = task.model_copy()

        # the order here matters
        fake_input._add_input_provenance(
            PromptInput(
                optimization_goal=self.goal_prompt,
                prompt_template=self.original_prompt,
                id=0,
            )
        )
        # we need to add the InitialTaskWorker as the input provenance
        fake_input._provenance = [task.prefix(1)[0]] + fake_input._provenance

        fake_input._add_worker_provenance(self.generator)
        self.publish_work(special_task, input_task=fake_input)


class PromptGenerationWorker(CachedLLMTaskWorker):
    """
    A worker class responsible for generating improved prompts based on an optimization goal.

    This class uses a Large Language Model (LLM) to analyze the provided prompt template
    and suggest improvements to better meet the specified optimization goal.

    Attributes:
        output_types (List[Type[Task]]): List containing ImprovedPrompt as the output type.
        prompt (str): The instruction prompt for the LLM to generate improved prompts.

    Methods:
        consume_work(task: PromptInput) -> ImprovedPrompt:
            Processes the input task and generates an improved prompt.

        post_process(response: ImprovedPrompt, input_task: PromptInput) -> ImprovedPrompt:
            Sanitizes the generated prompt to preserve required keywords.

    The worker ensures that:
    - Existing {{keywords}} in the original prompt are maintained for .format() expansion.
    - Literal curly braces in the prompt text are properly escaped using double braces.
    - The focus is on improving the structure and approach of the prompt rather than specific subject matter.
    - The generated prompt is a complete, standalone prompt that can be used as-is.

    The output is a JSON object with 'prompt_template' and 'comment' fields, providing
    the improved prompt and an explanation of the improvements made.
    """

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
    """
    A worker class responsible for preparing input data for prompt optimization.

    This class takes reference data and transforms it into appropriate Pydantic task instances
    that can be consumed by the target LLMTaskWorker class for which we are optimizing the prompt.

    Attributes:
        random_seed (int): The seed used for deterministic random selection of reference data.
        num_examples (int): The number of examples to produce in each iteration.
        _reference_data (List[Dict[str, Any]]): The raw reference data from which to create tasks.
        _module (Any): The module containing the target LLMTaskWorker class.

    Methods:
        __init__(module: Any, task_name: str, reference_data: List[Dict[str, Any]], **data):
            Initializes the worker with the necessary data and configuration.

        consume_work(task: ImprovedPrompt) -> None:
            Processes an improved prompt by selecting reference data and creating appropriate input tasks.

    The worker performs several key functions:
    1. It dynamically loads the appropriate Pydantic task class based on the target LLMTaskWorker.
    2. It selects a subset of the reference data using a deterministic random process.
    3. It transforms the selected reference data into instances of the appropriate Pydantic task class.
    4. It ensures that the created tasks maintain the necessary provenance information for the optimization process.

    This "massaging" of reference data is crucial because it allows the optimization process to use
    real-world data in a format that exactly matches what the target LLMTaskWorker expects. This ensures
    that the prompt optimization is performed under conditions that closely mimic actual usage scenarios.

    The worker uses a combination of the improved prompt's hash and its own random seed to ensure
    deterministic but varied selection of reference data across different optimization iterations.
    """

    random_seed: int = Field(
        42, description="The random seed to use for selecting reference data"
    )
    num_examples: int = Field(5, description="The number of examples to produce")
    _reference_data: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
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
    """
    A worker class that aggregates multiple PromptPerformanceOutput tasks into a single CombinedPromptCritique.

    This class is responsible for collecting the individual performance evaluations of a prompt
    across multiple examples and combining them into a single, aggregated critique and score.
    """

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


class AccumulateCritiqueOutput(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    iterations: int = Field(3, description="The number of iterations to run")
    output_types: List[Type[Task]] = [MultipleCombinedPromptCritique, PromptCritique]
    _state: Dict[str, PromptCritique] = PrivateAttr(default_factory=dict)
    _count: int = PrivateAttr(default=0)

    def consume_work_joined(self, tasks: List[CombinedPromptCritique]):
        if not tasks:
            self.print("No tasks received - this is a serious error")
            return

        for task in tasks:
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

        with self.lock:
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
    """
    A worker class that generates improved prompts based on aggregated critiques and validates them.

    This class uses a more powerful LLM to create an improved prompt template based on the
    critiques and scores of previous iterations. Crucially, it also validates the generated
    prompt by attempting to instantiate it with the target LLMTaskWorker class.

    The validation process involves:
    1. Temporarily injecting the new prompt into the target LLMTaskWorker.
    2. Attempting to generate a full prompt using real input data.
    3. Catching any errors that occur during this process.

    If the validation fails, the error is captured and is fed back to the LLM to generate a
    prompts works correctly for the target LLMTaskWorker class.
    """

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
    llm_fast: LLMInterface,
    llm_reason: LLMInterface,
    args: argparse.Namespace,
    debug: bool = False,
):
    """
    Orchestrates the prompt optimization process for a given LLMTaskWorker class.

    This function sets up and executes a multi-step, iterative process to optimize the prompt
    of a specified LLMTaskWorker class. It uses a combination of faster and more advanced LLMs
    to generate, evaluate, and improve prompts based on real-world data and specified goals.

    Parameters:
        llm_fast (LLMInterface): A faster LLM used for initial prompt evaluations.
        llm_reason (LLMInterface): A more advanced LLM used for in-depth analysis and improvements.
        args (argparse.Namespace): Command-line arguments specifying optimization parameters.

    The optimization process follows these main steps:
    1. Load the target LLMTaskWorker class and its associated debug log data.
    2. Set up a graph of specialized workers for different aspects of optimization:
       - PromptGenerationWorker: Generates new prompt variations.
       - PrepareInputWorker: Prepares real-world data for testing.
       - The target LLMTaskWorker: Used to test prompts with real data.
       - OutputAdapter: Adapts LLMTaskWorker output for analysis.
       - PromptPerformanceWorker: Evaluates prompt performance.
       - JoinPromptPerformanceOutput: Aggregates performance data.
       - AccumulateCritiqueOutput: Accumulates and ranks critiques over iterations.
       - PromptImprovementWorker: Creates improved prompts based on critiques.
    3. Execute multiple iterations of this process, each time:
       - Generating new prompts or improving existing ones.
       - Testing these prompts against real-world data.
       - Evaluating and scoring the performance of each prompt.
       - Accumulating critiques and suggestions for improvement.
    4. Output the top-performing prompts along with their scores and critiques.

    The function handles loading necessary modules, setting up the optimization graph,
    injecting prompt awareness into the target LLMTaskWorker, and managing the flow of
    tasks through the optimization process.

    Output:
    - Writes the top-performing prompts to text files.
    - Saves detailed metadata about each top prompt to JSON files.

    Note:
    This process requires a well-prepared debug log with diverse, representative examples
    to ensure effective optimization across various use cases.
    """
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
                llm_opt_provider = config.get("llm_opt_provider")
                llm_opt_model = config.get("llm_opt_model")
                llm_opt_max_tokens = config.get("llm_opt_max_tokens", 4096)
                output_dir = config.get("output_dir", ".")

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
        llm_opt_model = args.llm_opt_model
        llm_opt_provider = args.llm_opt_provider
        llm_opt_max_tokens = 4096
        output_dir = args.output_dir if args.output_dir else "."

    # Write out configuration if requested
    if args.output_config:
        config_data = {
            "python_file": python_file,
            "class_name": class_name,
            "debug_log": debug_log,
            "goal_prompt": goal_prompt,
            "search_path": search_path,
            "num_iterations": num_iterations,
            "llm_opt_provider": llm_opt_provider,
            "llm_opt_model": llm_opt_model,
            "llm_opt_max_tokens": llm_opt_max_tokens,
            "output_dir": output_dir,
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

    # Create the LLM interface to be used to generation in the class that we are optimizing
    llm_for_optimization = llm_fast
    if llm_opt_provider and llm_opt_model:
        llm_for_optimization = llm_from_config(
            provider=llm_opt_provider,
            model_name=llm_opt_model,
            max_tokens=llm_opt_max_tokens,
        )

    # Then, load and instantiate the class
    llm_class: Optional[LLMTaskWorker] = instantiate_llm_class_from_module(
        module=module, class_name=class_name, llm=llm_for_optimization
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

    setup_logging(level=logging.DEBUG if debug else logging.INFO)

    # strict needs to be false because we are injecting false provenance
    graph = Graph(name="Prompt Optimization Graph", strict=False)
    generation = PromptGenerationWorker(llm=llm_reason)

    # we need a reference to generation so that we can fake input provenance
    distributor = PromptDistributor(
        generator=generation, goal_prompt=goal_prompt, original_prompt=llm_class.prompt
    )

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
        distributor,
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
    graph.set_dependency(distributor, generation).next(prepare_input).next(
        llm_class
    ).next(adapt_output).next(prompt_analysis).next(joined_worker).next(
        accumulate_critique
    ).next(
        improvement_worker
    ).next(
        prepare_input
    )
    # this allows the original prompt to bypass the new prompt generation
    graph.set_dependency(distributor, prepare_input)
    graph.set_sink(accumulate_critique, PromptCritique)

    # inject a mock cache
    if debug:
        from planai.testing.helpers import MockCache, inject_mock_cache

        inject_mock_cache(graph, MockCache())

    # create two new prompts
    prompts = []
    for i, example in enumerate(data[:2]):
        # test whether we can create the input task before we spin up the whole graph
        if i == 0:
            try:
                task = create_input_task(module, task_name, example)
            except Exception as e:
                print(
                    f"Error creating input task from {str(example)[:100]}... - did you provide the right debug log: {e}"
                )
                exit(1)
            try:
                _ = llm_class.get_full_prompt(task)
            except Exception as e:
                print(
                    f"Error creating full prompt from {task.name} - did you provide the right debug log: {e}"
                )
                exit(1)

        prompt_template = llm_class.prompt
        prompts.append(
            PromptInput(
                optimization_goal=goal_prompt,
                prompt_template=prompt_template,
                id=i,
            )
        )

    input_tasks: List[Tuple[TaskWorker, Task]] = [
        (distributor, PromptInputs(inputs=prompts))
    ]

    # Make sure to pick the prompt from the upstream workers and reflect it in the cache key
    inject_prompt_awareness(llm_class)

    graph.run(
        initial_tasks=input_tasks,
        run_dashboard=False,
        display_terminal=not debug,
    )

    output = graph.get_output_tasks()
    write_results(llm_class.name, output, output_dir=output_dir)


def write_results(
    class_name: str, output: List[PromptCritique], output_dir: str = "."
) -> None:
    """
    Writes the results from prompt optimization to a text file and a JSON file.

    Parameters:
        class_name (str): The name of the worker class for which the prompt was optimized.
        output (List[Task]): A list of Task objects containing the prompt data and scores.
    """

    # when we run multiple iterations, we will get a list of PromptCritique objects
    output = sorted(output, key=attrgetter("score"), reverse=True)

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
        base_filename = Path(output_dir) / f"{class_name}_prompt_{index}"

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

    def new_format_prompt(task: Task) -> str:
        input_prompt: Optional[ImprovedPrompt] = task.find_input_task(ImprovedPrompt)
        if input_prompt is None:
            raise ValueError("No input task found for ImprovedPrompt")
        with llm_class.lock:
            llm_class.prompt = input_prompt.prompt_template
            return original_format_prompt(task)

    if hasattr(llm_class, "extra_cache_key"):
        original_extra_cache_key = llm_class.extra_cache_key

        def new_extra_cache_key(task: Task) -> str:
            input_prompt: Optional[ImprovedPrompt] = task.find_input_task(
                ImprovedPrompt
            )
            if input_prompt is None:
                raise ValueError("No input task found")
            return original_extra_cache_key(task) + input_prompt.prompt_template

        llm_class.__dict__["extra_cache_key"] = new_extra_cache_key

    # we need to use brute-force because of pydantic's checks
    llm_class.__dict__["format_prompt"] = new_format_prompt


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
