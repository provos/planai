# Prompt Optimization in PlanAI

## Introduction

Manually crafting and iterating on prompts for Large Language Models (LLMs) can be a time-consuming process that doesn't always yield optimal results. PlanAI introduces an automatic prompt optimization feature to address this challenge, leveraging the power of more advanced LLMs to improve prompt effectiveness.

## The Optimize-Prompt Tool

PlanAI's `optimize-prompt` tool automates the process of refining prompts for LLMTaskWorkers. It uses a more powerful LLM to assess the quality of outputs from a given prompt, suggest improvements, create new prompts, and iterate on real data until an improved prompt is developed.

### Key Features

1. **Automated Iteration**: The tool runs multiple optimization cycles (default is 3, customizable with `--num-iterations`) to progressively improve the prompt.

2. **Real Data Integration**: Utilizes debug logs containing actual input-output pairs from production runs to test and validate prompt improvements.

3. **Dynamic Class Loading**: Leverages PlanAI's use of Pydantic to dynamically load and use real production classes in the optimization workflow.

4. **Scoring Mechanism**: Employs an LLM with a scoring prompt to evaluate the accuracy and effectiveness of each iteration of the optimized prompt.

5. **Adaptability**: Designed to be agnostic to specific use cases, making it applicable to various LLM tasks.

## Preparation

Before using the optimize-prompt tool, you need to generate debug logs with real production data:

1. In your application, set `debug_mode=True` for all LLMTaskWorker classes that may need prompt optimization.
2. Run your application with a representative subset of real production data.
3. This will generate debug logs containing the input-output pairs needed for optimization.

Example of enabling debug mode:

```python
class YourLLMTaskWorker(LLMTaskWorker):
    debug_mode: bool = True
    # ... rest of your class definition
```

## Usage

To use the optimize-prompt tool, you can run a command like this:

```bash
planai --llm-provider openai --llm-model gpt-4o-mini --llm-reason-model gpt-4o optimize-prompt --python-file your_app.py --class-name YourLLMTaskWorker --search-path . --debug-log debug/YourLLMTaskWorker.json --goal-prompt "Your optimization goal here"
```

Here is an example of improving the DiffAnalyzer prompt of the releasenotes.py example application:
```bash
poetry run planai --llm-provider openai --llm-model gpt-4o --llm-reason-model o3-mini optimize-prompt --python-file releasenotes.py --class-name DiffAnalyzer --debug-log debug/DiffAnalyzer.json --search-path . --llm-opt-provider openai --llm-opt-model gpt-4o --goal-prompt "We need a change list description that accurately captures the change and can be understood in isolation. it should be clear about the component being changed and the main purpose of the code changes."
```

### Parameters Explained

- `--llm-provider`, `--llm-model`, `--llm-reason-model`: Specify the LLM provider and models to use for optimization.
- `--python-file`: The Python file containing your LLMTaskWorker class.
- `--class-name`: The name of the LLMTaskWorker class to optimize.
- `--search-path`: The path to search for additional modules.
- `--debug-log`: Path to the debug log file containing real input-output data.
- `--goal-prompt`: The optimization goal for the prompt.

### Example LLMTaskWorker Class

Here's a simple example of an LLMTaskWorker class with `debug_mode` set to True:

```python
from planai import LLMTaskWorker, Task
from pydantic import Field
from typing import List, Type

class SummaryInput(Task):
    text: str = Field(description="The text to summarize")

class Summary(Task):
    summary: str = Field(description="The generated summary")

class TextSummarizer(LLMTaskWorker):
    output_types: List[Type[Task]] = [Summary]
    debug_mode: bool = True
    prompt: str = """
    Summarize the following text in a concise manner:

    {text}

    Summary:
    """

    def consume_work(self, task: SummaryInput):
        return super().consume_work(task)

# Usage in your application
summarizer = TextSummarizer(llm=your_llm_instance)
```

When this `TextSummarizer` class is used in your application with `debug_mode=True`, it will generate debug logs that can be used with the `optimize-prompt` tool.

## How It Works

1. **Data Extraction**: The tool extracts input-output pairs from the provided debug log.

2. **Dynamic Class Loading**: It loads the specified LLMTaskWorker class and any associated Pydantic models.

3. **Initial Assessment**: The current prompt is evaluated using the provided goal and real data.

4. **Iteration**:
   - A more powerful LLM suggests improvements to the prompt.
   - The new prompt is tested against real data.
   - Results are scored for accuracy and effectiveness.
   - This process repeats for the specified number of iterations.

5. **Output**: The tool provides the optimized prompts along with performance metrics.

## Output

After running the optimization process, the tool will generate output files:

1. **Text Files**: The top three best-performing prompts are saved as separate .txt files. Each file includes the prompt text and its score.

   Example filename: `YourLLMTaskWorker_prompt_1.v2.txt`

2. **JSON Files**: Corresponding to each text file, a JSON file is created containing detailed metadata about the prompt, including critiques and the overall score.

   Example filename: `YourLLMTaskWorker_prompt_1.v2.json`

These files allow you to review the optimized prompts, their scores, and the reasoning behind the improvements.

## Benefits

- **Time-Saving**: Automates a traditionally manual and time-consuming process.
- **Data-Driven**: Uses real production data for optimization, ensuring relevance.
- **Continuous Improvement**: Allows for regular prompt refinement as new data becomes available.

## Limitations and Considerations

- Requires a comprehensive debug log with diverse examples for effective optimization.
- The tool's effectiveness may vary depending on the complexity of the task and the quality of the initial prompt.
- All Task classes in the input provenance chain must be available, or the tool will fail.

## Conclusion

The prompt optimization feature in PlanAI offers a powerful, data-driven approach to improving LLM task performance. By automating the refinement process and leveraging real production data, it enables developers to continually enhance their AI-driven applications with minimal manual intervention.