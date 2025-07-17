---
title: Quick Start
description: Build your first PlanAI workflow in minutes
---

This guide will walk you through creating your first PlanAI workflow. We'll build a simple data processing pipeline that combines traditional computation with AI analysis.

## Basic Workflow

Let's start with a simple example that processes data through multiple stages:

```python
from typing import List, Type

from planai import Graph, Task, TaskWorker


# Define our data models using Pydantic
class RawData(Task):
    content: str


class ProcessedData(Task):
    processed_content: str
    word_count: int


class AnalysisResult(Task):
    summary: str
    sentiment: str


# Create a simple data processor
class DataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]

    def consume_work(self, task: RawData):
        # Process the raw data
        cleaned = task.content.strip().lower()
        word_count = len(cleaned.split())

        # Publish the processed data
        self.publish_work(
            ProcessedData(processed_content=cleaned, word_count=word_count),
            input_task=task,
        )


class DataPrinter(TaskWorker):
    def consume_work(self, task: ProcessedData):
        print(task.processed_content)


# Create the workflow graph
graph = Graph(name="Simple Data Pipeline")
processor = DataProcessor()
printer = DataPrinter()
graph.add_workers(processor, printer)
graph.set_dependency(processor, printer)


# Run the workflow
initial_data = RawData(content="  Hello World! This is PlanAI.  ")
graph.run(initial_tasks=[(processor, initial_data)])

# Will print: Hello World! This is PlanAI. 
```

## Adding AI Capabilities

Now let's enhance our workflow by adding an AI-powered analyzer:

```python
from planai import LLMTaskWorker, llm_from_config


# Define an AI analyzer
class AIAnalyzer(LLMTaskWorker):
    prompt: str = """Analyze the following text and provide:
    1. A brief summary (one sentence)
    2. The overall sentiment (positive, negative, or neutral)
    
    Format your response as JSON with 'summary' and 'sentiment' fields."""

    llm_input_type: Type[Task] = ProcessedData
    output_types: List[Type[Task]] = [AnalysisResult]

class ResultPrinter(TaskWorker):
    def consume_work(self, task: AnalysisResult):
        self.print(task.summary)
        self.print(task.sentiment)

# Create the enhanced workflow
graph = Graph(name="AI-Enhanced Pipeline")

# Initialize workers
processor = DataProcessor()
analyzer = AIAnalyzer(
    llm=llm_from_config(
        provider="openai",
        model_name="gpt-4o",
    )
)
result_printer = ResultPrinter()

# Build the graph
graph.add_workers(processor, analyzer, result_printer)
# analyzer depends on processor and result_printer depends on analyzer
graph.set_dependency(processor, analyzer).next(result_printer)  

# Run with monitoring dashboard
initial_data = RawData(content="PlanAI makes it easy to build AI workflows!")
graph.run(
    initial_tasks=[(processor, initial_data)],
    run_dashboard=False,  # Set to True to open monitoring at http://localhost:5000
)

# Logs should show: 
# The text promotes PlanAI's ability to simplify the creation of AI workflows.
# positive
```

## Working with Multiple Inputs

PlanAI excels at handling workflows with multiple data sources:

```python
from typing import List, Type

from planai import Graph, InitialTaskWorker, JoinedTaskWorker, Task, TaskWorker


# Define a data source
class DataSource(Task):
    source_id: str
    data: str


class ProcessedData(Task):
    processed_data: str


class CombinedAnalysis(Task):
    sources_analyzed: int
    combined_summary: str


# Worker to process data
class DataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]

    def consume_work(self, task: DataSource):
        # We'll publish multiple tasks here
        for i in range(3):
            self.publish_work(
                ProcessedData(processed_data=f"{task.data} - processed {i}"),
                input_task=task,
            )


# Worker to join results
class ResultAggregator(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    output_types: List[Type[Task]] = [CombinedAnalysis]

    def consume_work_joined(self, tasks: List[ProcessedData]):
        combined_summary = f"Analyzed {len(tasks)} sources from {tasks[0].prefix(1)}"

        self.publish_work(
            CombinedAnalysis(
                sources_analyzed=len(tasks), combined_summary=combined_summary
            ),
            input_task=tasks[0],
        )


# Class DataPrinter
class DataPrinter(TaskWorker):
    def consume_work(self, task: CombinedAnalysis):
        self.print(task.combined_summary)


# Build the complete workflow
graph = Graph(name="Multi-Source Analysis")
processor = DataProcessor()
aggregator = ResultAggregator()
printer = DataPrinter()

graph.add_workers(processor, aggregator, printer)
graph.set_dependency(processor, aggregator).next(printer)

# Run the workflow
initial_data = [
    DataSource(source_id="source1", data="First dataset"),
    DataSource(source_id="source2", data="Second dataset"),
    DataSource(source_id="source3", data="Third dataset"),
]
graph.run(initial_tasks=[(processor, element) for element in initial_data])

# Will print:
# Analyzed 3 sources from (('InitialTaskWorker', 1),)
# Analyzed 3 sources from (('InitialTaskWorker', 2),)
# Analyzed 3 sources from (('InitialTaskWorker', 3),)
```

## Using Caching

For expensive operations (like LLM calls), use caching to improve performance:

```python
from planai import CachedLLMTaskWorker

class CachedAnalyzer(CachedLLMTaskWorker):
    prompt: str = "Analyze this text and provide insights"
    llm_input_type: Type[Task] = ProcessedData
    output_types: List[Type[Task]] = [AnalysisResult]

# The cached analyzer will automatically cache LLM responses
cached_analyzer = CachedAnalyzer(
    llm=llm_from_config("openai", "gpt-4"),
    cache_dir="./cache"  # Optional: specify cache location
)
```

## Next Steps

Congratulations! You've learned the basics of PlanAI. Here's what to explore next:

- **[Basic Usage Guide](/guide/usage/)**: Deep dive into all PlanAI features
- **[Prompts Guide](/guide/prompts/)**: Learn advanced prompt engineering techniques
- **[Task Workers](/features/taskworkers/)**: Explore different types of workers
- **[Examples Repository](https://github.com/provos/planai/tree/main/examples)**: See real-world implementations

### Example Projects

Check out these complete examples in the repository:

- **Textbook Q&A Generator**: Automatically generate questions and answers from textbook content
- **Deep Research Assistant**: Multi-stage research workflow with web search and analysis
- **Social Media Analyzer**: Analyze and summarize social media content

Remember to set your API keys as environment variables when working with LLMs:

```bash
export OPENAI_API_KEY="your-api-key"
```

Alternatively, PlanAI will automatically load environment variables from ```.env.local```.