---
title: Task Workers
description: Understanding the different types of TaskWorkers in PlanAI
---

TaskWorkers are the fundamental building blocks of PlanAI workflows. They define how to process input tasks and produce output tasks, forming the nodes in your workflow graph.

## Base TaskWorker

All workers inherit from the base `TaskWorker` class:

```python
from planai import TaskWorker, Task
from typing import List, Type

class MyWorker(TaskWorker):
    output_types: List[Type[Task]] = [OutputTask]

    def consume_work(self, task: InputTask):
        # Process the task
        result = self.process(task)

        # Publish output
        self.publish_work(OutputTask(data=result), input_task=task)
```

### Key Attributes

- **output_types**: List of task types this worker can produce
- **name**: Optional name for the worker (defaults to class name)

### Key Methods

- **consume_work(task)**: Process an input task
- **publish_work(task, input_task)**: Publish an output task with provenance

## Specialized TaskWorker Types

### InitialTaskWorker

Entry point for workflows, introducing external data:

```python
from typing import List, Type
from planai import TaskWorker, Task, Graph

# Need to define RawData and IngestTask types

class DataIngester(TaskWorker):
    output_types: List[Type[Task]] = [RawData]

    def consume_work(self, task: IngestTask):
        # Fetch data from external sources
        data_items = self.fetch_from_database(task.sql_query)

        # Publish each row as a separate task to the graph
        for item in data_items:
            self.publish_work(RawData(content=item), input_task=task)

# Additional set up for the graph needed for downstream processing of RawData
graph = Graph(name='Example')
worker = DataIngester()
graph.add_worker(worker)
graph.run(initial_tasks=[(worker, IngestTask(sql_query='SELECT * FROM table;'))])
```

Use cases:

- Reading from databases
- Fetching from APIs
- Loading files
- Generating synthetic data

### LLMTaskWorker

Integrates Large Language Models into your workflow:

```python
from planai import LLMTaskWorker, llm_from_config

class TextAnalyzer(LLMTaskWorker):
    prompt = "Analyze this text and extract key insights"
    llm_input_type: Type[Task] = TextData
    output_types: List[Type[Task]] = [AnalysisResult]

    # Optional: customize system prompt
    system_prompt = "You are an expert text analyst"

    # Optional: use XML format for complex text
    use_xml = True

# Graph setup omitted
llm = llm_from_config(provider='ollama', model_name='gemma3:4b')
worker = TextAnalyzer(llm=llm)
```

Features:

- Automatic prompt formatting
- Structured output with Pydantic
- Tool/function calling support
- Response streaming
- Token tracking

See the [LLM Integration guide](/features/llm-integration/) for detailed information.

### CachedTaskWorker

Provides caching for expensive operations:

```python
from planai import CachedTaskWorker

class ExpensiveProcessor(CachedTaskWorker):
    output_types: List[Type[Task]] = [ProcessedResult]

    def consume_work(self, task: InputData):
        # This expensive operation will be cached
        result = self.expensive_computation(task)
        self.publish_work(ProcessedResult(data=result), input_task=task)
```

Benefits:

- Automatic result caching
- Configurable cache backends
- TTL support
- Cache key customization

### CachedLLMTaskWorker

Combines LLM functionality with caching:

```python
from planai import CachedLLMTaskWorker

class CachedAnalyzer(CachedLLMTaskWorker):
    prompt = "Provide detailed analysis"
    llm_input_type = DocumentData
    output_types: List[Type[Task]] = [Analysis]
```

This helps during development to save model costs or avoiding repeat processing if a graph fails to run. Any changes
to the prompt, model or input_data will lead to a new cache key.

### JoinedTaskWorker

Aggregates results from multiple tasks:

```python
from planai import JoinedTaskWorker

class ResultAggregator(JoinedTaskWorker):
    join_type: Type[TaskWorker] = DataFetcher
    output_types: List[Type[Task]] = [AggregatedResult]

    def consume_work_joined(self, tasks: List[FetchedData]):
        # All tasks share the same provenance prefix
        combined_data = self.merge_results(tasks)

        self.publish_work(AggregatedResult(
            data=combined_data,
            source_count=len(tasks)
        ))
```

Use cases:

- Combining parallel search results
- Aggregating batch processing
- Consolidating multi-source data

### SubGraphWorker

Encapsulates an entire graph as a single worker:

```python
from planai import Graph, SubGraphWorker

# Create a subgraph
sub_graph = Graph(name="DataPipeline")
fetcher = DataFetcher()
processor = DataProcessor()
sub_graph.add_workers(fetcher, processor)
sub_graph.set_dependency(fetcher, processor)
sub_graph.set_entry(fetcher)
sub_graph.set_exit(processor)

# Use as a single worker
pipeline_worker = SubGraphWorker(
    name="DataPipeline",
    graph=sub_graph
)

# Add to main graph
main_graph = Graph(name="MainWorkflow")
main_graph.add_worker(pipeline_worker)
```

Benefits:

- Modular workflow design
- Reusable components
- Simplified testing
- Clear abstraction boundaries

## Creating Custom TaskWorkers

### Basic Pattern

```python
class CustomWorker(TaskWorker):
    # Define configuration
    output_types: List[Type[Task]] = [OutputTask]
    config_param: str = "default_value"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize resources
        self.connection = self.setup_connection()

    def consume_work(self, task: InputTask):
        try:
            # Process task
            result = self.process_data(task)

            # Publish result with provenance
            self.publish_work(
                OutputTask(data=result),
                input_task=task
            )
        except Exception as e:
            # Handle errors appropriately
            self.handle_error(task, e)
```

### Best Practices

1. **Type Safety**: Always specify output_types
2. **Error Handling**: Implement robust error handling
3. **Resource Management**: Clean up resources properly
4. **Logging**: Use appropriate logging levels
5. **Documentation**: Document expected inputs/outputs

### Advanced Features

#### Multiple Output Types

```python
class MultiOutputWorker(TaskWorker):
    output_types: List[Type[Task]] = [SuccessResult, ErrorResult, WarningResult]

    def consume_work(self, task: InputTask):
        if self.validate(task):
            self.publish_work(SuccessResult(data=task.data), input_task=task)
        elif self.has_warnings(task):
            self.publish_work(WarningResult(
                data=task.data,
                warnings=self.get_warnings(task)
            ), input_task=task)
        else:
            self.publish_work(ErrorResult(
                error="Validation failed",
                input_data=task.data
            ), input_task=task)
```

#### Conditional Processing

```python
class ConditionalWorker(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData, SkippedData]

    def should_process(self, task: InputData) -> bool:
        return task.priority > 5

    def consume_work(self, task: InputData):
        if self.should_process(task):
            result = self.heavy_processing(task)
            self.publish_work(ProcessedData(data=result), input_task=task)
        else:
            self.publish_work(SkippedData(
                reason="Low priority",
                original=task
            ), input_task=task)
```

#### Stateful Workers

Workers can maintain state across multiple invocations using `get_worker_state()`. This is particularly useful for circular dependencies or iterative processing:

**Example 1: Buffering Data**

```python
class BufferingWorker(TaskWorker):
    output_types: List[Type[Task]] = [AggregatedData]

    def consume_work(self, task: InputData):
        # Get state scoped to this task's provenance prefix
        state = self.get_worker_state(task.prefix(1))

        if "buffer" not in state:
            state["buffer"] = []

        state["buffer"].append(task)

        if len(state["buffer"]) >= 5:
            # Flush buffer for batch processing
            result = self.process_batch(state["buffer"])
            self.publish_work(
                AggregatedData(data=result),
                input_task=task
            )
            state["buffer"].clear()
```

**Example 2: Iterative Processing with Retry Counter**

```python
class RetryWorker(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData, FailedData]
    max_retries: int = 3

    def consume_work(self, task: InputData):
        # For circular dependencies, we need to use the original input task's provenance
        # because task.provenance() grows longer with each iteration. Using find_input_tasks
        # to get the first input task from the upstream worker ensures consistent state lookup.
        input_tasks = task.find_input_tasks(DataFetcher)
        if not input_tasks:
            raise ValueError("DataFetcher input task not found")

        prefix = input_tasks[0].provenance()
        state = self.get_worker_state(prefix)

        if "retry_count" not in state:
            state["retry_count"] = 0

        try:
            result = self.risky_operation(task)
            self.publish_work(
                ProcessedData(data=result),
                input_task=task
            )
        except Exception as e:
            state["retry_count"] += 1

            if state["retry_count"] < self.max_retries:
                # Retry by publishing back to an upstream worker
                self.publish_work(
                    InputData(data=task.data, retry=True),
                    input_task=task
                )
            else:
                # Max retries reached, publish failure
                self.publish_work(
                    FailedData(
                        data=task.data,
                        error=str(e),
                        retries=state["retry_count"]
                    ),
                    input_task=task
                )
```

**Important Notes:**

- State is automatically cleaned up when the provenance prefix completes
- State is scoped to a specific prefix, ensuring isolation between different task chains
- Useful for implementing retry logic and iterative refinement

## Worker Lifecycle

1. **Initialization**: Worker is created and added to graph
2. **Setup**: Resources are initialized
3. **Processing**: Worker consumes tasks from input queue
4. **Publishing**: Results are published to output queues
5. **Cleanup**: Resources are released when graph completes

## Performance Considerations

- **Concurrency**: Workers can process tasks in parallel
- **Batching**: Group operations for efficiency
- **Caching**: Use CachedTaskWorker for expensive operations
- **Resource Pooling**: Share expensive resources between tasks

## Next Steps

- Explore [LLM Integration](/features/llm-integration/) for AI-powered workers
- Learn about [Caching Strategies](/features/caching/)
- Understand [Subgraph Patterns](/features/subgraphs/)
- See [Examples](https://github.com/provos/planai/tree/main/examples) for real implementations
