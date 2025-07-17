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
        self.publish_work(OutputTask(data=result))
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
from planai import InitialTaskWorker

class DataIngester(InitialTaskWorker):
    output_types = [RawData]
    
    def generate_initial_tasks(self) -> List[Task]:
        # Fetch data from external sources
        data_items = self.fetch_from_database()
        
        return [
            RawData(content=item) 
            for item in data_items
        ]
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
    output_types = [AnalysisResult]
    
    # Optional: customize system prompt
    system_prompt = "You are an expert text analyst"
    
    # Optional: use XML format for complex text
    use_xml = True
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
    output_types = [ProcessedResult]
    
    def consume_work(self, task: InputData):
        # This expensive operation will be cached
        result = self.expensive_computation(task)
        self.publish_work(ProcessedResult(data=result))
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
    output_types = [Analysis]
    
    # Cache responses for 1 hour
    cache_ttl = 3600
```

Perfect for:
- Development (avoid repeated API calls)
- Deterministic LLM responses
- Cost optimization

### JoinedTaskWorker

Aggregates results from multiple tasks:

```python
from planai import JoinedTaskWorker

class ResultAggregator(JoinedTaskWorker):
    join_type: Type[TaskWorker] = DataFetcher
    output_types = [AggregatedResult]
    
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
    output_types = [OutputTask]
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
    output_types = [SuccessResult, ErrorResult, WarningResult]
    
    def consume_work(self, task: InputTask):
        if self.validate(task):
            self.publish_work(SuccessResult(data=task.data))
        elif self.has_warnings(task):
            self.publish_work(WarningResult(
                data=task.data,
                warnings=self.get_warnings(task)
            ))
        else:
            self.publish_work(ErrorResult(
                error="Validation failed",
                input_data=task.data
            ))
```

#### Conditional Processing

```python
class ConditionalWorker(TaskWorker):
    output_types = [ProcessedData, SkippedData]
    
    def should_process(self, task: InputData) -> bool:
        return task.priority > 5
    
    def consume_work(self, task: InputData):
        if self.should_process(task):
            result = self.heavy_processing(task)
            self.publish_work(ProcessedData(data=result))
        else:
            self.publish_work(SkippedData(
                reason="Low priority",
                original=task
            ))
```

#### Stateful Workers

```python
class StatefulWorker(TaskWorker):
    output_types = [AggregatedData]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer = []
        self.buffer_size = 10
    
    def consume_work(self, task: InputData):
        self.buffer.append(task)
        
        if len(self.buffer) >= self.buffer_size:
            # Process batch
            result = self.process_batch(self.buffer)
            self.publish_work(AggregatedData(data=result))
            self.buffer.clear()
```

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