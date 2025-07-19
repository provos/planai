---
title: Task
description: API reference for the Task base class
---

The `Task` class is the fundamental data unit in PlanAI. All data flowing through workflows must be represented as Task instances. Tasks carry both public data (validated by Pydantic) and private execution state, while maintaining a complete provenance chain of their processing history.

## Class Definition

```python
from planai import Task
from pydantic import BaseModel

class Task(BaseModel):
    """Base class for all tasks in PlanAI workflows"""
    pass
```

Task inherits from Pydantic's `BaseModel`, providing automatic validation, serialization, and type safety.

## Core Properties

### name
```python
@property
def name(self) -> str:
    """Returns the class name of the task"""
```

## Creating Tasks

Define custom tasks by inheriting from Task:

```python
from planai import Task
from typing import List, Optional
from datetime import datetime
from pydantic import Field

class UserQuery(Task):
    query: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ProcessedData(Task):
    original_query: str
    results: List[dict]
    processing_time: float
    metadata: Optional[dict] = None

class SearchQuery(Task):
    query: str = Field(..., description="A search query")
    metadata: Optional[str] = Field(None, description="Additional context")
```

## Task Lifecycle Methods

### copy_public

Create a safe copy of a task without private attributes:

```python
def copy_public(self, deep: bool = False) -> Task:
    """
    Creates a copy excluding private attributes. Safer than model_copy()
    for creating new tasks from existing ones.
    
    Args:
        deep: Whether to perform a deep copy of the public fields
        
    Returns:
        A new Task instance without private attributes
    """
```

Example usage in workflows:
```python
class DataProcessor(TaskWorker):
    def consume_work(self, task: ProcessedData):
        # Create a safe copy when publishing to multiple consumers
        clean_task = task.copy_public()
        self.publish_work(clean_task, input_task=task)
```

## Provenance Methods

Tasks maintain a complete history of their processing through the provenance system:

### find_input_task

Find a specific task type in the provenance chain:

```python
def find_input_task(self, task_type: Type[Task]) -> Optional[Task]:
    """
    Search provenance chain for a specific task type
    
    Args:
        task_type: The type of task to search for
        
    Returns:
        The most recent matching task or None
    """
```

Real-world example:
```python
class SearchSummarizer(LLMTaskWorker):
    def format_prompt(self, input_task: ConsolidatedPages) -> str:
        # Find the original plan from upstream
        plan: Plan = input_task.find_input_task(Plan)
        if plan is None:
            raise ValueError("Missing Plan task in provenance")
            
        # Find the search query that led to these pages
        query: SearchQuery = input_task.find_input_task(SearchQuery)
        
        return self.prompt.format(plan=plan.response, phase=query.metadata)
```

### find_input_tasks

Find all tasks of a specific type:

```python
def find_input_tasks(self, task_class: Type[Task]) -> List[Task]:
    """
    Find all input tasks of the specified class
    
    Args:
        task_class: The class of the tasks to find
        
    Returns:
        List of tasks of the specified class
    """
```

### previous_input_task

Get the immediately previous task:

```python
def previous_input_task(self) -> Optional[Task]:
    """
    Get the task that directly led to this one
    
    Returns:
        Previous task or None
    """
```

### prefix

Get a prefix of the provenance chain:

```python
def prefix(self, length: int) -> ProvenanceChain:
    """
    Get a prefix of specified length from task's provenance chain
    
    Args:
        length: The desired length of the prefix to extract
        
    Returns:
        ProvenanceChain tuple containing first 'length' elements
    """
```

### prefix_for_input_task

Get provenance prefix for joining operations:

```python
def prefix_for_input_task(self, worker_type: Type[TaskWorker]) -> Optional[ProvenanceChain]:
    """
    Get provenance prefix for a specific worker type
    
    Args:
        worker_type: Type of worker to find prefix for
        
    Returns:
        Provenance prefix string or None
    """
```

Example in joined workers:
```python
class AnalysisJoiner(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    
    def consume_work_joined(self, tasks: List[PhaseAnalysis]):
        # Tasks are automatically grouped by their provenance prefix
        # for the join_type worker
        combined = PhaseAnalyses(analyses=tasks)
        self.publish_work(combined, tasks[0])
```

## Private State Management

Tasks can carry private state that persists across the processing pipeline but isn't included in serialization:

### add_private_state

```python
def add_private_state(self, key: str, value: Any) -> None:
    """Store private state data"""
```

### get_private_state

This is very advanced functionality that most users of PlanAI should not
make use of. Most use cases can be satisfied by retrieving provenance task data vis ```find_input_task```. See also how to manage state in [TaskWorker](/api/taskworker/).

```python
def get_private_state(self, key: str) -> Any:
    """Retrieve and remove private state data"""
```

Example usage:
```python
class ExampleWorker(TaskWorker):
    def consume_work(self, task: Task):
        # Store original task for later retrieval
        new_task.add_private_state("metadata", task)
        # Process through subgraph...
        
class DownstreamWorker(TaskWorker):
    def consume_work(self, task: Task):
        # Retrieve original task
        original = task.get_private_state("metadata")
        # Continue processing...
```

### inject_input

Manually inject an input task into provenance. Should only be used when you know exactly what you are doing:

```python
def inject_input(self, input_task: Task) -> Task:
    """
    Inject an input task into the provenance chain
    
    Args:
        input_task: The input task to inject
        
    Returns:
        The task with injected input provenance
    """
```

## Utility Methods

### is_type

Type checking method:

```python
def is_type(self, task_class: Type[Task]) -> bool:
    """
    Check if this task is of the specified task class type
    
    Args:
        task_class: The task class type to check against
        
    Returns:
        True if the task is of the specified type, False otherwise
    """
```

### model_dump_xml

XML serialization for debugging:

```python
def model_dump_xml(self) -> str:
    """Formats the task as XML for debugging purposes"""
```

## Validation

Leverage Pydantic's validation features:

```python
from pydantic import Field, validator

class ValidatedTask(Task):
    score: float = Field(ge=0, le=100, description="Score between 0-100")
    email: str = Field(regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    
    @validator('score')
    def round_score(cls, v):
        return round(v, 2)
```

## Serialization

Tasks can be serialized to/from JSON:

```python
# Serialize to JSON
task = UserQuery(query="test", user_id="123")
json_str = task.model_dump_json()

# Deserialize from JSON
loaded_task = UserQuery.model_validate_json(json_str)

# Convert to dictionary
task_dict = task.model_dump()
```

## Real-World Usage Patterns

### Multi-Phase Research Pipeline

```python
class Request(Task):
    user_input: str = Field(..., description="User input for the LLM")

class Plan(Task):
    response: str = Field(..., description="A detailed plan for the task")

class SearchQueries(Task):
    queries: List[SearchQueryWithPhase] = Field(..., description="Search queries")

class PhaseAnalysis(Task):
    phase: str = Field(..., description="The phase of the plan")
    extraction: str = Field(..., description="Extracted information")

class FinalWriteup(Task):
    writeup: str = Field(..., description="Final writeup in markdown")

# Workers process tasks in sequence:
# Request -> Plan -> SearchQueries -> PhaseAnalysis -> FinalWriteup
```

### Finding Original Context

```python
class FinalNarrativeWorker(LLMTaskWorker):
    def format_prompt(self, input_task: PhaseAnalyses) -> str:
        # Find original request from the beginning of the pipeline
        request: Request = input_task.find_input_task(Request)
        plan: Plan = input_task.find_input_task(Plan)
        
        return self.prompt.format(
            user_query=request.user_input,
            plan=plan.response
        )
```

## Best Practices

1. **Keep Tasks Immutable**: Don't modify task data after creation
2. **Use copy_public()**: When republishing tasks, use `copy_public()` for safety
3. **Type Hints**: Always specify types for better type checking and IDE support
4. **Document Fields**: Use Field descriptions for clarity
5. **Validate Data**: Use Pydantic validators for data integrity
6. **Reasonable Size**: Keep task data reasonable in size for performance
7. **Leverage Provenance**: Use `find_input_task()` to access upstream context

## Advanced Usage

### Custom Methods

Add helper methods to your tasks:

```python
class AnalysisTask(Task):
    data: List[float]
    
    def average(self) -> float:
        return sum(self.data) / len(self.data) if self.data else 0
    
    def is_valid(self) -> bool:
        return len(self.data) > 0 and all(x >= 0 for x in self.data)
    
    def summary_stats(self) -> dict:
        return {
            "count": len(self.data),
            "average": self.average(),
            "valid": self.is_valid()
        }
```

### Nested Tasks

Create complex task structures:

```python
class DataPoint(BaseModel):
    x: float
    y: float
    label: Optional[str] = None

class Dataset(Task):
    name: str
    points: List[DataPoint]
    metadata: dict = Field(default_factory=dict)
    
    def add_point(self, x: float, y: float, label: str = None):
        self.points.append(DataPoint(x=x, y=y, label=label))
```

### Task Inheritance

Build task hierarchies for different types of processing:

```python
class BaseAnalysis(Task):
    timestamp: datetime = Field(default_factory=datetime.now)
    analyst: str
    confidence: float = Field(ge=0, le=1)
    
class TextAnalysis(BaseAnalysis):
    text: str
    word_count: int
    sentiment: Optional[str] = None
    
class ImageAnalysis(BaseAnalysis):
    image_path: str
    dimensions: tuple[int, int]
    detected_objects: List[str] = Field(default_factory=list)
```

## Working with TaskWorkers

Tasks are processed by TaskWorkers in a type-safe manner:

```python
class DataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]
    
    def consume_work(self, task: UserQuery):
        # Access original query context
        original_request = task.find_input_task(UserQuery)
        
        # Process the data
        results = self.process_query(task.query)
        
        # Create and publish new task
        processed = ProcessedData(
            original_query=task.query,
            results=results,
            processing_time=time.time() - start_time
        )
        
        # Use copy_public() for safety
        self.publish_work(processed.copy_public(), input_task=task)
```

## See Also

- [TaskWorker](/api/taskworker/) - Processing tasks
- [Graph](/api/graph/) - Orchestrating workflows
- [Provenance](/guide/provenance/) - Understanding task lineage
- [Pydantic Documentation](https://docs.pydantic.dev/) - Validation features