---
title: Task
description: API reference for the Task base class
---

The `Task` class is the fundamental data unit in PlanAI. All data flowing through workflows must be represented as Task instances.

## Class Definition

```python
from planai import Task
from pydantic import BaseModel

class Task(BaseModel):
    """Base class for all tasks in PlanAI workflows"""
    pass
```

Task inherits from Pydantic's `BaseModel`, providing automatic validation, serialization, and type safety.

## Creating Tasks

Define custom tasks by inheriting from Task:

```python
from planai import Task
from typing import List, Optional
from datetime import datetime

class UserQuery(Task):
    query: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ProcessedData(Task):
    original_query: str
    results: List[dict]
    processing_time: float
    metadata: Optional[dict] = None
```

## Provenance Methods

Tasks include methods for accessing their execution history:

### find_input_task

Find a specific task type in the provenance chain:

```python
def find_input_task(self, task_type: Type[Task]) -> Optional[Task]:
    """
    Search provenance chain for a specific task type
    
    Args:
        task_type: The type of task to search for
        
    Returns:
        The first matching task or None
    """
```

Example:
```python
def consume_work(self, task: ProcessedData):
    # Find the original user query
    original = task.find_input_task(UserQuery)
    if original:
        print(f"Processing query: {original.query}")
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

### copy_provenance

Get a copy of the full provenance chain:

```python
def copy_provenance(self) -> List[InputProvenance]:
    """
    Get complete provenance history
    
    Returns:
        List of InputProvenance objects
    """
```

### prefix_for_input_task

Get provenance prefix for a specific worker type:

```python
def prefix_for_input_task(self, worker_type: Type[TaskWorker]) -> Optional[str]:
    """
    Get provenance prefix for joining operations
    
    Args:
        worker_type: Type of worker to find prefix for
        
    Returns:
        Provenance prefix string or None
    """
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

## Best Practices

1. **Keep Tasks Immutable**: Don't modify task data after creation
2. **Use Type Hints**: Always specify types for better type checking
3. **Document Fields**: Use Field descriptions for clarity
4. **Validate Data**: Use Pydantic validators for data integrity
5. **Avoid Large Objects**: Keep task data reasonable in size

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

Build task hierarchies:

```python
class BaseAnalysis(Task):
    timestamp: datetime
    analyst: str
    
class TextAnalysis(BaseAnalysis):
    text: str
    word_count: int
    
class ImageAnalysis(BaseAnalysis):
    image_path: str
    dimensions: tuple[int, int]
```

## See Also

- [TaskWorker](/api/taskworker/) - Processing tasks
- [Provenance](/guide/provenance/) - Understanding task lineage
- [Pydantic Documentation](https://docs.pydantic.dev/) - Validation features