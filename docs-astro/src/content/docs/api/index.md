---
title: API Reference
description: Complete API reference for PlanAI framework
---

This section provides detailed API documentation for all PlanAI components. The framework is organized into several core modules:

## Core Components

### [Task](/api/task/)
Base class for all units of work that flow through PlanAI workflows. All data in PlanAI is represented as Task objects with built-in provenance tracking.

### [TaskWorker](/api/taskworker/)
Abstract base class for processing tasks. All workers inherit from this class and implement the `consume_work` method.

### [Graph](/api/graph/)
The workflow orchestrator that manages task execution, dependencies, and parallelism.

### LLM Workers
Specialized workers for integrating Large Language Models into workflows, including `LLMTaskWorker` and `CachedLLMTaskWorker`.

## Additional Components

### Integrations
- **llm_from_config**: Factory function for creating LLM instances
- **Tool**: Base class for LLM function calling/tools

### Caching
- **CachedTaskWorker**: Base class for workers with caching
- **CachedLLMTaskWorker**: LLM worker with response caching

### Advanced Workers
- **InitialTaskWorker**: Entry point for workflows
- **JoinedTaskWorker**: Aggregates multiple task results
- **SubGraphWorker**: Encapsulates graphs as workers

### Utilities
- **PydanticDictWrapper**: Utility for wrapping dictionaries as tasks
- **Provenance**: Classes for tracking task lineage

### Testing
- **WorkflowTestHelper**: Utilities for testing workflows
- **MockLLM**: Mock LLM for unit testing

## Import Structure

The main imports are available from the root module:

```python
from planai import (
    # Core classes
    Task,
    TaskWorker,
    Graph,
    
    # LLM integration
    LLMTaskWorker,
    CachedLLMTaskWorker,
    llm_from_config,
    Tool,
    
    # Advanced workers
    InitialTaskWorker,
    JoinedTaskWorker,
    CachedTaskWorker,
    SubGraphWorker,
    
    # Utilities
    Dispatcher,
    InputProvenance,
)
```

## Type Safety

PlanAI uses Python type hints extensively. All components are designed to work with type checkers like mypy:

```python
from typing import List, Type
from planai import TaskWorker, Task

class MyWorker(TaskWorker):
    output_types: List[Type[Task]] = [OutputTask]
    
    def consume_work(self, task: InputTask) -> None:
        # Type-safe processing
        pass
```

## Async Support

PlanAI supports asynchronous execution for improved performance:

```python
graph = Graph(name="Async Workflow")
# Workers run concurrently when possible
graph.run(initial_tasks, max_workers=10)
```

## Error Handling

All PlanAI components include comprehensive error handling:

```python
try:
    graph.run(initial_tasks)
except WorkflowError as e:
    # Handle workflow-specific errors
    pass
except Exception as e:
    # Handle general errors
    pass
```

## Next Steps

- Explore specific component documentation in the subsections
- See [Examples](https://github.com/provos/planai/tree/main/examples) for practical usage
- Review the [source code](https://github.com/provos/planai) for implementation details