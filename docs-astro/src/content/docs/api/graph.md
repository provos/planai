---
title: Graph
description: API reference for the Graph workflow orchestrator
---

The `Graph` class is the central orchestrator in PlanAI, managing task execution, worker dependencies, and workflow lifecycle.

## Class Definition

```python
from planai import Graph

class Graph:
    """
    Directed graph for orchestrating task execution
    
    Args:
        name: Name of the graph for identification
        use_process: Use process-based execution (default: False)
    """
    def __init__(self, name: str, use_process: bool = False):
        ...
```

## Core Methods

### add_worker / add_workers

Add TaskWorker instances to the graph:

```python
def add_worker(self, worker: TaskWorker) -> None:
    """Add a single worker to the graph"""
    
def add_workers(self, *workers: TaskWorker) -> None:
    """Add multiple workers to the graph"""
```

Example:
```python
graph = Graph(name="Data Pipeline")
graph.add_worker(processor)
graph.add_workers(fetcher, analyzer, reporter)
```

### set_dependency

Define execution dependencies between workers:

```python
def set_dependency(self, upstream: TaskWorker, downstream: TaskWorker) -> None:
    """
    Set dependency: downstream depends on upstream
    
    Args:
        upstream: Worker that produces tasks
        downstream: Worker that consumes tasks from upstream
    """
```

Example:
```python
graph.set_dependency(fetcher, processor)  # processor depends on fetcher
graph.set_dependency(processor, analyzer)  # analyzer depends on processor
```

### set_dependencies

Set multiple dependencies at once:

```python
def set_dependencies(self, worker: TaskWorker, dependencies: List[TaskWorker]) -> None:
    """Set multiple upstream dependencies for a worker"""
```

### run

Execute the workflow:

```python
def run(
    self,
    initial_tasks: List[Tuple[TaskWorker, Task]] = None,
    run_dashboard: bool = False,
    dashboard_port: int = 5000,
    dashboard_host: str = "127.0.0.1",
    max_workers: Optional[int] = None,
    completion_callback: Optional[Callable] = None
) -> None:
    """
    Run the graph execution
    
    Args:
        initial_tasks: List of (worker, task) tuples to start execution
        run_dashboard: Enable web monitoring dashboard
        dashboard_port: Port for web dashboard
        dashboard_host: Host for web dashboard
        max_workers: Maximum concurrent workers
        completion_callback: Callback when execution completes
    """
```

Example:
```python
# Basic execution
graph.run(initial_tasks=[(fetcher, FetchRequest(url="..."))])

# With monitoring
graph.run(
    initial_tasks=[(fetcher, FetchRequest(url="..."))],
    run_dashboard=True,
    dashboard_port=8080
)
```

### set_entry / set_exit

Define entry and exit points (for subgraphs):

```python
def set_entry(self, worker: TaskWorker) -> None:
    """Set the entry point worker"""
    
def set_exit(self, worker: TaskWorker) -> None:
    """Set the exit point worker"""
```

## Properties

### workers

Access all workers in the graph:

```python
@property
def workers(self) -> List[TaskWorker]:
    """Get all workers in the graph"""
```

### is_running

Check execution status:

```python
@property
def is_running(self) -> bool:
    """Check if graph is currently executing"""
```

## Advanced Features

### Execution Modes

```python
# Thread-based execution (default)
graph = Graph(name="ThreadedWorkflow")

# Process-based execution (for CPU-intensive tasks)
graph = Graph(name="ProcessWorkflow", use_process=True)
```

### Dynamic Worker Addition

Add workers during execution:

```python
class DynamicController(TaskWorker):
    def consume_work(self, task: ControlTask):
        if task.needs_extra_processing:
            extra_worker = ExtraProcessor()
            self.graph.add_worker(extra_worker)
            self.graph.set_dependency(self, extra_worker)
```

### Completion Handling

```python
def on_complete(graph: Graph):
    print(f"Workflow {graph.name} completed")
    # Perform cleanup or notifications

graph.run(
    initial_tasks=tasks,
    completion_callback=on_complete
)
```

## Monitoring Integration

### Built-in Dashboard

```python
# Enable web dashboard
graph.run(
    initial_tasks=tasks,
    run_dashboard=True,
    dashboard_port=5000
)
```

### Custom Monitoring

```python
class MonitoredGraph(Graph):
    def __init__(self, name: str):
        super().__init__(name)
        self.task_count = 0
        
    def on_task_complete(self, worker: TaskWorker, task: Task):
        self.task_count += 1
        print(f"Completed {self.task_count} tasks")
```

## Error Handling

```python
try:
    graph.run(initial_tasks)
except WorkflowException as e:
    print(f"Workflow error: {e}")
    # Access error details
    failed_worker = e.worker
    failed_task = e.task
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Name Your Graphs**: Use descriptive names for easier debugging
2. **Validate Dependencies**: Ensure no circular dependencies
3. **Limit Complexity**: Break large graphs into subgraphs
4. **Monitor Execution**: Use the dashboard during development
5. **Handle Errors**: Implement proper error handling in workers

## Example: Complete Workflow

```python
from planai import Graph, InitialTaskWorker, TaskWorker, Task

# Define workflow
graph = Graph(name="ETL Pipeline")

# Create workers
extractor = DataExtractor()
transformer = DataTransformer()
loader = DataLoader()
monitor = QualityMonitor()

# Build graph
graph.add_workers(extractor, transformer, loader, monitor)
graph.set_dependency(extractor, transformer)
graph.set_dependency(transformer, loader)
graph.set_dependency(transformer, monitor)

# Configure initial data
initial_tasks = [
    (extractor, ExtractRequest(source="database")),
    (extractor, ExtractRequest(source="api"))
]

# Run with monitoring
graph.run(
    initial_tasks=initial_tasks,
    run_dashboard=True,
    max_workers=4
)
```

## See Also

- [TaskWorker](/api/taskworker/) - Worker implementation
- [Task](/api/task/) - Task definition
- [Monitoring](/guide/monitoring/) - Dashboard usage
- [Subgraphs](/features/subgraphs/) - Nested workflows