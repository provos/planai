---
title: Graph
description: API reference for the Graph workflow orchestrator
---

The `Graph` class is the central orchestrator in PlanAI, managing task execution, worker dependencies, workflow lifecycle, and providing comprehensive monitoring and control capabilities.

## Class Definition

```python
from planai import Graph

class Graph(BaseModel):
    """
    A graph for orchestrating task workers and their dependencies.
    
    Args:
        name: Name identifier for the graph instance
        strict: If True, enforces strict validation of tasks in publish_work()
    """
    def __init__(self, name: str, strict: bool = False):
        ...
```

## Core Properties

### name
```python
name: str
```
Identifier for the graph instance, used in logging and monitoring.

### strict
```python
strict: bool = False
```
When True, enables strict validation that prevents common task reuse bugs by checking that provenance fields are empty during `publish_work()`.

### workers
```python
workers: Set[TaskWorker]
```
Set of all task workers registered in the graph.

### dependencies
```python
dependencies: Dict[TaskWorker, List[TaskWorker]]
```
Maps each worker to its list of downstream dependencies.

## Worker Management

### add_worker

Add a single TaskWorker to the graph:

```python
def add_worker(self, worker: TaskWorker) -> "Graph":
    """
    Adds a single task worker to the graph.
    
    Args:
        worker: The worker instance to add
        
    Returns:
        Graph instance for method chaining
        
    Raises:
        ValueError: If worker already exists in the graph
    """
```

Example:
```python
graph = Graph(name="Data Pipeline")
processor = DataProcessor()
graph.add_worker(processor)
```

### add_workers

Add multiple workers at once:

```python
def add_workers(self, *workers: TaskWorker) -> "Graph":
    """Add multiple workers to the graph"""
```

Example:
```python
graph.add_workers(fetcher, processor, analyzer, reporter)
```

### Worker Discovery

Find workers by their input/output types:

```python
def get_worker_by_input_type(self, input_type: Type[Task]) -> Optional[TaskWorker]:
    """Get a worker that consumes a specific input type"""
    
def get_worker_by_output_type(self, output_type: Type[Task]) -> Optional[TaskWorker]:
    """Get a worker that produces a specific output type"""
```

## Dependency Management

### set_dependency

Define execution dependencies between workers:

```python
def set_dependency(self, upstream: TaskWorker, downstream: TaskWorker) -> TaskWorker:
    """
    Set dependency: downstream depends on upstream
    
    Args:
        upstream: Worker that produces tasks  
        downstream: Worker that consumes tasks from upstream
        
    Returns:
        The downstream worker for method chaining
        
    Raises:
        ValueError: If workers aren't in graph or incompatible types
    """
```

The system automatically validates type compatibility between worker outputs and inputs.

Example:
```python
graph.set_dependency(fetcher, processor)  # processor depends on fetcher
graph.set_dependency(processor, analyzer) # analyzer depends on processor

# Method chaining
graph.set_dependency(fetcher, processor).next(analyzer).next(reporter)
```

## Entry and Exit Points

### set_entry

Define entry points for the workflow:

```python
def set_entry(self, *workers: TaskWorker) -> "Graph":
    """
    Set workers as entry points to the graph
    
    Args:
        *workers: Variable number of workers to set as entry points
        
    Returns:
        Graph instance for method chaining
    """
```

An entry worker does not need to be specified if the graph is executed with ```graph.run(initial_tasks=[(Worker1, Task1)])```.

### get_entry_workers

```python
def get_entry_workers(self) -> List[TaskWorker]:
    """Get all entry point workers"""
```

### set_exit / get_exit_worker

Define and retrieve exit points:

```python
def set_exit(self, worker: TaskWorker) -> None:
    """Set the exit worker for the graph"""
    
def get_exit_worker(self) -> Optional[TaskWorker]:
    """Get the exit worker"""
```

This is a convenience function to make it easier to return a fully instantiated graph from a utility function.

## Output Collection

### set_sink

Create data sinks for collecting specific output types:

```python
def set_sink(
    self,
    worker: TaskWorker,
    output_type: Type[Task],
    notify: Optional[Callable[[Dict[str, Any], Task], None]] = None,
) -> None:
    """
    Designates a worker as a data sink for collecting specific output tasks.
    
    Args:
        worker: The worker whose output should be collected
        output_type: The specific task type to collect
        notify: Optional callback for real-time notifications
        
    Raises:
        ValueError: If worker doesn't have the specified output type
    """
```

### get_output_tasks

Retrieve collected sink data:

```python
def get_output_tasks(self) -> List[Type[Task]]:
    """
    Retrieves all tasks collected by sink workers after graph execution.
    
    Returns:
        List of tasks collected by all sink workers
    """
```

Example workflow with sink:
```python
# Set up sink to collect results
graph.set_sink(final_worker, ResultTask)

# Run workflow
graph.run(initial_tasks=[(start_worker, StartTask())])

# Collect results
results = graph.get_output_tasks()
for result in results:
    print(f"Final result: {result.data}")
```

## Performance Optimization

### set_max_parallel_tasks

Limit concurrent execution for specific worker types:

```python
def set_max_parallel_tasks(
    self, 
    worker_class: Type[TaskWorker], 
    max_parallel_tasks: int
) -> None:
    """
    Set maximum number of parallel tasks for a worker class.
    
    Args:
        worker_class: The class of worker to limit
        max_parallel_tasks: Maximum concurrent tasks allowed
        
    Raises:
        ValueError: If worker_class isn't a TaskWorker subclass
        ValueError: If max_parallel_tasks <= 0
    """
```

Example for LLM rate limiting:
```python
# Limit expensive LLM operations
graph.set_max_parallel_tasks(LLMTaskWorker, 3)
graph.set_max_parallel_tasks(CachedLLMTaskWorker, 5)
```

## Execution Methods

### run

Complete workflow execution with setup and monitoring:

```python
def run(
    self,
    initial_tasks: Sequence[Tuple[TaskWorker, Task]],
    run_dashboard: bool = False,
    display_terminal: bool = True,
    dashboard_port: int = 5000,
) -> None:
    """
    Execute the graph by setting up dispatcher, workers, and processing initial tasks.
    
    Args:
        initial_tasks: List of (worker, task) tuples to start execution
        run_dashboard: Enable web monitoring dashboard
        display_terminal: Show terminal status display
        dashboard_port: Port for web dashboard
        
    Note:
        Blocks until all tasks complete unless dashboard is running
    """
```

### prepare

Set up execution environment without starting:

```python
def prepare(
    self,
    run_dashboard: bool = False,
    display_terminal: bool = True,
    dashboard_port: int = 5000,
) -> None:
    """
    Initialize graph for execution by setting up monitoring and worker components.
    
    Must be called before execute(). Sets up:
    - Task dispatcher for managing worker execution
    - Optional web dashboard for monitoring
    - Optional terminal-based status display
    - Worker parallel execution limits
    """
```

### execute

Start actual task processing:

```python
def execute(self, initial_tasks: Sequence[Tuple[TaskWorker, Task]]) -> None:
    """
    Execute graph with provided initial tasks.
    
    Should be called after prepare(). Blocks until completion.
    
    Args:
        initial_tasks: Sequence of worker-task pairs to start execution
    """
```

Staged execution example:
```python
# Set up environment
graph.prepare(run_dashboard=True, dashboard_port=8080)

# Start processing
initial = [(worker, Task(data="start"))]
graph.execute(initial)
```

## Dynamic Task Management

### add_work

Add tasks during execution:

```python
def add_work(
    self,
    worker: TaskWorker,
    task: Task,
    metadata: Optional[Dict] = None,
    status_callback: Optional[TaskStatusCallback] = None,
) -> ProvenanceChain:
    """
    Add work to a running graph.
    
    Args:
        worker: Target worker (must be an entry point)
        task: Task to process
        metadata: Optional metadata for tracking
        status_callback: Optional status update callback
        
    Returns:
        ProvenanceChain for tracking this work
        
    Raises:
        ValueError: If worker is not an entry point
    """
```

This can be called from a different thread as the thread in which ```graph.run()``` or ```graph.execute()``` have been called will block until the graph has finished processing.

### abort_work

Cancel in-progress work:

```python
def abort_work(self, provenance: ProvenanceChain) -> bool:
    """
    Abort work currently in progress.
    
    Args:
        provenance: Provenance chain identifying work to abort
        
    Returns:
        True if work was aborted, False if not found
    """
```

## Interactive Features

### User Input Support

```python
def set_user_request_callback(
    self, 
    callback: Callable[[Dict[str, Any], UserInputRequest], None]
) -> None:
    """Set callback function to handle user input requests"""

def wait_on_user_request(
    self, 
    request: UserInputRequest
) -> Tuple[Any, Optional[str]]:
    """
    Wait for user input request to be completed.
    
    Returns:
        Tuple of (user_data, mime_type)
    """
```

Example interactive workflow:
```python
def handle_user_input(metadata: Dict, request: UserInputRequest):
    # Custom UI integration
    response = my_ui.get_user_input(request.instruction)
    request._response_queue.put((response, "text/plain"))

graph.set_user_request_callback(handle_user_input)
```

## Monitoring and Debugging

### Dashboard Integration

Built-in web dashboard for real-time monitoring:

```python
# Enable monitoring dashboard
graph.run(
    initial_tasks=tasks,
    run_dashboard=True,
    dashboard_port=8080
)
```

Dashboard features:
- Real-time task execution status
- Worker performance metrics  
- Task queue visualization
- Interactive task inspection
- User input handling
- Log streaming

### Terminal Display

Built-in terminal status display features:
- Color-coded progress bars
- Completion/active/queued/failed counts
- Worker distance visualization
- Scrolling log display

### Logging

Integrated logging system:

```python
def print(self, *args):
    """
    Print messages with integrated logging support.
    
    Messages are:
    - Logged to the standard logger
    - Sent to dashboard if enabled
    - Displayed in terminal if enabled
    """
```

Usage in workers:
```python
class MyWorker(TaskWorker):
    def consume_work(self, task):
        self.print(f"Processing {task.name}")
```

## Advanced Features

### Dispatcher Management

```python
def register_dispatcher(self, dispatcher: Dispatcher) -> None:
    """Register external dispatcher for multi-graph scenarios"""
    
def get_dispatcher(self) -> Optional[Dispatcher]:
    """Get the current dispatcher instance"""
```

### Validation and State

```python
def validate_graph(self) -> None:
    """Validate graph structure and dependencies"""
    
def finalize(self):
    """Finalize graph by computing worker distances and validation"""
```

### Graceful Shutdown

```python
def shutdown(self, timeout: float = 5.0) -> bool:
    """
    Gracefully shut down the graph and all components.
    
    Args:
        timeout: Maximum time to wait for completion
        
    Returns:
        True if shutdown was successful, False if timeout
    """
```

## Real-World Examples

### Complex Research Pipeline

```python
from planai import Graph, LLMTaskWorker, JoinedTaskWorker, InitialTaskWorker

# Create sophisticated research workflow
graph = Graph(name="Research Pipeline", strict=True)

# Define workers
planner = PlanningWorker()
search_creator = SearchCreator()
search_splitter = SearchSplitter() 
fetcher = SearchFetcher()
analyzer = SearchAnalyzer()
joiner = AnalysisJoiner()
writer = FinalWriter()

# Build complex dependency graph
graph.add_workers(
    planner, search_creator, search_splitter, 
    fetcher, analyzer, joiner, writer
)

# Set up pipeline
graph.set_dependency(planner, search_creator)\
     .next(search_splitter)\
     .next(fetcher)\
     .next(analyzer)\
     .next(joiner)\
     .next(writer)

# Configure performance
graph.set_max_parallel_tasks(LLMTaskWorker, 3)
graph.set_max_parallel_tasks(SearchFetcher, 10)

# Set up result collection
graph.set_sink(writer, FinalReport, notify=save_report)

# Define entry points
graph.set_entry(planner)

# Execute with monitoring
graph.run(
    initial_tasks=[(planner, ResearchRequest(query="AI safety"))],
    run_dashboard=True,
    dashboard_port=8080
)
```

### Security Engineering Workflow

```python
# Multi-phase security analysis
graph = Graph(name="Security Engineering")

# Create specialized workers
request_worker = RequestWorker()
planner = SecurityPlanner()
command_worker = CommandWorker()
executor = CommandExecutor() 
inspector = ProgressInspector()
summarizer = SummaryWorker()

# Build iterative workflow
graph.add_workers(
    request_worker, planner, command_worker, 
    executor, inspector, summarizer
)

# Set up dependencies with loops
graph.set_dependency(request_worker, planner)\
     .next(command_worker)\
     .next(executor)\
     .next(inspector)

# Create feedback loops
executor.next(command_worker)  # Continue iteration
inspector.next(summarizer)     # Final summary

# Performance tuning
graph.set_max_parallel_tasks(LLMTaskWorker, 2)

# Interactive capabilities
graph.set_user_request_callback(handle_security_input)

# Execute with full monitoring
graph.run(
    initial_tasks=[(request_worker, SecurityRequest())],
    run_dashboard=True,
    display_terminal=False
)
```

### Multi-Graph Coordination

```python
# Shared dispatcher for multiple graphs
dispatcher = Dispatcher(web_port=8080)

# Create specialized graphs
research_graph = Graph(name="Research")
analysis_graph = Graph(name="Analysis") 

# Register with shared dispatcher
research_graph.register_dispatcher(dispatcher)
analysis_graph.register_dispatcher(dispatcher)

# Configure and run
research_graph.prepare()
analysis_graph.prepare()

# Start both workflows
research_graph.execute(research_tasks)
analysis_graph.execute(analysis_tasks)
```

## Best Practices

### 1. Graph Design
```python
# Use descriptive names
graph = Graph(name="CustomerDataProcessing")

# Enable strict mode for debugging
graph = Graph(name="Pipeline", strict=True)
```

### 2. Performance Optimization
```python
# Limit expensive operations
graph.set_max_parallel_tasks(LLMTaskWorker, 5)
graph.set_max_parallel_tasks(DatabaseWorker, 10)

# Use appropriate monitoring
if production:
    graph.run(run_dashboard=False, display_terminal=False)
else:
    graph.run(run_dashboard=True, dashboard_port=8080)
```

### 3. Monitoring Integration
```python
def task_notification(metadata: Dict, task: Task):
    # Send to external monitoring
    metrics.record_task_completion(task.name, metadata)

graph.set_sink(final_worker, ResultTask, notify=task_notification)
```

## Common Patterns

### Fan-out/Fan-in Processing
```python
# One source, multiple processors, single aggregator
graph.set_dependency(source, processor1)
graph.set_dependency(source, processor2) 
graph.set_dependency(source, processor3)

graph.set_dependency(processor1, aggregator)
graph.set_dependency(processor2, aggregator)
graph.set_dependency(processor3, aggregator)
```

### Conditional Routing
```python
# Router worker directs tasks to different paths
graph.set_dependency(router, path1_worker)
graph.set_dependency(router, path2_worker)
graph.set_dependency(router, path3_worker)
```

### Iterative Refinement
```python
# Feedback loops for improvement
graph.set_dependency(generator, evaluator)
graph.set_dependency(evaluator, generator)  # Loop back
graph.set_dependency(evaluator, finalizer)  # Exit condition
```

## Troubleshooting

### Common Issues

1. **Type Mismatches**: Ensure worker output types match downstream input types
2. **Missing Entry Points**: Set entry points with `set_entry()`
3. **Resource Limits**: Use `set_max_parallel_tasks()` to prevent overload
4. **Stuck Workflows**: Monitor with dashboard and check for infinite loops

### Debugging Tools

```python
# Enable strict validation
graph = Graph(name="Debug", strict=True)

# Use dashboard for visualization
graph.run(run_dashboard=True)

# Access execution statistics
stats = graph.get_dispatcher().get_execution_statistics()

# Enable detailed logging
logging.getLogger("planai").setLevel(logging.DEBUG)
```

## See Also

- [TaskWorker](/api/taskworker/) - Processing units
- [Task](/api/task/) - Data structures
- [Monitoring](/guide/monitoring/) - Dashboard usage
