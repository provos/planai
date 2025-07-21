---
title: TaskWorker
description: API reference for TaskWorker classes - the processing units of PlanAI workflows
---

The `TaskWorker` class is the fundamental processing unit in PlanAI. TaskWorkers consume tasks, process them, and can produce new tasks for downstream workers. The system ensures type safety between workers and maintains execution provenance throughout the workflow.

## Base TaskWorker Class

```python
from planai import TaskWorker, Task
from typing import List, Type
from abc import abstractmethod

class TaskWorker(BaseModel, ABC):
    """Base class for all task workers"""
    
    output_types: List[Type[Task]] = Field(default_factory=list)
    num_retries: int = Field(default=0)
    
    @abstractmethod
    def consume_work(self, task: Task):
        """Process a task - must be implemented by subclasses"""
        pass
```

## Core Properties

### output_types
```python
output_types: List[Type[Task]] = Field(default_factory=list)
```
Defines the types of tasks this worker can produce. Used for type checking and routing.

### num_retries
```python
num_retries: int = Field(default=0)
```
Number of times to retry failed tasks.

### name
```python
@property
def name(self) -> str:
    """Returns the class name of the worker"""
```

## Essential Methods

### consume_work (Abstract)

The core processing method that must be implemented by all TaskWorkers:

```python
@abstractmethod
def consume_work(self, task: Task):
    """
    Process a task. Must be implemented by subclasses.
    
    Args:
        task: The task to process
    """
    pass
```

Example implementation:
```python
class DataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]
    
    def consume_work(self, task: UserQuery):
        # Process the input
        results = self.process_query(task.query)
        
        # Create output task
        processed = ProcessedData(
            original_query=task.query,
            results=results,
            processing_time=time.time() - start_time
        )
        
        # Publish for downstream processing
        self.publish_work(processed, input_task=task)
```

### publish_work

Publish tasks to downstream workers:

```python
def publish_work(
    self,
    task: Task,
    input_task: Optional[Task],
    consumer: Optional[TaskWorker] = None,
):
    """
    Publish a work item to downstream consumers
    
    Args:
        task: The work item to publish (must be newly created)
        input_task: The input task that led to this work item
        consumer: Specific consumer to publish to (if multiple exist)
    """
```

**Critical:** The task must be a newly created object, not a reference to an existing task. Use `copy_public()` if needed.

### Type Safety Methods

#### get_task_class
```python
def get_task_class(self) -> Type[Task]:
    """Get the Task subclass this worker can consume based on consume_work signature"""
```

#### validate_task
```python
def validate_task(self, task_cls: Type[Task], consumer: TaskWorker) -> Tuple[bool, Optional[BaseException]]:
    """Validate that a consumer can handle a specific Task type"""
```

## Workflow Control Methods

### next
Chain workers together:
```python
def next(self, downstream: TaskWorker) -> TaskWorker:
    """Set dependency to downstream worker"""
```

Usage:
```python
graph.set_dependency(worker1, worker2).next(worker3).next(worker4)
```

### sink
Designate as workflow endpoint:
```python
def sink(self, output_type: Type[Task], notify: Optional[Callable] = None):
    """Mark this worker as a sink for collecting results"""
```

The output from the sink can be retrieved from the graph after execution has completed:
```python
graph.get_output_tasks()
```

## State Management

### Worker State
Store state associated with specific provenance chains:
```python
def get_worker_state(self, provenance: ProvenanceChain) -> Dict[str, Any]:
    """Get state for a specific provenance chain"""
```

### Task Metadata
Access task execution metadata:
```python
def get_metadata(self, task: Task) -> Dict[str, Any]:
    """Get metadata for task execution"""

def get_state(self, task: Task) -> Dict[str, Any]:
    """Get execution state for task"""
```

## Monitoring and Debugging

### Status Notifications
```python
def notify_status(self, task: Task, message: Optional[str] = None, object: Optional[BaseModel] = None):
    """Notify about task status updates"""
```

### Provenance Tracking
```python
def watch(self, prefix: ProvenanceChain) -> bool:
    """Watch for completion of a provenance prefix"""

def unwatch(self, prefix: ProvenanceChain) -> bool:
    """Remove watch for a provenance prefix"""

def trace(self, prefix: ProvenanceChain):
    """Set up tracing for dashboard visibility"""
```

## User Interaction

### Request User Input
```python
def request_user_input(
    self,
    task: Task,
    instruction: str,
    accepted_mime_types: List[str] = ["text/html"],
) -> Tuple[Any, Optional[str]]:
    """Request input from user during task execution"""
```

## LLMTaskWorker

Specialized worker for AI-powered processing:

```python
from planai import LLMTaskWorker

class PlanWorker(LLMTaskWorker):
    output_types: List[Type[Task]] = [Plan]
    llm_output_type: Type[Task] = Plan
    llm_input_type: Type[Task] = Request
    
    system_prompt: str = "You are an expert research agent..."
    prompt: str = "Please provide a detailed step-by-step plan..."
    
    use_xml: bool = True
    temperature: float = 0.7
```

### Key Features

#### Automatic LLM Integration
- **Structured Output**: Automatically converts LLM responses to Task objects
- **Type Safety**: Validates input/output types
- **Error Handling**: Robust error handling for LLM failures

#### Customization Hooks

##### format_prompt
```python
def format_prompt(self, task: Task) -> str:
    """Customize prompt based on input task"""
    request: Request = task.find_input_task(Request)
    return self.prompt.format(user_request=request.user_input)
```

##### pre_process
```python
def pre_process(self, task: Task) -> Optional[Task]:
    """Pre-process input before sending to LLM"""
    # Filter or transform the task data
    return task
```

##### post_process
```python
def post_process(self, response: Optional[Task], input_task: Task):
    """Post-process LLM response"""
    if response:
        self.publish_work(response, input_task=input_task)
        # Optionally publish additional tasks
        self.publish_work(
            StatusUpdate(message="Plan created"),
            input_task=input_task
        )
```

##### extra_validation
```python
def extra_validation(self, response: Task, input_task: Task) -> Optional[str]:
    """Additional validation of LLM response"""
    if not response.plan or len(response.plan) < 100:
        return "Plan too short"
    return None
```

### Real-World Example

```python
class SearchSummarizer(LLMTaskWorker):
    output_types: List[Type[Task]] = [PhaseAnalysis]
    llm_input_type: Type[Task] = ConsolidatedPages
    llm_output_type: Type[Task] = PhaseAnalysisInterim
    
    use_xml: bool = True
    system_prompt: str = dedent("""
        You are a master research scientist, adept at synthesizing 
        complex information from multiple sources...""").strip()
    prompt: str = dedent("""
        Summarize the pages according to: {plan} .... other instructions ...
        You are in this phase: {phase}
        ... more prompt instructions ...
    """).strip()
    
    def format_prompt(self, input_task: ConsolidatedPages) -> str:
        plan: Plan = input_task.find_input_task(Plan)
        query: SearchQuery = input_task.find_input_task(SearchQuery)
        
        return self.prompt.format(
            plan=plan.response, 
            phase=query.metadata
        )
    
    def post_process(self, response: PhaseAnalysisInterim, input_task: ConsolidatedPages):
        query: SearchQuery = input_task.find_input_task(SearchQuery)
        
        analysis = PhaseAnalysis(
            phase=query.metadata,
            extraction=response.extraction
        )
        
        self.publish_work(analysis, input_task=input_task)
```

### CachedLLMTaskWorker

For expensive LLM operations that benefit from caching:

```python
from planai import CachedLLMTaskWorker

class ExpensiveAnalysis(CachedLLMTaskWorker):
    # Automatically caches based on input task and prompt
    pass
```

## CachedTaskWorker

Specialized worker for expensive operations that benefit from persistent caching:

```python
from planai import CachedTaskWorker

class ExpensiveProcessor(CachedTaskWorker):
    output_types: List[Type[Task]] = [ProcessedResult]
    cache_dir: str = "./cache"
    cache_size_limit: int = 25_000_000_000  # 25GB limit
    
    def consume_work(self, task: ExpensiveTask):
        # Expensive computation that will be cached
        result = self.expensive_computation(task.data)
        processed = ProcessedResult(result=result)
        self.publish_work(processed, input_task=task)
```

### Key Features

#### Automatic Caching
- **Disk-Based**: Uses diskcache for persistent storage across runs
- **Content-Based Keys**: Generates cache keys from task content and worker configuration
- **Size Management**: Automatic cache size limits and eviction

#### Configuration Options

##### cache_dir
```python
cache_dir: str = Field("./cache", description="Directory to store the cache")
```
Directory where cache files are stored.

##### cache_size_limit
```python
cache_size_limit: int = Field(25_000_000_000, description="Cache size limit in bytes")
```
Maximum cache size in bytes before eviction begins.

### Lifecycle Hooks

#### pre_consume_work
```python
def pre_consume_work(self, task: Task):
    """Called before processing, even if cached"""
    # Set up state, update counters, etc.
    pass
```

#### post_consume_work
```python
def post_consume_work(self, task: Task):
    """Called after processing, even if cached"""
    # Clean up state, log completion, etc.
    pass
```

### Customization Methods

#### extra_cache_key
```python
def extra_cache_key(self, task: Task) -> str:
    """Add custom information to cache key"""
    return f"{self.custom_setting}_{task.priority}"
```

### Real-World Example

```python
class DocumentAnalyzer(CachedTaskWorker):
    output_types: List[Type[Task]] = [AnalysisResult]
    cache_dir: str = "./analysis_cache"
    model_version: str = "v2.1"
    
    def extra_cache_key(self, task: DocumentTask) -> str:
        # Include model version in cache key
        return f"model_{self.model_version}"
    
    def pre_consume_work(self, task: DocumentTask):
        self.notify_status(task, "Analyzing document...")
    
    def consume_work(self, task: DocumentTask):
        # Expensive analysis operation
        analysis = self.analyze_document(task.document)
        
        result = AnalysisResult(
            document_id=task.document_id,
            analysis=analysis,
            confidence=0.95
        )
        
        self.publish_work(result, input_task=task)
    
    def post_consume_work(self, task: DocumentTask):
        self.notify_status(task, "Analysis complete")
```

### Cache Behavior

#### Cache Hit
When input matches cached data:
1. `pre_consume_work()` is called
2. Cached results are published directly 
3. `consume_work()` is **skipped**
4. `post_consume_work()` is called

#### Cache Miss
When no cached data exists:
1. `pre_consume_work()` is called
2. `consume_work()` executes normally
3. Results are cached for future use
4. `post_consume_work()` is called

### Cache Key Generation

Cache keys are generated from:
- Task content (all public fields)
- Worker name and output types
- Custom cache key from `extra_cache_key()`
- Protocol version for cache compatibility

### Best Practices for Caching

#### 1. Deterministic Operations
Only cache deterministic operations:
```python
def consume_work(self, task: Task):
    # Good: deterministic computation
    result = self.mathematical_analysis(task.data)
    
    # Bad: includes current time/random elements
    # result = self.analysis_with_timestamp(task.data)
```

#### 2. Appropriate Cache Keys
Include relevant configuration in cache keys:
```python
def extra_cache_key(self, task: Task) -> str:
    return f"threshold_{self.threshold}_model_{self.model_version}"
```

#### 3. State Management
Use lifecycle hooks for consistent state:
```python
def pre_consume_work(self, task: Task):
    # Always runs - good for state setup
    self.processed_count += 1

def post_consume_work(self, task: Task):
    # Always runs - good for cleanup
    self.cleanup_temp_files()
```

## JoinedTaskWorker

Specialized worker for aggregating multiple tasks from upstream workers:

```python
from planai import JoinedTaskWorker, InitialTaskWorker

class AnalysisJoiner(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    output_types: List[Type[Task]] = [PhaseAnalyses]
    enable_trace: bool = True  # Enable dashboard tracing
    
    def consume_work_joined(self, tasks: List[PhaseAnalysis]):
        """Process aggregated tasks"""
        combined = PhaseAnalyses(analyses=tasks)
        self.publish_work(combined, input_task=tasks[0])
```

### Key Features

#### Automatic Aggregation
- Automatically groups tasks by provenance prefix
- Waits for all tasks from the join_type worker to complete
- Delivers sorted results for reproducibility

#### join_type
Specifies which upstream worker to join on:
```python
join_type: Type[TaskWorker] = SomeUpstreamWorker
```

Tasks are grouped by their provenance prefix for this worker type. ```InitialTaskWorker``` can be used to reference the provenance of the initial tasks submitted to the graph. For example, if the first worker ends up splitting the work by publishing multiple different outputs, they can be joined together using the ```InitialTaskWorker```.

#### enable_trace
```python
enable_trace: bool = Field(default=False)
```
Enable dashboard tracing for the join operation. Only needed in rare debugging cases. Should not be enabled by default.

### Validation Rules
- `join_type` must be a TaskWorker subclass
- `join_type` must be in the upstream path
- Cannot join on immediate upstream worker (would only ever have one result)

## Lifecycle Methods

### init
```python
def init(self):
    """Called when graph starts execution"""
    # Initialize resources, connections, etc.
    pass
```

### completed
```python
def completed(self):
    """Called when worker finishes processing all work"""
    # Cleanup resources
    pass
```

## Real-World Usage Patterns

### Simple Data Processing

```python
class SearchSplitter(TaskWorker):
    output_types: List[Type[Task]] = [SearchQuery]

    def consume_work(self, task: SearchQueries):
        for query in task.queries:
            search_task = SearchQuery(
                query=query.query, 
                metadata=query.phase
            )
            self.publish_work(search_task, input_task=task)
```

### Multi-Output Processing

```python
class PlanWorker(LLMTaskWorker):
    output_types: List[Type[Task]] = [Plan, Response]
    
    def post_process(self, response: Plan, input_task: Request):
        # Publish the plan for downstream processing
        self.publish_work(response, input_task=input_task)
        
        # Also publish a status update
        status = Response(
            response_type="thinking",
            phase="plan",
            message=response.response
        )
        self.publish_work(status, input_task=input_task)
```

### Complex Workflow Coordination

```python
class SearchCreator(LLMTaskWorker):
    def format_prompt(self, input_task: Plan) -> str:
        # Find original request from beginning of pipeline
        request: Request = input_task.find_input_task(Request)
        if request is None:
            raise ValueError("Missing Request task in provenance")
        
        return self.prompt.format(request=request.user_input)
    
    def pre_consume_work(self, task: Plan):
        self.notify_status(task, "Creating search queries for each phase")
```

## Best Practices

### 1. Type Safety
```python
class TypedWorker(TaskWorker):
    output_types: List[Type[Task]] = [OutputTask]  # Always specify
    
    def consume_work(self, task: InputTask):  # Use type hints
        # Implementation
        pass
```

### 2. Safe Task Publishing
```python
def consume_work(self, task: SomeTask):
    # Create new task objects
    result = ProcessedTask(data=task.process())
    self.publish_work(result, input_task=task)
    
    # Use copy_public() if republishing existing tasks
    clean_task = existing_task.copy_public()
    self.publish_work(clean_task, input_task=task)
```

### 3. Resource Management
```python
def init(self):
    """Initialize resources when graph starts"""
    self.connection = create_connection()

def completed(self):
    """Cleanup when done"""
    if hasattr(self, 'connection'):
        self.connection.close()
```

### 5. Provenance Usage
```python
def consume_work(self, task: FinalTask):
    # Find context from earlier in the pipeline
    original_request = task.find_input_task(UserRequest)
    config = task.find_input_task(ConfigTask)
    
    # Use provenance for decision making
    if original_request.priority == "high":
        self.process_urgently(task)
```

## Advanced Patterns

### Conditional Processing
```python
class ConditionalProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ResultA, ResultB]
    
    def consume_work(self, task: InputTask):
        if task.condition:
            result = ResultA(data=task.data)
        else:
            result = ResultB(data=task.data)
        
        self.publish_work(result, input_task=task)
```

### Multi-Consumer Publishing
```python
class Broadcaster(TaskWorker):
    output_types: List[Type[Task]] = [BroadcastTask]
    
    def consume_work(self, task: InputTask):
        for i in range(3):
            # Create separate instances for each consumer
            broadcast = BroadcastTask(
                data=task.data,
                instance=i
            )
            self.publish_work(broadcast, input_task=task)
```

### State-Driven Processing
```python
class StatefulProcessor(TaskWorker):
    output_tasks: List[Type[Task]] = [NormalResult, AggregatedResult]
    def consume_work(self, task: StateTask):
        # this state is conditioned on the input provenance
        # tasks with a different prefix will get their own state
        state = self.get_worker_state(task.prefix(1))
        
        if 'counter' not in state:
            state['counter'] = 0
        
        state['counter'] += 1
        
        if state['counter'] >= 5:
            # Process after collecting 5 tasks
            result = AggregatedResult(count=state['counter'])
            self.publish_work(result, input_task=task)
        else:
            # Regular processing
            result = NormalResult()
            self.publish_work(result, input_task=task)
```

This can be useful when looping in the graph and needing to terminate processing
after the specific number of retries.

## Integration with Graphs

### Worker Registration
```python
graph = Graph(name="Processing Pipeline")

# Add workers
worker1 = DataLoader()
llm = llm_from_config(provider="ollama", model_name="gemma3:27b")
worker2 = DataProcessor(llm=llm)
worker3 = ResultWriter()

graph.add_workers(worker1, worker2, worker3)

# Set dependencies
graph.set_dependency(worker1, worker2).next(worker3)

# Configure endpoints
graph.set_entry(worker1)
graph.set_sink(worker3, ResultTask)
```

### Parallel Execution Control
```python
# Limit parallel LLM calls
graph.set_max_parallel_tasks(LLMTaskWorker, 3)

# Limit specific worker types
graph.set_max_parallel_tasks(ExpensiveWorker, 1)
```

## Error Handling and Retries

### Retry Configuration
```python
class RetryableWorker(TaskWorker):
    num_retries: int = 3
    
    def consume_work(self, task: Task):
        if self.should_retry(task):
            raise RetryableError("Temporary failure")
        # Process normally
```

### Graceful Degradation
```python
def consume_work(self, task: ComplexTask):
    try:
        result = self.complex_processing(task)
    except ProcessingError:
        # Fall back to simpler processing
        result = self.simple_processing(task)
    
    self.publish_work(result, input_task=task)
```

## See Also

- [Task](/api/task/) - Data units processed by TaskWorkers
- [Graph](/api/graph/) - Orchestrating TaskWorker workflows  
- [Provenance](/guide/provenance/) - Understanding task lineage
- [LLM Integration](/features/llm-integration/) - Working with language models 