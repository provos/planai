---
title: Provenance in PlanAI
description: Understanding and leveraging task lineage tracking in your workflows
---

In PlanAI, provenance refers to the lineage of a task, providing a historical record of its execution within a workflow. It allows tracing the origin and transformations of a task as it moves through different workers in the graph.

## Benefits of Provenance Tracking

### Debugging and Auditing
By tracing the history of a task, you can identify the specific workers and inputs that contributed to its final state, which is crucial for debugging errors or auditing decisions made by AI agents within the workflow.

### Explainability
Provenance can be used to explain the results of a workflow by showing the steps involved and how data was transformed at each stage. This is important for understanding and trusting the output of AI-driven workflows.

## How PlanAI Uses Provenance

PlanAI leverages provenance information in several ways:

- **Task Lineage**: Each task in PlanAI carries its full execution history, including the sequence of workers it passed through and the inputs it was derived from.
- **Debugging and Monitoring**: The PlanAI dashboard displays the provenance of each task, allowing developers to track its progress and identify any bottlenecks or failures.
- **AI Decision Tracking**: When AI agents make decisions within a workflow, the provenance records the rationale and data behind those decisions, promoting transparency and accountability.

## Working with Provenance

### Accessing Provenance Information

Every task in PlanAI has built-in methods to access its provenance:

```python
class AnalysisWorker(TaskWorker):
    def consume_work(self, task: DataTask):
        # Get the full provenance chain
        full_provenance = task.copy_provenance()
        
        # Find a specific task type in the provenance
        original_input = task.find_input_task(UserInput)
        
        # Get the immediately previous task
        previous = task.previous_input_task()
        
        # Get provenance prefix for a specific worker type
        prefix = task.prefix_for_input_task(DataFetcher)
```

### Provenance Chain Example

Consider a workflow where data flows through multiple stages:

```python
UserInput → DataFetcher → DataProcessor → AIAnalyzer → ResultFormatter
```

At the `ResultFormatter` stage, you can access any previous stage:

```python
class ResultFormatter(TaskWorker):
    def consume_work(self, task: AnalysisResult):
        # Access the original user input
        user_input = task.find_input_task(UserInput)
        
        # Access intermediate processing results
        raw_data = task.find_input_task(FetchedData)
        processed_data = task.find_input_task(ProcessedData)
        
        # Format result with full context
        formatted_result = self.format_with_context(
            analysis=task,
            original_request=user_input,
            data_source=raw_data.source
        )
```

## Advanced Provenance Features

### Task Joining Using Provenance

PlanAI allows workers to join multiple tasks based on their provenance prefix. This is particularly useful in workflows where you need to consolidate results from multiple parallel tasks.

#### Search Result Consolidation
When multiple search queries are executed in parallel, a join worker can wait for all results with the same search query provenance and combine them for analysis.

```python
class SearchResultAggregator(JoinedTaskWorker):
    join_type: Type[TaskWorker] = SearchInitiator
    output_types: List[Type[Task]] = [ConsolidatedResults]
    
    def consume_work_joined(self, tasks: List[SearchResult]):
        # All tasks share the same provenance from SearchInitiator
        consolidated = self.merge_search_results(tasks)
        self.publish_work(ConsolidatedResults(data=consolidated), input_task=task)
```

#### Batch Processing
Tasks can be grouped by their origin (e.g., all tasks derived from a specific input) and processed together.

To implement a join, workers can:

1. Specify a worker type to join on using `join_type`
2. Receive tasks through `consume_work_joined()` when all tasks sharing the same provenance prefix are complete
3. Process the consolidated results as a single unit

### Context Retrieval Using Provenance

Workers can traverse the provenance chain of a task to access contextual information from earlier stages in the workflow. Common use cases include:

#### LLM Context Enhancement
Retrieving original user queries or intermediate results to provide better context to language models:

```python
class ContextAwareLLMWorker(LLMTaskWorker):
    def format_prompt(self, task: ProcessedData) -> str:
        # Get original user query for context
        user_query = task.find_input_task(UserQuery)
        
        # Get intermediate analysis
        initial_analysis = task.find_input_task(InitialAnalysis)
        
        return f"""
        Original Query: {user_query.question}
        Initial Analysis: {initial_analysis.summary}
        
        Current Data: {task.content}
        
        Please provide a detailed response considering the full context.
        """
```

#### Decision Tracking
Accessing the rationale or inputs that led to specific outcomes earlier in the workflow:

```python
class AuditReporter(TaskWorker):
    def consume_work(self, task: FinalResult):
        # Trace all decisions made
        decisions = []
        
        # Find all AI decision points
        if ai_decision := task.find_input_task(AIDecision):
            decisions.append({
                "stage": "AI Analysis",
                "input": ai_decision.input_data,
                "output": ai_decision.decision,
                "reasoning": ai_decision.reasoning
            })
        
        # Generate audit report
        self.generate_audit_report(task, decisions)
```

#### Metadata Propagation
Carrying important metadata or configuration through the workflow:

```python
class MetadataAwareWorker(TaskWorker):
    def consume_work(self, task: DataTask):
        # Get configuration from the start of the workflow
        config = task.find_input_task(WorkflowConfig)
        
        # Apply configuration-specific processing
        if config.enable_caching:
            result = self.process_with_cache(task)
        else:
            result = self.process_without_cache(task)
        
        self.publish_work(result, input_task=task)
```

## Best Practices

1. **Use Provenance for Debugging**: When errors occur, trace the provenance to understand what led to the failure
2. **Maintain Context**: Use provenance to maintain important context throughout your workflow
3. **Document Dependencies**: When accessing provenance, document which previous tasks your worker depends on

Overall, the provenance tracking in PlanAI makes it easy to create reproducible, and explainable AI-driven workflows. Provenance allows developers to fully understand all the computation that happened at any given points in the graph.