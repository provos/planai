---
applyTo: '**/*.py'
---
# PlanAI Best Practices for LLM-Assisted Coding

This document outlines best practices for using PlanAI, a Python framework for building complex, AI-enhanced workflows using a graph-based, data-flow architecture. PlanAI assumes TaskWorker classes to be unique. They don't get instantiated multiple times.

## Core Concepts & Best Practices

### 1. Graph-Based Architecture

* **Concept:** Workflows are defined as `planai.Graph` objects. A graph consists of nodes (`TaskWorker` subclasses) connected by dependencies, defining the data flow. Execution is parallel, constrained only by these data dependencies.
* **Best Practice:** Design workflows modularly. Break down complex processes into distinct `TaskWorker` units representing logical steps.
* **Instantiation:** Always instantiate a graph using `graph = Graph(name="YourWorkflowName")`.
* **Strict Mode:** Consider setting `strict=True` during graph initialization (`Graph(name="...", strict=True)`) to enforce stricter validation, like preventing the publishing of tasks that still hold provenance data, which can help catch bugs related to task reuse.

### 2. TaskWorkers: The Building Blocks

* **Concept:** Workers are Python classes inheriting from `planai.TaskWorker` (or specialized subclasses) that perform specific processing steps.
* **Core Logic:** Implement the primary logic within the `consume_work(self, task: YourInputTask)` method. Use type hints for the input `task` parameter for clarity and static analysis benefits.
* **Output:** Use `self.publish_work(output_task, input_task=task)` to send results downstream. It's crucial to pass the `input_task` to maintain provenance. Use `task.copy_public()` when creating `output_task` from the `input_task` if you don't want to pass along private state or provenance, especially in `strict` mode graphs[cite: 4654, 4655, 4656, 4657, 4658].
* **Input Type Declaration:** Define expected input types by type-hinting the `task` parameter in `consume_work`. For `LLMTaskWorker`, use `llm_input_type: Type[YourInputTask] = YourInputTask` if not overriding `consume_work`.
* **Output Type Declaration:** Explicitly declare output types using `output_types: List[Type[Task]] = [YourOutputTask1, YourOutputTask2]`. This enables type-aware routing.
* **Logging/Printing:** Use `self.print("Log message")` within workers instead of `print()`. This integrates with the graph's logging and dashboard.
* **Status Notifications:** Use `self.notify_status(task, "Descriptive status message")` to provide real-time updates visible in the dashboard or via callbacks.
* **Error Handling:** Implement `try...except` blocks within `consume_work` for robust error handling. Failed tasks are automatically tracked by the dispatcher.
* **Retries:** Set `num_retries: int = N` on a worker class to automatically retry failed tasks `N` times.

### 3. Tasks: Data Flow Units

* **Concept:** Data moves through the graph encapsulated in `planai.Task` objects, typically defined as Pydantic models.
* **Best Practice:** Define specific Pydantic models for each distinct data structure flowing between workers. This enforces type safety and improves code readability.
    ```python
    from planai import Task
    from pydantic import Field
    from typing import List

    class UserQuery(Task):
        query_text: str = Field(description="The user's input query")
        user_id: str

    class SearchResults(Task):
        query: str
        results: List[str] = Field(description="List of URLs")
    ```
* **Provenance:** Tasks automatically carry their execution history (`_provenance`) and the chain of tasks that led to them (`_input_provenance`).

### 4. Type Safety with Pydantic

* **Concept:** PlanAI heavily relies on Pydantic for defining `Task` data structures and ensuring type correctness during data flow.
* **Best Practice:** Define all `Task` subclasses using Pydantic models with clear field descriptions. Use Python's typing hints (`List`, `Optional`, `Type`, etc.).
* **Benefits:** Enables automatic data validation, clear interface definitions between workers, and type-aware routing by the dispatcher.

### 5. Defining Workflows

* **Steps:**
    1.  Instantiate `Graph()`.
    2.  Instantiate all necessary `TaskWorker` subclasses.
    3.  Add workers using `graph.add_workers(worker1, worker2, ...)`.
    4.  Define dependencies using `graph.set_dependency(upstream_worker, downstream_worker)`. Chain dependencies using `.next(another_worker)`.
    5.  Identify entry points using `graph.set_entry(entry_worker1, entry_worker2, ...)`. These workers receive the initial tasks.
    6.  Optionally define sinks using `graph.set_sink(exit_worker, OutputTaskType, notify=callback_func)` to collect final results or trigger notifications.
    7.  Finalize graph structure analysis (optional but recommended for complex graphs): `graph.finalize()`.
    8.  Run the workflow using `graph.run(initial_tasks=[(entry_worker, initial_task_data)], ...)` or `graph.prepare(...)` followed by `graph.execute(...)`.
    9.  If a sink was specified without a callback, the results can be collected from the graph via `graph.get_output_tasks()`.

* **Example Graph Setup:**
    ```python
    graph = Graph(name="ExampleWorkflow")
    fetcher = DataFetcher()
    processor = DataProcessor()
    analyzer = AIAnalyzer(llm=my_llm)
    final_output = FinalOutputWorker()

    graph.add_workers(fetcher, processor, analyzer, final_output)

    graph.set_dependency(fetcher, processor)\
         .next(analyzer)\
         .next(final_output) # Chaining dependencies

    graph.set_entry(fetcher) # Define entry point
    graph.set_sink(final_output, FinalResult) # Define sink

    initial_work = [(fetcher, FetchRequest(url="...") )]
    graph.run(initial_tasks=initial_work, display_terminal=False)
    results = graph.get_output_tasks() # Collect results
    ```

### 6. LLM Integration (`LLMTaskWorker`)

* **Concept:** Subclass `planai.LLMTaskWorker` or `planai.CachedLLMTaskWorker` for workers interacting with LLMs.
* **Initialization:** Requires an `llm: LLMInterface` instance (use `planai.llm_from_config`) and a `prompt: str`[cite: 2871, 3989].
* **Input/Output Types:** Define `llm_input_type` if not overriding `consume_work`. Define `output_types` (or `llm_output_type` if the Pydantic model differs from the final worker output)[cite: 2871, 4505, 4508].
* **Prompt Customization:**
    * **Basic:** Provide a static `prompt` string during initialization.
    * **Dynamic:** Override `format_prompt(self, task: YourInputTask) -> str`. Use f-strings or `.format()` with data accessed from the `task` (potentially using `task.find_input_task(OtherTask)`)[cite: 2836, 2837].
    * **System Prompt:** Set `system_prompt: str = "Your system message"` for context.
    * **XML Input:** Set `use_xml: bool = True` if the input data contains complex structures or newlines that are better represented as XML for the LLM.
* **Pre-Processing Input:** Override `pre_process(self, task: YourInputTask) -> Optional[Task]` to modify or filter the data sent to the LLM. Return `None` to omit the `{task}` section from the default prompt template entirely[cite: 2838, 2839, 2840]. Use `planai.PydanticDictWrapper` to dynamically shape data for the LLM.
* **Post-Processing Output:** Override `post_process(self, response: LLMOutputType, input_task: YourInputTask)` to modify the LLM's response before publishing. Call `super().post_process(...)` to proceed with publishing.
* **Validation:** Override `extra_validation(self, response: LLMOutputType, input_task: YourInputTask) -> Optional[str]` to add custom validation logic. Return an error string if validation fails, triggering a retry with the LLM.
* **Debug Mode:** Set `debug_mode: bool = True` to save LLM prompts and responses to JSON files in `debug_dir` (default: `./debug/`) for analysis and optimization[cite: 4506, 4538].
* **Parallelism:** Limit concurrent LLM calls using `graph.set_max_parallel_tasks(LLMTaskWorker, N)`.

### 7. Input Provenance

* **Concept:** PlanAI automatically tracks the lineage of each task. Any worker can inspect the history of the task it receives.
* **Accessing Provenance:** Inside `consume_work`, use:
    * `task.copy_provenance()`: Get the full list of (worker_name, id) tuples.
    * `task.find_input_task(SpecificTaskType)`: Find the most recent instance of `SpecificTaskType` in the input chain.
    * `task.previous_input_task()`: Get the immediate predecessor task.
    * `task.prefix_for_input_task(SpecificWorkerType)`: Get the provenance chain up to the first occurrence of `SpecificWorkerType`.
* **Use Case:** Essential for tasks needing context from earlier steps (e.g., `JoinedTaskWorker`, providing original query to later LLM calls).

### 8. Advanced Workers

* **`CachedTaskWorker` / `CachedLLMTaskWorker`:** Automatically caches results based on the input task hash and worker configuration (including prompt for LLM workers). Avoids recomputing identical tasks[cite: 2884, 3995]. Customize caching behavior by overriding `extra_cache_key(self, task: Task) -> str`.
* **`JoinedTaskWorker`:** Waits for multiple upstream tasks sharing a common ancestor (defined by `join_type`) to complete before processing them together[cite: 2885, 4475]. Implement logic in `consume_work_joined(self, tasks: List[JoinedTaskType])`.
* **`MergedTaskWorker`:** Similar to `JoinedTaskWorker`, but merges tasks of different types (`merged_types`) based on a `join_type`. Delivers results as a dictionary `Dict[str, List[Task]]` to `consume_work_merged`[cite: 4551, 4559].
* **`SubGraphWorker`:** Encapsulates an entire `Graph` as a single worker within a larger graph. Define using `planai.SubGraphWorker(graph=subgraph, entry_worker=sub_entry, exit_worker=sub_exit)`[cite: 4296, 4314]. Useful for complex, reusable workflow components.

### 9. Monitoring & Debugging

* **Dashboard:** Enable the web dashboard with `graph.run(..., run_dashboard=True)`. Access at `http://localhost:5000` (default) for real-time monitoring of tasks, provenance, logs, and stats[cite: 2872, 2877, 5533]. Use `dashboard_port` argument to change the port.
* **Terminal Display:** Use `graph.run(..., display_terminal=True)` for a live text-based status overview in the console.
* **Logging:** Use `self.print(...)` within workers. Configure global logging with `planai.utils.setup_logging()`.
* **LLM Debug Logs:** Set `debug_mode=True` on `LLMTaskWorker` instances to log prompts/responses.