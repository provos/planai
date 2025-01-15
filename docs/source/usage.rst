Usage
=====

PlanAI is a powerful framework for creating complex, AI-enhanced workflows using a graph-based architecture. This guide will walk you through the basic concepts and provide examples of how to use PlanAI effectively.

Basic Concepts
--------------

1. **TaskWorker**: The fundamental building block of a PlanAI workflow. It defines how to process input and produce output.
2. **Graph**: A structure that defines the workflow by connecting TaskWorkers.
3. **Task**: The unit of work that flows through the graph. It carries data and provenance information.
4. **Input Provenance**: A mechanism to track the history and origin of each Task as it moves through the workflow.
5. **LLMTaskWorker**: A special type of TaskWorker that integrates with Language Models.

Creating a Simple Workflow
--------------------------

Here's a basic example of how to create and execute a simple workflow:

.. code-block:: python

    from planai import Graph, TaskWorker, Task

    # Define custom TaskWorkers
    class DataFetcher(TaskWorker):
        output_types = [FetchedData]

        def consume_work(self, task: FetchRequest):
            # Fetch data from some source
            data = self.fetch_data(task.url)
            self.publish_work(FetchedData(data=data))

    class DataProcessor(TaskWorker):
        output_types = [ProcessedData]

        def consume_work(self, task: FetchedData):
            # Process the fetched data
            processed_data = self.process(task.data)
            self.publish_work(ProcessedData(data=processed_data))

    # Create a graph
    graph = Graph(name="Data Processing Workflow")

    # Initialize tasks
    fetcher = DataFetcher()
    processor = DataProcessor()

    # Add tasks to the graph and set dependencies
    graph.add_workers(fetcher, processor)
    graph.set_dependency(fetcher, processor)

    # Run the graph
    initial_request = FetchRequest(url="https://example.com/data")
    graph.run(initial_tasks=[(fetcher, initial_request)])

Integrating AI with LLMTaskWorker
---------------------------------

PlanAI allows you to easily integrate AI capabilities into your workflow using LLMTaskWorker:

.. code-block:: python

    from planai import LLMTaskWorker, llm_from_config

    class AIAnalyzer(LLMTaskWorker):
        prompt="Analyze the processed data and provide insights."
        llm_input_type: Type[Task] = ProcessedData
        output_types = [AnalysisResult]


    # Initialize LLM
    llm = llm_from_config(provider="openai", model_name="gpt-4")

    # Add to workflow
    ai_analyzer = AIAnalyzer(
        llm=llm,
    )
    graph.add_worker(ai_analyzer)
    graph.set_dependency(processor, ai_analyzer)

Monitoring Dashboard
--------------------

PlanAI includes a built-in web-based monitoring dashboard that provides real-time insights into your graph execution. This feature can be enabled by setting ``run_dashboard=True`` when calling the ``graph.run()`` method.

Key features of the monitoring dashboard:

- **Real-time Updates**: The dashboard uses server-sent events (SSE) to provide live updates on task statuses without requiring page refreshes.
- **Task Categories**: Tasks are organized into three categories: Queued, Active, and Completed, allowing for easy tracking of workflow progress.
- **Detailed Task Information**: Each task displays its ID, type, and assigned worker. Users can click on a task to view additional details such as provenance and input provenance.

To enable the dashboard:

.. code-block:: python

    graph.run(initial_tasks, run_dashboard=True)

When enabled, the dashboard will be accessible at ``http://localhost:5000`` by default. The application will continue running until manually terminated, allowing for ongoing monitoring of long-running workflows. The default can be changed by setting the ``dashboard_port`` parameter.

Note: Enabling the dashboard will block the main thread, so it's recommended for development and debugging purposes. For production use, consider implementing a separate monitoring solution.

Advanced Features
-----------------

Input Provenance
^^^^^^^^^^^^^^^^

PlanAI provides powerful input provenance tracking capabilities, allowing you to trace the lineage of each Task:

.. code-block:: python

    class AnalysisTask(TaskWorker):
        output_types = [AnalysisResult]

        def consume_work(self, task: ProcessedData):
            # Access the full provenance chain
            provenance = task.copy_provenance()

            # Find a specific input task
            original_data = task.find_input_task(FetchedData)

            # Get the immediately previous input task
            previous_task = task.previous_input_task()

            # Get the provenance chain for a specific task type
            fetch_provenance = task.prefix_for_input_task(DataFetcher)

            # Perform analysis using the provenance information
            result = self.analyze(task.data, original_data, provenance)
            self.publish_work(AnalysisResult(result=result), input_task=task)

Input provenance allows you to:
- Trace the full history of a Task
- Find specific input tasks in the provenance chain
- Access the immediately previous input task
- Get the provenance chain for a specific task type

This feature is particularly useful for complex workflows where understanding the origin and transformation of data is crucial.

Caching Results
^^^^^^^^^^^^^^^

Use CachedTaskWorker to avoid redundant computations:

.. code-block:: python

    from planai import CachedTaskWorker

    class CachedProcessor(CachedTaskWorker):
        output_types = [ProcessedData]

        def consume_work(self, task: FetchedData):
            # Processing logic here
            pass

Joining Multiple Results
^^^^^^^^^^^^^^^^^^^^^^^^

JoinedTaskWorker allows you to combine results from multiple upstream tasks:

.. code-block:: python

    from planai import JoinedTaskWorker

    class DataAggregator(JoinedTaskWorker):
        output_types: List[Type[Task]] = [AggregatedData]
        join_type: Type[TaskWorker] = InitialTaskWorker

        def consume_work(self, task: ProcessedData):
            super().consume_work(task)

        def consume_work_joined(self, tasks: List[ProcessedData]):
            # Aggregation logic here
            pass

When instantiating DataAggregator, you need to specify a TaskWorker as join_type.

Best Practices
--------------

1. **Modular Design**: Break down complex tasks into smaller, reusable TaskWorkers.
2. **Type Safety**: Use Pydantic models for input and output types to ensure data consistency.
3. **Error Handling**: Implement proper error handling in your TaskWorkers to make workflows robust.
4. **Logging**: Utilize PlanAI's logging capabilities to monitor workflow execution.
5. **Testing**: Write unit tests for individual TaskWorkers and integration tests for complete workflows.

For more detailed examples and advanced usage, please refer to the `examples/` directory in the PlanAI repository.
