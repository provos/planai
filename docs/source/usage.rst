Usage
=====

PlanAI is a powerful framework for creating complex, AI-enhanced workflows using a graph-based architecture. This guide will walk you through the basic concepts and provide examples of how to use PlanAI effectively.

Basic Concepts
--------------

1. **TaskWorker**: The fundamental building block of a PlanAI workflow.
2. **Graph**: A structure that defines the workflow by connecting TaskWorkers.
3. **LLMTaskWorker**: A special type of TaskWorker that integrates with Language Models.

Creating a Simple Workflow
--------------------------

Here's a basic example of how to create and execute a simple workflow:

.. code-block:: python

    from planai import Graph, TaskWorker, TaskWorkItem
    from pydantic import Field

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
        output_types = [AnalysisResult]
        
        def consume_work(self, task: ProcessedData):
            super().consume_work(task)

    # Initialize LLM
    llm = llm_from_config(provider="openai", model_name="gpt-4")

    # Add to workflow
    ai_analyzer = AIAnalyzer(
        llm=llm,
        prompt="Analyze the processed data and provide insights.",
    )
    graph.add_worker(ai_analyzer)
    graph.set_dependency(processor, ai_analyzer)

Advanced Features
-----------------

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
        output_types = [AggregatedData]
        
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
