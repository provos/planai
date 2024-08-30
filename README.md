# PlanAI

**PlanAI** is an innovative system designed for complex task automation through a sophisticated graph-based architecture. It integrates traditional computations and cutting-edge AI technologies to enable versatile and efficient workflow management.

## Key Features

- **Graph-Based Architecture**: Construct dynamic workflows comprising interconnected TaskWorkers for highly customizable automation.
- **Hybrid TaskWorkers**: Combine conventional computations (e.g., API calls) with powerful LLM-driven operations, leveraging Retrieval-Augmented Generation (RAG) capabilities.
- **Type Safety with Pydantic**: Ensure data integrity and type consistency across workflows with Pydantic-validated input and output.
- **Intelligent Data Routing**: Utilize type-aware routing to efficiently manage data flow between nodes, adapting to multiple downstream consumers.

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/planai.git
   cd planai
   ```

2. Install dependencies:

   ```bash
   poetry install
   ```

## Usage

PlanAI allows you to create complex, AI-enhanced workflows using a graph-based architecture. Here's a basic example of how to use PlanAI:

```python
from planai import Graph, TaskWorker, LLMTaskWorker, llm_from_config
from planai.task import TaskWorkItem
from pydantic import Field

# Define custom TaskWorkers
class CustomDataProcessor(TaskWorker):
    output_types = [ProcessedData]
    
    def consume_work(self, task: RawData):
        # Process the raw data
        processed_data = self.process(task.data)
        self.publish_work(ProcessedData(data=processed_data))

# Define an LLM-powered task
class AIAnalyzer(LLMTaskWorker):
    output_types = [AnalysisResult]
    
    def consume_work(self, task: ProcessedData):
        analysis = self.llm.analyze(task.data)
        self.publish_work(AnalysisResult(analysis=analysis))

# Create a graph
graph = Graph(name="Data Analysis Workflow")

# Initialize tasks
data_processor = CustomDataProcessor()
ai_analyzer = AIAnalyzer(llm=llm_from_config(provider="openai", model_name="gpt-4"))

# Add tasks to the graph and set dependencies
graph.add_workers(data_processor, ai_analyzer)
graph.set_dependency(data_processor, ai_analyzer)

# Run the graph
initial_data = RawData(data="Some raw data")
graph.run(initial_tasks=[(data_processor, initial_data)])
```

This example demonstrates:

1. Creating custom `TaskWorker` classes for specific operations.
2. Utilizing `LLMTaskWorker` for AI-powered tasks.
3. Building a `Graph` to define the workflow.
4. Setting up dependencies between tasks.
5. Executing the workflow with initial data.

PlanAI supports more advanced features like:

- Caching results with `CachedTaskWorker`
- Joining multiple task results with `JoinedTaskWorker`
- Integrating with various LLM providers (OpenAI, Ollama, etc.)

For more detailed examples and advanced usage, please refer to the `examples/` directory in the repository.