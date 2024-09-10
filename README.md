# PlanAI

[![PyPI version](https://badge.fury.io/py/planai.svg)](https://badge.fury.io/py/planai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/planai.svg)](https://pypi.org/project/planai/)
[![Documentation Status](https://readthedocs.org/projects/planai/badge/?version=latest)](https://docs.getplanai.com/en/latest/?badge=latest)

**PlanAI** is an innovative system designed for complex task automation through a sophisticated graph-based architecture. It integrates traditional computations and cutting-edge AI technologies to enable versatile and efficient workflow management.

## Table of Contents
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example: Textbook Q&A Generation](#example-textbook-qa-generation)
- [Monitoring Dashboard](#monitoring-dashboard)
- [Advanced Features](#advanced-features)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Key Features

- **Graph-Based Architecture**: Construct dynamic workflows comprising interconnected TaskWorkers for highly customizable automation.
- **Hybrid TaskWorkers**: Combine conventional computations (e.g., API calls) with powerful LLM-driven operations, leveraging Retrieval-Augmented Generation (RAG) capabilities.
- **Type Safety with Pydantic**: Ensure data integrity and type consistency across workflows with Pydantic-validated input and output.
- **Intelligent Data Routing**: Utilize type-aware routing to efficiently manage data flow between nodes, adapting to multiple downstream consumers.
- **Input Provenance Tracking**: Trace the lineage and origin of each Task as it flows through the workflow, enabling detailed analysis and debugging of complex processes.

## Requirements

- Python 3.10+
- Poetry (for development)

## Installation

You can install PlanAI using pip:

```bash
pip install planai
```

For development, clone the repository and install dependencies:

```bash
git clone https://github.com/provos/planai.git
cd planai
poetry install
```

## Usage

PlanAI allows you to create complex, AI-enhanced workflows using a graph-based architecture. Here's a basic example:

```python
from planai import Graph, TaskWorker, Task, LLMTaskWorker, llm_from_config

# Define custom TaskWorkers
class CustomDataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]

    def consume_work(self, task: RawData):
        processed_data = self.process(task.data)
        self.publish_work(ProcessedData(data=processed_data))

# Define an LLM-powered task
class AIAnalyzer(LLMTaskWorker):
    prompt: str ="Analyze the provided data and derive insights"
    output_types: List[Type[Task]] = [AnalysisResult]

    def consume_work(self, task: ProcessedData):
        super().consume_work(task)

# Create and run the workflow
graph = Graph(name="Data Analysis Workflow")
data_processor = CustomDataProcessor()
ai_analyzer = AIAnalyzer(
   llm=llm_from_config(provider="openai", model_name="gpt-4"))

graph.add_workers(data_processor, ai_analyzer)
graph.set_dependency(data_processor, ai_analyzer)

initial_data = RawData(data="Some raw data")
graph.run(initial_tasks=[(data_processor, initial_data)])
```

## Example: Textbook Q&A Generation

PlanAI has been used to create a system for generating high-quality question and answer pairs from textbook content. This example demonstrates PlanAI's capability to manage complex, multi-step workflows involving AI-powered text processing and content generation. The application processes textbook content through a series of steps including text cleaning, relevance filtering, question generation and evaluation, and answer generation and selection. For a detailed walkthrough of this example, including code and explanation, please see the [examples/textbook](examples/textbook) directory. The resulting dataset, generated from "World History Since 1500: An Open and Free Textbook," is available in our [World History 1500 Q&A repository](https://github.com/provos/world-history-1500-qa), showcasing the practical application of PlanAI in educational content processing and dataset creation.

## Monitoring Dashboard

PlanAI includes a built-in web-based monitoring dashboard that provides real-time insights into your graph execution. This feature can be enabled by setting `run_dashboard=True` when calling the `graph.run()` method.

Key features of the monitoring dashboard:

- **Real-time Updates**: The dashboard uses server-sent events (SSE) to provide live updates on task statuses without requiring page refreshes.
- **Task Categories**: Tasks are organized into three categories: Queued, Active, and Completed, allowing for easy tracking of workflow progress.
- **Detailed Task Information**: Each task displays its ID, type, and assigned worker. Users can click on a task to view additional details such as provenance and input provenance.

To enable the dashboard:

```python
graph.run(initial_tasks, run_dashboard=True)
```

When enabled, the dashboard will be accessible at `http://localhost:5000` by default. The application will continue running until manually terminated, allowing for ongoing monitoring of long-running workflows.

Note: Enabling the dashboard will block the main thread, so it's recommended for development and debugging purposes. For production use, consider implementing a separate monitoring solution.

## Advanced Features

PlanAI supports advanced features like:

- Caching results with `CachedTaskWorker`
- Joining multiple task results with `JoinedTaskWorker`
- Integrating with various LLM providers (OpenAI, Ollama, etc.)

For more detailed examples and advanced usage, please refer to the `examples/` directory in the repository.

## Documentation

Full documentation for PlanAI is available at [https://docs.getplanai.com/](https://docs.getplanai.com/)

## Contributing

We welcome contributions to PlanAI! Please see our [Contributing Guide](CONTRIBUTING.md) for more details on how to get started.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

For any questions or support, please open an issue on our [GitHub issue tracker](https://github.com/provos/planai/issues).