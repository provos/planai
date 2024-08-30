# PlanAI

[![PyPI version](https://badge.fury.io/py/planai.svg)](https://badge.fury.io/py/planai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/planai.svg)](https://pypi.org/project/planai/)
[![Documentation Status](https://readthedocs.org/projects/planai/badge/?version=latest)](https://planai.readthedocs.io/en/latest/?badge=latest)


**PlanAI** is an innovative system designed for complex task automation through a sophisticated graph-based architecture. It integrates traditional computations and cutting-edge AI technologies to enable versatile and efficient workflow management.

## Table of Contents
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Advanced Features](#advanced-features)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Roadmap](#roadmap)

## Key Features

- **Graph-Based Architecture**: Construct dynamic workflows comprising interconnected TaskWorkers for highly customizable automation.
- **Hybrid TaskWorkers**: Combine conventional computations (e.g., API calls) with powerful LLM-driven operations, leveraging Retrieval-Augmented Generation (RAG) capabilities.
- **Type Safety with Pydantic**: Ensure data integrity and type consistency across workflows with Pydantic-validated input and output.
- **Intelligent Data Routing**: Utilize type-aware routing to efficiently manage data flow between nodes, adapting to multiple downstream consumers.

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
from planai import Graph, TaskWorker, LLMTaskWorker, llm_from_config
from planai.task import TaskWorkItem
from pydantic import Field

# Define custom TaskWorkers
class CustomDataProcessor(TaskWorker):
    output_types = [ProcessedData]
    
    def consume_work(self, task: RawData):
        processed_data = self.process(task.data)
        self.publish_work(ProcessedData(data=processed_data))

# Define an LLM-powered task
class AIAnalyzer(LLMTaskWorker):
    output_types = [AnalysisResult]
    
    def consume_work(self, task: ProcessedData):
        super().consume_work(task)

# Create and run the workflow
graph = Graph(name="Data Analysis Workflow")
data_processor = CustomDataProcessor()
ai_analyzer = AIAnalyzer(llm=llm_from_config(provider="openai", model_name="gpt-4"))

graph.add_workers(data_processor, ai_analyzer)
graph.set_dependency(data_processor, ai_analyzer)

initial_data = RawData(data="Some raw data")
graph.run(initial_tasks=[(data_processor, initial_data)])
```

## Advanced Features

PlanAI supports advanced features like:

- Caching results with `CachedTaskWorker`
- Joining multiple task results with `JoinedTaskWorker`
- Integrating with various LLM providers (OpenAI, Ollama, etc.)

For more detailed examples and advanced usage, please refer to the `examples/` directory in the repository.

## Documentation

Full documentation for PlanAI is available at [https://planai.readthedocs.io/](https://planai.readthedocs.io/)

## Contributing

We welcome contributions to PlanAI! Please see our [Contributing Guide](CONTRIBUTING.md) for more details on how to get started.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

For any questions or support, please open an issue on our [GitHub issue tracker](https://github.com/provos/planai/issues).