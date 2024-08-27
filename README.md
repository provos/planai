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

### Usage

Define your TaskWorkers, connect them using the graph-based framework, and execute your workflow with type-safe routing and data management. For detailed examples, refer to the `examples/` directory.
