---
title: Installation
description: Get PlanAI up and running in your development environment
---

PlanAI can be installed using pip or poetry. Choose the method that best fits your project setup.

## Requirements

- Python 3.10 or higher
- pip or Poetry package manager

## Install with pip

The simplest way to install PlanAI is using pip:

```bash
pip install planai
```

## Install with Poetry

If you're using Poetry for dependency management:

```bash
poetry add planai
```

## Development Installation

To contribute to PlanAI or work with the latest development version:

1. Clone the repository:
```bash
git clone https://github.com/provos/planai.git
cd planai
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Verify Installation

To verify that PlanAI is installed correctly, you can run:

```bash
python -c "import planai; print(planai.Task)"
```

Or check the CLI:

```bash
planai --help
```

## Optional Dependencies

PlanAI has several optional dependencies for specific features:

### LLM Providers

To use specific LLM providers, you may need to install additional packages:

- **OpenAI**: Included by default via `llm-interface`
- **Ollama**: Requires Ollama to be installed and running locally
- **Other providers**: Check the [llm-interface documentation](https://github.com/provos/llm-interface) for supported providers

### Web Monitoring Dashboard

The monitoring dashboard requires Flask, which is included in the default installation.

## Next Steps

Now that you have PlanAI installed, you're ready to:

- Follow the [Quick Start guide](/getting-started/quickstart/) to build your first workflow
- Explore the [Basic Usage guide](/guide/usage/) for detailed examples
- Check out the [Examples](https://github.com/provos/planai/tree/main/examples) in the repository