# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PlanAI is a Python framework for building and orchestrating AI workflows using a graph-based task execution model. It combines traditional computation with LLM capabilities through a type-safe, composable architecture.

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies (using Poetry)
poetry install

# Install pre-commit hooks (required for all contributions)
pre-commit install
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/planai/test_graph.py

# Run with coverage
poetry run pytest --cov=planai

# Run regression tests (longer-running tests)
poetry run pytest --run-regression

# Run JavaScript tests for web interface
npm test
```

### Code Quality
```bash
# Format code with Black
poetry run black src/ tests/

# Run linting
poetry run flake8 src/ tests/

# Type checking
poetry run mypy src/

# Run all pre-commit checks
pre-commit run --all-files
```

### Documentation
```bash
# Build Sphinx documentation
cd docs && make html

# View built docs
open docs/_build/html/index.html
```

### CLI Usage
```bash
# Run the PlanAI CLI
poetry run planai --help

# Monitor a graph execution
poetry run planai monitor --web  # Web interface on localhost:5000
poetry run planai monitor        # Terminal interface
```

## Architecture Overview

### Core Concepts

1. **Graph-Based Execution**: Tasks flow through a directed graph of TaskWorkers. The Graph class manages execution, dependencies, and parallelism.

2. **Task/TaskWorker Pattern**:
   - `Task` (src/planai/task.py): Pydantic models representing units of work
   - `TaskWorker` (src/planai/task_worker.py): Abstract processors that consume and produce tasks
   - Workers are typed to specific input/output task types for type safety

3. **Key Worker Types**:
   - `InitialTaskWorker`: Entry point for external data
   - `LLMTaskWorker`: Integrates LLM capabilities with optional tool calling
   - `CachedTaskWorker`: Provides caching layer for expensive operations
   - `JoinedTaskWorker`: Combines results from multiple task types
   - `SubGraphWorker`: Enables graph composition

4. **Provenance System**: Every task maintains its execution history through the provenance system, enabling debugging and workflow analysis.

### Directory Structure

- `src/planai/`: Core framework code
  - `graph.py`: Graph execution engine
  - `task.py`, `task_worker.py`: Base classes
  - `llm_task_worker.py`: LLM integration
  - `joined_task_worker.py`: Task joining logic
  - `cached/`: Caching implementations
  - `integrations/`: External service integrations
  - `cli.py`: Command-line interface

- `examples/`: Reference implementations showing different patterns
- `tests/`: Comprehensive test suite with fixtures and utilities

### Key Design Patterns

1. **Type Safety**: Extensive use of Pydantic models and Python type hints throughout
2. **Composition**: Workers can be composed to build complex workflows
3. **Async Support**: Graph execution supports concurrent task processing
4. **Monitoring**: Built-in web and terminal interfaces for execution monitoring
5. **Caching**: Multiple caching strategies for performance optimization

### LLM Integration

The framework uses the `llm-interface` library for provider-agnostic LLM access. Key features:
- Support for OpenAI, Ollama, and other providers
- Tool/function calling capabilities
- Response caching for development efficiency
- Structured output with Pydantic models

### Testing Philosophy

- Unit tests for all core components
- Integration tests for worker interactions
- Regression tests (marked with `@pytest.mark.regression`) for complex scenarios
- Use fixtures in `tests/conftest.py` for common test setups

When implementing new features:
1. Ensure type annotations are complete
2. Add appropriate unit tests
3. Run the full test suite before committing
4. Follow the existing code style (enforced by Black)
5. Update documentation if adding public APIs