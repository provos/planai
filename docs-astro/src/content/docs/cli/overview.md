---
title: CLI Overview
description: Command-line interface reference for PlanAI
---

PlanAI provides a powerful command-line interface for monitoring workflows, optimizing prompts, and managing your AI automation tasks.

## Installation

The CLI is automatically available after installing PlanAI:

```bash
pip install planai
```

Verify installation:

```bash
planai --help
```

## Global Options

These options are available for all commands:

```bash
planai [global-options] <command> [command-options]
```

### LLM Configuration

Configure the LLM provider for commands that use AI:

```bash
# Specify provider and model
planai --llm-provider openai --llm-model gpt-4 <command>

# Use different model for reasoning tasks
planai --llm-provider openai --llm-model gpt-4 --llm-reason-model gpt-4 <command>

# Use local Ollama models
planai --llm-provider ollama --llm-model llama2 <command>
```

### Environment Variables

Set defaults using environment variables:

```bash
export PLANAI_LLM_PROVIDER=openai
export PLANAI_LLM_MODEL=gpt-4
export OPENAI_API_KEY=your-api-key
```

## Available Commands

### cache

Examine the planai cache

```bash
# Check out the cached tasks
planai cache ./cache

# Filter cache based on the Output Task
planai cache --output-task-filter PageResult ./cache
```

Options:

- `--clear`: Clear the cache
- `--output-task-filter`: Filter the output based on the corresponding output task

### optimize-prompt

Automatically optimize prompts using AI and production data:

```bash
planai --llm-provider openai --llm-model gpt-4o-mini --llm-reason-model gpt-4 \
  optimize-prompt \
  --python-file app.py \
  --class-name MyLLMWorker \
  --search-path . \
  --debug-log debug/MyLLMWorker.json \
  --goal-prompt "Improve accuracy while reducing token usage"
```

Required arguments:

- `--python-file`: Python file containing the LLMTaskWorker
- `--class-name`: Name of the LLMTaskWorker class to optimize
- `--search-path`: Python path for imports
- `--debug-log`: Debug log file with production data
- `--goal-prompt`: Optimization goal description

Optional arguments:

- `--num-iterations`: Number of optimization iterations (default: 3)
- `--output-dir`: Directory for optimized prompts (default: current directory)
- `--max-samples`: Maximum debug samples to use (default: all)

See the [Prompt Optimization guide](/cli/prompt-optimization/) for detailed usage.

### version

Display PlanAI version information:

```bash
planai version
```

Output:

```
PlanAI version 0.6.1
```

### help

Get help for any command:

```bash
# General help
planai --help

# Command-specific help
planai optimize-prompt --help
planai cache --help
```

## Common Workflows

### Development Workflow

During development, use the terminal dashboard to track execution which is enabled by default:

```bash
# Run your workflow and watch the terminal output
python my_workflow.py
```

Alternatively, you can pass `run_dashboard=True` to the Graph `run` or `prepare` method.
By default, this will create a web based dashboard on port `5000`.

### Prompt Optimization Workflow

1. Enable debug mode in your LLMTaskWorker:

```python
class MyWorker(LLMTaskWorker):
    debug_mode = True  # Generates debug logs
```

2. Run your workflow to collect data

3. Optimize the prompt:

```bash
planai --llm-provider openai --llm-model gpt-4o-mini \
  optimize-prompt \
  --python-file my_worker.py \
  --class-name MyWorker \
  --debug-log debug/MyWorker.json \
  --goal-prompt "Improve response quality"
```

## Next Steps

- Learn about [Prompt Optimization](/cli/prompt-optimization/) in detail
- Explore [Monitoring](/guide/monitoring/) capabilities
- See [Examples](https://github.com/provos/planai/tree/main/examples) using the CLI
