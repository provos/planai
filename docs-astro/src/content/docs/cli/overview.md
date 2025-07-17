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

### monitor

Monitor running workflows in real-time:

```bash
# Terminal-based monitoring
planai monitor

# Web-based dashboard
planai monitor --web

# Custom port for web interface
planai monitor --web --port 8080

# Monitor specific workflow
planai monitor --workflow "Data Processing Pipeline"
```

Options:
- `--web`: Launch web-based dashboard (default: terminal interface)
- `--port`: Port for web dashboard (default: 5000)
- `--workflow`: Filter by workflow name
- `--refresh`: Refresh interval in seconds (terminal mode)

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

### help

Get help for any command:

```bash
# General help
planai --help

# Command-specific help
planai optimize-prompt --help
planai monitor --help
```

## Common Workflows

### Development Workflow

During development, use the monitor to track execution:

```bash
# In one terminal, run your workflow
python my_workflow.py

# In another terminal, monitor execution
planai monitor --web
```

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

### Production Monitoring

Set up monitoring for production workflows:

```bash
# Export metrics to file
planai monitor --export metrics.json

# Run with custom configuration
planai --config production.yaml monitor --web
```

## Configuration Files

Create a `.planai.yaml` configuration file:

```yaml
# .planai.yaml
llm:
  provider: openai
  model: gpt-4
  reason_model: gpt-4

monitor:
  port: 8080
  refresh_interval: 2

optimize:
  num_iterations: 5
  output_dir: ./optimized_prompts
```

Load configuration:

```bash
planai --config .planai.yaml <command>
```

## Exit Codes

PlanAI CLI uses standard exit codes:

- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: Configuration error
- `4`: Runtime error

## Debugging

Enable verbose output for debugging:

```bash
# Verbose mode
planai -v optimize-prompt ...

# Very verbose mode
planai -vv optimize-prompt ...

# Debug mode (includes stack traces)
PLANAI_DEBUG=1 planai optimize-prompt ...
```

## Shell Completion

Enable tab completion for your shell:

### Bash
```bash
eval "$(_PLANAI_COMPLETE=bash_source planai)"
```

### Zsh
```bash
eval "$(_PLANAI_COMPLETE=zsh_source planai)"
```

### Fish
```fish
_PLANAI_COMPLETE=fish_source planai | source
```

Add to your shell configuration file to make it permanent.

## Integration with Scripts

Use PlanAI CLI in scripts:

```bash
#!/bin/bash

# Check if workflow completed successfully
if planai monitor --workflow "ETL Pipeline" --wait --timeout 300; then
    echo "Workflow completed successfully"
    planai export-results --workflow "ETL Pipeline" --output results.json
else
    echo "Workflow failed or timed out"
    exit 1
fi
```

## Next Steps

- Learn about [Prompt Optimization](/cli/prompt-optimization/) in detail
- Explore [Monitoring](/guide/monitoring/) capabilities
- See [Examples](https://github.com/provos/planai/tree/main/examples) using the CLI