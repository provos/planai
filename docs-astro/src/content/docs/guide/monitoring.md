---
title: Monitoring Dashboard
description: Real-time monitoring and debugging of PlanAI workflows
---

PlanAI includes a built-in web-based monitoring dashboard that provides real-time insights into your graph execution. This feature helps you understand workflow behavior, debug issues, and track performance.

## Overview

The monitoring dashboard offers:

- **Real-time Updates**: Live task status updates using server-sent events (SSE)
- **Task Organization**: Tasks grouped into Queued, Active, and Completed categories
- **Detailed Information**: Click on any task to view provenance and input details
- **Performance Metrics**: Track execution times and identify bottlenecks

## Enabling the Dashboard

### Web Dashboard

To enable the web-based dashboard, set `run_dashboard=True` when calling `graph.run()`:

```python
graph.run(initial_tasks, run_dashboard=True)
```

By default, the dashboard will be accessible at `http://localhost:5000`. You can customize the port:

```python
graph.run(
    initial_tasks, 
    run_dashboard=True,
    dashboard_port=8080  # Custom port
)
```

### Terminal Monitoring

For a lightweight terminal-based monitoring option, use the CLI:

```bash
planai monitor  # Terminal interface
```

Or with the web interface:

```bash
planai monitor --web  # Opens web dashboard
```

## Dashboard Features

### Task Status Tracking

The dashboard displays tasks in three categories:

1. **Queued**: Tasks waiting to be processed
2. **Active**: Tasks currently being processed
3. **Completed**: Tasks that have finished processing

Each task card shows:
- Task ID
- Task type
- Assigned worker
- Processing status

### Task Details View

Click on any task to view detailed information:

```python
# Example of information available in task details
{
    "task_id": "abc123",
    "type": "ProcessedData",
    "worker": "DataProcessor",
    "status": "completed",
    "provenance": [...],  # Full provenance chain
    "input_provenance": [...],  # Input task lineage
    "data": {...},  # Task data (if available)
    "processing_time": 1.23  # seconds
}
```

### Real-time Updates

The dashboard uses server-sent events for live updates without page refreshes:

```python
# The dashboard automatically subscribes to these events
# No manual configuration needed
```

## Using the Dashboard for Debugging

### Identifying Bottlenecks

Look for patterns in the Active tasks section:
- Tasks stuck in Active state may indicate processing issues
- Large queues for specific workers suggest bottlenecks

### Tracing Task Flow

Use the provenance information to understand task flow:

1. Click on a completed task
2. View its input provenance to see what led to this task
3. Trace back through the workflow to understand the full pipeline

### Monitoring LLM Tasks

For LLMTaskWorker tasks, the dashboard shows:
- LLM processing status
- Token usage (when available)
- Response times

## Production Considerations

### Performance Impact

The dashboard has minimal performance impact:
- Uses asynchronous updates
- Lightweight SSE connections
- Optional feature that can be disabled

### Security

For production deployments:

```python
# Restrict dashboard access to localhost only
graph.run(
    initial_tasks,
    run_dashboard=True,
    dashboard_host="127.0.0.1"  # Local access only
)
```

### Alternative Monitoring

For production environments, consider:

1. **Logging Integration**: Export metrics to your logging system
2. **Custom Monitoring**: Implement custom monitoring using PlanAI's hooks
3. **Metrics Export**: Send metrics to monitoring services (Prometheus, DataDog, etc.)

Example custom monitoring:

```python
class MonitoredWorker(TaskWorker):
    def consume_work(self, task: Task):
        start_time = time.time()
        
        try:
            # Your processing logic
            result = self.process(task)
            
            # Log success metrics
            self.log_metric("task_processed", 1, {
                "worker": self.name,
                "task_type": type(task).__name__,
                "duration": time.time() - start_time
            })
            
            self.publish_work(result)
            
        except Exception as e:
            # Log failure metrics
            self.log_metric("task_failed", 1, {
                "worker": self.name,
                "error": str(e)
            })
            raise
```

## Best Practices

1. **Development**: Always enable the dashboard during development
2. **Testing**: Use the dashboard to verify workflow behavior
3. **Production**: Consider security implications before exposing the dashboard
4. **Debugging**: Save dashboard screenshots when debugging complex issues
5. **Performance**: Monitor the Active tasks count to identify processing limits

## CLI Monitoring Commands

Monitor your workflows from the command line:

```bash
# Start web dashboard
planai monitor --web

# Terminal-based monitoring
planai monitor

# Custom port
planai monitor --web --port 8080

# Monitor specific workflow
planai monitor --workflow "Data Processing Pipeline"
```

Note: Enabling the dashboard will block the main thread, so it's recommended primarily for development and debugging purposes. For production use, implement a separate monitoring solution that fits your infrastructure.