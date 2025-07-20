---
title: Caching
description: Optimize performance and reduce costs with PlanAI's caching features
---

PlanAI makes it easy to cache the results of TaskWorkers based on the input data. This can help improve performance and reduce costs, especially when working with expensive operations like LLM API calls or complex computations.

## Overview

PlanAI offers two main caching approaches:

1. **CachedTaskWorker**: Caching for traditional compute based workers
2. **CachedLLMTaskWorker**: Caching for LLM operations

Both use the `diskcache` library for persistent, thread-safe caching.

## CachedTaskWorker

Cache results from any expensive operation:

```python
from planai import CachedTaskWorker
from typing import List, Type

class ExpensiveComputation(CachedTaskWorker):
    output_types: List[Type[Task]] = [ComputedResult]
    
    def consume_work(self, task: InputData):
        # This expensive operation will be cached
        result = self.complex_calculation(task.data)
        
        self.publish_work(ComputedResult(
            value=result,
            computation_time=self.elapsed_time
        ), input_task=task)
    
    def complex_calculation(self, data):
        # Simulate expensive computation
        import time
        time.sleep(5)
        return data ** 2
```

### Cache Key Generation

By default, cache keys are generated from the task's data. Customize this behavior:

```python
class CustomCacheWorker(CachedTaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]
    
    def extra_cache_key(self, task: InputTask) -> str:
        # Custom cache key based on workflow
        weekday = datetime.now().strftime("%A")
        return weekday
    
    def consume_work(self, task: InputTask):
        # Processing logic
        pass
```

### Cache Configuration

```python
class ConfiguredCacheWorker(CachedTaskWorker):
    output_types: List[Type[Task]] = [Result]
    
    # Cache configuration
    cache_dir = "./my_cache"  # Custom cache directory
    cache_size_limit = 1000  # Maximum cache entries
    
    def consume_work(self, task: InputData):
        # Cached processing
        pass
```

## CachedLLMTaskWorker

Specialized caching for LLM operations with additional features:

```python
from planai import CachedLLMTaskWorker, llm_from_config

class CachedAnalyzer(CachedLLMTaskWorker):
    prompt = "Analyze this document and provide insights"
    llm_input_type: Type[Task] = Document
    output_types: List[Type[Task]] = [Analysis]
    
    # LLM-specific cache settings
    cache_ttl = 86400  # Cache for 24 hours
    include_model_in_key = True  # Include model name in cache key

# Usage
analyzer = CachedAnalyzer(
    llm=llm_from_config("openai", "gpt-4"),
    cache_dir="./llm_cache"
)
```

### Benefits for LLM Caching

1. **Development Efficiency**: Avoid repeated API calls during testing
2. **Cost Reduction**: Reuse expensive LLM responses
3. **Deterministic Testing**: Consistent responses for unit tests

### Versioned Caching

Handle cache invalidation with versions:

```python
class VersionedCacheWorker(CachedTaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]
    
    # Algorithm version - increment to invalidate cache
    algorithm_version = "v2.1"
    
    def extra_cache_key(self, task: InputTask) -> str:
        return algorithm_version
```

## Next Steps

- Learn about [Task Workers](/features/taskworkers/) that can be cached
- Explore [LLM Integration](/features/llm-integration/) with caching
- See caching examples in the [repository](https://github.com/provos/planai/tree/main/examples)