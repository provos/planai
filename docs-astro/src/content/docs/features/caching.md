---
title: Caching
description: Optimize performance and reduce costs with PlanAI's caching features
---

PlanAI provides powerful caching capabilities to improve performance and reduce costs, especially when working with expensive operations like LLM API calls or complex computations.

## Overview

PlanAI offers two main caching approaches:

1. **CachedTaskWorker**: General-purpose caching for any TaskWorker
2. **CachedLLMTaskWorker**: Specialized caching for LLM operations

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
        ))
    
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
    output_types = [ProcessedData]
    
    def get_cache_key(self, task: InputTask) -> str:
        # Custom cache key based on specific fields
        return f"{task.category}:{task.id}:{task.version}"
    
    def consume_work(self, task: InputTask):
        # Processing logic
        pass
```

### Cache Configuration

```python
class ConfiguredCacheWorker(CachedTaskWorker):
    output_types = [Result]
    
    # Cache configuration
    cache_dir = "./my_cache"  # Custom cache directory
    cache_ttl = 3600  # Time-to-live in seconds (1 hour)
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
    output_types = [Analysis]
    
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
4. **Rate Limit Management**: Reduce API request frequency

## Advanced Caching Patterns

### Conditional Caching

Cache based on specific conditions:

```python
class ConditionalCacheWorker(CachedTaskWorker):
    output_types = [Result]
    
    def should_cache(self, task: InputTask) -> bool:
        # Only cache expensive operations
        return task.complexity > 5
    
    def consume_work(self, task: InputTask):
        if self.should_cache(task):
            # Check cache first
            cached = self.get_from_cache(task)
            if cached:
                self.publish_work(cached)
                return
        
        # Process normally
        result = self.process(task)
        
        if self.should_cache(task):
            self.save_to_cache(task, result)
        
        self.publish_work(result)
```

### Versioned Caching

Handle cache invalidation with versions:

```python
class VersionedCacheWorker(CachedTaskWorker):
    output_types = [ProcessedData]
    
    # Algorithm version - increment to invalidate cache
    algorithm_version = "v2.1"
    
    def get_cache_key(self, task: InputTask) -> str:
        base_key = super().get_cache_key(task)
        return f"{self.algorithm_version}:{base_key}"
```

### Multi-Level Caching

Implement memory and disk caching:

```python
from functools import lru_cache

class MultiLevelCacheWorker(CachedTaskWorker):
    output_types = [Result]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_cache = {}
    
    @lru_cache(maxsize=100)
    def get_from_memory(self, key: str):
        return self.memory_cache.get(key)
    
    def consume_work(self, task: InputTask):
        cache_key = self.get_cache_key(task)
        
        # Check memory cache first
        if result := self.get_from_memory(cache_key):
            self.publish_work(result)
            return
        
        # Check disk cache
        if result := self.get_from_cache(cache_key):
            self.memory_cache[cache_key] = result
            self.publish_work(result)
            return
        
        # Compute and cache
        result = self.process(task)
        self.memory_cache[cache_key] = result
        self.save_to_cache(cache_key, result)
        self.publish_work(result)
```

## Cache Management

### Clearing Cache

```python
class ManagedCacheWorker(CachedTaskWorker):
    output_types = [Result]
    
    def clear_cache(self):
        """Clear all cached entries"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    def clear_old_entries(self, days: int = 7):
        """Clear entries older than specified days"""
        import time
        cutoff = time.time() - (days * 86400)
        
        with self.get_cache() as cache:
            for key in list(cache.keys()):
                if cache.touch(key) < cutoff:
                    del cache[key]
```

### Cache Statistics

Monitor cache performance:

```python
class MonitoredCacheWorker(CachedTaskWorker):
    output_types = [Result]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def consume_work(self, task: InputTask):
        cache_key = self.get_cache_key(task)
        
        if cached := self.get_from_cache(cache_key):
            self.cache_hits += 1
            self.log_cache_hit_rate()
            self.publish_work(cached)
            return
        
        self.cache_misses += 1
        result = self.process(task)
        self.save_to_cache(cache_key, result)
        self.publish_work(result)
    
    def log_cache_hit_rate(self):
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = self.cache_hits / total * 100
            self.logger.info(f"Cache hit rate: {hit_rate:.2f}%")
```

## Best Practices

### 1. Cache Key Design

- Include all parameters that affect the output
- Use stable serialization for complex objects
- Consider including version information
- Keep keys reasonably short

### 2. TTL Strategy

```python
class SmartTTLWorker(CachedTaskWorker):
    output_types = [Result]
    
    def get_ttl(self, task: InputTask) -> int:
        # Dynamic TTL based on task properties
        if task.data_type == "static":
            return 86400 * 30  # 30 days
        elif task.data_type == "dynamic":
            return 3600  # 1 hour
        else:
            return 300  # 5 minutes default
```

### 3. Cache Warming

Pre-populate cache for common requests:

```python
class WarmableCacheWorker(CachedTaskWorker):
    output_types = [Result]
    
    def warm_cache(self, common_tasks: List[InputTask]):
        """Pre-compute and cache common tasks"""
        for task in common_tasks:
            if not self.is_cached(task):
                self.consume_work(task)
```

### 4. Error Handling

Don't cache errors by default:

```python
class ErrorAwareCacheWorker(CachedTaskWorker):
    output_types = [Result, ErrorResult]
    
    def consume_work(self, task: InputTask):
        try:
            result = self.process(task)
            self.save_to_cache(task, result)
            self.publish_work(result)
        except Exception as e:
            # Don't cache errors
            error_result = ErrorResult(error=str(e))
            self.publish_work(error_result)
```

## Performance Considerations

### Cache Size Management

```python
class SizeLimitedCacheWorker(CachedTaskWorker):
    output_types = [Result]
    
    # Limit cache size
    cache_size_limit = 1000  # Maximum entries
    cache_disk_limit = 1024 * 1024 * 1024  # 1GB
    
    def evict_old_entries(self):
        """Implement LRU eviction"""
        with self.get_cache() as cache:
            if len(cache) > self.cache_size_limit:
                # Remove least recently used
                cache.evict()
```

### Compression

For large cached objects:

```python
import zlib
import pickle

class CompressedCacheWorker(CachedTaskWorker):
    output_types = [LargeResult]
    
    def save_to_cache(self, key: str, value: Task):
        # Compress before caching
        serialized = pickle.dumps(value)
        compressed = zlib.compress(serialized)
        
        with self.get_cache() as cache:
            cache[key] = compressed
    
    def get_from_cache(self, key: str) -> Optional[Task]:
        with self.get_cache() as cache:
            if compressed := cache.get(key):
                serialized = zlib.decompress(compressed)
                return pickle.loads(serialized)
        return None
```

## Debugging Caching Issues

Enable cache debugging:

```python
class DebugCacheWorker(CachedTaskWorker):
    output_types = [Result]
    debug_cache = True
    
    def consume_work(self, task: InputTask):
        cache_key = self.get_cache_key(task)
        self.logger.debug(f"Cache key: {cache_key}")
        
        if cached := self.get_from_cache(cache_key):
            self.logger.debug(f"Cache HIT for {cache_key}")
            self.publish_work(cached)
        else:
            self.logger.debug(f"Cache MISS for {cache_key}")
            result = self.process(task)
            self.save_to_cache(cache_key, result)
            self.publish_work(result)
```

## Next Steps

- Learn about [Task Workers](/features/taskworkers/) that can be cached
- Explore [LLM Integration](/features/llm-integration/) with caching
- See caching examples in the [repository](https://github.com/provos/planai/tree/main/examples)