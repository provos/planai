---
title: LLM Integration
description: Integrating Large Language Models into PlanAI workflows
---

PlanAI provides seamless integration with Large Language Models (LLMs) through the `LLMTaskWorker` class and the `llm-interface` library. This allows you to combine traditional computation with AI-powered processing in your workflows.

## Setting Up LLM Providers

PlanAI supports multiple LLM providers through the `llm-interface` library:

### OpenAI

```python
from planai import llm_from_config

# Using environment variable (recommended)
# Set OPENAI_API_KEY in your environment
llm = llm_from_config(
    provider="openai",
    model_name="gpt-4"
)
```

### Ollama (Local Models)

```python
# Requires Ollama running locally
llm = llm_from_config(
    provider="ollama",
    model_name="llama2"
)
```

### Other Providers

Check the [llm-interface documentation](https://github.com/provos/llm-interface) for additional supported providers.

## Basic LLMTaskWorker

The simplest way to add LLM capabilities to your workflow:

```python
from planai import LLMTaskWorker
from typing import Type

class TextSummarizer(LLMTaskWorker):
    prompt = "Summarize the following text in 2-3 sentences"
    llm_input_type: Type[Task] = TextDocument
    output_types: List[Type[Task]] = [Summary]

# Usage
summarizer = TextSummarizer(llm=llm)
```

## Advanced Configuration

### System Prompts

Set the context for your LLM:

```python
class ExpertAnalyzer(LLMTaskWorker):
    prompt = "Analyze this data and provide insights"
    system_prompt = """You are a data analysis expert with deep knowledge 
    of statistics and pattern recognition. Provide clear, actionable insights."""
    llm_input_type = DataSet
    output_types: List[Type[Task]] = [Analysis]
```

### Structured Output

PlanAI automatically handles structured output using Pydantic models:

```python
from pydantic import Field
from typing import List, Literal

class SentimentResult(Task):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    key_phrases: List[str]
    reasoning: str

class SentimentAnalyzer(LLMTaskWorker):
    prompt = """Analyze the sentiment of this text. 
    Identify key phrases that indicate the sentiment."""
    llm_input_type = TextData
    output_types: List[Type[Task]] = [SentimentResult]
```

The LLM will automatically return data in the format specified by your Pydantic model.

### Dynamic Prompts

Generate prompts based on input data:

```python
class ContextualAnalyzer(LLMTaskWorker):
    llm_input_type = Document
    output_types: List[Type[Task]] = [Analysis]
    
    def format_prompt(self, task: Document) -> str:
        # Access previous context from provenance
        context = task.find_input_task(UserQuery)
        
        return f"""
        User Query: {context.question}
        Document Type: {task.doc_type}
        
        Analyze this document in the context of the user's query.
        Focus on aspects relevant to: {context.focus_areas}
        """
```

### Input Pre-processing

Control how data is presented to the LLM:

```python
class FilteredAnalyzer(LLMTaskWorker):
    prompt = "Analyze this filtered data"
    llm_input_type = RawData
    output_types: List[Type[Task]] = [Analysis]
    
    def pre_process(self, task: RawData) -> Optional[Task]:
        # Filter sensitive information
        filtered_data = {
            "content": task.content,
            "metadata": self.filter_sensitive(task.metadata)
        }
        
        # Return a wrapper with filtered data
        from planai.utils import PydanticDictWrapper
        return PydanticDictWrapper(data=filtered_data)
```

## Tool Calling / Function Calling

Enable LLMs to call functions or use tools:

```python
from planai import Tool
from typing import Literal

class WeatherTool(Tool):
    """Get current weather for a location"""
    
    name: str = "get_weather"
    
    def execute(self, location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
        # Implementation to fetch weather
        return {
            "location": location,
            "temperature": 22,
            "unit": unit,
            "condition": "sunny"
        }

class AssistantWorker(LLMTaskWorker):
    prompt = "Help the user with their request"
    llm_input_type = UserRequest
    output_types: List[Type[Task]] = [AssistantResponse]
    
    # Enable tool usage
    tools = [WeatherTool()]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tools are automatically registered with the LLM
```

## Streaming Responses

For real-time applications, enable response streaming:

```python
class StreamingAnalyzer(LLMTaskWorker):
    prompt = "Provide a detailed analysis"
    llm_input_type = Document
    output_types: List[Type[Task]] = [Analysis]
    stream = True
    
    def handle_stream(self, chunk: str):
        # Process streaming chunks
        self.update_progress(chunk)
```

## Token Management

Monitor and control token usage:

```python
class TokenAwareAnalyzer(LLMTaskWorker):
    prompt = "Analyze this document"
    llm_input_type = Document
    output_types: List[Type[Task]] = [Analysis]
    
    # Set token limits
    max_tokens = 2000
    
    def consume_work(self, task: Document):
        # Pre-check document size
        estimated_tokens = self.estimate_tokens(task.content)
        
        if estimated_tokens > 3000:
            # Truncate or summarize first
            task = self.prepare_for_context_limit(task)
        
        super().consume_work(task)
```

## Error Handling

Implement robust error handling for LLM operations:

```python
class RobustAnalyzer(LLMTaskWorker):
    prompt = "Analyze this data"
    llm_input_type = DataTask
    output_types: List[Type[Task]] = [Analysis, ErrorResult]
    
    # Retry configuration
    max_retries = 3
    retry_delay = 1.0
    
    def consume_work(self, task: DataTask):
        try:
            super().consume_work(task)
        except RateLimitError:
            # Handle rate limiting
            self.delay_and_retry(task)
        except InvalidResponseError as e:
            # LLM returned invalid format
            self.publish_work(ErrorResult(
                error="Invalid LLM response",
                details=str(e),
                input_task_id=task.id
            ))
```

## Caching LLM Responses

Use `CachedLLMTaskWorker` for development and cost optimization:

```python
from planai import CachedLLMTaskWorker

class CachedAnalyzer(CachedLLMTaskWorker):
    prompt = "Perform expensive analysis"
    llm_input_type = ComplexData
    output_types: List[Type[Task]] = [Analysis]
    
    # Cache configuration
    cache_ttl = 3600  # 1 hour
    cache_key_prefix = "analysis_v1"
```

## Best Practices

### 1. Prompt Engineering

- Be specific and clear in your prompts
- Use examples for complex tasks
- Test prompts with different inputs
- Version your prompts

### 2. Cost Optimization

- Use caching during development
- Choose appropriate models for tasks
- Implement token limits
- Batch similar requests when possible

### 3. Reliability

- Implement retry logic
- Handle API errors gracefully
- Provide fallback options
- Monitor success rates

### 4. Performance

- Use streaming for long responses
- Pre-process large inputs
- Consider parallel processing
- Optimize context usage

## Advanced Patterns

### Multi-Stage LLM Processing

```python
class ResearchAssistant(LLMTaskWorker):
    prompt = "Generate search queries for this topic"
    llm_input_type = ResearchTopic
    output_types: List[Type[Task]] = [SearchQueries]

class Synthesizer(LLMTaskWorker):
    prompt = "Synthesize these search results into a report"
    llm_input_type = SearchResults
    output_types: List[Type[Task]] = [Report]

# Connect in workflow
graph.add_workers(researcher, search_engine, synthesizer)
graph.set_dependencies(...)
```

### Ensemble LLM Approaches

```python
class EnsembleAnalyzer(TaskWorker):
    output_types: List[Type[Task]] = [ConsolidatedAnalysis]
    
    def __init__(self, llms: List[LLM]):
        super().__init__()
        self.analyzers = [
            LLMAnalyzer(llm=llm, variant=i) 
            for i, llm in enumerate(llms)
        ]
    
    def consume_work(self, task: InputData):
        # Get analysis from multiple LLMs
        results = []
        for analyzer in self.analyzers:
            result = analyzer.process(task)
            results.append(result)
        
        # Consolidate results
        consolidated = self.merge_analyses(results)
        self.publish_work(consolidated)
```

### Context-Aware Chains

```python
class ContextChainWorker(LLMTaskWorker):
    prompt = "Continue the analysis"
    llm_input_type = IntermediateResult
    output_types: List[Type[Task]] = [FinalResult]
    
    def format_prompt(self, task: IntermediateResult) -> str:
        # Build context from provenance chain
        context_items = []
        
        # Get all previous analyses
        current = task
        while current:
            if analysis := current.find_input_task(Analysis):
                context_items.append(f"Previous: {analysis.summary}")
            current = current.previous_input_task()
        
        context = "\n".join(reversed(context_items))
        
        return f"""
        Context from previous analyses:
        {context}
        
        Current data: {task.data}
        
        Continue the analysis building on previous insights.
        """
```

## Monitoring and Debugging

### Enable Debug Mode

```python
class DebugAnalyzer(LLMTaskWorker):
    prompt = "Analyze this data"
    llm_input_type = DataTask
    output_types: List[Type[Task]] = [Analysis]
    debug_mode = True  # Logs prompts and responses
```

### Token Usage Tracking

```python
class TokenTrackingWorker(LLMTaskWorker):
    def consume_work(self, task: Task):
        result = super().consume_work(task)
        
        # Log token usage
        self.log_metrics({
            "prompt_tokens": self.last_prompt_tokens,
            "completion_tokens": self.last_completion_tokens,
            "total_cost": self.estimate_cost()
        })
        
        return result
```

## Next Steps

- Learn about [Prompt Engineering](/guide/prompts/) for better results
- Explore [Caching Strategies](/features/caching/) for cost optimization
- See [Examples](https://github.com/provos/planai/tree/main/examples) using LLMs
- Read about [Prompt Optimization](/cli/prompt-optimization/) tools