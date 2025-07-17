---
title: Prompts
description: Master prompt engineering in PlanAI for optimal LLM performance
---

Prompts are the most important component for influencing the performance of an LLM. PlanAI provides a flexible prompting system that helps maintain consistency while allowing customization.

## Default Template

By default, PlanAI will automatically format the input task into a prompt using this template:

```
Here is your input data:
{task}

Here are your instructions:
{instructions}
```

The `{task}` placeholder is automatically filled with the formatted input task data, and `{instructions}` is filled with the prompt you provide to the LLMTaskWorker.

## Customizing Prompts

There are several ways to customize how prompts are generated:

### 1. Basic Prompt Customization

Provide your instructions when creating an LLMTaskWorker:

```python
worker = LLMTaskWorker(
    llm=llm,
    prompt="Analyze this text and extract key topics",
    output_types=[TopicsTask]
)
```

### 2. Dynamic Prompt Generation

Override `format_prompt()` to generate prompts based on the input task:

```python
class CustomWorker(LLMTaskWorker):
    def format_prompt(self, task: InputTask) -> str:
        return f"Analyze this {task.content_type} and provide a summary"
```

The `format_prompt()` method can also be used to fill template parameters in your prompt:

```python
class StatementAnalyzer(LLMTaskWorker):
    prompt = "Think about whether {statement} is a good idea"
    
    def format_prompt(self, task: Task) -> str:
        statement = task.find_input(StatementTask)
        return self.prompt.format(statement=statement.statement)
```

### 3. Input Pre-processing

Override `pre_process()` to modify how the input task is presented:

```python
def pre_process(self, task: InputTask) -> Optional[Task]:
    # Remove sensitive fields or reformat data
    return ProcessedTask(content=task.filtered_content)
```

The `pre_process()` method serves two important purposes:

1. **Data transformation**: Convert or filter input data before it reaches the LLM
2. **Input format control**: Determine how input data is presented to the LLM

You can use `PydanticDictWrapper` to inject a different Pydantic object:

```python
from planai.utils import PydanticDictWrapper

def pre_process(self, task: Task) -> Optional[Task]:
    custom_data = {
        "filtered": task.content,
        "metadata": task.get_metadata()
    }
    return PydanticDictWrapper(data=custom_data)
```

If `pre_process()` returns `None`, PlanAI will not provide any input data in the default template. 
In this case, your class should provide all necessary context through `format_prompt()`:

```python
class FullyCustomPrompt(LLMTaskWorker):
    def pre_process(self, task: Task) -> Optional[Task]:
        # Signal that we'll handle all input formatting
        return None
    
    def format_prompt(self, task: Task) -> str:
        # Provide complete prompt with all necessary context
        return f"""
        System: {task.system_context}
        Input: {task.content}
        Question: {task.question}
        """
```

### 4. XML Serialization

For tasks containing complex text data (like markdown or text with newlines), 
you can enable XML serialization of the input data:

```python
worker = LLMTaskWorker(
    llm=llm,
    prompt="Analyze this markdown document",
    use_xml=True,
    output_types=[AnalysisTask]
)
```

This will format the input task as XML instead of JSON, which can be easier for the LLM
to process when dealing with text that contains newlines or special characters.

## System Prompts

You can also customize the system prompt that sets the context for the LLM:

```python
worker = LLMTaskWorker(
    llm=llm,
    prompt="Analyze the text",
    system_prompt="You are an expert analyst specialized in text classification",
    output_types=[AnalysisTask]
)
```

## Structured Output

PlanAI supports structured output using Pydantic models. The LLM will be instructed to return data in the format specified by your output types:

```python
class SentimentAnalysis(Task):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str

class SentimentAnalyzer(LLMTaskWorker):
    prompt = "Analyze the sentiment of this text"
    output_types: List[Type[Task]] = [SentimentAnalysis]
```

## Advanced Prompt Techniques

### Few-Shot Prompting

Include examples in your prompts to improve performance:

```python
class FewShotAnalyzer(LLMTaskWorker):
    prompt = """Classify the following text into a category.

Examples:
- "The product arrived damaged" -> category: complaint
- "Thank you for the excellent service" -> category: praise
- "How do I reset my password?" -> category: question

Now classify this text: {text}"""
    
    def format_prompt(self, task: Task) -> str:
        return self.prompt.format(text=task.content)
```

### Chain-of-Thought Prompting

Encourage step-by-step reasoning:

```python
class ReasoningAnalyzer(LLMTaskWorker):
    prompt = """Solve this problem step by step.

Problem: {problem}

Please think through this carefully:
1. First, identify what we're trying to solve
2. List the given information
3. Work through the solution step by step
4. Verify your answer

Provide your final answer in the required format."""
```

### Context Window Management

For tasks with large inputs, manage the context window effectively:

```python
class LargeDocumentAnalyzer(LLMTaskWorker):
    def pre_process(self, task: DocumentTask) -> Optional[Task]:
        # Truncate or summarize large documents
        max_chars = 10000
        if len(task.content) > max_chars:
            # Return truncated version or extract key sections
            return ProcessedDocument(
                content=task.content[:max_chars],
                was_truncated=True
            )
        return task
```

## Best Practices

1. **Be specific and clear** in your instructions
2. **Include examples** if the task is complex
3. **Consider using pre-processing** to simplify complex input data
4. **Test different system prompts** to find what works best
5. **Use format_prompt()** for dynamic instructions based on input
6. **Monitor token usage** and optimize prompts for efficiency
7. **Use structured output** with Pydantic models for consistent results
8. **Version your prompts** and track performance improvements

## Automatic Prompt Optimization

PlanAI includes a CLI tool for automatic prompt optimization. See the [Prompt Optimization guide](/cli/prompt-optimization/) for details on how to use AI to improve your prompts based on production data.