---
title: Subgraphs
description: Building modular, reusable workflows with SubGraphWorker
---

PlanAI allows you to create nested workflows by encapsulating an entire graph as a single TaskWorker using `SubGraphWorker`. This enables modular, reusable, and composable subgraphs within larger workflows.

## Overview

Subgraphs provide:

- **Modularity**: Break complex workflows into manageable components
- **Reusability**: Use the same subgraph in multiple workflows
- **Abstraction**: Hide implementation details behind a simple interface
- **Testing**: Test components in isolation

## Basic Subgraph

Create a subgraph and use it as a single worker:

```python
from planai import Graph, SubGraphWorker
from planai.graph_task import SubGraphWorker

# 1. Define the subgraph
sub_graph = Graph(name="DataProcessingPipeline")

# Add workers to the subgraph
validator = DataValidator()
transformer = DataTransformer()
enricher = DataEnricher()

sub_graph.add_workers(validator, transformer, enricher)
sub_graph.set_dependency(validator, transformer)
sub_graph.set_dependency(transformer, enricher)

# 2. Define entry and exit points
sub_graph.set_entry(validator)  # Where data enters
sub_graph.set_exit(enricher)    # Where data exits

# 3. Wrap as a SubGraphWorker
pipeline_worker = SubGraphWorker(
    name="DataPipeline",
    graph=sub_graph
)

# 4. Use in a main graph
main_graph = Graph(name="MainWorkflow")
main_graph.add_worker(pipeline_worker)
```

## Entry and Exit Points

Subgraphs must have exactly one entry and one exit worker:

```python
# Entry point: receives input tasks
sub_graph.set_entry(first_worker)

# Exit point: produces output tasks
sub_graph.set_exit(last_worker)
```

The input/output types must be importable:

```python
# Define tasks in a separate module for reusability
# tasks.py
from planai import Task

class PipelineInput(Task):
    data: str
    config: dict

class PipelineOutput(Task):
    processed_data: str
    metadata: dict

# In your subgraph module
from .tasks import PipelineInput, PipelineOutput

class FirstWorker(TaskWorker):
    # Must accept PipelineInput
    def consume_work(self, task: PipelineInput):
        ...

class LastWorker(TaskWorker):
    # Must produce PipelineOutput
    output_types: List[Type[Task]] = [PipelineOutput]
```

## Complex Subgraph Example

Build a reusable text analysis pipeline:

```python
# text_analysis_subgraph.py
from planai import Graph, TaskWorker, LLMTaskWorker, Task
from typing import List, Type

# Define interface tasks
class TextInput(Task):
    text: str
    analysis_type: str

class AnalysisOutput(Task):
    original_text: str
    word_count: int
    sentiment: str
    key_topics: List[str]
    summary: str

# Workers for the subgraph
class TextPreprocessor(TaskWorker):
    output_types: List[Type[Task]] = [PreprocessedText]
    
    def consume_work(self, task: TextInput):
        cleaned = self.clean_text(task.text)
        word_count = len(cleaned.split())
        
        self.publish_work(PreprocessedText(
            text=cleaned,
            word_count=word_count,
            original=task.text
        ))

class SentimentAnalyzer(LLMTaskWorker):
    prompt = "Analyze the sentiment of this text"
    llm_input_type = PreprocessedText
    output_types: List[Type[Task]] = [SentimentResult]

class TopicExtractor(LLMTaskWorker):
    prompt = "Extract 3-5 key topics from this text"
    llm_input_type = PreprocessedText
    output_types: List[Type[Task]] = [TopicsResult]

class Summarizer(LLMTaskWorker):
    prompt = "Provide a concise summary"
    llm_input_type = PreprocessedText
    output_types: List[Type[Task]] = [SummaryResult]

class ResultAggregator(JoinedTaskWorker):
    join_type = TextPreprocessor
    output_types: List[Type[Task]] = [AnalysisOutput]
    
    def consume_work_joined(self, tasks: List[Task]):
        # Combine all analysis results
        sentiment = next(t for t in tasks if isinstance(t, SentimentResult))
        topics = next(t for t in tasks if isinstance(t, TopicsResult))
        summary = next(t for t in tasks if isinstance(t, SummaryResult))
        preprocessed = next(t for t in tasks if isinstance(t, PreprocessedText))
        
        self.publish_work(AnalysisOutput(
            original_text=preprocessed.original,
            word_count=preprocessed.word_count,
            sentiment=sentiment.sentiment,
            key_topics=topics.topics,
            summary=summary.summary
        ))

# Create the analysis subgraph
def create_text_analysis_subgraph(llm):
    graph = Graph(name="TextAnalysis")
    
    # Initialize workers
    preprocessor = TextPreprocessor()
    sentiment = SentimentAnalyzer(llm=llm)
    topics = TopicExtractor(llm=llm)
    summarizer = Summarizer(llm=llm)
    aggregator = ResultAggregator()
    
    # Build graph structure
    graph.add_workers(preprocessor, sentiment, topics, summarizer, aggregator)
    graph.set_dependency(preprocessor, sentiment)
    graph.set_dependency(preprocessor, topics)
    graph.set_dependency(preprocessor, summarizer)
    graph.set_dependency(sentiment, aggregator)
    graph.set_dependency(topics, aggregator)
    graph.set_dependency(summarizer, aggregator)
    
    # Set entry and exit
    graph.set_entry(preprocessor)
    graph.set_exit(aggregator)
    
    return graph
```

## Using Subgraphs in Workflows

Integrate subgraphs into larger workflows:

```python
# main_workflow.py
from planai import Graph, SubGraphWorker, llm_from_config
from text_analysis_subgraph import create_text_analysis_subgraph, TextInput, AnalysisOutput

# Create main workflow
main_graph = Graph(name="DocumentProcessor")

# Initialize components
doc_loader = DocumentLoader()
llm = llm_from_config("openai", "gpt-4")

# Create and wrap the subgraph
analysis_subgraph = create_text_analysis_subgraph(llm)
text_analyzer = SubGraphWorker(
    name="TextAnalyzer",
    graph=analysis_subgraph
)

# Create report generator that uses analysis results
report_generator = ReportGenerator()

# Build the workflow
main_graph.add_workers(doc_loader, text_analyzer, report_generator)
main_graph.set_dependency(doc_loader, text_analyzer)
main_graph.set_dependency(text_analyzer, report_generator)

# Run the workflow
main_graph.run(initial_tasks=[(doc_loader, DocumentPath(path="report.pdf"))])
```

## Nested Subgraphs

Subgraphs can contain other subgraphs:

```python
# Create a higher-level subgraph
meta_graph = Graph(name="MetaAnalysis")

# Add multiple analysis subgraphs
english_analyzer = SubGraphWorker(
    name="EnglishAnalyzer",
    graph=create_text_analysis_subgraph(english_llm)
)

spanish_analyzer = SubGraphWorker(
    name="SpanishAnalyzer", 
    graph=create_text_analysis_subgraph(spanish_llm)
)

# Language detector to route tasks
language_detector = LanguageDetector()
result_merger = ResultMerger()

meta_graph.add_workers(language_detector, english_analyzer, spanish_analyzer, result_merger)
# Set up routing based on detected language
```

## Testing Subgraphs

Test subgraphs by running the full graph with `MockLLM` and `inject_mock_cache`:

```python
from planai import Graph, SubGraphWorker
from planai.testing import MockLLM, MockLLMResponse, MockCache, inject_mock_cache

def test_text_analysis_subgraph():
    mock_llm = MockLLM(responses=[
        MockLLMResponse(pattern="Analyze the sentiment.*", response=SentimentResult(sentiment="positive")),
        MockLLMResponse(pattern="Extract.*key topics.*", response=TopicsResult(topics=["product"])),
        MockLLMResponse(pattern="Provide a concise summary", response=SummaryResult(summary="Positive review")),
    ])
    mock_cache = MockCache(dont_store=True)

    # Build and wrap the subgraph
    subgraph = create_text_analysis_subgraph(mock_llm)
    analyzer = SubGraphWorker(name="TestAnalyzer", graph=subgraph)

    # Embed in a test graph with a sink
    graph = Graph(name="TestGraph")
    graph.add_workers(analyzer)
    graph.set_sink(analyzer, AnalysisOutput)
    inject_mock_cache(graph, mock_cache)

    test_input = TextInput(text="This is a great product! I love it.", analysis_type="full")
    graph.run(initial_tasks=[(analyzer, test_input)], run_dashboard=False, display_terminal=False)

    results = graph.get_output_tasks()
    assert len(results) == 1
    assert isinstance(results[0], AnalysisOutput)
    assert results[0].sentiment == "positive"
```

See the [Testing guide](/guide/testing/) for full details on all available testing utilities.

## Best Practices

### 1. Clear Interfaces

Define clear input/output contracts:

```python
# Good: Clear, documented interfaces
class SubgraphInput(Task):
    """Input for data processing subgraph"""
    raw_data: str
    processing_config: ProcessingConfig
    
class SubgraphOutput(Task):
    """Output from data processing subgraph"""
    processed_data: ProcessedData
    quality_metrics: QualityMetrics
    processing_time: float
```

### 2. Error Handling

Handle errors within subgraphs:

```python
class ErrorHandlingWorker(TaskWorker):
    output_types: List[Type[Task]] = [SuccessResult, ErrorResult]
    
    def consume_work(self, task: InputTask):
        try:
            result = self.process(task)
            self.publish_work(SuccessResult(data=result), input_task=task)
        except ValidationError as e:
            self.publish_work(ErrorResult(
                error_type="validation",
                message=str(e),
                input_data=task
            ))
```

### 3. Configuration

Make subgraphs configurable:

```python
def create_configurable_subgraph(config: SubgraphConfig):
    graph = Graph(name=config.name)
    
    # Configure workers based on config
    if config.enable_caching:
        processor = CachedProcessor()
    else:
        processor = StandardProcessor()
    
    # Add workers based on config
    if config.include_validation:
        validator = DataValidator(rules=config.validation_rules)
        graph.add_worker(validator)
    
    return graph
```

### 4. Documentation

Document subgraph behavior:

```python
class DocumentedSubgraph:
    """
    Text Analysis Subgraph
    
    This subgraph performs comprehensive text analysis including:
    - Sentiment analysis
    - Topic extraction
    - Summarization
    
    Input: TextInput with 'text' and 'analysis_type'
    Output: AnalysisOutput with sentiment, topics, and summary
    
    Example:
        subgraph = create_text_analysis_subgraph(llm)
        analyzer = SubGraphWorker("Analyzer", subgraph)
        result = analyzer.process(TextInput(text="...", analysis_type="full"))
    """
```

## Advanced Patterns

### Dynamic Subgraphs

Create subgraphs dynamically based on configuration:

```python
class DynamicSubgraphBuilder:
    def build_analysis_pipeline(self, stages: List[str], llm):
        graph = Graph(name="DynamicAnalysis")
        
        previous_worker = None
        entry_worker = None
        
        for stage in stages:
            worker = self.create_worker_for_stage(stage, llm)
            graph.add_worker(worker)
            
            if previous_worker:
                graph.set_dependency(previous_worker, worker)
            else:
                entry_worker = worker
            
            previous_worker = worker
        
        graph.set_entry(entry_worker)
        graph.set_exit(previous_worker)
        
        return graph
```

### Subgraph Libraries

Build reusable subgraph libraries:

```python
# subgraph_library.py
class SubgraphLibrary:
    @staticmethod
    def create_nlp_pipeline(llm, languages=["en"]):
        """Natural Language Processing pipeline"""
        ...
    
    @staticmethod
    def create_data_validation_pipeline(rules):
        """Data validation and cleaning pipeline"""
        ...
    
    @staticmethod
    def create_ml_preprocessing_pipeline(features):
        """Machine learning preprocessing pipeline"""
        ...
```

## Limitations

Current limitations of subgraphs:

1. **Single Entry/Exit**: Only one entry and one exit point allowed
2. **Type Requirements**: Input/output types must be importable
3. **Provenance**: Subgraph internal provenance is encapsulated
4. **Monitoring**: Internal subgraph execution requires special handling for monitoring

## Next Steps

- Explore [Task Workers](/features/taskworkers/) for building subgraph components
- Learn about [Provenance](/guide/provenance/) in nested workflows
- See subgraph examples in the [repository](https://github.com/provos/planai/tree/main/examples)