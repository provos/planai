from typing import List, Type

from planai import Graph, Task, TaskWorker


# Define our data models using Pydantic
class RawData(Task):
    content: str


class ProcessedData(Task):
    processed_content: str
    word_count: int


class AnalysisResult(Task):
    summary: str
    sentiment: str


# Create a simple data processor
class DataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]

    def consume_work(self, task: RawData):
        # Process the raw data
        cleaned = task.content.strip().lower()
        word_count = len(cleaned.split())

        # Publish the processed data
        self.publish_work(
            ProcessedData(processed_content=cleaned, word_count=word_count),
            input_task=task,
        )


from planai import LLMTaskWorker, llm_from_config  # noqa: E402


# Define an AI analyzer
class AIAnalyzer(LLMTaskWorker):
    prompt: str = """Analyze the following text and provide:
    1. A brief summary (one sentence)
    2. The overall sentiment (positive, negative, or neutral)
    
    Format your response as JSON with 'summary' and 'sentiment' fields."""

    llm_input_type: Type[Task] = ProcessedData
    output_types: List[Type[Task]] = [AnalysisResult]


class ResultPrinter(TaskWorker):
    def consume_work(self, task: AnalysisResult):
        self.print(task.summary)
        self.print(task.sentiment)


# Create the enhanced workflow
graph = Graph(name="AI-Enhanced Pipeline")

# Initialize workers
processor = DataProcessor()
analyzer = AIAnalyzer(
    llm=llm_from_config(
        provider="openai",
        model_name="gpt-4o",
    )
)
result_printer = ResultPrinter()

# Build the graph
graph.add_workers(processor, analyzer, result_printer)
# analyzer depends on processor and result_printer depends on analyzer
graph.set_dependency(processor, analyzer).next(result_printer)

# Run with monitoring dashboard
initial_data = RawData(content="PlanAI makes it easy to build AI workflows!")
graph.run(
    initial_tasks=[(processor, initial_data)],
    run_dashboard=False,  # Set to True to open monitoring at http://localhost:5000
)

# Logs should show:
# The text promotes PlanAI's ability to simplify the creation of AI workflows.
# positive
