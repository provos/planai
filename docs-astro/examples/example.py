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


class DataPrinter(TaskWorker):
    def consume_work(self, task: ProcessedData):
        print(task.processed_content)


# Create the workflow graph
graph = Graph(name="Simple Data Pipeline")
processor = DataProcessor()
printer = DataPrinter()
graph.add_workers(processor, printer)
graph.set_dependency(processor, printer)


# Run the workflow
initial_data = RawData(content="  Hello World! This is PlanAI.  ")
graph.run(initial_tasks=[(processor, initial_data)])

# Will print: Hello World! This is PlanAI.
