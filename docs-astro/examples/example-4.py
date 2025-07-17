# flake8: noqa: E821
from typing import List, Type

from planai import Graph, Task, TaskWorker


# Define custom TaskWorkers
class DataFetcher(TaskWorker):
    output_types: List[Type[Task]] = [FetchedData]

    def consume_work(self, task: FetchRequest):
        # Fetch data from some source - needs to be implemented
        data = self.fetch_data(task.url)
        self.publish_work(FetchedData(data=data), input_task=task)


class DataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]

    def consume_work(self, task: FetchedData):
        # Process the fetched data
        processed_data = self.process(task.data)
        self.publish_work(ProcessedData(data=processed_data), input_task=task)


# Create a graph
graph = Graph(name="Data Processing Workflow")

# Initialize tasks
fetcher = DataFetcher()
processor = DataProcessor()

# Add tasks to the graph and set dependencies
graph.add_workers(fetcher, processor)
graph.set_dependency(fetcher, processor)

# Let the graph collect all tasks published by the processor with the type ProcessedData
graph.set_sink(processor, ProcessedData)

# Run the graph
initial_request = FetchRequest(url="https://example.com/data")
graph.run(initial_tasks=[(fetcher, initial_request)])

# Get the outputs
outputs = graph.get_output_tasks()
print(outputs)
