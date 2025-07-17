from typing import List, Type

from planai import Graph, InitialTaskWorker, JoinedTaskWorker, Task, TaskWorker


# Define a data source
class DataSource(Task):
    source_id: str
    data: str


class ProcessedData(Task):
    processed_data: str


class CombinedAnalysis(Task):
    sources_analyzed: int
    combined_summary: str


# Worker to process data
class DataProcessor(TaskWorker):
    output_types: List[Type[Task]] = [ProcessedData]

    def consume_work(self, task: DataSource):
        # We'll publish multiple tasks here
        for i in range(3):
            self.publish_work(
                ProcessedData(processed_data=f"{task.data} - processed {i}"),
                input_task=task,
            )


# Worker to join results
class ResultAggregator(JoinedTaskWorker):
    join_type: Type[TaskWorker] = InitialTaskWorker
    output_types: List[Type[Task]] = [CombinedAnalysis]

    def consume_work_joined(self, tasks: List[ProcessedData]):
        combined_summary = f"Analyzed {len(tasks)} sources from {tasks[0].prefix(1)}"

        self.publish_work(
            CombinedAnalysis(
                sources_analyzed=len(tasks), combined_summary=combined_summary
            ),
            input_task=tasks[0],
        )


# Class DataPrinter
class DataPrinter(TaskWorker):
    def consume_work(self, task: CombinedAnalysis):
        self.print(task.combined_summary)


# Build the complete workflow
graph = Graph(name="Multi-Source Analysis")
processor = DataProcessor()
aggregator = ResultAggregator()
printer = DataPrinter()

graph.add_workers(processor, aggregator, printer)
graph.set_dependency(processor, aggregator).next(printer)

# Run the workflow
initial_data = [
    DataSource(source_id="source1", data="First dataset"),
    DataSource(source_id="source2", data="Second dataset"),
    DataSource(source_id="source3", data="Third dataset"),
]
graph.run(initial_tasks=[(processor, element) for element in initial_data])

# Will print:
# Analyzed 3 sources from (('InitialTaskWorker', 1),)
# Analyzed 3 sources from (('InitialTaskWorker', 2),)
# Analyzed 3 sources from (('InitialTaskWorker', 3),)
