Provenance in PlanAI
====================

In PlanAI, provenance refers to the lineage of a task, providing a historical record of its execution within a workflow. It allows tracing the origin and transformations of a task as it moves through different workers in the graph.

Benefits of Provenance Tracking
-------------------------------

- **Debugging and Auditing**: By tracing the history of a task, you can identify the specific workers and inputs that contributed to its final state, which is crucial for debugging errors or auditing decisions made by AI agents within the workflow.
- **Explainability**: Provenance can be used to explain the results of a workflow by showing the steps involved and how data was transformed at each stage. This is important for understanding and trusting the output of AI-driven workflows.

How PlanAI Uses Provenance
--------------------------

PlanAI leverages provenance information in several ways:

- **Task Lineage**: Each task in PlanAI carries its full execution history, including the sequence of workers it passed through and the inputs it was derived from.
- **Debugging and Monitoring**: The PlanAI dashboard displays the provenance of each task, allowing developers to track its progress and identify any bottlenecks or failures.
- **AI Decision Tracking**: When AI agents make decisions within a workflow, the provenance records the rationale and data behind those decisions, promoting transparency and accountability.

Overall, the provenance tracking capabilities of PlanAI are essential for building robust, reproducible, and explainable AI-driven workflows. It provides developers with the tools they need to understand, debug, and audit complex automated processes.

Advanced Provenance Features
----------------------------

Task Joining Using Provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PlanAI allows workers to join multiple tasks based on their provenance prefix. This is particularly useful in workflows where you need to consolidate results from multiple parallel tasks. For example:

- **Search Result Consolidation**: When multiple search queries are executed in parallel, a join worker can wait for all results with the same search query provenance and combine them for analysis.
- **Batch Processing**: Tasks can be grouped by their origin (e.g., all tasks derived from a specific input) and processed together.

To implement a join, workers can:

1. Specify a worker type to join on using ``join_type``
2. Receive tasks through ``consume_work_joined()`` when all tasks sharing the same provenance prefix are complete
3. Process the consolidated results as a single unit

Context Retrieval Using Provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workers can traverse the provenance chain of a task to access contextual information from earlier stages in the workflow. Common use cases include:

- **LLM Context Enhancement**: Retrieving original user queries or intermediate results to provide better context to language models
- **Decision Tracking**: Accessing the rationale or inputs that led to specific outcomes earlier in the workflow
- **Metadata Propagation**: Carrying important metadata or configuration through the workflow

Tasks provide methods to find specific input tasks in their provenance chain, making it easy to access relevant historical context when needed.