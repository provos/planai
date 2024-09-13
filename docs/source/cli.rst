CLI
===

Automatic Prompt Optimization
-----------------------------

One of the main features of the PlanAI CLI is the automatic prompt optimization tool. This feature helps refine prompts for Large Language Models (LLMs) by leveraging more advanced LLMs to improve prompt effectiveness.

Key aspects of the optimize-prompt tool:

- Automates the process of iterating and improving prompts
- Uses real production data from debug logs for optimization
- Dynamically loads and uses production classes in the workflow
- Employs an LLM-based scoring mechanism to evaluate prompt effectiveness
- Adapts to various LLM tasks

To use the optimize-prompt tool, ensure you have generated debug logs with real production data by setting `debug_mode=True` for your LLMTaskWorker classes. Then, run a command similar to:

.. code-block:: bash

   planai --llm-provider openai --llm-model gpt-4o-mini --llm-reason-model gpt-4 optimize-prompt --python-file your_app.py --class-name YourLLMTaskWorker --search-path . --debug-log debug/YourLLMTaskWorker.json --goal-prompt "Your optimization goal here"

The tool will generate optimized prompts as text files along with corresponding JSON files containing metadata about the improvements.

For more detailed information on the prompt optimization feature, refer to the PlanAI documentation.

.. argparse::
   :module: planai.cli
   :func: create_parser
   :prog: planai

.. automodule:: planai.cli
   :members:
   :undoc-members:
   :show-inheritance:
