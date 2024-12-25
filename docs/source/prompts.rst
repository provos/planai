Prompts
=======

Prompts are the most important component for influencing the performance of an LLM. PlanAI provides a flexible prompting system that helps maintain consistency while allowing customization.

Default Template
----------------
By default, PlanAI will automatically format the input task into a prompt using this template:

.. code-block:: text

    Here is your input data:
    {task}

    Here are your instructions:
    {instructions}

The ``{task}`` placeholder is automatically filled with the formatted input task data, and ``{instructions}`` is filled with the prompt you provide to the LLMTaskWorker.

Customizing Prompts
-------------------
There are several ways to customize how prompts are generated:

1. Basic Prompt Customization
   Provide your instructions when creating an LLMTaskWorker:

   .. code-block:: python

       worker = LLMTaskWorker(
           llm=llm,
           prompt="Analyze this text and extract key topics",
           output_types=[TopicsTask]
       )

2. Dynamic Prompt Generation
   Override ``format_prompt()`` to generate prompts based on the input task:

   .. code-block:: python

       class CustomWorker(LLMTaskWorker):
           def format_prompt(self, task: InputTask) -> str:
               return f"Analyze this {task.content_type} and provide a summary"

   The ``format_prompt()`` method can also be used to fill template parameters in your prompt:

   .. code-block:: python

       class StatementAnalyzer(LLMTaskWorker):
           prompt="Think about whether {statement} is a good idea",

           def format_prompt(self, task: Task) -> str:
               statement = task.find_input(StatementTask)
               return self.prompt.format(statement=statement.statement)

3. Input Pre-processing
   Override ``pre_process()`` to modify how the input task is presented:

   .. code-block:: python

       def pre_process(self, task: InputTask) -> Optional[Task]:
           # Remove sensitive fields or reformat data
           return ProcessedTask(content=task.filtered_content)

   The ``pre_process()`` method serves two important purposes:

   1. Data transformation: Convert or filter input data before it reaches the LLM
   2. Input format control: Determine how input data is presented to the LLM

   You can use ``PydanticDictWrapper`` to inject a different Pydantic object:

   .. code-block:: python

       from planai.utils import PydanticDictWrapper

       def pre_process(self, task: Task) -> Optional[Task]:
           custom_data = {
               "filtered": task.content,
               "metadata": task.get_metadata()
           }
           return PydanticDictWrapper(data=custom_data)

   If ``pre_process()`` returns ``None``, PlanAI will not provide any input data in the default template. 
   In this case, your class should provide all necessary context through ``format_prompt()``:

   .. code-block:: python

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

System Prompts
--------------
You can also customize the system prompt that sets the context for the LLM:

.. code-block:: python

    worker = LLMTaskWorker(
        llm=llm,
        prompt="Analyze the text",
        system_prompt="You are an expert analyst specialized in text classification",
        output_types=[AnalysisTask]
    )

Best Practices
--------------
1. Be specific and clear in your instructions
2. Include examples if the task is complex
3. Consider using pre-processing to simplify complex input data
4. Test different system prompts to find what works best
5. Use format_prompt() for dynamic instructions based on input

