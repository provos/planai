## Example: Textbook Question and Answer Generation

PlanAI has been used to create a system for generating question and answer pairs from textbook content. This example demonstrates PlanAI's capabilities in processing educational material and automating complex workflows.

### Project Overview

The application processes textbook content to create question and answer pairs suitable for educational purposes or model training. It uses a series of AI-powered workers to:

1. Clean and format text
2. Identify relevant content
3. Generate questions
4. Evaluate question quality
5. Generate and select answers

The workflow is managed using the PlanAI framework, which allows for parallel processing of tasks while maintaining control over LLM API usage.

### Key Components

- **Text Cleaning (CleanText)**: Removes irrelevant content and improves text formatting.
- **Relevance Filtering (InterestingText)**: Identifies text chunks suitable for Q&A generation.
- **Question Generation (CreateQuestions)**: Produces multiple questions from each relevant text chunk.
- **Question Evaluation (QuestionEvaluationWorker)**: Assesses and improves question quality.
- **Answer Generation (QuestionAnswer)**: Creates multiple potential answers for each question.
- **Answer Evaluation (AnswerEvaluator)**: Selects the best answer from the generated options.
- **Output Handling (PrintOutput)**: Manages the final output of Q&A pairs.

### Workflow

1. The input text is divided into chunks and processed by the CleanText worker.
2. InterestingText worker filters out irrelevant content.
3. CreateQuestions generates multiple questions for each relevant chunk.
4. QuestionEvaluationWorker assesses each question and suggests improvements if needed.
5. QuestionAnswer generates two potential answers for each approved question.
6. AnswerEvaluator selects the best answer based on accuracy and clarity.
7. PrintOutput handles the final Q&A pairs, printing them and saving to a file.

### Usage

The application can be run from the command line, specifying the input file:

```bash
python textbook_app.py --file path/to/your/textbook.pdf
