"""
Textbook Question and Answer Generation Application

This application processes textbook content to generate high-quality question and answer pairs
suitable for retrieval augmented generation or fine-tuning foundation models. It employs a
series of AI-powered workers to clean text, identify relevant content, create questions,
evaluate their quality, generate answers, and select the best responses.

The workflow is managed using the PlanAI framework, which allows for efficient parallel
processing of tasks while maintaining control over LLM API usage.

Key components:
- Text cleaning and relevance filtering
- Question generation and evaluation
- Answer generation and selection
- Output formatting and storage

Note: This application is designed to process PDF textbooks and may require adjustment
for other input formats.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import List, Literal, Optional, TextIO, Type

from dotenv import load_dotenv
from pydantic import Field, PrivateAttr
from utils import create_semantic_chunks, pdf_to_text

from planai import CachedLLMTaskWorker, Graph, Task, TaskWorker, llm_from_config
from planai.utils import setup_logging

load_dotenv()


# Models
class InputChunk(Task):
    """Represents a chunk of text from the input textbook to be processed."""

    text: str = Field(description="The text chunk to process.")
    id: int = Field(description="The unique identifier for the text chunk.")


class IsInteresting(Task):
    """Holds the analysis result of whether a text chunk contains information suitable for Q&A generation."""

    analysis: str = Field(
        description="The analysis on whether the text contains interesting information."
    )
    is_interesting: bool = Field(
        description="Whether the text contains interesting information and should be processed further."
    )


class Question(Task):
    """Represents a generated closed textbook question."""

    question: str = Field(description="The generated closed textbook question.")


class Questions(Task):
    """Contains a list of generated closed textbook questions."""

    questions: List[str] = Field(
        description="The generated closed textbook questions for which the task contains a good answer."
    )


class QuestionEvaluation(Task):
    """Holds the evaluation results for a generated question, including analysis and potential improvements."""

    analysis: str = Field(
        description="A concise analysis of the question's strengths and weaknesses, referencing the above criteria. 2-3 sentences."
    )
    rating: Literal["excellent", "good", "fair", "poor"] = Field(
        description="The rating of the question: excellent/good/fair/poor"
    )
    improved_question: Optional[str] = Field(
        description="An improved question if needed or null if the question is excellent."
    )
    is_satisfactory: bool = Field(
        description="Whether the question meets the criteria for a high-quality closed textbook question."
    )


class Answers(Task):
    """Contains two possible answers generated for a given question."""

    answer1: str = Field(description="The first possible answer to the question.")
    answer2: str = Field(description="The second possible answer to the question.")


class EvaluatedAnswer(Task):
    """Represents the selected best answer along with an explanation for the choice."""

    best_answer: str = Field(description="The selected best answer.")
    explanation: str = Field(description="Explanation for why this answer was chosen.")


class QuestionAndAnswer(Task):
    """Represents a final question-answer pair ready for output."""

    question: str = Field(description="The generated closed textbook question.")
    answer: str = Field(description="The answer to the question.")


# Workers


class CleanText(CachedLLMTaskWorker):
    """
    Cleans and formats input text chunks by removing irrelevant content such as page numbers,
    headers, and footers, ensuring only the main textbook content is processed further.
    """

    output_types: List[Type[Task]] = [InputChunk]
    llm_output_type: Type[Task] = InputChunk
    prompt: str = dedent(
        """
    Clean and format the given text to extract only the relevant content, following these guidelines:

    1. Remove metadata, headers, footers, page numbers, and other non-content elements.
    2. Correct OCR errors and formatting issues to improve readability.
    3. Remove irrelevant content such as image captions, copyright information, and chapter headings.
    4. Ensure the cleaned text is coherent and self-contained.
    5. Preserve the original meaning and information of the text.
    6. Do not add any new information or alter the content's meaning.
    7. If the text contains multiple sections or topics, separate them with line breaks.

    Provide your response in the following JSON format:

    {{
    \"text\": \"The cleaned and formatted text chunk.\"
    }}
    """
    ).strip()

    def consume_work(self, task: InputChunk):
        super().consume_work(task)


class InterestingText(CachedLLMTaskWorker):
    """
    Analyzes text chunks to determine if they contain information suitable for Q&A generation,
    filtering out content that may not yield useful questions (e.g., index pages).
    """

    output_types: List[Type[Task]] = [InputChunk]
    llm_output_type: Type[Task] = IsInteresting
    prompt: str = dedent(
        """
    Analyze the given text chunk and determine if it contains information suitable for creating closed-book questions in a Q&A dataset. Consider the following criteria:

    1. Specific facts, data, or details that can be directly queried
    2. Unique or distinctive information not commonly known
    3. Names, dates, places, or events that can form the basis of factual questions
    4. Defined concepts, terms, or processes that can be explained or described
    5. Cause-and-effect relationships or sequences of events
    6. Comparisons or contrasts between ideas, events, or entities

    Importantly, consider whether the information in the text is:
    a) Substantial enough to form the basis of a question
    b) Specific to the text rather than general knowledge
    c) Self-contained within the given chunk

    Provide your response in the following JSON format:

    {
        "analysis": "A concise explanation of why the text is or isn't suitable for closed-book Q&A generation. If suitable, mention 1-2 specific pieces of information that could be used for questions. If not suitable, explain why. Limit to 2-3 sentences.",
        "is_interesting": true or false
    }

    The 'is_interesting' field should be true only if the text contains specific, substantial information that can be directly used to create closed-book questions. Exclude text that is too general, lacks specific facts, or is primarily metadata (like acknowledgments or table of contents).
    """
    ).strip()

    def consume_work(self, task: InputChunk):
        super().consume_work(task)

    def post_process(self, response: IsInteresting, input_task: InputChunk):
        if response is not None and not response.is_interesting:
            self.print(f"Skipping uninteresting text: {input_task.text[:100]}...")
            return

        # we just republish the input task to the next LLM in the graph
        super().post_process(input_task.model_copy(), input_task=input_task)


class CreateQuestions(CachedLLMTaskWorker):
    """
    Generates three closed textbook questions based on the given text chunk. This number
    balances between generating sufficient questions without overwhelming the system.
    """

    debug_mode: bool = False
    output_types: List[Type[Task]] = [Question]
    llm_output_type: Type[Task] = Questions
    prompt: str = dedent(
        """
You are an expert educator tasked with crafting high-quality closed textbook questions based on the given text. Your goal is to develop questions that:

1. Can be answered directly and unambiguously from the information provided in the text.
2. Test specific knowledge rather than general understanding, while also encouraging deeper analysis.
3. Are clear, concise, and require minimal interpretation.
4. Vary in focus, complexity, and approach, incorporating diverse question types and cognitive skills.
5. Cover different aspects of the information presented in the text, ensuring all major themes and concepts are addressed.

Guidelines for question creation:
- Focus on key facts, dates, names, events, concepts, or relationships from the text.
- Ensure each question targets a different piece of information or aspect.
- Frame questions to test general knowledge applicable beyond the specific text.
- Ensure that the correct answer is found in or directly inferable from the given text without external context.
- Use a range of question types such as "who", "what", "when", "where", "which", "how", and "why" questions to cover logical reasoning and diverse cognitive levels.
- Include at least one question that requires synthesis of information across multiple text segments.
- Encourage critical thinking by connecting themes or exploring implications.
- Frame questions so they can stand alone as general knowledge questions without needing supplementary context.
- Avoid overly broad or general questions and refrain from using phrases like "according to the text".

Based on the given text, generate exactly three high-quality closed textbook questions, ordered from least to most complex. Utilize a variety of question approaches while ensuring clarity and factual accuracy. Provide your response in the following JSON format:

{
    "questions": [
        "Question 1",
        "Question 2",
        "Question 3"
    ]
}
    """
    ).strip()

    def consume_work(self, task: InputChunk):
        super().consume_work(task)

    def post_process(self, response: Optional[Questions], input_task: InputChunk):
        if response is None:
            self.print(f"No questions generated for text: {input_task.text[:100]}...")
            return
        for question in response.questions:
            self.publish_work(Question(question=question), input_task=input_task)


class QuestionEvaluationWorker(CachedLLMTaskWorker):
    """
    Evaluates the quality of generated questions, providing analysis, ratings, and potential
    improvements to ensure high-quality output.
    """

    debug_mode: bool = False
    output_types: List[Type[Task]] = [Question]
    llm_output_type: Type[Task] = QuestionEvaluation
    prompt: str = dedent(
        """
You are an expert educator and your task is to evaluate a question to determine if it is a good closed-textbook question. A good question should independently stand on its own, allowing it to be answered using only the provided information in the text chunk.

Text chunk:
{input_text}

Question to evaluate:
{question}

A good closed-textbook question must satisfy the following criteria:
1. Specificity: It targets precise, factual information found in the text.
2. Clarity: It is clear and unambiguous, ensuring the reader knows exactly what is being asked.
3. Answerability: The answer must be present in or directly inferred from the text.
4. Relevance: It relates to the central or significant information in the text.
5. Closed-ended nature: It should have a definite, fact-based answer without inviting subjective opinion.
6. Independence: It should stand alone without referencing a specific text part or needing additional context.

Evaluation Format:
{{
    "analysis": "Evaluate the question by addressing each criterion, highlighting strengths and weaknesses. Note explicitly if the answer isn't found in the text.",
    "rating": "excellent/good/fair/poor",
    "improved_question": "Suggest an improved version of the question, or use null if the question is rated excellent.",
    "is_satisfactory": true or false
}}

Mark 'is_satisfactory' as true ONLY if all these conditions are satisfied:
1. The question is rated as excellent or good.
2. The response to the question is located or inferred directly from the text.
3. The question does not reference the text or any specific source.
4. It requires no additional context for understanding.

Consider broader themes or implications that may be touched by the question, encouraging diverse interpretations to enrich the analysis. Conclude whether the question invites engagement with the broader topic or remains narrowly focused.

Examples:
- Strong example: "What primary function do mitochondria serve according to the text?"
- Weak example: "Discuss external theories of mitochondria not found in the text."
    """
    ).strip()

    def format_prompt(self, task: Question) -> str:
        input_chunk: Optional[InputChunk] = task.find_input_task(InputChunk)
        if input_chunk is None:
            raise ValueError("InputChunk not found in task dependencies")
        return self.prompt.format(input_text=input_chunk.text, question=task.question)

    def consume_work(self, task: Question):
        super().consume_work(task)

    def post_process(
        self, response: Optional[QuestionEvaluation], input_task: Question
    ):
        if response is None:
            self.print(
                f"No evaluation generated for question: {input_task.question[:100]}..."
            )
            return
        if response.is_satisfactory:
            self.print(f"Question is satisfactory: {input_task.question}")
            self.print(f"Analysis: {response.analysis}")
            self.print(f"Rating: {response.rating}")
            self.print("---------")
            self.publish_work(
                Question(question=input_task.question), input_task=input_task
            )
        elif response.improved_question:
            self.print(f"Question needs improvement: {input_task.question}")
            self.print(f"Improvement suggestion: {response.improved_question}")
            self.print(f"Analysis: {response.analysis}")
            self.print(f"Rating: {response.rating}")
            self.print("---------")
            self.publish_work(
                Question(question=response.improved_question), input_task=input_task
            )
        else:
            self.print(f"Question is unsatisfactory: {input_task.question}")
            self.print(f"Analysis: {response.analysis}")
            self.print(f"Rating: {response.rating}")
            self.print("However, no improvement suggestion was provided.")
            self.print("---------")


def get_input_text(input_chunk: InputChunk, chunks: List[str]) -> str:
    """
    Retrieve a segment of text from a list of chunks based on the provided input chunk.

    Parameters:
        input_chunk (InputChunk): The input chunk containing the current text and its ID.
        chunks (List[str]): A list of text chunks from which to retrieve the segment.

    Returns:
        str: A string containing the concatenated text from the specified range of chunks.
              If the chunks list is empty, returns the text from the input_chunk.
    """
    if chunks:
        min_id = max(0, input_chunk.id - 1)
        max_id = min(len(chunks), input_chunk.id + 2)
        input_text = "\n".join(chunks[min_id:max_id])
    else:
        input_text = input_chunk.text
    return input_text


class QuestionAnswer(CachedLLMTaskWorker):
    """
    Generates two comprehensive and distinct answers for a given question based on the
    provided text, allowing for selection of the best response.
    """

    debug_mode: bool = False
    output_types: List[Type[Task]] = [Answers]
    prompt: str = dedent(
        """
You are an expert educator tasked with generating two comprehensive and distinct answers for a given question based on the provided text. Your goal is to create answers that:

1. Are derived solely from the information in the provided text, ensuring completeness and accuracy.
2. Are clear, well-structured, and easy to understand with the use of markdown formatting for readability.
3. Demonstrate a depth of understanding by exploring different aspects and implications of the text.
4. Significantly vary in their approach, detail, or perspective to ensure distinct answers.

**Text:**
{input_text}

**Question:**
{question}

**Guidelines for Answer Generation:**
- Both answers must be correct and contain only information from the text.
- Use detailed examples, context, and specific information to support your answers.
- Ensure that each answer covers a different aspect or theme of the question by varying in structure, such as using bullets or paragraphs, or focusing on different implications.
- Include pertinent details such as dates, names, or events explicitly mentioned in the text.
- Organize your answers logically, explaining cause-and-effect relationships or historical context if present.
- Avoid any external knowledge or interpretations beyond the text.

**Output Format:**
Generate two possible answers for the question. Each should be at least 3-4 sentences long, addressing multiple aspects of the question. Format your response as follows:

{{
    "answer1": "First comprehensive answer, using markdown for clarity and structured differently",
    "answer2": "Second comprehensive and distinct answer, with another structure or focus using markdown"
}}

Ensure:
- Both answers are factually correct and comprehensive.
- The answers differ significantly in approach or focus.
- No external information is included.
- Markdown enhances readability and structure.
- Each answer provides a thorough and satisfying response to the question.
    """
    ).strip()

    _chunks: List[str] = PrivateAttr(default=[])

    def set_chunks(self, chunks: List[str]):
        self._chunks = chunks

    def format_prompt(self, task: Question) -> str:
        input_chunk: Optional[InputChunk] = task.find_input_task(InputChunk)
        if input_chunk is None:
            raise ValueError("InputChunk not found in task dependencies")
        input_text = get_input_text(input_chunk, self._chunks)
        return self.prompt.format(input_text=input_text, question=task.question)

    def consume_work(self, task: Question):
        super().consume_work(task)

    def post_process(self, response: Optional[Answers], input_task: Question):
        if response is None:
            self.print(
                f"No answers generated for question: {input_task.question[:100]}..."
            )
            return

        super().post_process(response, input_task)


class AnswerEvaluator(CachedLLMTaskWorker):
    """
    Evaluates the two generated answers for each question and selects the best one based on
    accuracy, completeness, and clarity.
    """

    debug_mode: bool = False
    output_types: List[Type[Task]] = [QuestionAndAnswer]
    llm_output_type: Type[Task] = EvaluatedAnswer
    prompt: str = dedent(
        """
You are an expert educator tasked with evaluating two potential answers to a given question. Your objective is to select the answer that fulfills the following criteria:

1. Provides a direct and comprehensive response to the question based on the input text.
2. Is written with clarity and is easy to comprehend.
3. Extracts and explains the most pertinent information from the original text.
4. Balances conciseness with comprehensiveness, avoiding verbosity while covering necessary details.
5. Offers a unique perspective or interpretation that adds depth to the analysis.

Question:
{question}

Answer 1:
{answer1}

Answer 2:
{answer2}

Original Text:
{input_text}

Evaluation Guidelines:
- Verify the accuracy of each answer by cross-referencing the original text and highlighting any discrepancies.
- Assess the clarity and readability of each answer, ensuring it is well-structured.
- Determine how directly and thoroughly each answer tackles the question, focusing on both its directness and comprehensiveness.
- Evaluate the completeness, relevance, and unique perspectives presented in each answer.

Please provide your decision in JSON format with the following structure:

{{
    "best_answer": "<The full text of the selected best answer without any additional commentary or text>",
    "explanation": "<A concise explanation (2-3 sentences) supporting your choice, focused on the evaluation criteria>"
}}

In your explanation, justify the superiority of the chosen answer by considering both content quality and presentation. Highlight any unique insights or perspectives in your evaluation. Select only one answer as the best and copy it directly from the input without including the text "Answer 1" or "Answer 2" as part of it.
    """
    ).strip()

    _chunks: List[str] = PrivateAttr(default=[])

    def set_chunks(self, chunks: List[str]):
        self._chunks = chunks

    def format_prompt(self, task: Answers) -> str:
        question_task: Optional[Question] = task.find_input_task(Question)
        input_chunk: Optional[InputChunk] = task.find_input_task(InputChunk)
        if question_task is None or input_chunk is None:
            raise ValueError("Required input tasks not found in task dependencies")
        input_text = get_input_text(input_chunk, self._chunks)
        return self.prompt.format(
            question=question_task.question,
            answer1=task.answer1,
            answer2=task.answer2,
            input_text=input_text,
        )

    def consume_work(self, task: Answers):
        super().consume_work(task)

    def post_process(self, response: Optional[EvaluatedAnswer], input_task: Answers):
        if response is None:
            self.print("No evaluation generated for answers.")
            return

        question_task: Optional[Question] = input_task.find_input_task(Question)
        if question_task is None:
            raise ValueError("Question task not found in input_task dependencies")

        self.print(f"Question: {question_task.question}")
        self.print(f"Best Answer: {response.best_answer}")
        self.print(f"Explanation: {response.explanation}")
        self.print("---------")

        question_and_answer = QuestionAndAnswer(
            question=question_task.question, answer=response.best_answer
        )

        super().post_process(question_and_answer, input_task)


class PrintOutput(TaskWorker):
    """
    Handles the final output of the processed question-answer pairs, printing them to the
    console and saving them to a JSON file.
    """

    output_dir: str = Field("output", description="The directory to save output in")
    _output_file: TextIO = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"output_{current_time}.json"
        self._output_file = open(output_file, "w", encoding="utf-8")

    def consume_work(self, task: QuestionAndAnswer):
        input_chunk = task.find_input_task(InputChunk)
        self.print(f"Text chunk: {input_chunk.text}")
        self.print(f"Question: {task.question}")
        self.print(f"Answer: {task.answer}")
        self.print("---------")

        self._output_file.write(task.model_dump_json(indent=2) + ",\n")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Textbook app")
    parser.add_argument(
        "--file", type=str, help="Path to the file to process", required=False
    )
    parser.add_argument(
        "--chunk-output",
        type=str,
        help="Path to save chunks as JSON and exit without running the pipeline",
        required=False,
    )
    parser.add_argument(
        "--chunk-input",
        type=str,
        help="Path to read chunks from JSON instead of processing the file",
        required=False,
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run the pipeline in test mode with a small subset of data",
    )
    args = parser.parse_args()

    if not args.file and not args.chunk_input:
        print("Please provide either a file to process or a JSON file with chunks.")
        exit(1)

    setup_logging()

    # Initialize the PlanAI graph for task management
    graph = Graph(name="Textbook Analysis")

    # Initialize workers
    fast_llm = llm_from_config(provider="openai", model_name="gpt-4o-mini")
    reasoning_llm = llm_from_config(provider="openai", model_name="gpt-4o-2024-08-06")

    clean_text_worker = CleanText(llm=fast_llm)
    interesting_worker = InterestingText(llm=fast_llm)
    create_questions_worker = CreateQuestions(llm=reasoning_llm)
    question_evaluation_worker = QuestionEvaluationWorker(llm=fast_llm)
    question_answer_worker = QuestionAnswer(llm=reasoning_llm)
    answer_evaluation_worker = AnswerEvaluator(llm=fast_llm)
    print_worker = PrintOutput()

    # Add workers to the graph and set up the processing pipeline
    graph.add_workers(
        clean_text_worker,
        interesting_worker,
        create_questions_worker,
        question_evaluation_worker,
        question_answer_worker,
        answer_evaluation_worker,
        print_worker,
    )
    graph.set_dependency(clean_text_worker, interesting_worker).next(
        create_questions_worker
    ).next(question_evaluation_worker).next(question_answer_worker).next(
        answer_evaluation_worker
    ).next(
        print_worker
    )

    # Limit the number of parallel LLM tasks for performance management
    graph.set_max_parallel_tasks(CachedLLMTaskWorker, 4)

    # Process chunks based on input arguments
    if args.chunk_output:
        chunks = save_chunks(args.file, args.chunk_output)
        print(f"Chunks saved to {args.chunk_output}")
        return
    elif args.chunk_input:
        chunks = load_chunks_from_file(args.chunk_input)
        print(f"Chunks loaded from {args.chunk_input}")
    else:
        chunks = process_file(args.file)

    if args.test_run:
        start_index = max(0, len(chunks) // 2 - 5)
        end_index = min(len(chunks), len(chunks) // 2 + 5)
        chunks = chunks[start_index:end_index]
        print(f"Running test pipeline with {len(chunks)} chunks...")

    # Inject chunks into the workers that require them
    # This will give them extra context for generating answers.
    question_answer_worker.set_chunks(chunks)
    answer_evaluation_worker.set_chunks(chunks)

    # Create initial tasks for processing
    input_work = [
        (clean_text_worker, InputChunk(text=chunk, id=id))
        for id, chunk in enumerate(chunks)
    ]

    # Run the graph with the initial tasks
    print(f"Processing {len(input_work)} chunks...")

    graph.run(initial_tasks=input_work, run_dashboard=True)


def save_chunks(file_path: str, output_path: str):
    text = pdf_to_text(file_path)
    chunks = create_semantic_chunks(
        text, max_sentences=80, buffer_size=1, threshold_percentile=55
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)


def load_chunks_from_file(input_path: str) -> List[InputChunk]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_file(file_path: str) -> List[str]:
    text = pdf_to_text(file_path)
    return create_semantic_chunks(
        text, max_sentences=80, buffer_size=1, threshold_percentile=55
    )


if __name__ == "__main__":
    main()
