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
            print(f"Skipping uninteresting text: {input_task.text[:100]}...")
            return

        # we just republish the input task to the next LLM in the graph
        super().post_process(input_task.model_copy(), input_task=input_task)


class CreateQuestions(CachedLLMTaskWorker):
    """
    Generates three closed textbook questions based on the given text chunk. This number
    balances between generating sufficient questions without overwhelming the system.
    """

    output_types: List[Type[Task]] = [Question]
    llm_output_type: Type[Task] = Questions
    prompt: str = dedent(
        """
    You are an expert educator tasked with creating high-quality closed textbook questions based on the given text. Your goal is to generate questions that:

    1. Can be answered directly and unambiguously from the information provided in the text
    2. Test specific knowledge rather than general understanding
    3. Are clear, concise, and require minimal interpretation
    4. Vary in both focus and complexity
    5. Cover different aspects of the information presented in the text

    Guidelines for question creation:
    - Focus on key facts, dates, names, events, concepts, or relationships from the text
    - Ensure each question targets a different piece of information or aspect
    - Frame questions as if they are testing general knowledge, not specific to any text
    - Ensure that the correct answer is found in or directly inferable from the given text
    - Use a variety of question types, such as "who", "what", "when", "where", "which", or "how" questions
    - Do not include the answer in the question itself
    - Avoid overly broad or general questions
    - Do not use phrases like "according to the text" or "in the passage"

    Based on the given text, generate exactly three high-quality closed textbook questions. Provide your response in the following JSON format:

    {
        "questions": [
            "Question 1",
            "Question 2",
            "Question 3"
        ]
    }

    Ensure that:
    - Each question is a complete, properly formatted sentence ending with a question mark
    - The questions vary in their focus and target different information from the text
    - At least one question requires synthesizing information from different parts of the text
    - The questions are ordered from least to most complex
    - No question directly references the source text or any specific passage
    - Each question can stand alone as a general knowledge question about the subject matter
    """
    ).strip()

    def consume_work(self, task: InputChunk):
        super().consume_work(task)

    def post_process(self, response: Optional[Questions], input_task: InputChunk):
        if response is None:
            print(f"No questions generated for text: {input_task.text[:100]}...")
            return
        for question in response.questions:
            self.publish_work(Question(question=question), input_task=input_task)


class QuestionEvaluationWorker(CachedLLMTaskWorker):
    """
    Evaluates the quality of generated questions, providing analysis, ratings, and potential
    improvements to ensure high-quality output.
    """

    output_types: List[Type[Task]] = [Question]
    llm_output_type: Type[Task] = QuestionEvaluation
    prompt: str = dedent(
        """
    You are an expert educator tasked with evaluating the quality of a closed textbook question. Analyze the following question in the context of the given text chunk to determine if it meets the criteria for a good closed textbook question.

    Text chunk:
    {input_text}

    Question to evaluate:
    {question}

    Criteria for a good closed textbook question:
    1. Specificity: The question targets specific, factual information.
    2. Clarity: The question is clear, concise, and unambiguous.
    3. Answerability: The answer can be found in or directly inferred from the given text.
    4. Relevance: The question relates to important or central information in the text.
    5. Closed-ended nature: The question has a definite, fact-based answer that doesn't invite opinion.
    6. Independence: The question stands alone without referencing any specific text or requiring additional context.

    Provide your evaluation in the following JSON format:

    {{
        "analysis": "A concise analysis of the question's strengths and weaknesses, referencing the above criteria. Explicitly state if the answer is not found in the text. 2-3 sentences.",
        "rating": "excellent/good/fair/poor",
        "improved_question": "An improved version of the question, or null if the question is excellent.",
        "is_satisfactory": true or false
    }}

    The 'is_satisfactory' field should be true ONLY if ALL of the following conditions are met:
    1. The question is rated excellent or good.
    2. The answer to the question can be found in or directly inferred from the given text.
    3. The question itself does not reference any specific text, passage, or source.
    4. The question is self-contained and doesn't require additional context to understand.

    Important:
    - The question must not contain any reference to "the text," "the passage," or any other specific source.
    - Evaluate the question's quality both as a standalone question and in terms of its answerability from the given text.
    - If the question is well-formed but not answerable from the given text, it should be rated as unsatisfactory.
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
            print(
                f"No evaluation generated for question: {input_task.question[:100]}..."
            )
            return
        if response.is_satisfactory:
            print(f"Question is satisfactory: {input_task.question}")
            print(f"Analysis: {response.analysis}")
            print(f"Rating: {response.rating}")
            print("---------")
            self.publish_work(
                Question(question=input_task.question), input_task=input_task
            )
        elif response.improved_question:
            print(f"Question needs improvement: {input_task.question}")
            print(f"Improvement suggestion: {response.improved_question}")
            print(f"Analysis: {response.analysis}")
            print(f"Rating: {response.rating}")
            print("---------")
            self.publish_work(
                Question(question=response.improved_question), input_task=input_task
            )
        else:
            print(f"Question is unsatisfactory: {input_task.question}")
            print(f"Analysis: {response.analysis}")
            print(f"Rating: {response.rating}")
            print("However, no improvement suggestion was provided.")
            print("---------")


class QuestionAnswer(CachedLLMTaskWorker):
    """
    Generates two comprehensive and distinct answers for a given question based on the
    provided text, allowing for selection of the best response.
    """

    output_types: List[Type[Task]] = [Answers]
    prompt: str = dedent(
        """
    You are an expert educator tasked with generating two comprehensive and distinct answers for a given question based on the provided text. Your goal is to create answers that:

    1. Are directly derived from the information in the given text
    2. Are accurate, factual, and comprehensive
    3. Are clear, well-structured, and easy to understand
    4. Vary significantly in their approach, detail, or perspective
    5. Do not include any information not present in the text
    6. Demonstrate depth of understanding and analysis

    Text:
    {input_text}

    Question:
    {question}

    Guidelines for answer generation:
    - Ensure both answers are correct and based solely on the information in the text
    - Make the answers distinct from each other in terms of detail, focus, or structure
    - Aim for completeness while maintaining clarity and relevance
    - Include specific details, examples, or context from the text to support your answers
    - Use markdown formatting to enhance readability (e.g., for lists, emphasis, or headings)
    - Organize information logically, using paragraphs or bullet points as appropriate
    - Consider including relevant dates, names, or events mentioned in the text
    - Explain any cause-and-effect relationships or historical context if applicable
    - Do not include any external knowledge or information not present in the text

    Generate two possible answers for the given question. Each answer should be at least 3-4 sentences long and cover multiple aspects of the question. Provide your response in the following JSON format:

    {{
        "answer1": "First comprehensive answer, formatted in markdown for clarity and structure",
        "answer2": "Second comprehensive answer, taking a different approach or focus, also formatted in markdown"
    }}

    Ensure that:
    - Both answers are factually correct and comprehensive based on the text
    - The answers are significantly distinct from each other in approach or focus
    - No answer contains information not present in the given text
    - Markdown formatting is used to enhance readability and structure
    - Each answer provides a complete and satisfying response to the question
    """
    ).strip()

    def format_prompt(self, task: Question) -> str:
        input_chunk: Optional[InputChunk] = task.find_input_task(InputChunk)
        if input_chunk is None:
            raise ValueError("InputChunk not found in task dependencies")
        return self.prompt.format(input_text=input_chunk.text, question=task.question)

    def consume_work(self, task: Question):
        super().consume_work(task)

    def post_process(self, response: Optional[Answers], input_task: Question):
        if response is None:
            print(f"No answers generated for question: {input_task.question[:100]}...")
            return

        super().post_process(response, input_task)


class AnswerEvaluator(CachedLLMTaskWorker):
    """
    Evaluates the two generated answers for each question and selects the best one based on
    accuracy, completeness, and clarity.
    """

    output_types: List[Type[Task]] = [QuestionAndAnswer]
    llm_output_type: Type[Task] = EvaluatedAnswer
    prompt: str = dedent(
        """
    You are an expert educator tasked with evaluating two possible answers to a given question and selecting the best one. Your goal is to choose the answer that:

    1. Most accurately and completely addresses the question
    2. Is clearly written and easy to understand
    3. Provides the most relevant information from the original text
    4. Is concise while still being comprehensive

    Question:
    {question}

    Answer 1:
    {answer1}

    Answer 2:
    {answer2}

    Original Text:
    {input_text}

    Guidelines for evaluation:
    - Compare both answers against the original text for accuracy
    - Consider the clarity and conciseness of each answer
    - Evaluate how well each answer directly addresses the question
    - Assess the relevance and completeness of the information provided

    Select the best answer and provide your evaluation. Your response should be in the following JSON format:

    {{
        "best_answer": "The full text of the selected best answer",
        "explanation": "A brief explanation (2-3 sentences) of why this answer was chosen over the other, referencing the evaluation criteria"
    }}

    Ensure that:
    - You choose only one answer as the best
    - Your explanation clearly justifies why the chosen answer is superior
    - You consider both the content and the presentation of the answers
    """
    ).strip()

    def format_prompt(self, task: Answers) -> str:
        question_task: Optional[Question] = task.find_input_task(Question)
        input_chunk: Optional[InputChunk] = task.find_input_task(InputChunk)
        if question_task is None or input_chunk is None:
            raise ValueError("Required input tasks not found in task dependencies")
        return self.prompt.format(
            question=question_task.question,
            answer1=task.answer1,
            answer2=task.answer2,
            input_text=input_chunk.text,
        )

    def consume_work(self, task: Answers):
        super().consume_work(task)

    def post_process(self, response: Optional[EvaluatedAnswer], input_task: Answers):
        if response is None:
            print("No evaluation generated for answers.")
            return

        question_task: Optional[Question] = input_task.find_input_task(Question)
        if question_task is None:
            raise ValueError("Question task not found in input_task dependencies")

        print(f"Question: {question_task.question}")
        print(f"Best Answer: {response.best_answer}")
        print(f"Explanation: {response.explanation}")
        print("---------")

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
        print(f"Text chunk: {input_chunk.text}")
        print(f"Question: {task.question}")
        print(f"Answer: {task.answer}")
        print("---------")

        self._output_file.write(task.model_dump_json(indent=2) + ",\n")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Textbook app")
    parser.add_argument(
        "--file", type=str, help="Path to the file to process", required=True
    )
    args = parser.parse_args()

    setup_logging()

    # Initialize the PlanAI graph for task management
    graph = Graph(name="Textbook Analysis")

    main_model = "llama3.1:70b"
    # main_model = "phi3:14b"

    # we use a small and fast llm to determine whether a text chunk is interesting
    local_llm = llm_from_config(
        provider="ollama", model_name=main_model, host="http://localhost:11435/"
    )
    clean_text_worker = CleanText(llm=local_llm)
    interesting_worker = InterestingText(llm=local_llm)

    # we use a more powerful llm to generate questions from interesting text chunks
    # while developing we use a smaller model to speed up the process but in production we can use a larger model
    reasoning_llm = llm_from_config(
        provider="ollama", model_name=main_model, host="http://localhost:11435/"
    )
    create_questions_worker = CreateQuestions(llm=reasoning_llm)

    question_evaluation_worker = QuestionEvaluationWorker(llm=reasoning_llm)
    question_answer_worker = QuestionAnswer(llm=reasoning_llm)
    answer_evaluation_worker = AnswerEvaluator(llm=reasoning_llm)

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

    # Process the input file
    print(f"Processing file: {args.file}")
    text = pdf_to_text(args.file)
    chunks = create_semantic_chunks(
        text, max_sentences=80, buffer_size=1, threshold_percentile=55
    )

    # Create initial tasks for processing
    input_work = [
        (clean_text_worker, InputChunk(text=chunk)) for chunk in chunks[40:58]
    ]

    # Run the graph with the initial tasks
    print(f"Processing {len(input_work)} chunks...")
    graph.run(initial_tasks=input_work, run_dashboard=True)


if __name__ == "__main__":
    main()
