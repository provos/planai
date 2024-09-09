import argparse
from textwrap import dedent
from typing import List, Literal, Optional, Type

from dotenv import load_dotenv
from pydantic import Field

from planai import CachedLLMTaskWorker, Graph, Task, TaskWorker, llm_from_config
from planai.utils import setup_logging

from utils import create_semantic_chunks, pdf_to_text

load_dotenv()


# Models
class InputChunk(Task):
    text: str = Field(description="The text chunk to process.")


class IsInteresting(Task):
    analysis: str = Field(
        description="The analysis on whether the text contains interesting information."
    )
    is_interesting: bool = Field(
        description="Whether the text contains interesting information and should be processed further."
    )


class Question(Task):
    question: str = Field(description="The generated closed textbook question.")


class Questions(Task):
    questions: List[str] = Field(
        description="The generated closed textbook questions for which the task contains a good answer."
    )


class QuestionEvaluation(Task):
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


# Workers


class InterestingText(CachedLLMTaskWorker):
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
        "improvement_question": "An improved version of the question if needed, or null if the question is excellent.",
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
        else:
            print(f"Question needs improvement: {input_task.question}")
            print(f"Improvement suggestion: {response.improved_question}")
            print(f"Analysis: {response.analysis}")
            print(f"Rating: {response.rating}")
            print("---------")
            self.publish_work(
                Question(question=response.improved_question), input_task=input_task
            )


class PrintOutput(TaskWorker):

    def consume_work(self, task: Question):
        input_chunk = task.find_input_task(InputChunk)
        print(f"Text chunk: {input_chunk.text}")
        print(f"Question: {task.question}")
        print("---------")


def main():
    parser = argparse.ArgumentParser(description="Textbook app")
    parser.add_argument(
        "--file", type=str, help="Path to the file to process", required=True
    )
    args = parser.parse_args()

    setup_logging()

    graph = Graph(name="Textbook Analysis")

    # main_model = "llama3.1:70b"
    main_model = "phi3:14b"

    # we use a small and fast llm to determine whether a text chunk is interesting
    local_llm = llm_from_config(
        provider="ollama", model_name=main_model, host="http://localhost:11435/"
    )
    interesting_worker = InterestingText(llm=local_llm)

    # we use a more powerful llm to generate questions from interesting text chunks
    # while developing we use a smaller model to speed up the process but in production we can use a larger model
    reasoning_llm = llm_from_config(
        provider="ollama", model_name=main_model, host="http://localhost:11435/"
    )
    create_questions_worker = CreateQuestions(llm=reasoning_llm)

    question_evaluation_worker = QuestionEvaluationWorker(llm=reasoning_llm)

    print_worker = PrintOutput()

    graph.add_workers(
        interesting_worker,
        create_questions_worker,
        question_evaluation_worker,
        print_worker,
    )
    graph.set_dependency(interesting_worker, create_questions_worker).next(
        question_evaluation_worker
    ).next(print_worker)

    print(f"Processing file: {args.file}")
    text = pdf_to_text(args.file)
    chunks = create_semantic_chunks(
        text, max_sentences=80, buffer_size=1, threshold_percentile=55
    )

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}")

    input_work = [
        (interesting_worker, InputChunk(text=chunk)) for chunk in chunks[25:50]
    ]

    print(f"Processing {len(input_work)} chunks...")
    graph.run(initial_tasks=input_work, run_dashboard=True)


if __name__ == "__main__":
    main()
