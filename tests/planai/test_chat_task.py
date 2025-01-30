import unittest
from datetime import datetime

from llm_interface.testing.mock_llm import MockLLM, MockLLMResponse

from planai.chat_task import ChatMessage, ChatTask, ChatTaskWorker
from planai.testing.helpers import InvokeTaskWorker


class TestChatTaskWorker(unittest.TestCase):
    def setUp(self):
        # Create mock responses for different chat scenarios
        self.responses = [
            MockLLMResponse(
                pattern=r"Hello, how are you\?$",  # Make pattern more specific
                response_string="I'm doing well, thank you for asking!",
            ),
            MockLLMResponse(
                pattern=r"What is the capital of France\?",
                response_string="The capital of France is Paris.",
            ),
            # Updated pattern for multi-turn conversation to match full context
            MockLLMResponse(
                pattern=r"System:.*Today is.*\nUser: Hello, how are you\?\nUser: Great!.*follow-up question",
                response_string="I understand you want to know more. Let me elaborate...",
            ),
        ]

        self.mock_llm = MockLLM(responses=self.responses)
        self.invoke_worker = InvokeTaskWorker(ChatTaskWorker, llm=self.mock_llm)

    def test_basic_chat(self):
        """Test basic chat interaction with a single message."""
        chat_task = ChatTask(
            messages=[ChatMessage(role="user", content="Hello, how are you?")]
        )

        # Process the chat task using InvokeTaskWorker
        published_tasks = self.invoke_worker.invoke(chat_task)

        # Verify response
        self.invoke_worker.assert_published_task_count(1)
        response: ChatMessage = published_tasks[0]

        self.assertIsInstance(response, ChatMessage)
        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.content, "I'm doing well, thank you for asking!")

    def test_system_prompt_formatting(self):
        """Test that system prompt includes current date."""
        chat_task = ChatTask(
            messages=[
                ChatMessage(role="user", content="What is the capital of France?")
            ]
        )

        worker = ChatTaskWorker(llm=self.mock_llm)

        # Get today's date in the expected format
        today = datetime.today().strftime("%Y-%m-%d")

        # Verify the formatted messages included the correct system prompt
        formatted_messages = worker._format_messages(chat_task.messages)

        self.assertEqual(
            formatted_messages[0],
            {
                "role": "system",
                "content": f"You are a helpful AI assistant. Today is {today}.",
            },
        )

    def test_multi_turn_conversation(self):
        """Test handling of multi-turn conversations."""
        chat_task = ChatTask(
            messages=[
                ChatMessage(role="user", content="Hello, how are you?"),
                ChatMessage(
                    role="assistant", content="I'm doing well, thank you for asking!"
                ),
                ChatMessage(
                    role="user",
                    content="Great! I have a follow-up question about your previous response.",
                ),
            ]
        )

        # Process the chat task using InvokeTaskWorker
        published_tasks = self.invoke_worker.invoke(chat_task)

        # Verify response
        self.invoke_worker.assert_published_task_count(1)
        response: ChatMessage = published_tasks[0]

        self.assertIsInstance(response, ChatMessage)
        self.assertEqual(response.role, "assistant")
        self.assertEqual(
            response.content,
            "I understand you want to know more. Let me elaborate...",
        )

    def test_message_format(self):
        """Test proper formatting of messages for LLM."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?"),
        ]
        chat_task = ChatTask(messages=messages)

        formatted = self.invoke_worker.worker._format_messages(chat_task.messages)

        # Check that system message is first
        self.assertEqual(formatted[0]["role"], "system")

        # Check that other messages are formatted correctly
        for i, msg in enumerate(messages):
            self.assertEqual(formatted[i + 1]["role"], msg.role)
            self.assertEqual(formatted[i + 1]["content"], msg.content)


if __name__ == "__main__":
    unittest.main()
