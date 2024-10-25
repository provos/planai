import unittest
from unittest.mock import MagicMock, Mock, patch

from openai import ContentFilterFinishReasonError, LengthFinishReasonError

from planai.openai import OpenAIWrapper


class TestOpenAIWrapper(unittest.TestCase):
    def setUp(self):
        # Mock the OpenAI client
        self.mock_client = Mock()

        # Patch the OpenAI class to use the mock client
        patcher = patch("planai.openai.OpenAI", return_value=self.mock_client)
        self.addCleanup(patcher.stop)
        self.mock_openai = patcher.start()

        self.api_key = "test_api_key"
        self.openai_wrapper = OpenAIWrapper(api_key=self.api_key)

    def test_generate_basic_prompt(self):
        # Mock the response from the client
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response content"))]
        )

        prompt = "Test prompt"
        response = self.openai_wrapper.generate(prompt=prompt)

        self.assertEqual(response, {"response": "Test response content", "done": True})

        # Verify that the client's chat.completions.create was called with correct parameters
        self.mock_client.chat.completions.create.assert_called_once()
        args, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo")
        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": prompt}])

    def test_generate_with_system_message(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="System message response"))]
        )

        prompt = "Test prompt"
        system = "System message"
        response = self.openai_wrapper.generate(prompt=prompt, system=system)

        self.assertEqual(
            response, {"response": "System message response", "done": True}
        )

        self.mock_client.chat.completions.create.assert_called_once()
        args, kwargs = self.mock_client.chat.completions.create.call_args
        expected_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        self.assertEqual(kwargs["messages"], expected_messages)

    def test_generate_with_json_format(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"key": "value"}'))]
        )

        prompt = "Test prompt"
        response = self.openai_wrapper.generate(prompt=prompt, format="json")

        self.assertEqual(response, {"response": '{"key": "value"}', "done": True})

        self.mock_client.chat.completions.create.assert_called_once()
        args, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_chat_basic(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Chat response content"))]
        )

        messages = [{"role": "user", "content": "Hello, assistant!"}]
        response = self.openai_wrapper.chat(messages=messages)

        self.assertEqual(response, {"message": {"content": "Chat response content"}})

        self.mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=messages, max_tokens=4096
        )

    def test_chat_with_length_error(self):
        self.mock_client.chat.completions.create.side_effect = LengthFinishReasonError(
            completion=MagicMock()
        )

        messages = [{"role": "user", "content": "Test message"}]
        response = self.openai_wrapper.chat(messages=messages)

        self.assertEqual(
            response,
            {
                "error": "Response exceeded the maximum allowed length.",
                "content": None,
                "done": False,
            },
        )

    def test_chat_with_content_filter_error(self):
        self.mock_client.chat.completions.create.side_effect = (
            ContentFilterFinishReasonError()
        )

        messages = [{"role": "user", "content": "Test message"}]
        response = self.openai_wrapper.chat(messages=messages)

        self.assertEqual(
            response,
            {
                "error": "Content was rejected by the content filter.",
                "content": None,
                "done": False,
            },
        )

    def test_chat_with_response_schema(self):
        # Mock the beta chat completion with parsing
        mock_parsed_content = {"parsed": "data"}
        mock_message = MagicMock(parsed=mock_parsed_content)
        # Configure __contains__ to allow 'in' checks
        mock_message.__contains__.side_effect = lambda key: key in mock_message.__dict__

        mock_response = Mock(choices=[Mock(message=mock_message)])
        self.mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "Test message"}]
        response = self.openai_wrapper.chat(
            messages=messages, response_schema="DummySchema"
        )

        self.assertEqual(response, {"message": {"content": mock_parsed_content}})

    def test_chat_with_options(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Chat response with options"))]
        )

        messages = [{"role": "user", "content": "Test message"}]
        options = {"temperature": 0.7}
        response = self.openai_wrapper.chat(messages=messages, options=options)

        self.assertEqual(
            response, {"message": {"content": "Chat response with options"}}
        )

        self.mock_client.chat.completions.create.assert_called_once()
        args, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.7)

    def test_generate_exception_propagation(self):
        self.mock_client.chat.completions.create.side_effect = Exception(
            "Test exception"
        )

        with self.assertRaises(Exception) as context:
            self.openai_wrapper.generate(prompt="Test prompt")

        self.assertEqual(str(context.exception), "Test exception")

    def test_chat_exception_propagation(self):
        self.mock_client.chat.completions.create.side_effect = Exception(
            "Test exception"
        )

        with self.assertRaises(Exception) as context:
            self.openai_wrapper.chat(
                messages=[{"role": "user", "content": "Test message"}]
            )

        self.assertEqual(str(context.exception), "Test exception")


if __name__ == "__main__":
    unittest.main()