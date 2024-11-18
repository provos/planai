import unittest
from typing import List, Optional
from planai.llm_tool import Tool, create_tool, tool


class TestLLMTool(unittest.TestCase):
    def test_basic_tool_creation(self):
        """Test basic Tool class instantiation and execution."""

        def dummy_func(x: int, y: int) -> int:
            return x + y

        tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
            },
            func=dummy_func,
        )

        result = tool.execute(x=1, y=2)
        self.assertEqual(result, 3)
        self.assertEqual(tool.name, "add")
        self.assertEqual(tool.description, "Add two numbers")

    def test_create_tool_from_function(self):
        """Test creating a Tool instance from a function using create_tool."""

        def get_weather(location: str, units: str = "celsius") -> str:
            """Get the weather for a specific location.

            Args:
                location: The city or location to get weather for
                units: Temperature units (celsius or fahrenheit)
            """
            return f"Weather in {location} in {units}"

        weather_tool = create_tool(get_weather)

        # Test tool metadata
        self.assertEqual(weather_tool.name, "get_weather")
        self.assertEqual(
            weather_tool.description, "Get the weather for a specific location."
        )

        # Test parameters schema
        self.assertEqual(weather_tool.parameters["type"], "object")
        self.assertIn("location", weather_tool.parameters["properties"])
        self.assertIn("units", weather_tool.parameters["properties"])
        self.assertEqual(weather_tool.parameters["required"], ["location"])

        # Test parameter descriptions
        self.assertEqual(
            weather_tool.parameters["properties"]["location"]["description"],
            "The city or location to get weather for",
        )

        # Test execution
        result = weather_tool.execute(location="Paris", units="fahrenheit")
        self.assertEqual(result, "Weather in Paris in fahrenheit")

    def test_tool_decorator(self):
        """Test the @tool decorator."""

        @tool(name="multiply", description="Multiply two numbers")
        def multiply(x: float, y: float = 1.0) -> float:
            """Multiply two floating point numbers.

            Args:
                x: First number
                y: Second number (default: 1.0)
            """
            return x * y

        # Test tool metadata
        self.assertEqual(multiply.name, "multiply")
        self.assertEqual(multiply.description, "Multiply two numbers")

        # Test parameters
        self.assertEqual(multiply.parameters["type"], "object")
        self.assertIn("x", multiply.parameters["properties"])
        self.assertIn("y", multiply.parameters["properties"])
        self.assertEqual(multiply.parameters["required"], ["x"])

        # Test execution
        result = multiply.execute(x=2.0, y=3.0)
        self.assertEqual(result, 6.0)

    def test_complex_types(self):
        """Test tool creation with complex type hints."""

        def process_data(
            items: List[str], filter_value: Optional[int] = None
        ) -> List[str]:
            """Process a list of items with optional filtering.

            Args:
                items: List of strings to process
                filter_value: Optional value to filter items
            """
            return items

        data_tool = create_tool(process_data)

        # Test parameters schema
        self.assertEqual(data_tool.parameters["properties"]["items"]["type"], "array")
        self.assertEqual(
            data_tool.parameters["properties"]["items"]["items"]["type"], "string"
        )
        self.assertNotIn("filter_value", data_tool.parameters["required"])

        # Test execution
        result = data_tool.execute(items=["a", "b", "c"])
        self.assertEqual(result, ["a", "b", "c"])

    def test_docstring_parsing(self):
        """Test proper parsing of docstrings for descriptions."""

        def complex_function(
            param1: int, param2: str, param3: Optional[float] = None
        ) -> dict:
            """This is a complex function that does something.

            This is a longer description that spans
            multiple lines and should be ignored.

            Args:
                param1: First parameter description
                    with multiple lines
                param2: Second parameter description
                param3: Optional third parameter
                    also with multiple lines
            """
            return {"result": param1}

        complex_tool = create_tool(complex_function)

        # Test main description
        self.assertEqual(
            complex_tool.description, "This is a complex function that does something."
        )

        # Test parameter descriptions
        params = complex_tool.parameters["properties"]
        self.assertEqual(
            params["param1"]["description"],
            "First parameter description with multiple lines",
        )
        self.assertEqual(
            params["param2"]["description"], "Second parameter description"
        )
        self.assertEqual(
            params["param3"]["description"],
            "Optional third parameter also with multiple lines",
        )

    def test_custom_name_and_description(self):
        """Test overriding name and description."""

        def simple_func(x: int) -> int:
            """Original description."""
            return x

        # Test with create_tool
        tool1 = create_tool(
            simple_func, name="custom_name", description="custom description"
        )
        self.assertEqual(tool1.name, "custom_name")
        self.assertEqual(tool1.description, "custom description")

        # Test with decorator
        @tool(name="decorated_name", description="decorated description")
        def decorated_func(x: int) -> int:
            """Original description."""
            return x

        self.assertEqual(decorated_func.name, "decorated_name")
        self.assertEqual(decorated_func.description, "decorated description")

    def test_invalid_function(self):
        """Test handling of functions without type hints."""

        def no_types(x, y):
            """Function without type hints."""
            return x + y

        tool = create_tool(no_types)

        # Should default to string type when no type hints are provided
        self.assertEqual(tool.parameters["properties"]["x"]["type"], "string")
        self.assertEqual(tool.parameters["properties"]["y"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
