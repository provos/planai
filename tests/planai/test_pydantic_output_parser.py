import unittest
from pydantic import BaseModel
from planai.pydantic_output_parser import (
    MinimalPydanticOutputParser,
    OutputParserException,
)


class SimpleModel(BaseModel):
    name: str
    age: int


class ComplexModel(BaseModel):
    id: int
    data: SimpleModel
    tags: list[str]


class TestMinimalPydanticOutputParser(unittest.TestCase):
    def setUp(self):
        self.simple_parser = MinimalPydanticOutputParser(SimpleModel)
        self.complex_parser = MinimalPydanticOutputParser(ComplexModel)

    def test_parse_valid_simple_json(self):
        valid_json = '{"name": "Alice", "age": 30}'
        result = self.simple_parser.parse(valid_json)
        self.assertIsInstance(result, SimpleModel)
        self.assertEqual(result.name, "Alice")
        self.assertEqual(result.age, 30)

    def test_parse_valid_markdown_json(self):
        markdown_json = """```json
{
  "name": "Bob",
  "age": 25
}
```"""
        result = self.simple_parser.parse(markdown_json)
        self.assertIsInstance(result, SimpleModel)
        self.assertEqual(result.name, "Bob")
        self.assertEqual(result.age, 25)

    def test_parse_valid_complex_json(self):
        valid_json = (
            '{"id": 1, "data": {"name": "Bob", "age": 25}, "tags": ["user", "admin"]}'
        )
        result = self.complex_parser.parse(valid_json)
        self.assertIsInstance(result, ComplexModel)
        self.assertEqual(result.id, 1)
        self.assertEqual(result.data.name, "Bob")
        self.assertEqual(result.data.age, 25)
        self.assertEqual(result.tags, ["user", "admin"])

    def test_parse_invalid_json(self):
        invalid_json = '{"name": "Charlie", "age": }'
        with self.assertRaises(OutputParserException) as context:
            self.simple_parser.parse(invalid_json)
        self.assertIn("Invalid JSON", str(context.exception))

    def test_parse_mismatched_json(self):
        mismatched_json = '{"name": "David", "height": 180}'
        with self.assertRaises(OutputParserException) as context:
            self.simple_parser.parse(mismatched_json)
        self.assertIn("Failed to parse SimpleModel", str(context.exception))

    def test_parse_invalid_types(self):
        invalid_types_json = '{"name": "Eve", "age": "thirty"}'
        with self.assertRaises(OutputParserException) as context:
            self.simple_parser.parse(invalid_types_json)
        self.assertIn("Failed to parse SimpleModel", str(context.exception))

    def test_parse_extra_fields(self):
        extra_fields_json = '{"name": "Frank", "age": 40, "city": "New York"}'
        result = self.simple_parser.parse(extra_fields_json)
        self.assertIsInstance(result, SimpleModel)
        self.assertEqual(result.name, "Frank")
        self.assertEqual(result.age, 40)
        with self.assertRaises(AttributeError):
            result.city

    def test_parse_missing_fields(self):
        missing_fields_json = '{"name": "Grace"}'
        with self.assertRaises(OutputParserException) as context:
            self.simple_parser.parse(missing_fields_json)
        self.assertIn("Failed to parse SimpleModel", str(context.exception))


if __name__ == "__main__":
    unittest.main()
