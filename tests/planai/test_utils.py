import random
import string
import unittest
from datetime import datetime

from planai.utils import dict_dump_xml


class TestDictDumpXml(unittest.TestCase):
    def test_basic_dict(self):
        test_dict = {"key": "value"}
        result = dict_dump_xml(test_dict)
        self.assertIn("<key>value</key>", result)

    def test_special_characters(self):
        test_dict = {
            "special": "§ß&<>\"'äöü",
            "emoji": "🌟🔥🌈",
            "control": "\n\t\r",
        }
        result = dict_dump_xml(test_dict)
        self.assertIsInstance(result, str)
        self.assertIn("§ß&amp;&lt;&gt;&quot;'äöü", result)
        self.assertIn("🌟🔥🌈", result)

    def test_nested_structure(self):
        test_dict = {
            "level1": {
                "level2": {"level3": "deep value"},
                "array": [1, 2, {"nested": "value"}],
            }
        }
        result = dict_dump_xml(test_dict)
        self.assertIn("<level1>", result)
        self.assertIn("<level2>", result)
        self.assertIn("<level3>deep value</level3>", result)

    def test_complex_types(self):
        test_dict = {
            "none": None,
            "number": float("inf"),
            "datetime": datetime(2024, 1, 1),
            "bytes": b"binary data",
            "set": {1, 2, 3},
        }
        result = dict_dump_xml(test_dict)
        self.assertIsInstance(result, str)
        self.assertIn("<none/>", result)
        self.assertIn("<number>inf</number>", result)
        self.assertIn("<datetime>2024-01-01", result)
        self.assertIn("<bytes>", result)
        self.assertIn("<set>", result)

    def test_custom_root(self):
        test_dict = {"key": "value"}
        result = dict_dump_xml(test_dict, root="custom")
        self.assertTrue(result.startswith("<custom>"))
        self.assertTrue(result.endswith("</custom>\n"))

    def test_empty_dict(self):
        test_dict = {}
        result = dict_dump_xml(test_dict)
        self.assertIn("<root/>", result.strip())

    def test_unicode_keys(self):
        test_dict = {"키": "값", "ключ": "значение"}
        result = dict_dump_xml(test_dict)
        self.assertIn("<키>값</키>", result)
        self.assertIn("<ключ>значение</ключ>", result)

    def test_random_bytes(self):
        test_dict = {"random": b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"}
        result = dict_dump_xml(test_dict)
        self.assertIn("<random>", result)
        self.assertIn("</random>", result)

    def _generate_random_string(self, length: int) -> str:
        """Generate a string with random characters including edge cases."""
        # Include normal chars, unicode, control chars, and potentially problematic chars
        chars = (
            string.printable
            + "".join(chr(i) for i in range(0x80, 0x110000, 997))  # sparse unicode
            + "\x00\x01\x02\x03\x1F"  # control chars
            + "<>\"'&"  # XML special chars
            + "🌟🔥🌈"  # emojis
        )
        return "".join(random.choice(chars) for _ in range(length))

    def _generate_random_dict(self, depth: int, max_items: int) -> dict:
        """Generate a nested dictionary with random content."""
        result = {}
        items = random.randint(1, max_items)
        for _ in range(items):
            key = self._generate_random_string(random.randint(1, 20))
            value_type = random.choice(["dict", "list", "string", "number", "none"])

            if value_type == "dict" and depth > 1:
                value = self._generate_random_dict(depth - 1, max_items)
            elif value_type == "list":
                value = [
                    self._generate_random_string(random.randint(1, 50))
                    for _ in range(random.randint(1, 5))
                ]
            elif value_type == "string":
                value = self._generate_random_string(random.randint(1, 100))
            elif value_type == "number":
                value = random.uniform(-1000, 1000)
            else:
                value = None

            result[key] = value
        return result

    def test_stress_test_random_dicts(self):
        """Stress test with randomly generated dictionaries."""
        test_cases = [
            (1, 5),  # Small shallow dict
            (3, 5),  # Medium depth dict
            (5, 3),  # Deep dict with fewer items
            (2, 10),  # Wide dict with many items
        ]

        for depth, max_items in test_cases:
            with self.subTest(depth=depth, max_items=max_items):
                random_dict = self._generate_random_dict(depth, max_items)
                try:
                    result = dict_dump_xml(random_dict)
                    # Basic validation of the result
                    self.assertIsInstance(result, str)
                    self.assertTrue(result.startswith("<root>"))
                    self.assertTrue(result.endswith("</root>\n"))
                except Exception as e:
                    self.fail(
                        f"Failed with depth={depth}, max_items={max_items}: {random_dict}: {str(e)}"
                    )


if __name__ == "__main__":
    unittest.main()
