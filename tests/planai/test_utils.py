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

    def test_invalid_input(self):
        test_cases = [
            ("Function value", {"func": lambda x: x}),
            ("Circular reference", {}),
        ]

        # Create circular reference
        d = test_cases[1][1]
        d["circular"] = d

        for case_name, test_dict in test_cases:
            with self.subTest(case_name=case_name):
                result = dict_dump_xml(test_dict)
                self.assertIsInstance(result, str)
                self.assertIn("<error>Failed to convert to XML</error>", result)

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


if __name__ == "__main__":
    unittest.main()
