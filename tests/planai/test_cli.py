import unittest

from planai.cli import create_parser, parse_comma_separated_list


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()

    def test_base_arguments(self):
        args = self.parser.parse_args(
            ["--llm-provider", "openai", "--llm-model", "gpt-4"]
        )
        self.assertEqual(args.llm_provider, "openai")
        self.assertEqual(args.llm_model, "gpt-4")
        self.assertIsNone(args.command)

    def test_optimize_prompt_command(self):
        args = self.parser.parse_args(
            [
                "--llm-provider",
                "openai",
                "optimize-prompt",
                "--python-file",
                "test.py",
                "--class-name",
                "TestClass",
                "--debug-log",
                "debug.json",
                "--goal-prompt",
                "test prompt",
                "--num-iterations",
                "5",
            ]
        )
        self.assertEqual(args.command, "optimize-prompt")
        self.assertEqual(args.python_file, "test.py")
        self.assertEqual(args.class_name, "TestClass")
        self.assertEqual(args.debug_log, "debug.json")
        self.assertEqual(args.goal_prompt, "test prompt")
        self.assertEqual(args.num_iterations, 5)

    def test_cache_command(self):
        args = self.parser.parse_args(
            [
                "cache",
                "cache_dir",
                "--output-task-filter",
                "test",
                "--search-dirs",
                "dir1,dir2,dir3",
            ]
        )
        self.assertEqual(args.command, "cache")
        self.assertEqual(args.cache_dir, "cache_dir")
        self.assertEqual(args.output_task_filter, "test")
        self.assertEqual(args.search_dirs, ["dir1", "dir2", "dir3"])

    def test_parse_comma_separated_list(self):
        result = parse_comma_separated_list("a,b,c")
        self.assertEqual(result, ["a", "b", "c"])

        result = parse_comma_separated_list(" a , b , c ")
        self.assertEqual(result, ["a", "b", "c"])

    def test_invalid_arguments(self):
        # Test with unknown argument
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--not-a-real-argument"])

        # Test with invalid subcommand
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["not-a-real-command"])

        # Test cache command without required positional argument
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["cache"])


if __name__ == "__main__":
    unittest.main()
