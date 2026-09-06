# test_filesystem.py

import tempfile
import unittest
from pathlib import Path

from planai.tools.filesystem import Workspace, hash_files, make_file_tools


class TestWorkspaceJail(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.workspace = Workspace(self.tempdir.name)

    def test_root_is_created_and_resolved(self):
        self.assertTrue(self.workspace.root.exists())
        self.assertTrue(self.workspace.root.is_absolute())

    def test_nested_path_accepted(self):
        resolved = self.workspace.resolve("a/b/c.txt")
        self.assertEqual(resolved, self.workspace.root / "a" / "b" / "c.txt")

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            self.workspace.resolve("/etc/passwd")

    def test_empty_path_rejected(self):
        with self.assertRaises(ValueError):
            self.workspace.resolve("")

    def test_dotdot_escape_rejected(self):
        with self.assertRaises(ValueError):
            self.workspace.resolve("../escape.txt")
        with self.assertRaises(ValueError):
            self.workspace.resolve("sub/../../escape.txt")

    def test_symlink_escape_rejected(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        link = Path(self.tempdir.name) / "evil"
        link.symlink_to(outside.name)
        with self.assertRaises(ValueError):
            self.workspace.resolve("evil/secret.txt")

    def test_symlink_within_root_accepted(self):
        target_dir = Path(self.tempdir.name) / "real"
        target_dir.mkdir()
        link = Path(self.tempdir.name) / "link"
        link.symlink_to(target_dir)
        resolved = self.workspace.resolve("link/file.txt")
        self.assertEqual(resolved, target_dir.resolve() / "file.txt")

    def test_parent_dirs_created_on_write(self):
        tools = {t.name: t for t in make_file_tools(self.workspace)}
        result = tools["write_file"].execute(path="a/b/c.txt", content="hi")
        self.assertNotIn("Error", result)
        self.assertTrue((Path(self.tempdir.name) / "a" / "b" / "c.txt").exists())

    def test_error_messages_do_not_leak_host_path(self):
        with self.assertRaises(ValueError) as ctx:
            self.workspace.resolve("/etc/passwd")
        self.assertNotIn(self.tempdir.name, str(ctx.exception))


class FileToolsTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.workspace = Workspace(self.tempdir.name)
        self.tools = {t.name: t for t in make_file_tools(self.workspace)}

    def write(self, rel_path: str, content: str):
        full = Path(self.tempdir.name) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return full


class TestReadFile(FileToolsTestCase):
    def test_line_numbering(self):
        self.write("file.txt", "one\ntwo\nthree\n")
        result = self.tools["read_file"].execute(path="file.txt")
        lines = result.splitlines()
        self.assertEqual(lines[0], "     1\tone")
        self.assertEqual(lines[1], "     2\ttwo")
        self.assertEqual(lines[2], "     3\tthree")

    def test_offset_and_limit_window(self):
        self.write("file.txt", "\n".join(f"line{i}" for i in range(1, 11)) + "\n")
        result = self.tools["read_file"].execute(path="file.txt", offset=3, limit=2)
        lines = result.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("line3", lines[0])
        self.assertIn("line4", lines[1])
        self.assertTrue(lines[0].startswith("     3\t"))

    def test_truncation_note(self):
        self.write("big.txt", "x" * 1000)
        result = self.tools["read_file"].execute(path="big.txt", offset=0, limit=0)
        # use a small max_read_chars via a fresh tool set
        tools = {t.name: t for t in make_file_tools(self.workspace, max_read_chars=50)}
        truncated = tools["read_file"].execute(path="big.txt")
        self.assertLess(len(result), len(truncated) + 10000)  # sanity: result exists
        self.assertIn("truncated", truncated.lower())
        self.assertIn("offset", truncated.lower())

    def test_missing_file_error(self):
        result = self.tools["read_file"].execute(path="nope.txt")
        self.assertTrue(result.startswith("Error:"))

    def test_binary_file_error(self):
        full = Path(self.tempdir.name) / "binary.bin"
        full.write_bytes(bytes([0xFF, 0xFE, 0x00, 0x80, 0x81]))
        result = self.tools["read_file"].execute(path="binary.bin")
        self.assertTrue(result.startswith("Error:"))

    def test_jail_error_propagated(self):
        result = self.tools["read_file"].execute(path="../outside.txt")
        self.assertTrue(result.startswith("Error:"))


class TestWriteFile(FileToolsTestCase):
    def test_write_and_confirm(self):
        result = self.tools["write_file"].execute(path="out.txt", content="a\nb\n")
        self.assertIn("2", result)
        full = Path(self.tempdir.name) / "out.txt"
        self.assertEqual(full.read_text(), "a\nb\n")

    def test_write_overwrites(self):
        self.write("out.txt", "old content")
        self.tools["write_file"].execute(path="out.txt", content="new")
        full = Path(self.tempdir.name) / "out.txt"
        self.assertEqual(full.read_text(), "new")

    def test_omitted_when_read_only(self):
        tools = {t.name: t for t in make_file_tools(self.workspace, read_only=True)}
        self.assertNotIn("write_file", tools)
        self.assertNotIn("edit_file", tools)
        self.assertIn("read_file", tools)
        self.assertIn("list_files", tools)
        self.assertIn("grep_files", tools)


class TestEditFile(FileToolsTestCase):
    def test_edit_unique_match(self):
        self.write("file.txt", "hello world")
        result = self.tools["edit_file"].execute(
            path="file.txt", old_string="world", new_string="there"
        )
        self.assertIn("1", result)
        full = Path(self.tempdir.name) / "file.txt"
        self.assertEqual(full.read_text(), "hello there")

    def test_edit_zero_matches_error(self):
        self.write("file.txt", "hello world")
        result = self.tools["edit_file"].execute(
            path="file.txt", old_string="missing", new_string="x"
        )
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("0", result)

    def test_edit_multiple_matches_error(self):
        self.write("file.txt", "foo foo foo")
        result = self.tools["edit_file"].execute(
            path="file.txt", old_string="foo", new_string="bar"
        )
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("3", result)

    def test_edit_replace_all(self):
        self.write("file.txt", "foo foo foo")
        result = self.tools["edit_file"].execute(
            path="file.txt",
            old_string="foo",
            new_string="bar",
            replace_all=True,
        )
        self.assertIn("3", result)
        full = Path(self.tempdir.name) / "file.txt"
        self.assertEqual(full.read_text(), "bar bar bar")

    def test_omitted_when_read_only(self):
        tools = {t.name: t for t in make_file_tools(self.workspace, read_only=True)}
        self.assertNotIn("edit_file", tools)


class TestListFiles(FileToolsTestCase):
    def test_format_and_sorting(self):
        self.write("b.txt", "22")
        self.write("a.txt", "1")
        self.write("sub/c.txt", "333")
        result = self.tools["list_files"].execute()
        lines = result.splitlines()
        rel_paths = [line.split(None, 1)[1] for line in lines]
        self.assertEqual(rel_paths, sorted(rel_paths))
        self.assertIn("a.txt", rel_paths)
        self.assertIn("sub/c.txt", rel_paths)

    def test_pattern_filters(self):
        self.write("a.py", "x")
        self.write("b.txt", "x")
        result = self.tools["list_files"].execute(pattern="*.py")
        self.assertIn("a.py", result)
        self.assertNotIn("b.txt", result)

    def test_cap(self):
        for i in range(10):
            self.write(f"file{i}.txt", "x")
        tools = {t.name: t for t in make_file_tools(self.workspace, max_list_entries=3)}
        result = tools["list_files"].execute()
        lines = [line for line in result.splitlines() if line.strip()]
        self.assertIn("truncated", result.lower())
        # 3 file lines + note line(s)
        file_lines = [line for line in lines if not line.startswith("[")]
        self.assertEqual(len(file_lines), 3)

    def test_missing_directory_error(self):
        result = self.tools["list_files"].execute(path="nope")
        self.assertTrue(result.startswith("Error:"))

    def test_empty_directory(self):
        result = self.tools["list_files"].execute()
        self.assertEqual(result, "(no files found)")


class TestGrepFiles(FileToolsTestCase):
    def test_format(self):
        self.write("a.md", "hello world\nfoo bar\n")
        result = self.tools["grep_files"].execute(pattern="foo")
        self.assertEqual(result, "a.md:2: foo bar")

    def test_glob_filters_files(self):
        self.write("a.md", "needle\n")
        self.write("b.txt", "needle\n")
        result = self.tools["grep_files"].execute(pattern="needle")
        self.assertIn("a.md", result)
        self.assertNotIn("b.txt", result)

    def test_invalid_regex(self):
        result = self.tools["grep_files"].execute(pattern="(unclosed")
        self.assertTrue(result.startswith("Error:"))

    def test_no_matches(self):
        self.write("a.md", "hello\n")
        result = self.tools["grep_files"].execute(pattern="zzz")
        self.assertEqual(result, "(no matches found)")

    def test_cap(self):
        self.write("a.md", "\n".join("match" for _ in range(20)))
        tools = {t.name: t for t in make_file_tools(self.workspace, max_grep_matches=5)}
        result = tools["grep_files"].execute(pattern="match")
        self.assertIn("truncated", result.lower())
        match_lines = [line for line in result.splitlines() if line.startswith("a.md:")]
        self.assertEqual(len(match_lines), 5)


class TestHashFiles(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.workspace = Workspace(self.tempdir.name)

    def write(self, rel_path, content):
        full = Path(self.tempdir.name) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return full

    def test_empty_when_no_match(self):
        self.assertEqual(hash_files(self.workspace, ["**/*.md"]), "")

    def test_stable_across_runs(self):
        self.write("a.md", "content")
        h1 = hash_files(self.workspace, ["**/*.md"])
        h2 = hash_files(self.workspace, ["**/*.md"])
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, "")

    def test_changes_with_content(self):
        self.write("a.md", "content")
        h1 = hash_files(self.workspace, ["**/*.md"])
        self.write("a.md", "different content")
        h2 = hash_files(self.workspace, ["**/*.md"])
        self.assertNotEqual(h1, h2)

    def test_changes_with_new_file(self):
        self.write("a.md", "content")
        h1 = hash_files(self.workspace, ["**/*.md"])
        self.write("b.md", "more content")
        h2 = hash_files(self.workspace, ["**/*.md"])
        self.assertNotEqual(h1, h2)


class TestToolSchemas(unittest.TestCase):
    def test_schemas_have_descriptions_for_all_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Workspace(tmpdir)
            tools = make_file_tools(workspace)
            self.assertGreaterEqual(len(tools), 5)
            for t in tools:
                schema = t.to_dict()
                fn = schema["function"]
                self.assertTrue(fn["name"])
                self.assertTrue(fn["description"])
                params = fn["parameters"]
                self.assertEqual(params["type"], "object")
                for pname, pschema in params["properties"].items():
                    self.assertIn(
                        "description",
                        pschema,
                        f"Parameter {pname} of tool {fn['name']} missing description",
                    )
                    self.assertTrue(pschema["description"])


if __name__ == "__main__":
    unittest.main()


class TestSymlinkEscapeInWalks(unittest.TestCase):
    """list_files, grep_files and hash_files must not follow symlinks out of the jail."""

    def setUp(self):
        import tempfile

        self.outside_dir = tempfile.TemporaryDirectory()
        self.root_dir = tempfile.TemporaryDirectory()
        outside = Path(self.outside_dir.name) / "secret.md"
        outside.write_text("top secret needle")
        root = Path(self.root_dir.name)
        (root / "inside.md").write_text("inside needle")
        (root / "link.md").symlink_to(outside)
        (root / "linkdir").symlink_to(Path(self.outside_dir.name))
        self.ws = Workspace(root)

    def tearDown(self):
        self.outside_dir.cleanup()
        self.root_dir.cleanup()

    def _tool(self, name):
        return next(t for t in make_file_tools(self.ws) if t.name == name)

    def test_list_skips_symlinked_files_and_directories(self):
        listing = self._tool("list_files").execute(path=".", pattern="**/*")
        self.assertIn("inside.md", listing)
        self.assertNotIn("link.md", listing)
        self.assertNotIn("secret.md", listing)

    def test_grep_skips_symlinked_content(self):
        hits = self._tool("grep_files").execute(
            pattern="needle", path=".", glob="**/*.md"
        )
        self.assertIn("inside.md", hits)
        self.assertNotIn("top secret", hits)
        self.assertNotIn("link.md", hits)

    def test_hash_ignores_symlinked_content(self):
        before = hash_files(self.ws, ["**/*.md"])
        (Path(self.outside_dir.name) / "secret.md").write_text("changed outside")
        self.assertEqual(before, hash_files(self.ws, ["**/*.md"]))
        (Path(self.root_dir.name) / "inside.md").write_text("changed inside")
        self.assertNotEqual(before, hash_files(self.ws, ["**/*.md"]))
