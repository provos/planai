# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""File system tools that let an LLM read, write, and search files inside a
sandboxed per-job working directory (a :class:`Workspace`).

These are meant to be handed to :class:`planai.llm_task.LLMTaskWorker` (or more
commonly :class:`planai.workspace_task.WorkspaceLLMTaskWorker`) as ``tools`` so
that many concurrent jobs, each with their own working directory, can safely let
an LLM operate on files without risking access outside of the assigned directory.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Union

from llm_interface import Tool
from llm_interface.llm_tool import create_tool

__all__ = ["Workspace", "make_file_tools", "hash_files"]


class Workspace:
    """A sandboxed working directory that file tools are jailed to.

    All paths handed to :meth:`resolve` are interpreted as relative to the
    workspace root and are rejected with a :class:`ValueError` if they would
    escape that root, whether through an absolute path, a ``..`` component, or
    a symlink that points outside of the root.
    """

    def __init__(self, root: Union[str, "Path"]):
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        # resolve() follows symlinks and normalizes the path so that every
        # subsequent comparison against self.root is done against the real path.
        self.root: Path = root_path.resolve()

    def resolve(self, rel_path: str) -> Path:
        """Resolve a workspace-relative path to an absolute path inside the root.

        Args:
            rel_path: A path relative to the workspace root.

        Returns:
            Path: The resolved absolute path, guaranteed to be inside the workspace root.

        Raises:
            ValueError: If the path is empty, absolute, contains a ``..`` component,
                or resolves (following symlinks) to a location outside the workspace root.
        """
        if not rel_path:
            raise ValueError("Path must not be empty")

        candidate = Path(rel_path)
        if candidate.is_absolute():
            raise ValueError(f"Absolute paths are not allowed: {rel_path!r}")
        if ".." in candidate.parts:
            raise ValueError(f"Path must not contain '..': {rel_path!r}")

        try:
            # strict=False resolves symlinks and normalizes the path as far as
            # possible even if the final component does not yet exist (e.g. a
            # file that write_file is about to create).
            resolved = (self.root / candidate).resolve(strict=False)
        except OSError as e:
            raise ValueError(f"Could not resolve path: {rel_path!r}") from e

        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path escapes the workspace root: {rel_path!r}") from None

        return resolved


def jailed_files(ws: Workspace, base: Path, pattern: str) -> List[Path]:
    """Files under ``base`` matching ``pattern`` whose real location is inside the
    workspace. Glob follows symlinks, so a link pointing outside the root would
    otherwise be readable; such entries are skipped."""
    files = []
    for path in base.glob(pattern):
        if not path.is_file():
            continue
        try:
            ws.resolve(path.relative_to(ws.root).as_posix())
        except ValueError:
            continue
        files.append(path)
    return sorted(files)


def hash_files(workspace: Union[Workspace, str, Path], globs: List[str]) -> str:
    """Compute a stable hash over the content of files matching the given globs.

    The hash is a sha1 digest over the sorted (relative path, content) pairs of every
    file matched by any of the glob patterns, so it changes whenever a matched file's
    content (or the set of matched files) changes, and is stable across runs otherwise.

    Args:
        workspace: The Workspace (or a path to one) whose files to hash.
        globs: A list of glob patterns (e.g. ``["**/*.py"]``), evaluated relative to
            the workspace root.

    Returns:
        str: A hex sha1 digest, or an empty string if no files match any pattern.
    """
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)

    matched = {}
    for pattern in globs:
        for path in jailed_files(ws, ws.root, pattern):
            rel = path.relative_to(ws.root).as_posix()
            matched[rel] = path

    if not matched:
        return ""

    digest = hashlib.sha1()
    for rel in sorted(matched.keys()):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(matched[rel].read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def make_file_tools(
    workspace: Union[Workspace, str, Path],
    *,
    read_only: bool = False,
    max_read_chars: int = 100_000,
    max_list_entries: int = 500,
    max_grep_matches: int = 200,
) -> List[Tool]:
    """Create llm_interface Tool objects bound to a single sandboxed workspace.

    Args:
        workspace: The Workspace to jail all file operations to (or a path to one,
            which will be created if missing).
        read_only: If True, omit the write_file and edit_file tools.
        max_read_chars: Maximum number of characters read_file returns before truncating.
        max_list_entries: Maximum number of entries list_files returns before truncating.
        max_grep_matches: Maximum number of matches grep_files returns before truncating.

    Returns:
        List[Tool]: Tool objects usable as the ``tools`` argument to an LLMTaskWorker.
    """
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)

    def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
        """Read the text content of a file in the workspace, with line numbers.

        Returns the file's content formatted like ``cat -n``, i.e. each line is
        prefixed with its 1-based line number. Use offset and limit to read a
        large file in smaller windows.

        Args:
            path: Workspace-relative path to the file to read.
            offset: 1-based line number to start reading from. 0 or 1 both mean
                start at the first line of the file.
            limit: Maximum number of lines to return. 0 means return every line
                from offset to the end of the file (subject to the character limit).
        """
        try:
            target = ws.resolve(path)
        except ValueError as e:
            return f"Error: {e}"

        if not target.exists():
            return f"Error: File not found: {path}"
        if not target.is_file():
            return f"Error: Not a file: {path}"

        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text (binary?): {path}"
        except OSError as e:
            return f"Error: Could not read file {path}: {e.strerror or e}"

        lines = text.splitlines()
        start = max(offset - 1, 0) if offset > 1 else 0
        end = start + limit if limit > 0 else len(lines)
        selected = lines[start:end]

        numbered = "\n".join(
            f"{start + i + 1:6d}\t{line}" for i, line in enumerate(selected)
        )

        if len(numbered) > max_read_chars:
            numbered = (
                numbered[:max_read_chars]
                + f"\n\n[Output truncated at {max_read_chars} characters. "
                "Use the offset and limit parameters to read this file in smaller windows.]"
            )
        return numbered

    def write_file(path: str, content: str) -> str:
        """Write text content to a file in the workspace, overwriting it if it exists.

        Parent directories are created automatically as needed.

        Args:
            path: Workspace-relative path to the file to write.
            content: The full text content to write to the file.
        """
        try:
            target = ws.resolve(path)
        except ValueError as e:
            return f"Error: {e}"

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return f"Error: Could not write file {path}: {e.strerror or e}"

        byte_count = len(content.encode("utf-8"))
        line_count = len(content.splitlines())
        return f"Wrote {byte_count} bytes ({line_count} lines) to {path}"

    def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace an exact snippet of text within an existing file.

        By default, old_string must occur exactly once in the file; use replace_all
        to replace every occurrence instead.

        Args:
            path: Workspace-relative path to the file to edit.
            old_string: The exact, literal text to find in the file. Must be unique
                in the file unless replace_all is true.
            new_string: The text to replace old_string with.
            replace_all: When true, replace every occurrence of old_string instead
                of requiring exactly one match.
        """
        try:
            target = ws.resolve(path)
        except ValueError as e:
            return f"Error: {e}"

        if not target.exists():
            return f"Error: File not found: {path}"
        if not target.is_file():
            return f"Error: Not a file: {path}"
        if not old_string:
            return "Error: old_string must not be empty"

        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text (binary?): {path}"
        except OSError as e:
            return f"Error: Could not read file {path}: {e.strerror or e}"

        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path} (0 matches)"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string is not unique in {path} ({count} matches); "
                "pass replace_all=true or include more surrounding context"
            )

        if replace_all:
            new_text = text.replace(old_string, new_string)
            replacements = count
        else:
            new_text = text.replace(old_string, new_string, 1)
            replacements = 1

        try:
            target.write_text(new_text, encoding="utf-8")
        except OSError as e:
            return f"Error: Could not write file {path}: {e.strerror or e}"

        return f"Replaced {replacements} occurrence(s) in {path}"

    def list_files(path: str = ".", pattern: str = "**/*") -> str:
        """List files under a directory in the workspace matching a glob pattern.

        Only files are listed (not directories), sorted by relative path, each
        shown with its size in bytes.

        Args:
            path: Workspace-relative directory to list. Use "." for the workspace root.
            pattern: Glob pattern, relative to path, selecting which files to include,
                e.g. "**/*" for every file recursively or "*.py" for top-level Python files.
        """
        try:
            target = ws.resolve(path)
        except ValueError as e:
            return f"Error: {e}"

        if not target.exists():
            return f"Error: Directory not found: {path}"
        if not target.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            matches = jailed_files(ws, target, pattern)
        except (re.error, ValueError) as e:
            return f"Error: Invalid pattern: {e}"

        truncated = len(matches) > max_list_entries
        lines = []
        for p in matches[:max_list_entries]:
            rel = p.relative_to(ws.root).as_posix()
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            lines.append(f"{size:>10}  {rel}")

        if not lines:
            return "(no files found)"

        output = "\n".join(lines)
        if truncated:
            output += f"\n\n[Output truncated at {max_list_entries} entries.]"
        return output

    def grep_files(pattern: str, path: str = ".", glob: str = "**/*.md") -> str:
        """Search for a regular expression across text files in the workspace.

        Args:
            pattern: Python regular expression to search for within each line of
                each matching file.
            path: Workspace-relative directory to search within. Use "." for the
                workspace root.
            glob: Glob pattern, relative to path, selecting which files to search,
                e.g. "**/*.md" for every Markdown file recursively.
        """
        try:
            target = ws.resolve(path)
        except ValueError as e:
            return f"Error: {e}"

        if not target.exists():
            return f"Error: Directory not found: {path}"
        if not target.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regular expression: {e}"

        try:
            candidates = jailed_files(ws, target, glob)
        except ValueError as e:
            return f"Error: Invalid pattern: {e}"

        results: List[str] = []
        truncated = False
        for file_path in candidates:
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            rel = file_path.relative_to(ws.root).as_posix()
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{lineno}: {line}")
                    if len(results) >= max_grep_matches:
                        truncated = True
                        break
            if truncated:
                break

        if not results:
            return "(no matches found)"

        output = "\n".join(results)
        if truncated:
            output += f"\n\n[Output truncated at {max_grep_matches} matches.]"
        return output

    tools = [create_tool(read_file)]
    if not read_only:
        tools.append(create_tool(write_file))
        tools.append(create_tool(edit_file))
    tools.append(create_tool(list_files))
    tools.append(create_tool(grep_files))
    return tools
