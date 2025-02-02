import argparse
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Type

from git import Repo
from pydantic import Field, PrivateAttr

from planai import (
    CachedLLMTaskWorker,
    CachedTaskWorker,
    Graph,
    InitialTaskWorker,
    JoinedTaskWorker,
    Task,
    TaskWorker,
    llm_from_config,
)
from planai.utils import setup_logging


class InitialCommit(Task):
    """Task containing the initial repository information"""

    repo_path: str = Field(description="Path to the git repository")
    from_tag: Optional[str] = Field(None, description="Starting tag for commit range")


class CommitDiff(Task):
    """Task containing the diff for a commit"""

    commit_hash: str = Field(description="The commit hash")
    diff: str = Field(description="The git diff output")


class DiffAnalysis(Task):
    """Task containing the analysis of a diff"""

    commit_hash: str = Field(description="The commit hash")
    description: str = Field(description="Analysis of what the changes accomplished")


class ChangeCollection(Task):
    """Task containing all analyzed changes"""

    changes: List[DiffAnalysis] = Field(
        description="Collection of all analyzed changes"
    )


class ReleaseNotes(Task):
    """Task containing the final release notes"""

    notes: str = Field(description="The generated release notes in markdown format")


class CommitCollector(TaskWorker):
    """Worker that collects all commits from a repository"""

    output_types: List[Type[Task]] = [CommitDiff]

    def consume_work(self, task: InitialCommit):
        commits = get_commits(task.repo_path, task.from_tag)
        for commit in commits:
            self.publish_work(CommitDiff(commit_hash=commit, diff=""), input_task=task)


class DiffWorker(CachedTaskWorker):
    """Worker that gets the diff for each commit"""

    output_types: List[Type[Task]] = [CommitDiff]
    repo_path: str = Field(description="Path to the git repository")
    _empty_tree_hash: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        # Generate the empty tree hash using git
        repo = Repo(self.repo_path)
        self._empty_tree_hash = repo.git.hash_object("-t", "tree", "/dev/null")

    def consume_work(self, task: CommitDiff):
        repo = Repo(self.repo_path)
        commit = repo.commit(task.commit_hash)
        # Get the parent commit to generate the diff
        parent = commit.parents[0] if commit.parents else None
        if parent:
            diff = commit.diff(parent, create_patch=True)
        else:
            # For initial commit with no parent, use the generated empty tree hash
            diff = commit.diff(self._empty_tree_hash, create_patch=True)

        # Convert the diff to a string representation
        diff_text = "\n".join(d.diff.decode("utf-8") for d in diff)

        self.publish_work(
            CommitDiff(commit_hash=task.commit_hash, diff=diff_text), input_task=task
        )


class DiffAnalyzer(CachedLLMTaskWorker):
    """Worker that analyzes each diff and describes the changes"""

    output_types: List[Type[Task]] = [DiffAnalysis]
    llm_input_type: Type[Task] = CommitDiff
    use_xml: bool = True
    prompt: str = dedent(
        """
        Analyze the provided git diff and describe what the changes accomplished.
        Focus on the functional changes and their purpose, not on technical details.
        Be concise but comprehensive.
        Ignore files changed or statistics.

        The description should be 1-2 sentences that clearly explain what was changed and why.

        Provide your response in JSON format:
        {
            "description": "A clear description of what the changes accomplished"
        }
    """
    ).strip()


class ChangeCollector(JoinedTaskWorker):
    """Worker that collects all analyzed changes"""

    join_type: Type[TaskWorker] = InitialTaskWorker

    output_types: List[Type[Task]] = [ChangeCollection]

    def consume_work_joined(self, tasks: List[DiffAnalysis]):
        self.publish_work(ChangeCollection(changes=tasks), input_task=tasks[0])


class ReleaseNotesGenerator(CachedLLMTaskWorker):
    """Worker that generates the final release notes"""

    output_types: List[Type[Task]] = [ReleaseNotes]
    llm_input_type: Type[Task] = ChangeCollection
    use_xml: bool = True
    prompt: str = dedent(
        """
        Generate comprehensive release notes from the provided changes.
        Group the changes into relevant categories (e.g., Features, Bug Fixes, Documentation).
        Use GitHub-style markdown formatting.

        Follow these guidelines:
        1. Start with a brief summary of the key changes
        2. Group changes into logical categories
        3. Use bullet points for each change
        4. Format as proper markdown

        Provide your response directly in markdown format suitable for GitHub.
    """
    ).strip()


def get_commits(repo_path: str, from_tag: Optional[str]) -> List[str]:
    """Get all commit hashes from the given tag to HEAD"""
    repo = Repo(repo_path)
    list_arg = f"{from_tag}..HEAD" if from_tag else "HEAD"
    commits = repo.git.rev_list(list_arg).split("\n")
    return [c.strip() for c in commits if c.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Generate release notes from git commits"
    )
    parser.add_argument(
        "--tag",
        help="Starting tag for release notes (e.g., v0.2.0)",
        required=False,
        default=None,
    )
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4", help="LLM model name")
    parser.add_argument("--repo", default=".", help="Path to git repository")
    parser.add_argument("--ollama-host", default="localhost:11434", help="Ollama host")
    args = parser.parse_args()

    # Initialize graph and workers
    graph = Graph(name="Release Notes Generator")
    llm = llm_from_config(
        provider=args.provider, model_name=args.model, host=args.ollama_host
    )

    commit_collector = CommitCollector()
    diff_worker = DiffWorker(repo_path=args.repo)
    analyzer = DiffAnalyzer(llm=llm)
    collector = ChangeCollector()
    generator = ReleaseNotesGenerator(llm=llm)

    # Set up the processing pipeline
    graph.add_workers(commit_collector, diff_worker, analyzer, collector, generator)
    graph.set_dependency(commit_collector, diff_worker).next(analyzer).next(
        collector
    ).next(generator)
    graph.set_sink(generator, ReleaseNotes)
    graph.set_max_parallel_tasks(CachedLLMTaskWorker, 2)

    # Create initial task with repository information
    initial_task = [
        (commit_collector, InitialCommit(repo_path=args.repo, from_tag=args.tag))
    ]

    setup_logging()

    # Run the graph
    graph.run(initial_tasks=initial_task, run_dashboard=False, display_terminal=True)

    # Get the release notes from the final task
    notes_task: ReleaseNotes = graph.get_output_tasks()[0]

    # Write release notes to file
    output_path = Path("RELEASE_NOTES.md")
    output_path.write_text(notes_task.notes)
    print(f"Release notes written to {output_path}")


if __name__ == "__main__":
    main()
