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
    diff: str = Field(description="The complete git show output")


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
            self.publish_work(
                CommitDiff(
                    commit_hash=commit,
                    diff="",
                ),
                input_task=task,
            )


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

    def show_commit(self, commit_hash: str) -> str:
        """Generate git-show like output for a commit"""
        repo = Repo(self.repo_path)

        # Use git.show() directly which gives us the complete diff output
        try:
            # The -U option ensures we get the full unified diff
            # --no-prefix removes a/ and b/ prefixes from filenames
            show_output = repo.git.show(
                commit_hash,
                "--no-prefix",
                "-U10",  # Include 10 lines of context
                no_color=True,
            )
            return show_output
        except Exception as e:
            print(f"Error getting diff for {commit_hash}: {e}")
            return ""

    def consume_work(self, task: CommitDiff):
        diff_text = self.show_commit(task.commit_hash)

        self.publish_work(
            CommitDiff(
                commit_hash=task.commit_hash,
                diff=diff_text,
            ),
            input_task=task,
        )


class DiffAnalyzer(CachedLLMTaskWorker):
    """Worker that analyzes each diff and describes the changes"""

    output_types: List[Type[Task]] = [DiffAnalysis]
    llm_input_type: Type[Task] = CommitDiff
    use_xml: bool = False
    prompt: str = dedent(
        """
        Analyze the provided git diff and produce a 1-2 sentence description that clearly explains what was changed and why.
        Focus on the component affected and the primary purpose of the change, ensuring that the description is self-contained and easily understood in isolation.
        Do not include file statistics or non-functional details.
    """
    ).strip()

    def post_process(self, response: DiffAnalysis, input_task: Task):
        commit: CommitDiff = input_task.find_input_task(CommitDiff)
        if commit is None:
            raise ValueError("Input task not found")
        response.commit_hash = commit.commit_hash
        return super().post_process(response, input_task)


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
    debug_mode: bool = True
    prompt: str = dedent(
        """
        Generate clear and concise release notes suitable for GitHub. Start with a brief summary of the key changes that covers all major updates.
        Then, using GitHub-style markdown, organize the release notes by grouping changes into bullet-point lists under clearly defined categories such as Features, Bug Fixes, and Documentation.
        For each bullet point, provide a succinct, factual, and matter-of-fact description of the change and its significance without including commit hashes or hyperbolic language.
        Ensure that every change is accurately represented and, if relevant, note any impact or further implications concisely.
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
    graph.set_max_parallel_tasks(
        CachedLLMTaskWorker, 2 if args.provider == "ollama" else 4
    )

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
