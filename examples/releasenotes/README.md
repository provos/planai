## Release Notes Generator

This example demonstrates how to use PlanAI to automatically generate comprehensive release notes from Git commit history. The application processes commits between specified Git tags, analyzes the changes, and produces well-structured release notes in GitHub-style markdown format.

### Key Components

- **CommitCollector**: Retrieves commit hashes between a specified tag and HEAD
- **DiffWorker**: Extracts detailed diffs for each commit
- **DiffAnalyzer**: Uses LLM to analyze and describe the changes in each commit
- **ChangeCollector**: Aggregates all analyzed changes
- **ReleaseNotesGenerator**: Creates structured release notes with categorized changes

### Features

1. Processes all commits from a specified tag to HEAD
2. Generates detailed descriptions of changes using AI
3. Groups changes into logical categories (e.g., Features, Bug Fixes)
4. Outputs GitHub-compatible markdown
5. Supports different LLM providers (OpenAI, Ollama)

### Usage

Run the application with:

```
poetry run python releasenotes.py --provider ollama --model phi4:latest --repo {your-repo} --tag {last-release-tag}
```