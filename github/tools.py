"""
LangChain tools given to the NVIDIA NIM Extractor Agent.

The agent uses these tools to:
1. List all files in a repo → decide which ones to read
2. Fetch specific file content → read actual code

This is the "GitHub API call capability" inside each parallel agent node.
"""

import asyncio
import concurrent.futures
from typing import Annotated

from langchain_core.tools import tool
from github.fetcher import GitHubFetcher

_fetcher = GitHubFetcher()


def _run(coro):
    """Run an async coroutine from sync context (tools are called synchronously by the agent)."""
    try:
        loop = asyncio.get_running_loop()
        # We're inside an existing event loop — run in a thread
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=20)
    except RuntimeError:
        # No running loop — just run directly
        return asyncio.run(coro)


@tool
def list_repo_files(
    owner: Annotated[str, "GitHub username or org name"],
    repo:  Annotated[str, "Repository name"],
) -> str:
    """
    List all file paths inside a GitHub repository.
    Use this first to understand the repo structure before fetching specific files.
    """
    files = _run(_fetcher.get_file_tree(owner, repo))
    if not files:
        return f"Could not retrieve file tree for {owner}/{repo}."
    return "Files in repo:\n" + "\n".join(files)


@tool
def fetch_file_from_github(
    owner:    Annotated[str, "GitHub username or org name"],
    repo:     Annotated[str, "Repository name"],
    filepath: Annotated[str, "File path inside the repo, e.g. 'src/main.py' or 'package.json'"],
) -> str:
    """
    Fetch and return the content of a specific file from a GitHub repository.
    Use this after listing files to read actual code and understand what was implemented.
    """
    content = _run(_fetcher.get_file(owner, repo, filepath))
    if content:
        return f"=== {filepath} ===\n{content}"
    return f"File '{filepath}' not found or empty in {owner}/{repo}."


# Exported list for the agent
GITHUB_TOOLS = [list_repo_files, fetch_file_from_github]
