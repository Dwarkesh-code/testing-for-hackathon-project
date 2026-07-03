"""
LangChain tools exposed to the Extractor react-agent (llm/extractor.py).

The extractor is pre-fed key_files from main.py, but for repos where the
pre-fetch missed something relevant, the agent can pull more on its own
using these two tools. Kept deliberately narrow (list + fetch) — no write
access, no arbitrary shell, nothing that can wander outside the target repo.
"""

import asyncio
from langchain_core.tools import tool
from github.fetcher import GitHubFetcher

_fetcher = GitHubFetcher()


def _run_async(coro):
    """
    Tools are called synchronously by create_react_agent (which itself runs
    inside asyncio.to_thread from main.py's analyze_repo_node — a plain OS
    thread, no running event loop). asyncio.run() is safe here.
    """
    return asyncio.run(coro)


@tool
def list_repo_files(owner: str, repo: str, branch: str = "main") -> str:
    """
    List file paths in a GitHub repository (up to 80 files, recursive).
    Use this when you need to see what's in the repo beyond the pre-fetched
    key files — e.g. to find a config file, a specific module, or tests.
    Returns a newline-separated list of file paths.
    """
    files = _run_async(_fetcher.get_file_tree(owner, repo, branch))
    if not files:
        return "No files found (repo may be empty or branch name incorrect)."
    return "\n".join(files)


@tool
def fetch_file_from_github(owner: str, repo: str, path: str) -> str:
    """
    Fetch the content of a specific file from a GitHub repository.
    Use this after list_repo_files to inspect a file's actual code.
    Returns up to the first 2500 characters of the file content.
    """
    content = _run_async(_fetcher.get_file(owner, repo, path))
    if not content:
        return f"Could not fetch '{path}' — file may not exist or is binary/empty."
    return content


GITHUB_TOOLS = [list_repo_files, fetch_file_from_github]
