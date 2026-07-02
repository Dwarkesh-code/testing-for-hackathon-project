"""
main.py — LangGraph Orchestrator for the GitHub Proof-of-Work Pipeline

Flow:
  fetch_repos → filter_repos → fetch_readmes → router
                                                  ↓
                                    [fan-out via Send]
                               ┌──────┬──────┬──────┐
                           analyze  analyze  analyze  ... (one per top repo, all parallel)
                               └──────┴──────┴──────┘
                                          ↓
                                    END (results accumulated)

Run:  python main.py <github_username>
"""

import asyncio
import json
import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from github.fetcher import GitHubFetcher
from github.filters import filter_repos, filter_commits
from llm.router import select_top_repos
from llm.extractor import extract_skills
from llm.vision import analyze_best_screenshot
from llm.synthesizer import synthesize_portfolio
from config import config

fetcher = GitHubFetcher()


# ── LangGraph State Definitions ────────────────────────────────────────────

class PipelineState(TypedDict):
    """Main graph state — flows through the sequential nodes before fan-out."""
    username:          str
    leetcode:          Optional[str]
    linkedin:          Optional[str]
    credly:            Optional[str]
    raw_repos:         List[dict]   # all repos from GitHub API
    filtered_repos:    List[dict]   # after Python rule filter
    repos_with_readmes: List[dict]  # filtered_repos + readme_snippet added
    top_repos:         List[dict]   # selected by Router LLM
    repo_results:      Annotated[List[dict], operator.add]  # accumulated from parallel nodes
    final_portfolio:   dict


class RepoState(TypedDict):
    """State passed to each parallel analyze_repo node."""
    username:    str
    repo:        dict   # single repo dict (metadata + readme_snippet)
    chain_index: int    # index used for model round-robin rotation


# ── Node: Fetch all repos from GitHub ──────────────────────────────────────

async def fetch_repos_node(state: PipelineState) -> dict:
    print(f"\n[Step 1] Fetching repos for: {state['username']}")
    raw = await fetcher.get_repos(state["username"])
    print(f"[Step 1] Found {len(raw)} public repos")
    return {"raw_repos": raw}


# ── Node: Python filter (no LLM) ───────────────────────────────────────────

async def filter_repos_node(state: PipelineState) -> dict:
    print(f"\n[Step 2] Applying Python filter (remove forks, empty repos)...")
    filtered = filter_repos(state["raw_repos"])
    print(f"[Step 2] {len(filtered)} repos kept as candidates")
    return {"filtered_repos": filtered}


# ── Node: Fetch READMEs for all candidates (async parallel fetch) ───────────

async def fetch_readmes_node(state: PipelineState) -> dict:
    repos = state["filtered_repos"]
    print(f"\n[Step 3] Fetching READMEs for {len(repos)} candidates...")
    tasks = [fetcher.get_readme(state["username"], r["name"]) for r in repos]
    readmes = await asyncio.gather(*tasks)
    repos_with_readmes = [
        {**repo, "readme_snippet": readme}
        for repo, readme in zip(repos, readmes)
    ]
    return {"repos_with_readmes": repos_with_readmes}


# ── Node: Router LLM (Groq — fast) ─────────────────────────────────────────

async def router_node(state: PipelineState) -> dict:
    print(f"\n[Step 4] Router LLM (Groq) picking top repos from {len(state['repos_with_readmes'])} candidates...")
    top_names = await asyncio.to_thread(select_top_repos, state["repos_with_readmes"])
    name_set  = set(top_names)
    top_repos = [r for r in state["repos_with_readmes"] if r["name"] in name_set]
    print(f"[Step 4] Selected: {[r['name'] for r in top_repos]}")
    return {"top_repos": top_repos}


# ── Node: Parallel Repo Analyzer (NVIDIA NIM agent) ────────────────────────

async def analyze_repo_node(state: RepoState) -> dict:
    """
    One instance of this runs per top repo — all in parallel via LangGraph Send.

    Steps inside each parallel chain:
      1. Fetch commits from GitHub API
      2. Python noise filter on commits
      3. Pre-fetch top code files for rich code context
      4. Extractor agent (NVIDIA NIM primary round-robin, Groq fallback)
      5. Returns structured skill evidence JSON
    """
    username    = state["username"]
    repo        = state["repo"]
    chain_index = state.get("chain_index", 0)
    print(f"\n  [Parallel → {repo['name']}] Fetching commits and key files...")

    # Fetch + filter commits
    raw_commits     = await fetcher.get_commits(username, repo["name"])
    filtered_commits = filter_commits(raw_commits)
    print(f"  [Parallel → {repo['name']}] {len(raw_commits)} raw → {len(filtered_commits)} after filter")

    # Pre-fetch key code files to give the LLM rich context
    file_tree = await fetcher.get_file_tree(username, repo["name"])
    code_exts = (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".html")
    candidate_files = [
        f for f in file_tree
        if f.endswith(code_exts) and not any(skip in f.lower() for skip in ("test", "spec", "min.", "lock"))
    ][:config.MAX_KEY_FILES]

    key_files = {}
    if candidate_files:
        print(f"  [Parallel → {repo['name']}] Pre-fetching {len(candidate_files)} key file(s): {candidate_files}")
        contents = await asyncio.gather(*[fetcher.get_file(username, repo["name"], f) for f in candidate_files])
        key_files = {f: c for f, c in zip(candidate_files, contents) if c}

    # Extractor runs in a thread
    print(f"  [Parallel → {repo['name']}] Running Extractor Agent (Chain Index: {chain_index})...")
    result = await asyncio.to_thread(
        extract_skills, username, repo, filtered_commits, key_files, chain_index
    )
    result["repo_url"] = repo.get("html_url", "")

    # Vision screenshot evaluation
    print(f"  [Parallel → {repo['name']}] Analyzing best screenshot...")
    best_img = await asyncio.to_thread(
        analyze_best_screenshot,
        repo["name"],
        repo.get("description") or "",
        repo.get("readme_snippet") or "",
        file_tree,
        repo.get("html_url", "")
    )
    result["best_screenshot"] = best_img

    skills_count = len(result.get("skills_demonstrated", []))
    print(f"  [Parallel → {repo['name']}] Done — {skills_count} skill(s), screenshot: {bool(best_img)}")

    # Return as a list so operator.add accumulates correctly across parallel nodes
    return {"repo_results": [result]}


# ── Fan-out Edge: one Send per top repo → all parallel ─────────────────────

def fan_out_to_repos(state: PipelineState):
    """Dynamically creates one parallel node per selected top repo."""
    return [
        Send("analyze_repo", {
            "username":    state["username"],
            "repo":        repo,
            "chain_index": idx,
        })
        for idx, repo in enumerate(state["top_repos"])
    ]


# ── Node: Final Synthesizer LLM ────────────────────────────────────────────

async def synthesize_node(state: PipelineState) -> dict:
    print(f"\n[Step 5] Synthesizing final executive portfolio across {len(state['repo_results'])} analyzed repos...")
    final_json = await asyncio.to_thread(
        synthesize_portfolio,
        state["username"],
        state["repo_results"],
        state.get("leetcode"),
        state.get("linkedin"),
        state.get("credly")
    )
    print(f"[Step 5] Portfolio synthesis complete!")
    return {"final_portfolio": final_json}


# ── Build the LangGraph StateGraph ─────────────────────────────────────────

def build_pipeline():
    graph = StateGraph(PipelineState)

    # Sequential nodes
    graph.add_node("fetch_repos",    fetch_repos_node)
    graph.add_node("filter_repos",   filter_repos_node)
    graph.add_node("fetch_readmes",  fetch_readmes_node)
    graph.add_node("router",         router_node)

    # Parallel node (one instance spawned per top repo)
    graph.add_node("analyze_repo",   analyze_repo_node)

    # Final Synthesizer node
    graph.add_node("synthesize",     synthesize_node)

    # Sequential edges
    graph.set_entry_point("fetch_repos")
    graph.add_edge("fetch_repos",   "filter_repos")
    graph.add_edge("filter_repos",  "fetch_readmes")
    graph.add_edge("fetch_readmes", "router")

    # Fan-out: router → N parallel analyze_repo nodes
    graph.add_conditional_edges("router", fan_out_to_repos, ["analyze_repo"])

    # All parallel nodes converge into synthesize
    graph.add_edge("analyze_repo", "synthesize")
    graph.add_edge("synthesize", END)

    compiled = graph.compile()

    # ── Print graph structure in terminal ──────────────────────────────────
    print("\n📊 Pipeline Graph Structure (ASCII):")
    print("─" * 50)
    compiled.get_graph().print_ascii()
    print("─" * 50)
    print("\n📋 Pipeline Graph (Mermaid format — paste at mermaid.live to visualize):")
    print("─" * 50)
    print(compiled.get_graph().draw_mermaid())
    print("─" * 50)

    return compiled


# ── Entry point ─────────────────────────────────────────────────────────────

async def run_pipeline(
    username: str,
    leetcode: Optional[str] = None,
    linkedin: Optional[str] = None,
    credly: Optional[str] = None
) -> dict:
    print(f"\n{'═'*50}")
    print(f"  GitHub Proof-of-Work Pipeline: {username}")
    print(f"{'═'*50}")

    pipeline = build_pipeline()

    final_state = await pipeline.ainvoke({
        "username":           username,
        "leetcode":           leetcode,
        "linkedin":           linkedin,
        "credly":             credly,
        "raw_repos":          [],
        "filtered_repos":     [],
        "repos_with_readmes": [],
        "top_repos":          [],
        "repo_results":       [],
        "final_portfolio":    {},
    })

    output = final_state.get("final_portfolio") or {
        "developer": {"name": username},
        "top_projects": final_state.get("repo_results", [])
    }

    print(f"\n{'═'*50}")
    print(f"  Pipeline & Synthesis Complete!")
    print(f"{'═'*50}\n")

    return output


if __name__ == "__main__":
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else "torvalds"
    result = asyncio.run(run_pipeline(username))
    print(json.dumps(result, indent=2))
