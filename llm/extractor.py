"""
Extractor Agent — NVIDIA NIM primary, Groq fallback.

Each parallel chain calls this with its own chain_index:
  chain_index 0 → nvidia/nemotron-3-super-120b-a12b
  chain_index 1 → meta/llama-3.3-70b-instruct
  chain_index 2 → nvidia/llama-3.3-nemotron-super-49b-v1.5

If a call fails with a rate-limit-shaped error, we first retry the SAME
provider with a DIFFERENT key from the rotation pool (this is the whole
point of having 8 keys). Only if that also fails do we move down to the
next model/provider in the fallback list.

Note: with_fallbacks() not used here because create_react_agent() requires
a BaseChatModel with .bind_tools() — RunnableWithFallbacks lacks that.
Instead we build a priority list of agents and try each, with a manual
same-provider retry step in between.
"""

import json
from typing import List, Dict, Any, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from github.tools import GITHUB_TOOLS
from config import config


EXTRACTOR_SYSTEM_PROMPT = """\
You are an expert technical interviewer and code analyst.

Your task: analyze a GitHub repository and extract VERIFIED, CONCRETE technical skills \
demonstrated by the developer. Do not guess — only claim skills you can prove from the code.

You have been given:
1. Repo metadata (name, language, topics)
2. Filtered commit history (noise removed, meaningful commits only)
3. Pre-fetched key files (actual code content)
4. Tools to fetch additional files if you need deeper inspection

Process:
1. Study the pre-fetched files carefully
2. If more context is needed, use `list_repo_files` then `fetch_file_from_github`
3. Cross-reference with commit messages to confirm skills

── Confidence tier rules (mandatory — do not invent a number instead) ─
Every skill must be tagged with exactly one confidence_tier:
  - "verified_in_code"           → you can point to a specific file + function/class
                                    name where this skill is directly implemented
  - "verified_in_commit_message" → only mentioned in a commit message, not seen in
                                    the pre-fetched code
  - "inferred_from_context"      → inferred only from repo language/topics/description,
                                    no direct evidence seen

GOOD evidence (verified_in_code — cites real proof):
  "async def fetch_repos() in github/fetcher.py uses asyncio.to_thread to wrap
   synchronous urllib calls — demonstrates async/sync bridging."

BAD evidence (never do this — restates the skill instead of proving it):
  "The developer shows strong async programming skills."
─────────────────────────────────────────────────────────────────────

Output ONLY this JSON (no markdown, no explanation):
{
  "repo_name": "repo name",
  "summary": "2-sentence summary of what this project does and its technical scope",
  "skills_demonstrated": [
    {
      "skill": "Specific technology or pattern (e.g. FastAPI async routing, RAG pipeline)",
      "evidence": "Exact proof from the code — cite file name, function name, or commit SHA",
      "commit_sha": "optional 7-char SHA",
      "confidence_tier": "verified_in_code"
    }
  ]
}

Extract 2 to 4 skills. Only claim what you can prove from the code.\
"""


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(t in msg for t in ("429", "rate limit", "rate_limit", "quota", "too many requests"))


def _build_agents(chain_index: int) -> List[Tuple[str, str, Optional[str], Any]]:
    """
    Build ordered list of (provider, model_name, key_used, agent) to try.
    NVIDIA models assigned by chain_index (round-robin) → then Groq fallbacks.
    """
    agents = []

    nvidia_key = config.NVIDIA.get()
    groq_key   = config.GROQ.get()

    if nvidia_key:
        nvidia_model = config.NVIDIA_EXTRACTOR_MODELS[
            chain_index % len(config.NVIDIA_EXTRACTOR_MODELS)
        ]
        try:
            llm = ChatNVIDIA(model=nvidia_model, api_key=nvidia_key, temperature=0.1)
            agents.append(("nvidia", nvidia_model, nvidia_key, create_react_agent(llm, GITHUB_TOOLS)))
        except Exception as e:
            print(f"[Extractor] Could not init NVIDIA model {nvidia_model}: {e}")

        for fallback_model in config.NVIDIA_EXTRACTOR_FALLBACKS:
            try:
                llm = ChatNVIDIA(model=fallback_model, api_key=nvidia_key, temperature=0.1)
                agents.append(("nvidia", fallback_model, nvidia_key, create_react_agent(llm, GITHUB_TOOLS)))
            except Exception as e:
                print(f"[Extractor] Could not init NVIDIA fallback model {fallback_model}: {e}")

    if groq_key:
        for model in config.GROQ_EXTRACTOR_FALLBACKS:
            try:
                llm = ChatGroq(model=model, api_key=groq_key, temperature=0.1)
                agents.append(("groq", model, groq_key, create_react_agent(llm, GITHUB_TOOLS)))
            except Exception as e:
                print(f"[Extractor] Could not init Groq model {model}: {e}")

    return agents


def _build_single_agent(provider: str, model_name: str, key: str):
    if provider == "nvidia":
        llm = ChatNVIDIA(model=model_name, api_key=key, temperature=0.1)
    else:
        llm = ChatGroq(model=model_name, api_key=key, temperature=0.1)
    return create_react_agent(llm, GITHUB_TOOLS)


def _parse_result(content: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON from agent response."""
    text = content.strip()
    if "```" in text:
        text = text.split("```")[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _try_invoke(agent, messages) -> Dict[str, Any]:
    result = agent.invoke(messages)
    last_content = result["messages"][-1].content.strip()
    parsed = _parse_result(last_content)
    if "skills_demonstrated" not in parsed:
        raise ValueError("invalid structure — missing skills_demonstrated")
    return parsed


def extract_skills(
    username:        str,
    repo:            Dict,
    filtered_commits: List[Dict],
    key_files:       Dict[str, str] = {},
    chain_index:     int = 0,
) -> Dict[str, Any]:
    """
    Run extractor agent for a single repo.
    Tries NVIDIA first (by chain_index rotation). On a rate-limit error,
    retries once with a different NVIDIA key before falling back to Groq
    models (same retry pattern applied to Groq too).

    Parameters:
        key_files: pre-fetched file contents {filepath: content}
                   passed from main.py so agent doesn't need to fetch these again
    """
    commit_list = []
    for c in filtered_commits:
        meta = c.get("commit", {})
        commit_list.append({
            "sha":     c.get("sha", "")[:7],
            "message": meta.get("message", "").split("\n")[0],
            "author":  meta.get("author", {}).get("name", ""),
            "date":    meta.get("author", {}).get("date", ""),
        })

    files_context = ""
    if key_files:
        files_context = "\n\n── Pre-fetched Key Files ──────────────────────────\n"
        for path, content in key_files.items():
            files_context += f"\n=== {path} ===\n{content[:1500]}\n"
        files_context += "──────────────────────────────────────────────────\n"
        files_context += "Use your tools to fetch more files only if you need additional context.\n"
    else:
        files_context = "\nNo files were pre-fetched. Use list_repo_files then fetch_file_from_github to inspect the code.\n"

    user_prompt = (
        f"Repository to analyze:\n"
        f"  Owner:       {username}\n"
        f"  Repo:        {repo['name']}\n"
        f"  Description: {repo.get('description') or 'None'}\n"
        f"  Language:    {repo.get('language') or 'Unknown'}\n"
        f"  Topics:      {repo.get('topics', [])}\n"
        f"  Stars:       {repo.get('stargazers_count', 0)}\n\n"
        f"Top commits (noise already filtered):\n"
        f"{json.dumps(commit_list, indent=2)}\n"
        f"{files_context}"
    )

    messages = {
        "messages": [
            SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    }

    agents = _build_agents(chain_index)
    for provider, model_name, key_used, agent in agents:
        try:
            print(f"  [Extractor → {repo['name']}] Using model: {model_name}")
            return _try_invoke(agent, messages)
        except Exception as e:
            if _is_rate_limit_error(e):
                retry_key = (
                    config.NVIDIA.get_excluding(key_used) if provider == "nvidia"
                    else config.GROQ.get_excluding(key_used)
                )
                if retry_key and retry_key != key_used:
                    print(f"  [Extractor] {model_name} rate-limited — retrying with a different {provider} key...")
                    try:
                        retry_agent = _build_single_agent(provider, model_name, retry_key)
                        return _try_invoke(retry_agent, messages)
                    except Exception as e2:
                        print(f"  [Extractor] Retry with different key also failed: {type(e2).__name__}: {str(e2)[:100]}")
            print(f"  [Extractor] {model_name} failed: {type(e).__name__}: {str(e)[:100]} — trying next...")
            continue

    print(f"  [Extractor] All models failed for {repo['name']} — using fallback output")
    return {
        "repo_name": repo["name"],
        "summary":   repo.get("description") or f"A {repo.get('language', 'software')} project.",
        "skills_demonstrated": [
            {
                "skill":      f"{repo.get('language', 'Software')} Development",
                "evidence":   f"Primary language used in {repo['name']}.",
                "commit_sha": None,
                "confidence_tier": "inferred_from_context",
            }
        ],
    }
