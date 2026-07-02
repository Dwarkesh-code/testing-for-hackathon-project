"""
Router LLM — uses Groq (llama-3.3-70b-versatile) for speed.

Input:  list of repos with README snippets
Output: names of the top N most impressive repos

This is a single LangChain call, not an agent loop.
Speed matters here — we want this done fast so parallel chains can start.
"""

import json
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import config


def select_top_repos(repos_with_readmes: List[Dict], top_n: int = None) -> List[str]:
    """
    Calls Groq LLM with compact repo summaries.
    Returns list of selected repo names (strings).
    Falls back to star-ranking if LLM response can't be parsed.
    """
    if top_n is None:
        top_n = config.MAX_TOP_REPOS

    # If already fewer than needed, skip the LLM call
    if len(repos_with_readmes) <= top_n:
        return [r["name"] for r in repos_with_readmes]

    # Build compact payload — only what the router needs
    summaries = []
    for r in repos_with_readmes:
        summaries.append({
            "name":        r["name"],
            "description": r.get("description") or "",
            "language":    r.get("language") or "Unknown",
            "topics":      r.get("topics", []),
            "stars":       r.get("stargazers_count", 0),
            "readme":      r.get("readme_snippet", "")[:300],
        })

    system_prompt = (
        "You are a senior technical recruiter evaluating GitHub repositories.\n"
        "Pick the most technically impressive and skill-demonstrating projects.\n"
        "Prefer: complex architectures, real functionality, multiple technologies, active development.\n"
        "Avoid: tutorial repos, simple scripts, empty/incomplete projects.\n\n"
        f"Return ONLY valid JSON — no explanation, no markdown:\n"
        f'{{ "selected": ["repo_name_1", "repo_name_2"] }}\n'
        f"Select exactly {top_n} repos."
    )

    user_prompt = (
        f"Evaluate these {len(summaries)} repositories and select the top {top_n}:\n\n"
        + json.dumps(summaries, indent=2)
    )

    # Build router LLM with fallbacks
    # Primary: qwen/qwen3-32b (60 RPM — fastest free tier on Groq)
    groq_key = config.GROQ.get()
    router_llms = [
        ChatGroq(model=m, api_key=groq_key, temperature=0.1)
        for m in config.GROQ_ROUTER_MODELS
    ]
    llm = router_llms[0].with_fallbacks(router_llms[1:])

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        text = response.content.strip()

        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()

        data = json.loads(text)
        selected = data.get("selected", [])

        if selected:
            return selected[:top_n]

    except Exception as e:
        print(f"[Router] Parse/call error: {e} — using fallback ranking")

    # Fallback: return top repos by star count
    return [r["name"] for r in repos_with_readmes[:top_n]]
