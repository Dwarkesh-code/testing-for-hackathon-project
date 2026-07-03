import re
from datetime import datetime, timezone
from typing import List, Dict
from config import config

# ── Commit message patterns ─────────────────────────────────────────────────

# Commits matching these are NOISE — skip them
NOISE_PATTERNS = [
    r"\btypo\b",
    r"\bwip\b",
    r"\bfmt\b",
    r"\bchore\b",
    r"update\s*readme",
    r"merge\s*pull\s*request",
    r"merge\s*branch",
    r"\bformatting\b",
    r"\bbump\b",
    r"\bminor\b",
    r"fix\s*typo",
    r"update\s*dep",
    r"initial\s*commit",
    r"^\s*v?\d+\.\d+",           # version bumps like "v1.2.3"
]

# Commits with these words are HIGH SIGNAL — keep and score higher
HIGH_SIGNAL_PATTERNS = [
    r"\bimplement\b",
    r"\bfeature\b",
    r"\badd\b",
    r"\brefactor\b",
    r"\boptimize\b",
    r"fix\s+bug",
    r"\bmodule\b",
    r"\bapi\b",
    r"\bauth\b",
    r"\bdatabase\b",
    r"\bpipeline\b",
    r"\basync\b",
    r"\bcache\b",
    r"\bintegrat\b",
    r"\bbuild\b",
]


def filter_repos(raw_repos: List[Dict]) -> List[Dict]:
    """
    Step 1 of the pipeline (pure Python, no LLM).

    Rules:
    - Skip forks (not the developer's own work)
    - Skip empty repos (size == 0)
    - Skip repos with no name

    Then rank by: stars + recency + has description
    Return top config.MAX_CANDIDATE_REPOS candidates for the Router LLM to decide from.
    """
    candidates = []
    for r in raw_repos:
        if r.get("fork"):
            continue
        if r.get("size", 0) == 0:
            continue
        if not r.get("name"):
            continue
        candidates.append(r)

    def activity_score(r: Dict) -> float:
        stars   = r.get("stargazers_count", 0) * 2.0
        has_desc = 1.0 if r.get("description") else 0.0
        try:
            dt = datetime.fromisoformat(r["updated_at"].replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - dt).days
            recency = max(0.0, (365 - days_ago) * 0.1)
        except Exception:
            recency = 0.0
        return stars + has_desc + recency

    candidates.sort(key=activity_score, reverse=True)
    return candidates[:config.MAX_CANDIDATE_REPOS]


def filter_commits(raw_commits: List[Dict]) -> List[Dict]:
    """
    Step 2 inside each parallel chain (pure Python, no LLM).

    Rules:
    - Skip commits shorter than 10 chars
    - Skip commits matching NOISE_PATTERNS
    - Score remaining by HIGH_SIGNAL_PATTERNS
    - Return top config.MAX_COMMITS by signal score
    """
    scored = []
    for c in raw_commits:
        commit_meta = c.get("commit", {})
        # Use only the first line of the commit message
        full_msg = commit_meta.get("message", "").strip()
        msg = full_msg.split("\n")[0].strip()

        if len(msg) < 10:
            continue

        msg_lower = msg.lower()

        # Noise check
        if any(re.search(p, msg_lower) for p in NOISE_PATTERNS):
            continue

        # Signal score
        signal = sum(2 for p in HIGH_SIGNAL_PATTERNS if re.search(p, msg_lower))

        scored.append({"commit": c, "score": signal})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [item["commit"] for item in scored[:config.MAX_COMMITS]]
