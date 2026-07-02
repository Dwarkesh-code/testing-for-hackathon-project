"""
Final Synthesizer — combines all parallel repo skill analyses, screenshots,
and external profiles into a stunning executive Proof-of-Work portfolio JSON.

Two stages:
  Stage A (deterministic, no LLM): aggregate skills across all analyzed repos,
           count how many repos/commits demonstrate each one, weight by
           confidence_tier, and compute verified_score with a real formula
           instead of letting an LLM invent a number.
  Stage B (LLM): takes the clean, pre-scored data from Stage A and writes the
           final copy (bios, headlines, taglines) — with explicit good/bad
           examples and banned generic phrases so output can't collapse into
           filler text.
"""

import json
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from config import config


TIER_WEIGHT = {
    "verified_in_code": 3,
    "verified_in_commit_message": 2,
    "inferred_from_context": 1,
}

BANNED_PHRASES = [
    "passionate about", "proven track record", "strong skills in",
    "detail-oriented", "hardworking", "results-driven", "team player",
]


def _aggregate_skills(repo_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage A — pure Python, no LLM call. Groups skills (case-insensitive exact
    match on skill name) across all repos, weights occurrences by confidence
    tier, and produces a deterministic verified_score.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for r in repo_results:
        for s in r.get("skills_demonstrated", []):
            skill_name = (s.get("skill") or "").strip()
            key = skill_name.lower()
            if not key:
                continue
            tier = s.get("confidence_tier", "inferred_from_context")
            weight = TIER_WEIGHT.get(tier, 1)
            g = groups.setdefault(key, {
                "skill": skill_name,
                "weight_total": 0,
                "evidences": [],
                "_repos": set(),
            })
            g["weight_total"] += weight
            g["evidences"].append({
                "repo": r.get("repo_name", ""),
                "evidence": s.get("evidence", ""),
                "tier": tier,
            })
            g["_repos"].add(r.get("repo_name", ""))

    result = []
    for g in groups.values():
        repo_count = len(g["_repos"])
        g["repo_count"] = repo_count
        g["verified_score"] = min(100, 40 + g["weight_total"] * 8 + repo_count * 5)
        del g["_repos"]
        result.append(g)

    result.sort(key=lambda x: x["verified_score"], reverse=True)
    return result


SYNTHESIZER_SYSTEM_PROMPT = """\
You are writing the copy for a developer's executive Proof-of-Work portfolio page.

You are given pre-aggregated, VERIFIED skill data — it has already been counted and
scored deterministically. Your job is ONLY to turn this into clear, specific, well
written text. You are a copywriter here, not a scorer.

── Hard rules ──────────────────────────────────────────────────────
1. Never invent facts, technologies, or numbers that are not present in the input data.
2. Every sentence in a project's deep_summary must reference something concrete from
   that project's evidence — a specific file, function, technology, or pattern. No
   generic filler sentences.
3. Never use these banned phrases, or close paraphrases of them: {banned}
4. Do not repeat the same sentence structure across different projects' summaries —
   vary how each one opens.
5. verified_score values in core_competencies are given to you already in the input —
   copy them exactly, do not recalculate or invent new ones.

── Example: WEAK bio (never write like this) ───────────────────────
"A passionate full-stack developer with strong skills in Python and web development,
showing a proven track record of building projects."
(Generic — could describe literally any developer. Rejected.)

── Example: STRONG bio (write like this) ────────────────────────────
"Built a three-stage LangGraph pipeline that fans out parallel repo analysis across
rotating NVIDIA NIM and Groq model pools, keeping the whole run under free-tier rate
limits — visible in the round-robin chain_index assignment in config.py."
(Specific, cites real evidence, could only describe this developer.)
──────────────────────────────────────────────────────────────────

Output ONLY valid JSON matching exactly this schema:
{{
  "developer": {{
    "name": "Developer Name",
    "headline": "High-impact professional headline synthesizing their stack",
    "executive_bio": "3-4 sentence synthesized professional overview based on verified contributions.",
    "profiles": {{
      "github": "...",
      "leetcode": "...",
      "linkedin": "...",
      "credly": "..."
    }}
  }},
  "core_competencies": [
    {{
      "category": "e.g. AI & NLP Engineering / Systems Architecture / Frontend Dev",
      "skills": ["Skill 1", "Skill 2", "Skill 3"],
      "verified_score": 95
    }}
  ],
  "top_projects": [
    {{
      "repo_name": "Project Name",
      "tagline": "Short punchy summary",
      "deep_summary": "Detailed technical scope",
      "best_screenshot_url": "URL or null",
      "screenshot_caption": "Caption or null",
      "live_repo_url": "URL",
      "verified_skills": [
        {{
          "skill": "Specific skill name",
          "evidence": "Concrete code proof",
          "commit_url": "Full clickable commit URL if commit_sha exists"
        }}
      ]
    }}
  ]
}}
Do not output markdown code fences or explanatory text. Output ONLY pure JSON.\
""".format(banned=", ".join(f'"{p}"' for p in BANNED_PHRASES))


def synthesize_portfolio(
    username: str,
    repo_results: List[Dict[str, Any]],
    leetcode: Optional[str] = None,
    linkedin: Optional[str] = None,
    credly: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stage A (deterministic) + Stage B (LLM copywriting) → final portfolio JSON.
    """
    aggregated_skills = _aggregate_skills(repo_results)

    payload = {
        "developer_name": username,
        "external_links": {
            "github": f"https://github.com/{username}",
            "leetcode": leetcode or "Not provided",
            "linkedin": linkedin or "Not provided",
            "credly": credly or "Not provided"
        },
        "aggregated_skills": aggregated_skills,  # pre-scored — LLM must not recompute
        "analyzed_repositories": repo_results
    }

    user_prompt = (
        f"Write the portfolio copy for developer '{username}' from this verified data:\n\n"
        + json.dumps(payload, indent=2)
    )

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    groq_key = config.GROQ.get()
    nvidia_key = config.NVIDIA.get()

    if groq_key:
        try:
            print("[Synthesizer] Stage B: writing executive portfolio copy via Groq...")
            llm = ChatGroq(model=config.SYNTHESIS_MODEL, api_key=groq_key, temperature=0.3)
            res = llm.invoke(messages)
            text = res.content.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
        except Exception as e:
            print(f"[Synthesizer] Groq failed: {e}. Falling back to NVIDIA...")

    if nvidia_key:
        try:
            llm = ChatNVIDIA(model=config.EXTRACTOR_MODEL, api_key=nvidia_key, temperature=0.3)
            res = llm.invoke(messages)
            text = res.content.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
        except Exception as e:
            print(f"[Synthesizer] NVIDIA failed: {e}")

    # Fallback structure if both LLM calls fail — still uses real Stage A scores,
    # so it stays "verified" even without a copywriting pass.
    projects = []
    for r in repo_results:
        skills = []
        for s in r.get("skills_demonstrated", []):
            sha = s.get("commit_sha")
            commit_url = f"https://github.com/{username}/{r['repo_name']}/commit/{sha}" if sha else None
            skills.append({
                "skill": s.get("skill", "Software Development"),
                "evidence": s.get("evidence", ""),
                "commit_url": commit_url
            })
        img = r.get("best_screenshot") or {}
        projects.append({
            "repo_name": r.get("repo_name", ""),
            "tagline": r.get("summary", "")[:80],
            "deep_summary": r.get("summary", ""),
            "best_screenshot_url": img.get("url"),
            "screenshot_caption": img.get("caption"),
            "live_repo_url": r.get("repo_url", f"https://github.com/{username}/{r['repo_name']}"),
            "verified_skills": skills
        })

    return {
        "developer": {
            "name": username,
            "headline": "Full-Stack Software Developer & Technical Contributor",
            "executive_bio": f"Verified code contributor on GitHub across {len(projects)} high-signal repositories.",
            "profiles": {
                "github": f"https://github.com/{username}",
                "leetcode": leetcode or "",
                "linkedin": linkedin or "",
                "credly": credly or ""
            }
        },
        "core_competencies": [
            {
                "category": g["skill"],
                "skills": [g["skill"]],
                "verified_score": g["verified_score"]
            }
            for g in aggregated_skills[:8]
        ],
        "top_projects": projects
    }
