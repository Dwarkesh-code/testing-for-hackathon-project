"""
LeetCode Fetcher — pulls public profile stats via LeetCode's GraphQL endpoint.

No auth needed for public profile data. Uses stdlib urllib only, matching
github/fetcher.py's dependency-light style.

Important limit (by design, not a bug): LeetCode's public API does NOT expose
submitted solution code — only aggregate stats. So evidence here is
necessarily different in kind from GitHub evidence:

  - Solve counts / difficulty split / contest rating → "verified_in_platform_stats"
    (a hard number straight from LeetCode's servers, not self-reported)
  - Skill-tag strength (e.g. "22 Dynamic Programming problems solved") →
    "inferred_from_context" (a real number, but skill-from-tag is an inference,
    not proof of a specific implementation the way a commit diff is)

We never claim "verified_in_code" for LeetCode data — there is no code to point to.
"""

import asyncio
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

GRAPHQL_URL = "https://leetcode.com/graphql"

_QUERY = """
query userProfile($username: String!) {
  matchedUser(username: $username) {
    username
    submitStats {
      acSubmissionNum {
        difficulty
        count
      }
    }
    tagProblemCounts {
      advanced { tagName problemsSolved }
      intermediate { tagName problemsSolved }
      fundamental { tagName problemsSolved }
    }
  }
  userContestRanking(username: $username) {
    attendedContestsCount
    rating
    globalRanking
    topPercentage
  }
}
"""


def _post(username: str) -> Optional[Dict[str, Any]]:
    payload = json.dumps({
        "query": _QUERY,
        "variables": {"username": username},
        "operationName": "userProfile",
    }).encode()

    req = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ProofOfWork-Pipeline",
            "Referer": f"https://leetcode.com/{username}/",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[LeetCode] HTTP {e.code} for username '{username}'")
        return None
    except Exception as e:
        print(f"[LeetCode] Error fetching '{username}': {e}")
        return None


def _top_tags(tag_counts: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    """Flatten advanced/intermediate/fundamental tag buckets, sort by count."""
    all_tags = []
    for bucket in ("advanced", "intermediate", "fundamental"):
        for t in tag_counts.get(bucket) or []:
            all_tags.append({"tag": t["tagName"], "count": t["problemsSolved"]})
    all_tags.sort(key=lambda x: x["count"], reverse=True)
    return all_tags[:limit]


async def get_leetcode_stats(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch + deterministically shape LeetCode stats into an evidence list,
    same {metric, value, confidence_tier} shape the synthesizer already
    understands for GitHub skills.

    Returns None on any failure (invalid username, private profile, API
    down) — pipeline degrades gracefully, same pattern as github/filters.py.
    """
    if not username:
        return None

    raw = await asyncio.to_thread(_post, username)
    if not raw or not raw.get("data") or not raw["data"].get("matchedUser"):
        print(f"[LeetCode] No profile found for '{username}' — skipping")
        return None

    user = raw["data"]["matchedUser"]
    contest = raw["data"].get("userContestRanking")

    ac_counts = {d["difficulty"]: d["count"] for d in user["submitStats"]["acSubmissionNum"]}
    total_solved = ac_counts.get("All", 0)
    easy   = ac_counts.get("Easy", 0)
    medium = ac_counts.get("Medium", 0)
    hard   = ac_counts.get("Hard", 0)

    top_tags = _top_tags(user.get("tagProblemCounts") or {})

    evidence: List[Dict[str, Any]] = [{
        "metric": "Problems Solved",
        "value": f"{total_solved} total ({easy} Easy / {medium} Medium / {hard} Hard)",
        "confidence_tier": "verified_in_platform_stats",
    }]

    if top_tags:
        top_names = ", ".join(f"{t['tag']} ({t['count']})" for t in top_tags[:3])
        evidence.append({
            "metric": "Strongest topic areas",
            "value": top_names,
            "confidence_tier": "inferred_from_context",
        })

    if contest and contest.get("attendedContestsCount", 0) > 0:
        evidence.append({
            "metric": "Contest Performance",
            "value": (
                f"Rating {round(contest['rating'])}, "
                f"global rank {contest['globalRanking']}, "
                f"top {contest['topPercentage']:.1f}% "
                f"across {contest['attendedContestsCount']} contests"
            ),
            "confidence_tier": "verified_in_platform_stats",
        })

    return {
        "username": username,
        "profile_url": f"https://leetcode.com/{username}/",
        "total_solved": total_solved,
        "by_difficulty": {"easy": easy, "medium": medium, "hard": hard},
        "top_tags": top_tags,
        "contest": {
            "rating": round(contest["rating"]) if contest else None,
            "global_ranking": contest["globalRanking"] if contest else None,
            "top_percentage": contest["topPercentage"] if contest else None,
            "attended": contest["attendedContestsCount"] if contest else 0,
        } if contest else None,
        "evidence": evidence,
    }
