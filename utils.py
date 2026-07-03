"""
utils.py — small shared helpers used across the pipeline.

normalize_handle() exists because the frontend input fields accept either a
bare username ("Dwarkesh-code") or a full profile link
("https://github.com/Dwarkesh-code", "github.com/Dwarkesh-code/",
"https://leetcode.com/u/Dwarkesh-code/"). Every downstream caller
(github/fetcher.py, leetcode/fetcher.py) builds API URLs by string-inserting
the raw value, so a pasted link has to be reduced to just the handle here,
once, before it enters the pipeline — instead of trusting every caller to
re-implement this.
"""

import re
from typing import Optional

# Matches github.com/<handle> or leetcode.com/<handle> or leetcode.com/u/<handle>,
# with or without scheme/www, case-insensitive host.
_GITHUB_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/?.*$", re.IGNORECASE
)
_LEETCODE_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?leetcode\.com/(?:u/)?([A-Za-z0-9_.-]+)/?.*$", re.IGNORECASE
)


def normalize_handle(raw: Optional[str], platform: str) -> Optional[str]:
    """
    Reduce a pasted GitHub/LeetCode username OR full profile URL down to
    just the handle. Returns None/"" untouched (optional fields), and
    returns non-matching input trimmed as-is (so a plain username still
    passes straight through).

    platform: "github" | "leetcode"
    """
    if not raw:
        return raw

    value = raw.strip()
    if not value:
        return value

    # Strip an accidental leading "@"
    if value.startswith("@"):
        value = value[1:]

    pattern = _GITHUB_URL_RE if platform == "github" else _LEETCODE_URL_RE
    match = pattern.match(value)
    if match:
        return match.group(1)

    # Not a recognized URL shape — treat as already a bare handle.
    # Still guard against a stray trailing slash from manual typing.
    return value.rstrip("/")
