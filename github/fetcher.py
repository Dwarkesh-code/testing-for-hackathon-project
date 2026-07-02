import asyncio
import base64
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from config import config


class GitHubFetcher:
    """
    Async wrapper around GitHub REST API.
    Uses urllib (stdlib only, no extra deps).
    Token rotation happens via config.GITHUB.get() on every request.
    """
    BASE = "https://api.github.com"

    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ProofOfWork-Pipeline"
        }
        token = config.GITHUB.get()
        if token:
            h["Authorization"] = f"token {token}"
        return h

    def _get(self, url: str) -> Any:
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            print(f"[GitHub] HTTP {e.code} → {url}\n  {body[:200]}")
            return None
        except Exception as e:
            print(f"[GitHub] Error → {url}: {e}")
            return None

    # ── Public async methods ────────────────────────────────────────────────

    async def get_repos(self, username: str) -> List[Dict]:
        """Fetch all public repos for a user, sorted by most recently updated."""
        url = f"{self.BASE}/users/{username}/repos?per_page=100&sort=updated&type=owner"
        result = await asyncio.to_thread(self._get, url)
        return result if isinstance(result, list) else []

    async def get_readme(self, owner: str, repo: str) -> str:
        """Fetch and base64-decode README. Returns first 2000 chars."""
        url = f"{self.BASE}/repos/{owner}/{repo}/readme"
        result = await asyncio.to_thread(self._get, url)
        if result and "content" in result:
            try:
                return base64.b64decode(result["content"]).decode("utf-8", errors="ignore")[:2000]
            except Exception:
                return ""
        return ""

    async def get_commits(self, owner: str, repo: str, per_page: int = 50) -> List[Dict]:
        """Fetch latest commits for a repo."""
        url = f"{self.BASE}/repos/{owner}/{repo}/commits?per_page={per_page}"
        result = await asyncio.to_thread(self._get, url)
        return result if isinstance(result, list) else []

    async def get_file(self, owner: str, repo: str, path: str) -> str:
        """Fetch a specific file's content. Returns first 2500 chars."""
        url = f"{self.BASE}/repos/{owner}/{repo}/contents/{path}"
        result = await asyncio.to_thread(self._get, url)
        if result and isinstance(result, dict) and "content" in result:
            try:
                return base64.b64decode(result["content"]).decode("utf-8", errors="ignore")[:2500]
            except Exception:
                return ""
        return ""

    async def get_file_tree(self, owner: str, repo: str, branch: str = "main") -> List[str]:
        """
        Get flat list of all file paths in a repo.
        Tries 'main' first, then 'master' as fallback.
        """
        for b in [branch, "master"]:
            url = f"{self.BASE}/repos/{owner}/{repo}/git/trees/{b}?recursive=1"
            result = await asyncio.to_thread(self._get, url)
            if result and "tree" in result:
                return [
                    f["path"] for f in result["tree"]
                    if f.get("type") == "blob"
                ][:80]
        return []
