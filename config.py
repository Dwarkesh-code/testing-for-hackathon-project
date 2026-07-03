import os
import itertools
import threading
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class KeyRotator:
    """
    Loads all API keys for a given env var prefix and rotates through them.
    Supports: GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
    This prevents hitting rate limits when running parallel chains.

    Thread-safe: analyze_repo_node runs parallel chains via asyncio.to_thread,
    which means multiple real OS threads can call .get() on the same rotator
    concurrently. itertools.cycle is not guaranteed thread-safe under that,
    so all access is guarded by a lock.
    """
    def __init__(self, prefix: str):
        self.keys = self._load(prefix)
        self._cycle = itertools.cycle(self.keys) if self.keys else None
        self._lock = threading.Lock()
        if not self.keys:
            print(f"[Config Warning] No keys found for prefix: {prefix}")

    def _load(self, prefix: str) -> List[str]:
        keys = []
        base = os.getenv(prefix)
        if base:
            keys.append(base)
        for i in range(1, 10):
            k = os.getenv(f"{prefix}_{i}")
            if k and k not in keys:
                keys.append(k)
        return keys

    def get(self) -> Optional[str]:
        """Returns next key in rotation (thread-safe)."""
        if not self._cycle:
            return None
        with self._lock:
            return next(self._cycle)

    def get_excluding(self, exclude: Optional[str] = None) -> Optional[str]:
        """
        Returns next key that isn't `exclude`. Used to retry with a different
        key from the same provider after a rate-limit error, before falling
        back to a different provider entirely. Falls back to the same key
        if only one key exists for this prefix.
        """
        if not self._cycle:
            return None
        with self._lock:
            if len(self.keys) <= 1:
                return next(self._cycle)
            for _ in range(len(self.keys)):
                k = next(self._cycle)
                if k != exclude:
                    return k
            return exclude

    def all(self) -> List[str]:
        return self.keys


class Config:
    # API Key pools — auto-rotates across all provided keys
    GITHUB = KeyRotator("GITHUB_TOKEN")
    GROQ   = KeyRotator("GROQ_API_KEY")
    NVIDIA = KeyRotator("NVIDIA_API_KEY")

    # ── Router models (Groq — speed priority, 1 call per run) ──────────────
    # qwen3-32b has 60 RPM (double others), best for router
    GROQ_ROUTER_MODELS = [
        "qwen/qwen3-32b",               # 60 RPM — primary
        "openai/gpt-oss-20b",           # 30 RPM — fallback 1
        "llama-3.1-8b-instant",         # 30 RPM, 14.4K/day — fallback 2
    ]

    # ── Extractor models (NVIDIA primary, Groq fallback) ───────────────────
    # Each parallel chain picks a different NVIDIA model (round-robin by index)
    # → distributes 40 RPM across multiple models, no single bottleneck
    NVIDIA_EXTRACTOR_MODELS = [
        "nvidia/nemotron-3-super-120b-a12b",     # NVIDIA best, tool calling
        "meta/llama-3.3-70b-instruct",           # reliable, fast tool calling
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",  # good reasoning + tools
    ]

    # ── Explicitly separated model roles ────────────────────────────────────
    # Previously all three of these shared one misleadingly-named "ROUTER_MODEL"
    # alias, which meant synthesis + vision-picking + extractor-fallback traffic
    # all piled onto a single Groq model/key pool. Split so each role is
    # independently tunable and easy to trace in usage dashboards.
    SYNTHESIS_MODEL      = "mixtral-8x7b-32768"                          # final portfolio copywriting only
    VISION_PICKER_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"   # vision-capable — actually looks at images
    EXTRACTOR_MODEL       = "meta/llama-3.3-70b-instruct"                 # NVIDIA fallback alias (synthesizer safety net)

    # If NVIDIA fails for any chain → try these Groq stable tool-calling models in order
    GROQ_EXTRACTOR_FALLBACKS = [
        "llama-3.3-70b-versatile",                       # 30 RPM — stable tool calling
        "qwen/qwen3-32b",                                # 60 RPM — fast & high rate limit
        "llama-3.1-8b-instant",                          # 30 RPM — reliable small tool caller
    ]

    # Pipeline limits
    MAX_CANDIDATE_REPOS = 12   # Python filter sends this many to the Router LLM
    MAX_TOP_REPOS       = 4    # Router selects this many for deep analysis
    MAX_COMMITS         = 15   # per repo, after noise filter
    MAX_KEY_FILES       = 4    # pre-fetched files per repo before LLM call


config = Config()
