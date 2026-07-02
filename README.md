# Proof-of-Work Portfolio Pipeline

**NYC CodeQuest 2026** — merged problem tracks **WEB-04 (Portfolio)** + **EDU-01 (Credentials)**.

Replaces degree/credential-based developer profiles with a verified, evidence-backed
portfolio generated directly from a developer's real GitHub activity — commits, code,
and project structure — instead of self-reported claims.

## Team

| Member | Role |
|---|---|
| Dwarkesh | AI pipeline & backend |
| Vishal | Frontend |
| Pratyush | Deck / n8n |
| Anii | Video & social |

## Architecture

```
fetch_repos → filter_repos (Python) → fetch_readmes → router (LLM)
                                                          │
                                              [fan-out — one per top repo]
                                    ┌──────────┬──────────┬──────────┐
                                analyze_repo analyze_repo analyze_repo  (parallel)
                                    └──────────┴──────────┴──────────┘
                                                          │
                                                     synthesize
                                                          │
                                                        final JSON
```

**Pipeline stages:**

1. **Fetch** — pull all public repos for a GitHub username via the GitHub REST API.
2. **Python filter** (no LLM) — drop forks, empty repos, and low-signal candidates. Rank by
   stars + recency + description presence. Keeps this cheap and deterministic before any
   LLM sees the data.
3. **Router LLM** (Groq) — given lightweight repo summaries (README snippet, language,
   topics), picks the top N most technically interesting repos for deep analysis.
4. **Parallel extractor agents** (NVIDIA NIM primary, Groq fallback) — one LangGraph node
   per selected repo, run concurrently via `Send`. Each agent:
   - Pulls filtered commits (noise-filtered by regex) and pre-fetched key code files
   - Extracts 2–4 skills, each tagged with a `confidence_tier`
     (`verified_in_code` / `verified_in_commit_message` / `inferred_from_context`) instead
     of an unexplained numeric score
   - Picks the best project screenshot using a vision-capable model that actually looks at
     the image content, not just the filename
5. **Synthesizer** — two stages:
   - **Stage A** (deterministic Python): aggregates skills across all analyzed repos,
     weights them by confidence tier, computes a real `verified_score`
   - **Stage B** (LLM): writes the final portfolio copy (headline, bio, project summaries)
     from the pre-scored Stage A data — cannot invent scores, must cite concrete evidence,
     explicit banned-phrase list to avoid generic filler text

Deliberately **not** a fully autonomous agent — a router + parallel fan-out pipeline,
chosen for predictability and debuggability under hackathon time constraints.

## Tech stack

- **Orchestration:** LangGraph (`StateGraph`, `Send`-based fan-out)
- **LLMs:** Groq (router, synthesis) + NVIDIA NIM (primary extractor, round-robin across
  models to spread rate limits)
- **Server:** aiohttp
- **Data source:** GitHub REST API

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GITHUB_TOKEN=...
GITHUB_TOKEN_1=...          # optional additional tokens for rotation

GROQ_API_KEY=...
GROQ_API_KEY_1=...          # additional keys — note: keys under the SAME Groq
                             # account share one daily token quota; use keys from
                             # separate accounts for real quota headroom

NVIDIA_API_KEY=...
NVIDIA_API_KEY_1=...
```

## Run

```bash
python server.py
```

Open `http://localhost:8080`, or `POST /api/generate-portfolio` with:

```json
{ "username": "octocat", "leetcode": "", "linkedin": "", "credly": "" }
```

## Known limitations

- Groq's per-model daily token limit (TPD) is shared across all keys under the same
  organization — meaningful rate-limit resilience requires keys from genuinely separate
  Groq accounts, not just multiple keys on one account.
- The vision screenshot picker is capped at 5 candidate images per call (model limit).
- Router and extractor both have LLM-output-parsing fallbacks (star-ranking / minimal
  skill entry) if the model returns malformed JSON, so the pipeline degrades gracefully
  rather than failing outright.
