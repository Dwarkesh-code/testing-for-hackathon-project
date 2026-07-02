from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class RepoMetadata:
    name: str
    full_name: str
    html_url: str
    updated_at: str
    description: Optional[str] = None
    language: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    stargazers_count: int = 0
    fork: bool = False
    size: int = 0
    default_branch: str = "main"

@dataclass
class FilteredRepo:
    metadata: RepoMetadata
    readme_snippet: str = ""
    activity_score: float = 0.0

@dataclass
class CommitInfo:
    sha: str
    message: str
    author: str
    date: str
    additions: int = 0
    deletions: int = 0
    files_changed: List[str] = field(default_factory=list)

@dataclass
class SkillEvidence:
    skill: str
    evidence: str
    confidence: float
    commit_sha: Optional[str] = None

@dataclass
class BestScreenshot:
    url: str
    description: str
    score: float

@dataclass
class RepoAnalysisResult:
    repo_name: str
    repo_url: str
    summary: str
    skills_demonstrated: List[SkillEvidence] = field(default_factory=list)
    key_files_analyzed: List[str] = field(default_factory=list)
    best_screenshot: Optional[BestScreenshot] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class GitHubPortfolio:
    username: str
    total_repos_analyzed: int
    top_repos: List[RepoAnalysisResult] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
