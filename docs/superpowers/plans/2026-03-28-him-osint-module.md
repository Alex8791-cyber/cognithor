# HIM (Human Investigation Module) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OSINT module that researches persons/projects/orgs, cross-verifies claims, and computes a Trust Score (0-100).

**Architecture:** Self-contained module at `src/jarvis/osint/` with internal orchestrator. Exposes 3 MCP tools (`investigate_person`, `investigate_project`, `investigate_org`). Uses existing `web_search`/`search_and_read` for web collection, `httpx` for GitHub/arXiv APIs, `vault_save` for report storage.

**Tech Stack:** Python 3.12+, Pydantic v2, httpx (existing dep), asyncio, SHA-256 for report signatures

**Spec:** `docs/superpowers/specs/2026-03-28-him-osint-module-design.md`

---

## File Structure

```
CREATE src/jarvis/osint/__init__.py          — Package exports
CREATE src/jarvis/osint/models.py            — All Pydantic v2 data models
CREATE src/jarvis/osint/collectors/__init__.py — Collector registry
CREATE src/jarvis/osint/collectors/base.py   — BaseCollector ABC + retry
CREATE src/jarvis/osint/collectors/github.py — GitHub REST API collector
CREATE src/jarvis/osint/collectors/web.py    — Web collector (uses MCP tools)
CREATE src/jarvis/osint/collectors/arxiv.py  — arXiv API collector
CREATE src/jarvis/osint/collectors/scholar.py — Stub
CREATE src/jarvis/osint/collectors/linkedin.py — Stub
CREATE src/jarvis/osint/collectors/crunchbase.py — Stub
CREATE src/jarvis/osint/collectors/social.py — Stub
CREATE src/jarvis/osint/gdpr_gatekeeper.py   — DSGVO compliance gate
CREATE src/jarvis/osint/evidence_aggregator.py — Cross-verification engine
CREATE src/jarvis/osint/trust_scorer.py      — 5-dimension scoring
CREATE src/jarvis/osint/him_reporter.py      — Report rendering
CREATE src/jarvis/osint/him_agent.py         — Main orchestrator
CREATE src/jarvis/mcp/osint_tools.py         — MCP tool registration
CREATE data/procedures/human-investigation.md — Planner skill
CREATE tests/test_osint/__init__.py
CREATE tests/test_osint/conftest.py          — Fixtures + terry_case
CREATE tests/test_osint/test_models.py       — Model validation tests
CREATE tests/test_osint/test_trust_scorer.py — Scoring tests
CREATE tests/test_osint/test_evidence_aggregator.py — Cross-verification tests
CREATE tests/test_osint/test_gdpr_gatekeeper.py — GDPR tests
CREATE tests/test_osint/test_him_agent.py    — Integration tests
CREATE tests/test_osint/test_collectors.py   — Collector tests
MODIFY src/jarvis/config.py                  — Add OsintConfig
MODIFY src/jarvis/core/gatekeeper.py         — Add investigate_* to ORANGE
MODIFY src/jarvis/gateway/phases/tools.py    — Register osint tools
```

---

### Task 1: Pydantic Models

**Files:**
- Create: `src/jarvis/osint/__init__.py`
- Create: `src/jarvis/osint/models.py`
- Create: `tests/test_osint/__init__.py`
- Create: `tests/test_osint/test_models.py`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p src/jarvis/osint/collectors
mkdir -p tests/test_osint
```

- [ ] **Step 2: Write `src/jarvis/osint/__init__.py`**

```python
"""Human Investigation Module (HIM) — OSINT research and trust scoring."""
```

- [ ] **Step 3: Write `tests/test_osint/__init__.py`**

```python
```

- [ ] **Step 4: Write the failing tests for models**

Create `tests/test_osint/test_models.py`:

```python
"""Tests for HIM data models."""
from __future__ import annotations

import pytest
from jarvis.osint.models import (
    ClaimType,
    ClaimResult,
    Evidence,
    Finding,
    GDPRScope,
    HIMReport,
    HIMRequest,
    TrustScore,
    VerificationStatus,
)
from datetime import datetime, timezone


def test_him_request_minimal():
    req = HIMRequest(
        target_name="Test User",
        requester_justification="Testing purposes",
    )
    assert req.target_type == "person"
    assert req.depth == "standard"
    assert req.claims == []


def test_him_request_full():
    req = HIMRequest(
        target_name="Terry Zhang",
        target_github="dinnar1407-code",
        claims=["works at Anthropic"],
        target_type="person",
        depth="deep",
        requester_justification="Verifying credentials",
    )
    assert req.target_github == "dinnar1407-code"
    assert len(req.claims) == 1


def test_claim_type_values():
    assert ClaimType.EMPLOYMENT == "employment"
    assert ClaimType.TECHNICAL == "technical"


def test_evidence_creation():
    ev = Evidence(
        source="github",
        source_type="github",
        content="User profile shows 3 repos",
        confidence=0.8,
        collected_at=datetime.now(timezone.utc),
        url="https://github.com/user",
    )
    assert ev.confidence == 0.8


def test_trust_score_label():
    ts = TrustScore(
        total=80,
        label="high",
        claim_accuracy=90.0,
        source_diversity=70.0,
        technical_substance=80.0,
        transparency=100.0,
        activity_recency=70.0,
    )
    assert ts.label == "high"


def test_finding_severity():
    f = Finding(
        title="No org membership",
        description="Claims to work at X but not in public orgs",
        severity="red_flag",
        source="github",
    )
    assert f.severity == "red_flag"
```

- [ ] **Step 5: Run tests to verify they fail**

```bash
python -m pytest tests/test_osint/test_models.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.osint'`

- [ ] **Step 6: Write `src/jarvis/osint/models.py`**

```python
"""HIM data models — Pydantic v2."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    EMPLOYMENT = "employment"
    EDUCATION = "education"
    TECHNICAL = "technical"
    FUNDING = "funding"
    AFFILIATION = "affiliation"
    ACHIEVEMENT = "achievement"


class VerificationStatus(str, Enum):
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


class HIMRequest(BaseModel):
    target_name: str
    target_github: str | None = None
    target_email: str | None = None
    target_linkedin: str | None = None
    target_twitter: str | None = None
    claims: list[str] = Field(default_factory=list)
    target_type: Literal["person", "project", "org"] = "person"
    depth: Literal["quick", "standard", "deep"] = "standard"
    requester_justification: str
    language: str = "en"


class Evidence(BaseModel):
    source: str
    source_type: str
    content: str
    confidence: float
    collected_at: datetime
    url: str | None = None


class ClaimResult(BaseModel):
    claim: str
    claim_type: ClaimType
    status: VerificationStatus
    confidence: float
    evidence: list[Evidence] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    explanation: str = ""


class Finding(BaseModel):
    title: str
    description: str
    severity: Literal["info", "warning", "red_flag"]
    source: str


class TrustScore(BaseModel):
    total: int
    label: Literal["high", "mixed", "low"]
    claim_accuracy: float
    source_diversity: float
    technical_substance: float
    transparency: float
    activity_recency: float


class HIMReport(BaseModel):
    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    target: str
    target_type: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    depth: str
    trust_score: TrustScore
    claims: list[ClaimResult] = Field(default_factory=list)
    key_findings: list[Finding] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    summary: str = ""
    recommendation: str = ""
    report_signature: str = ""
    raw_evidence_count: int = 0


class GDPRScope(BaseModel):
    is_public_figure: bool
    allowed_collectors: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    ttl_days: int = 30


class GDPRViolationError(Exception):
    pass
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
python -m pytest tests/test_osint/test_models.py -v
```
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/jarvis/osint/ tests/test_osint/
git commit -m "feat(osint): add HIM Pydantic v2 data models"
```

---

### Task 2: BaseCollector + Stubs

**Files:**
- Create: `src/jarvis/osint/collectors/__init__.py`
- Create: `src/jarvis/osint/collectors/base.py`
- Create: `src/jarvis/osint/collectors/scholar.py`
- Create: `src/jarvis/osint/collectors/linkedin.py`
- Create: `src/jarvis/osint/collectors/crunchbase.py`
- Create: `src/jarvis/osint/collectors/social.py`
- Create: `tests/test_osint/test_collectors.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_osint/test_collectors.py`:

```python
"""Tests for HIM collectors."""
from __future__ import annotations

import pytest
from jarvis.osint.collectors.base import BaseCollector, CollectorError
from jarvis.osint.collectors.scholar import ScholarCollector
from jarvis.osint.collectors.linkedin import LinkedInCollector
from jarvis.osint.collectors.crunchbase import CrunchbaseCollector
from jarvis.osint.collectors.social import SocialCollector


def test_stub_scholar_not_available():
    c = ScholarCollector()
    assert c.is_available() is False


def test_stub_linkedin_not_available():
    c = LinkedInCollector()
    assert c.is_available() is False


def test_stub_crunchbase_not_available():
    c = CrunchbaseCollector()
    assert c.is_available() is False


def test_stub_social_not_available():
    c = SocialCollector()
    assert c.is_available() is False


@pytest.mark.asyncio
async def test_stub_scholar_returns_empty():
    c = ScholarCollector()
    result = await c.collect("test", [])
    assert result == []


@pytest.mark.asyncio
async def test_stub_social_returns_empty():
    c = SocialCollector()
    result = await c.collect("test", [])
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_osint/test_collectors.py -v
```

- [ ] **Step 3: Write `src/jarvis/osint/collectors/__init__.py`**

```python
"""HIM Collectors — data source modules."""
```

- [ ] **Step 4: Write `src/jarvis/osint/collectors/base.py`**

```python
"""Base collector with retry logic."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx

from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class CollectorError(Exception):
    """Raised when a collector exhausts retries."""


class BaseCollector(ABC):
    source_name: str = "base"
    max_requests_per_minute: int = 30

    @abstractmethod
    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    async def _fetch_with_retry(
        self, url: str, headers: dict[str, str] | None = None, max_retries: int = 3
    ) -> dict[str, Any]:
        """GET url with exponential backoff. Raises CollectorError on exhaustion."""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(url, headers=headers or {})
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.HTTPError, Exception) as e:
                if attempt == max_retries - 1:
                    raise CollectorError(f"Failed after {max_retries} retries: {e}") from e
                wait = 2 ** attempt
                log.debug("collector_retry", source=self.source_name, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)
        raise CollectorError("Unreachable")
```

- [ ] **Step 5: Write the 4 stub collectors**

`src/jarvis/osint/collectors/scholar.py`:
```python
"""Google Scholar collector — STUB (Phase 2)."""
from __future__ import annotations

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence


class ScholarCollector(BaseCollector):
    source_name = "scholar"

    def is_available(self) -> bool:
        return False

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        return []
```

`src/jarvis/osint/collectors/linkedin.py`:
```python
"""LinkedIn collector — STUB (Phase 2)."""
from __future__ import annotations

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence


class LinkedInCollector(BaseCollector):
    source_name = "linkedin"

    def is_available(self) -> bool:
        return False

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        return []
```

`src/jarvis/osint/collectors/crunchbase.py`:
```python
"""Crunchbase collector — STUB (Phase 2)."""
from __future__ import annotations

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence


class CrunchbaseCollector(BaseCollector):
    source_name = "crunchbase"

    def is_available(self) -> bool:
        return False

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        return []
```

`src/jarvis/osint/collectors/social.py`:
```python
"""Social/Twitter collector — STUB (Phase 2)."""
from __future__ import annotations

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence


class SocialCollector(BaseCollector):
    source_name = "social"

    def is_available(self) -> bool:
        return False

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        return []
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_osint/test_collectors.py -v
```
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/jarvis/osint/collectors/ tests/test_osint/test_collectors.py
git commit -m "feat(osint): add BaseCollector ABC and 4 stub collectors"
```

---

### Task 3: GitHub Collector

**Files:**
- Create: `src/jarvis/osint/collectors/github.py`
- Modify: `tests/test_osint/test_collectors.py`

- [ ] **Step 1: Add GitHub collector tests to `tests/test_osint/test_collectors.py`**

Append to the file:

```python
from unittest.mock import AsyncMock, patch, MagicMock
from jarvis.osint.collectors.github import GitHubCollector
from datetime import datetime, timezone


def test_github_collector_available_without_token():
    c = GitHubCollector()
    assert c.is_available() is True


def test_github_collector_source_name():
    c = GitHubCollector()
    assert c.source_name == "github"


@pytest.mark.asyncio
async def test_github_collector_profile():
    mock_profile = {
        "login": "dinnar1407-code",
        "name": "Terry",
        "company": None,
        "bio": "Building Agent Nexus",
        "public_repos": 3,
        "followers": 12,
        "following": 5,
        "created_at": "2024-01-15T00:00:00Z",
        "html_url": "https://github.com/dinnar1407-code",
        "location": None,
    }
    mock_repos = [
        {
            "name": "agent-nexus",
            "description": "A2A protocol layer",
            "stargazers_count": 8,
            "updated_at": "2025-12-01T00:00:00Z",
            "html_url": "https://github.com/dinnar1407-code/agent-nexus",
            "fork": False,
            "language": "Python",
        }
    ]
    mock_orgs: list[dict] = []

    c = GitHubCollector()
    with patch.object(c, "_fetch_with_retry", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = [mock_profile, mock_repos, mock_orgs]
        evidence = await c.collect("dinnar1407-code", ["works at Anthropic"])

    assert len(evidence) >= 2  # profile + repos
    assert any("dinnar1407-code" in e.content for e in evidence)
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
python -m pytest tests/test_osint/test_collectors.py::test_github_collector_profile -v
```

- [ ] **Step 3: Write `src/jarvis/osint/collectors/github.py`**

```python
"""GitHub collector — uses REST API v3."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from jarvis.osint.collectors.base import BaseCollector, CollectorError
from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class GitHubCollector(BaseCollector):
    source_name = "github"
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN", "")

    def is_available(self) -> bool:
        return True  # Works without token (60 req/h)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            h["Authorization"] = f"token {self._token}"
        return h

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        evidence: list[Evidence] = []
        now = datetime.now(timezone.utc)
        try:
            profile = await self._fetch_with_retry(
                f"{self.BASE_URL}/users/{target}", headers=self._headers()
            )
            evidence.append(Evidence(
                source="github_profile",
                source_type="github",
                content=(
                    f"User: {profile.get('login')} | "
                    f"Name: {profile.get('name')} | "
                    f"Bio: {profile.get('bio', '')} | "
                    f"Company: {profile.get('company', '')} | "
                    f"Repos: {profile.get('public_repos', 0)} | "
                    f"Followers: {profile.get('followers', 0)} | "
                    f"Following: {profile.get('following', 0)} | "
                    f"Created: {profile.get('created_at', '')} | "
                    f"Location: {profile.get('location', '')}"
                ),
                confidence=0.9,
                collected_at=now,
                url=profile.get("html_url", ""),
            ))

            repos = await self._fetch_with_retry(
                f"{self.BASE_URL}/users/{target}/repos?sort=updated&per_page=30",
                headers=self._headers(),
            )
            for repo in repos[:15]:
                if repo.get("fork"):
                    continue
                evidence.append(Evidence(
                    source=f"github_repo:{repo['name']}",
                    source_type="github",
                    content=(
                        f"Repo: {repo['name']} | "
                        f"Description: {repo.get('description', '')} | "
                        f"Stars: {repo.get('stargazers_count', 0)} | "
                        f"Language: {repo.get('language', '')} | "
                        f"Updated: {repo.get('updated_at', '')}"
                    ),
                    confidence=0.85,
                    collected_at=now,
                    url=repo.get("html_url", ""),
                ))

            orgs = await self._fetch_with_retry(
                f"{self.BASE_URL}/users/{target}/orgs",
                headers=self._headers(),
            )
            for org in orgs:
                evidence.append(Evidence(
                    source=f"github_org:{org.get('login', '')}",
                    source_type="github",
                    content=f"Organization membership: {org.get('login', '')}",
                    confidence=0.9,
                    collected_at=now,
                    url=f"https://github.com/{org.get('login', '')}",
                ))

            # Claim-specific: check if target has repo matching claim keywords
            for claim in claims:
                claim_lower = claim.lower()
                for repo in repos:
                    repo_name = (repo.get("name", "") or "").lower()
                    repo_desc = (repo.get("description", "") or "").lower()
                    if any(w in repo_name or w in repo_desc for w in claim_lower.split() if len(w) > 3):
                        evidence.append(Evidence(
                            source=f"github_claim_match:{repo['name']}",
                            source_type="github",
                            content=f"Claim '{claim}' may relate to repo '{repo['name']}': {repo.get('description', '')}",
                            confidence=0.7,
                            collected_at=now,
                            url=repo.get("html_url", ""),
                        ))

        except CollectorError:
            log.warning("github_collector_failed", target=target[:30])
        except Exception:
            log.debug("github_collector_error", exc_info=True)
        return evidence
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_osint/test_collectors.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/osint/collectors/github.py tests/test_osint/test_collectors.py
git commit -m "feat(osint): GitHub collector with profile, repos, orgs, claim matching"
```

---

### Task 4: Web Collector

**Files:**
- Create: `src/jarvis/osint/collectors/web.py`
- Modify: `tests/test_osint/test_collectors.py`

- [ ] **Step 1: Add web collector tests**

Append to `tests/test_osint/test_collectors.py`:

```python
from jarvis.osint.collectors.web import WebCollector


def test_web_collector_needs_mcp():
    c = WebCollector(mcp_client=None)
    assert c.is_available() is False


def test_web_collector_available_with_mcp():
    c = WebCollector(mcp_client=MagicMock())
    assert c.is_available() is True


@pytest.mark.asyncio
async def test_web_collector_generates_claim_queries():
    mock_mcp = AsyncMock()
    mock_result = MagicMock()
    mock_result.is_error = False
    mock_result.content = "Terry Zhang works at Anthropic as a researcher."
    mock_mcp.call_tool = AsyncMock(return_value=mock_result)

    c = WebCollector(mcp_client=mock_mcp)
    evidence = await c.collect("Terry Zhang", ["works at Anthropic"])

    assert len(evidence) >= 1
    assert mock_mcp.call_tool.called
```

- [ ] **Step 2: Write `src/jarvis/osint/collectors/web.py`**

```python
"""Web collector — uses existing search_and_read MCP tool."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class WebCollector(BaseCollector):
    source_name = "web"

    def __init__(self, mcp_client: Any = None) -> None:
        self._mcp = mcp_client

    def is_available(self) -> bool:
        return self._mcp is not None

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        if not self._mcp:
            return []
        evidence: list[Evidence] = []
        now = datetime.now(timezone.utc)

        # Build claim-specific queries
        queries = [f'"{target}"']
        for claim in claims[:5]:
            queries.append(f"{target} {claim}")

        for query in queries[:6]:
            try:
                result = await self._mcp.call_tool(
                    "search_and_read",
                    {"query": query[:150], "num_results": 3, "language": "de"},
                )
                if result and not result.is_error and result.content:
                    text = result.content[:3000]
                    evidence.append(Evidence(
                        source=f"web_search:{query[:50]}",
                        source_type="web",
                        content=text,
                        confidence=0.6,
                        collected_at=now,
                    ))
            except Exception:
                log.debug("web_collector_query_failed", query=query[:40], exc_info=True)
        return evidence
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest tests/test_osint/test_collectors.py -v
git add src/jarvis/osint/collectors/web.py tests/test_osint/test_collectors.py
git commit -m "feat(osint): Web collector using search_and_read MCP tool"
```

---

### Task 5: arXiv Collector

**Files:**
- Create: `src/jarvis/osint/collectors/arxiv.py`
- Modify: `tests/test_osint/test_collectors.py`

- [ ] **Step 1: Add arXiv tests**

Append to `tests/test_osint/test_collectors.py`:

```python
from jarvis.osint.collectors.arxiv import ArxivCollector


def test_arxiv_collector_always_available():
    c = ArxivCollector()
    assert c.is_available() is True


@pytest.mark.asyncio
async def test_arxiv_collector_parses_atom():
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Deep Learning for NLP</title>
        <summary>We present a novel approach...</summary>
        <author><name>Test Author</name></author>
        <published>2025-01-15T00:00:00Z</published>
        <id>http://arxiv.org/abs/2501.12345</id>
      </entry>
    </feed>"""

    c = ArxivCollector()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_xml
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        evidence = await c.collect("Test Author", [])

    assert len(evidence) >= 1
    assert "Deep Learning" in evidence[0].content
```

- [ ] **Step 2: Write `src/jarvis/osint/collectors/arxiv.py`**

```python
"""arXiv collector — official API, no key required."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivCollector(BaseCollector):
    source_name = "arxiv"

    def is_available(self) -> bool:
        return True

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        evidence: list[Evidence] = []
        now = datetime.now(timezone.utc)
        query = f'au:"{target}"'
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    _ARXIV_API,
                    params={"search_query": query, "max_results": 10, "sortBy": "submittedDate"},
                )
                resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for entry in root.findall(f"{_ATOM_NS}entry"):
                title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
                summary = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()[:300]
                published = entry.findtext(f"{_ATOM_NS}published") or ""
                entry_id = entry.findtext(f"{_ATOM_NS}id") or ""
                authors = [
                    a.findtext(f"{_ATOM_NS}name") or ""
                    for a in entry.findall(f"{_ATOM_NS}author")
                ]
                evidence.append(Evidence(
                    source=f"arxiv:{entry_id.split('/')[-1]}",
                    source_type="arxiv",
                    content=(
                        f"Paper: {title} | "
                        f"Authors: {', '.join(authors[:5])} | "
                        f"Published: {published[:10]} | "
                        f"Abstract: {summary}"
                    ),
                    confidence=0.85,
                    collected_at=now,
                    url=entry_id,
                ))
        except Exception:
            log.debug("arxiv_collector_failed", exc_info=True)
        return evidence
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest tests/test_osint/test_collectors.py -v
git add src/jarvis/osint/collectors/arxiv.py tests/test_osint/test_collectors.py
git commit -m "feat(osint): arXiv collector with Atom XML parsing"
```

---

### Task 6: GDPR Gatekeeper

**Files:**
- Create: `src/jarvis/osint/gdpr_gatekeeper.py`
- Create: `tests/test_osint/test_gdpr_gatekeeper.py`

- [ ] **Step 1: Write tests**

Create `tests/test_osint/test_gdpr_gatekeeper.py`:

```python
"""Tests for GDPR Gatekeeper."""
from __future__ import annotations

import pytest
from jarvis.osint.gdpr_gatekeeper import GDPRGatekeeper
from jarvis.osint.models import HIMRequest, GDPRViolationError


def _req(**kw) -> HIMRequest:
    defaults = {
        "target_name": "Test User",
        "requester_justification": "Valid justification for testing",
    }
    defaults.update(kw)
    return HIMRequest(**defaults)


def test_missing_justification_raises():
    gk = GDPRGatekeeper()
    with pytest.raises(GDPRViolationError, match="justification"):
        gk.check(_req(requester_justification="short"))


def test_public_figure_all_collectors():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(), github_followers=100)
    assert scope.is_public_figure is True
    assert "social" in scope.allowed_collectors


def test_private_person_social_blocked():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(), github_followers=5)
    assert scope.is_public_figure is False
    assert "social" not in scope.allowed_collectors
    assert "linkedin" not in scope.allowed_collectors


def test_private_person_deep_blocked():
    gk = GDPRGatekeeper()
    with pytest.raises(GDPRViolationError, match="deep"):
        gk.check(_req(depth="deep"), github_followers=5)


def test_project_collectors():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(target_type="project"))
    assert "github" in scope.allowed_collectors
    assert "web" in scope.allowed_collectors


def test_org_collectors():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(target_type="org"))
    assert "web" in scope.allowed_collectors
    assert "github" in scope.allowed_collectors
```

- [ ] **Step 2: Write `src/jarvis/osint/gdpr_gatekeeper.py`**

```python
"""GDPR Gatekeeper — compliance check before investigations."""
from __future__ import annotations

from jarvis.osint.models import GDPRScope, GDPRViolationError, HIMRequest
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_ALL_COLLECTORS = ["github", "web", "arxiv", "scholar", "linkedin", "crunchbase", "social"]
_PRIVATE_PERSON = ["github", "scholar", "arxiv", "web"]
_PROJECT = ["github", "crunchbase", "web", "arxiv"]
_ORG = ["crunchbase", "web", "github"]


class GDPRGatekeeper:
    """Check GDPR compliance before running investigations."""

    def check(
        self,
        request: HIMRequest,
        github_followers: int = 0,
        has_papers: bool = False,
        has_public_talks: bool = False,
    ) -> GDPRScope:
        if len(request.requester_justification.strip()) < 10:
            raise GDPRViolationError(
                "requester_justification must be at least 10 characters"
            )

        if request.target_type == "project":
            return GDPRScope(
                is_public_figure=False,
                allowed_collectors=list(_PROJECT),
                restrictions=["data_minimisation"],
            )

        if request.target_type == "org":
            return GDPRScope(
                is_public_figure=False,
                allowed_collectors=list(_ORG),
                restrictions=["data_minimisation"],
            )

        # Person checks
        is_public = (
            github_followers >= 50
            or has_papers
            or has_public_talks
        )

        if not is_public and request.depth == "deep":
            raise GDPRViolationError(
                "depth='deep' not allowed for private persons"
            )

        allowed = list(_ALL_COLLECTORS) if is_public else list(_PRIVATE_PERSON)
        restrictions = ["data_minimisation"]
        if not is_public:
            restrictions.append("no_social_media")
            restrictions.append("no_deep_linkedin")

        log.info(
            "gdpr_check_passed",
            target=request.target_name[:30],
            is_public=is_public,
            collectors=len(allowed),
        )

        return GDPRScope(
            is_public_figure=is_public,
            allowed_collectors=allowed,
            restrictions=restrictions,
        )
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest tests/test_osint/test_gdpr_gatekeeper.py -v
git add src/jarvis/osint/gdpr_gatekeeper.py tests/test_osint/test_gdpr_gatekeeper.py
git commit -m "feat(osint): GDPR Gatekeeper with public figure detection and scope limits"
```

---

### Task 7: Evidence Aggregator

**Files:**
- Create: `src/jarvis/osint/evidence_aggregator.py`
- Create: `tests/test_osint/test_evidence_aggregator.py`

- [ ] **Step 1: Write tests**

Create `tests/test_osint/test_evidence_aggregator.py`:

```python
"""Tests for Evidence Aggregator."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from jarvis.osint.evidence_aggregator import EvidenceAggregator
from jarvis.osint.models import ClaimType, Evidence, VerificationStatus


def _ev(source_type: str, content: str, confidence: float = 0.7) -> Evidence:
    return Evidence(
        source=f"{source_type}_test",
        source_type=source_type,
        content=content,
        confidence=confidence,
        collected_at=datetime.now(timezone.utc),
    )


def test_employment_claim_classification():
    agg = EvidenceAggregator()
    ct = agg.classify_claim("works at Anthropic")
    assert ct == ClaimType.EMPLOYMENT


def test_education_claim_classification():
    agg = EvidenceAggregator()
    ct = agg.classify_claim("PhD from Stanford")
    assert ct == ClaimType.EDUCATION


def test_technical_claim_classification():
    agg = EvidenceAggregator()
    ct = agg.classify_claim("built the Agent Nexus framework")
    assert ct == ClaimType.TECHNICAL


def test_cross_verification_boosts_confidence():
    agg = EvidenceAggregator()
    evidence = [
        _ev("github", "User works at Anthropic org", 0.9),
        _ev("web", "Terry Zhang, Anthropic researcher", 0.6),
    ]
    results = agg.aggregate(evidence, ["works at Anthropic"])
    assert len(results) == 1
    assert results[0].confidence > 0.6  # Boosted by multiple sources


def test_self_report_confidence_cap():
    agg = EvidenceAggregator()
    evidence = [
        _ev("linkedin", "Senior Engineer at Anthropic", 0.4),
    ]
    results = agg.aggregate(evidence, ["works at Anthropic"])
    assert results[0].confidence <= 0.4


def test_contradiction_detection():
    agg = EvidenceAggregator()
    evidence = [
        _ev("github", "User has no Anthropic org membership", 0.9),
        _ev("web", "No mention of Terry at Anthropic found. Terry works at startup X.", 0.6),
    ]
    results = agg.aggregate(evidence, ["works at Anthropic"])
    assert results[0].status in (VerificationStatus.UNVERIFIED, VerificationStatus.CONTRADICTED)
```

- [ ] **Step 2: Write `src/jarvis/osint/evidence_aggregator.py`**

```python
"""Evidence Aggregator — cross-verification and claim scoring."""
from __future__ import annotations

from jarvis.osint.models import (
    ClaimResult,
    ClaimType,
    Evidence,
    VerificationStatus,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_CLAIM_KEYWORDS: dict[ClaimType, list[str]] = {
    ClaimType.EMPLOYMENT: ["works at", "employed", "position", "role at", "engineer at", "researcher at", "staff"],
    ClaimType.EDUCATION: ["phd", "doctorate", "studied", "graduate", "degree", "university", "stanford", "mit"],
    ClaimType.TECHNICAL: ["built", "created", "developed", "authored", "implemented", "designed", "architect"],
    ClaimType.FUNDING: ["backed", "funded", "raised", "seed", "series", "investor", "grant"],
    ClaimType.AFFILIATION: ["member", "affiliated", "associated", "collaborator", "partner", "advisor"],
    ClaimType.ACHIEVEMENT: ["award", "publication", "published", "prize", "won", "recognized"],
}

_SOURCE_PRIORITY: dict[str, float] = {
    "github": 0.9,
    "arxiv": 0.85,
    "scholar": 0.8,
    "crunchbase": 0.75,
    "web": 0.6,
    "linkedin": 0.4,
    "social": 0.3,
}

# Confidence caps by number of independent sources
_CONFIDENCE_CAPS = {1: 0.6, 2: 0.8}
_CONFIDENCE_CAP_3PLUS = 0.95
_SELF_REPORT_CAP = 0.4
_CONTRADICTION_PENALTY = 0.3


class EvidenceAggregator:
    """Cross-verify evidence and produce ClaimResults."""

    def classify_claim(self, claim: str) -> ClaimType:
        claim_lower = claim.lower()
        best_type = ClaimType.AFFILIATION
        best_count = 0
        for ct, keywords in _CLAIM_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in claim_lower)
            if matches > best_count:
                best_count = matches
                best_type = ct
        return best_type

    def aggregate(
        self, all_evidence: list[Evidence], claims: list[str]
    ) -> list[ClaimResult]:
        results: list[ClaimResult] = []
        for claim in claims:
            claim_type = self.classify_claim(claim)
            claim_lower = claim.lower()

            # Find relevant evidence for this claim
            relevant: list[Evidence] = []
            for ev in all_evidence:
                content_lower = ev.content.lower()
                claim_words = [w for w in claim_lower.split() if len(w) > 3]
                if any(w in content_lower for w in claim_words):
                    relevant.append(ev)

            if not relevant:
                results.append(ClaimResult(
                    claim=claim,
                    claim_type=claim_type,
                    status=VerificationStatus.UNVERIFIED,
                    confidence=0.0,
                    evidence=[],
                    sources_used=[],
                    explanation="No relevant evidence found",
                ))
                continue

            # Count independent source types
            source_types = set(ev.source_type for ev in relevant)
            n_sources = len(source_types)

            # Confidence cap by source count
            if n_sources >= 3:
                cap = _CONFIDENCE_CAP_3PLUS
            else:
                cap = _CONFIDENCE_CAPS.get(n_sources, 0.6)

            # Self-report only?
            self_report_only = source_types <= {"linkedin", "social"}
            if self_report_only:
                cap = _SELF_REPORT_CAP

            # Weighted confidence from evidence
            total_weight = 0.0
            weighted_conf = 0.0
            for ev in relevant:
                priority = _SOURCE_PRIORITY.get(ev.source_type, 0.5)
                weighted_conf += ev.confidence * priority
                total_weight += priority

            raw_confidence = weighted_conf / total_weight if total_weight else 0.0
            confidence = min(raw_confidence, cap)

            # Check for contradictions (negative signals)
            has_contradiction = any(
                "not" in ev.content.lower() or "no " in ev.content.lower()[:50]
                for ev in relevant
                if ev.source_type in ("github", "web")
            )
            if has_contradiction:
                confidence = max(0.0, confidence - _CONTRADICTION_PENALTY)

            # Determine status
            if has_contradiction and confidence < 0.3:
                status = VerificationStatus.CONTRADICTED
            elif confidence >= 0.7:
                status = VerificationStatus.CONFIRMED
            elif confidence >= 0.4:
                status = VerificationStatus.PARTIAL
            else:
                status = VerificationStatus.UNVERIFIED

            results.append(ClaimResult(
                claim=claim,
                claim_type=claim_type,
                status=status,
                confidence=round(confidence, 2),
                evidence=relevant,
                sources_used=list(source_types),
                explanation=self._explain(status, n_sources, has_contradiction, self_report_only),
            ))

        return results

    def _explain(
        self, status: VerificationStatus, n_sources: int,
        has_contradiction: bool, self_report: bool
    ) -> str:
        if status == VerificationStatus.CONFIRMED:
            return f"Confirmed by {n_sources} independent source(s)"
        if status == VerificationStatus.CONTRADICTED:
            return "Evidence contradicts this claim"
        if self_report:
            return "Only self-reported (LinkedIn/social), not independently verified"
        if status == VerificationStatus.PARTIAL:
            return f"Partially supported by {n_sources} source(s)"
        return "No sufficient evidence found"
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest tests/test_osint/test_evidence_aggregator.py -v
git add src/jarvis/osint/evidence_aggregator.py tests/test_osint/test_evidence_aggregator.py
git commit -m "feat(osint): Evidence Aggregator with cross-verification and claim scoring"
```

---

### Task 8: Trust Scorer

**Files:**
- Create: `src/jarvis/osint/trust_scorer.py`
- Create: `tests/test_osint/test_trust_scorer.py`

- [ ] **Step 1: Write tests**

Create `tests/test_osint/test_trust_scorer.py`:

```python
"""Tests for Trust Scorer."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from jarvis.osint.trust_scorer import TrustScorer
from jarvis.osint.models import (
    ClaimResult, ClaimType, Evidence, TrustScore, VerificationStatus,
)


def _cr(status: VerificationStatus, source_types: list[str], confidence: float = 0.8) -> ClaimResult:
    return ClaimResult(
        claim="test claim",
        claim_type=ClaimType.EMPLOYMENT,
        status=status,
        confidence=confidence,
        evidence=[
            Evidence(
                source=f"{st}_test", source_type=st, content="test",
                confidence=confidence, collected_at=datetime.now(timezone.utc),
            )
            for st in source_types
        ],
        sources_used=source_types,
        explanation="test",
    )


def test_perfect_trust_score():
    scorer = TrustScorer()
    claims = [
        _cr(VerificationStatus.CONFIRMED, ["github", "web", "arxiv"]),
        _cr(VerificationStatus.CONFIRMED, ["github", "arxiv"]),
    ]
    all_evidence = []
    for c in claims:
        all_evidence.extend(c.evidence)
    ts = scorer.score(claims, all_evidence)
    assert ts.total >= 75
    assert ts.label == "high"


def test_contradicted_claim_penalty():
    scorer = TrustScorer()
    claims = [
        _cr(VerificationStatus.CONTRADICTED, ["github"], 0.1),
    ]
    ts = scorer.score(claims, claims[0].evidence)
    assert ts.total < 40
    assert ts.label == "low"


def test_score_label_mapping():
    scorer = TrustScorer()
    assert scorer._label(75) == "high"
    assert scorer._label(74) == "mixed"
    assert scorer._label(40) == "mixed"
    assert scorer._label(39) == "low"


def test_terry_case_score_range():
    scorer = TrustScorer()
    claims = [
        _cr(VerificationStatus.PARTIAL, ["github"], 0.5),
        _cr(VerificationStatus.PARTIAL, ["web"], 0.4),
        _cr(VerificationStatus.CONFIRMED, ["github"], 0.7),
    ]
    all_ev = []
    for c in claims:
        all_ev.extend(c.evidence)
    ts = scorer.score(claims, all_ev)
    assert 30 <= ts.total <= 70
    assert ts.label == "mixed"


def test_transparency_namedropping_penalty():
    scorer = TrustScorer()
    claims = [
        _cr(VerificationStatus.PARTIAL, ["linkedin"], 0.4),
    ]
    ts = scorer.score(claims, claims[0].evidence)
    assert ts.transparency < 50
```

- [ ] **Step 2: Write `src/jarvis/osint/trust_scorer.py`**

```python
"""Trust Scorer — 5-dimension weighted scoring (0-100)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from jarvis.osint.models import (
    ClaimResult,
    Evidence,
    TrustScore,
    VerificationStatus,
)

_WEIGHTS = {
    "claim_accuracy": 0.35,
    "source_diversity": 0.20,
    "technical_substance": 0.25,
    "transparency": 0.10,
    "activity_recency": 0.10,
}

_STATUS_POINTS = {
    VerificationStatus.CONFIRMED: 100,
    VerificationStatus.PARTIAL: 50,
    VerificationStatus.UNVERIFIED: 20,
    VerificationStatus.CONTRADICTED: -50,
}

_DIVERSITY_MAP = {0: 0, 1: 30, 2: 55, 3: 75}


class TrustScorer:
    """Compute a 0-100 Trust Score from claims and evidence."""

    def score(
        self, claims: list[ClaimResult], all_evidence: list[Evidence]
    ) -> TrustScore:
        ca = self._claim_accuracy(claims)
        sd = self._source_diversity(all_evidence)
        ts = self._technical_substance(all_evidence)
        tr = self._transparency(claims)
        ar = self._activity_recency(all_evidence)

        total = (
            ca * _WEIGHTS["claim_accuracy"]
            + sd * _WEIGHTS["source_diversity"]
            + ts * _WEIGHTS["technical_substance"]
            + tr * _WEIGHTS["transparency"]
            + ar * _WEIGHTS["activity_recency"]
        )
        total_int = max(0, min(100, int(round(total))))

        return TrustScore(
            total=total_int,
            label=self._label(total_int),
            claim_accuracy=round(ca, 1),
            source_diversity=round(sd, 1),
            technical_substance=round(ts, 1),
            transparency=round(tr, 1),
            activity_recency=round(ar, 1),
        )

    def _label(self, score: int) -> str:
        if score >= 75:
            return "high"
        if score >= 40:
            return "mixed"
        return "low"

    def _claim_accuracy(self, claims: list[ClaimResult]) -> float:
        if not claims:
            return 0.0
        points = [_STATUS_POINTS.get(c.status, 0) for c in claims]
        avg = sum(points) / len(points)
        return max(0.0, min(100.0, avg))

    def _source_diversity(self, evidence: list[Evidence]) -> float:
        source_types = set(e.source_type for e in evidence if e.confidence > 0.3)
        n = len(source_types)
        if n >= 4:
            return min(100, 90 + (n - 4) * 5)
        return _DIVERSITY_MAP.get(n, 0)

    def _technical_substance(self, evidence: list[Evidence]) -> float:
        score = 0.0
        sources = {e.source_type for e in evidence}
        contents = " ".join(e.content.lower() for e in evidence)

        if "github" in sources and ("repo" in contents or "commit" in contents):
            score += 30
        if "arxiv" in sources or "scholar" in sources:
            score += 25
        if "github" in sources and "stars" in contents:
            score += 20
        if sources <= {"linkedin", "social", "web"}:
            if "linkedin" in sources:
                score = max(score, 5)
        return min(100.0, score)

    def _transparency(self, claims: list[ClaimResult]) -> float:
        if not claims:
            return 100.0
        transparent_count = 0
        for c in claims:
            if c.status == VerificationStatus.CONFIRMED:
                transparent_count += 1
            elif c.status == VerificationStatus.PARTIAL:
                # Partial = possible namedropping
                pass
            elif c.status == VerificationStatus.CONTRADICTED:
                return 0.0
        ratio = transparent_count / len(claims)
        if ratio >= 0.8:
            return 100.0
        if ratio >= 0.5:
            return 60.0
        return 30.0

    def _activity_recency(self, evidence: list[Evidence]) -> float:
        now = datetime.now(timezone.utc)
        most_recent = None
        for e in evidence:
            if e.collected_at and e.collected_at > (most_recent or datetime.min.replace(tzinfo=timezone.utc)):
                # Check content for date signals
                content = e.content
                for date_str in _extract_date_hints(content):
                    try:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        if most_recent is None or dt > most_recent:
                            most_recent = dt
                    except (ValueError, TypeError):
                        pass
        if most_recent is None:
            return 10.0
        age = now - most_recent
        if age < timedelta(days=90):
            return 100.0
        if age < timedelta(days=365):
            return 70.0
        if age < timedelta(days=365 * 3):
            return 40.0
        return 10.0


def _extract_date_hints(content: str) -> list[str]:
    """Extract ISO-ish date strings from content."""
    import re
    return re.findall(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?", content)
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest tests/test_osint/test_trust_scorer.py -v
git add src/jarvis/osint/trust_scorer.py tests/test_osint/test_trust_scorer.py
git commit -m "feat(osint): Trust Scorer with 5-dimension weighted scoring"
```

---

### Task 9: HIM Reporter

**Files:**
- Create: `src/jarvis/osint/him_reporter.py`

- [ ] **Step 1: Write `src/jarvis/osint/him_reporter.py`**

```python
"""HIM Reporter — renders Markdown and JSON reports."""
from __future__ import annotations

import hashlib
import json

from jarvis.osint.models import HIMReport, VerificationStatus


class HIMReporter:
    """Generate formatted reports from HIMReport objects."""

    def render_markdown(self, report: HIMReport) -> str:
        lines = [
            f"# HIM Report: {report.target} | Trust Score: {report.trust_score.total}/100",
            "",
            f"**Label:** {report.trust_score.label.upper()} | "
            f"**Type:** {report.target_type} | "
            f"**Depth:** {report.depth} | "
            f"**Evidence:** {report.raw_evidence_count} items",
            "",
            "## Summary",
            report.summary or "No summary available.",
            "",
            "## Claim Verification",
            "",
            "| Claim | Status | Confidence | Sources |",
            "|-------|--------|------------|---------|",
        ]
        for c in report.claims:
            status_icon = {
                VerificationStatus.CONFIRMED: "[OK]",
                VerificationStatus.PARTIAL: "[~]",
                VerificationStatus.UNVERIFIED: "[?]",
                VerificationStatus.CONTRADICTED: "[X]",
            }.get(c.status, "[-]")
            lines.append(
                f"| {c.claim[:60]} | {status_icon} {c.status.value} | "
                f"{c.confidence:.0%} | {', '.join(c.sources_used)} |"
            )

        if report.key_findings:
            lines.extend(["", "## Key Findings", ""])
            for f in report.key_findings:
                icon = {"info": "[i]", "warning": "[!]", "red_flag": "[!!]"}.get(f.severity, "-")
                lines.append(f"- {icon} **{f.title}**: {f.description} _(Source: {f.source})_")

        if report.red_flags:
            lines.extend(["", "## Red Flags", ""])
            for rf in report.red_flags:
                lines.append(f"- [!!] {rf}")

        lines.extend([
            "",
            "## Trust Score Breakdown",
            "",
            f"| Dimension | Score | Weight |",
            f"|-----------|-------|--------|",
            f"| Claim Accuracy | {report.trust_score.claim_accuracy:.0f} | 35% |",
            f"| Source Diversity | {report.trust_score.source_diversity:.0f} | 20% |",
            f"| Technical Substance | {report.trust_score.technical_substance:.0f} | 25% |",
            f"| Transparency | {report.trust_score.transparency:.0f} | 10% |",
            f"| Activity Recency | {report.trust_score.activity_recency:.0f} | 10% |",
            f"| **Total** | **{report.trust_score.total}** | |",
            "",
            "## Recommendation",
            report.recommendation or "No recommendation.",
            "",
            "---",
            f"*Generated: {report.generated_at.isoformat()} | "
            f"Report-ID: {report.report_id} | "
            f"Signature: {report.report_signature[:16]}...*",
        ])
        return "\n".join(lines)

    def render_json(self, report: HIMReport) -> str:
        return report.model_dump_json(indent=2)

    def render_quick(self, report: HIMReport) -> str:
        ts = report.trust_score
        status_line = f"Trust Score: {ts.total}/100 ({ts.label.upper()})"
        claims_line = ", ".join(
            f"{c.claim[:30]}={c.status.value}" for c in report.claims[:3]
        )
        flags = f"Red Flags: {len(report.red_flags)}" if report.red_flags else "No red flags"
        return f"{report.target} | {status_line} | Claims: [{claims_line}] | {flags}"

    @staticmethod
    def sign_report(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
```

- [ ] **Step 2: Commit**

```bash
git add src/jarvis/osint/him_reporter.py
git commit -m "feat(osint): HIM Reporter with Markdown, JSON, and quick summary rendering"
```

---

### Task 10: HIM Agent (Orchestrator)

**Files:**
- Create: `src/jarvis/osint/him_agent.py`
- Create: `tests/test_osint/test_him_agent.py`
- Create: `tests/test_osint/conftest.py`

- [ ] **Step 1: Write test fixtures**

Create `tests/test_osint/conftest.py`:

```python
"""Test fixtures for HIM OSINT tests."""
from __future__ import annotations

import json
import pytest

TERRY_CASE = {
    "request": {
        "target_name": "dinnar1407-code",
        "target_github": "dinnar1407-code",
        "claims": [
            "works at Anthropic",
            "most robust A2A implementation",
            "pre-alpha refinement phase",
        ],
        "target_type": "person",
        "depth": "standard",
        "requester_justification": "Received collaboration request via GitHub, verifying credentials",
    },
    "expected": {
        "trust_score_range": [30, 70],
        "trust_label": "mixed",
        "must_have_red_flags": True,
    },
    "mock_github_response": {
        "login": "dinnar1407-code",
        "name": "Terry",
        "company": None,
        "bio": "Building Agent Nexus",
        "public_repos": 3,
        "followers": 12,
        "following": 5,
        "created_at": "2024-01-15T00:00:00Z",
        "html_url": "https://github.com/dinnar1407-code",
        "location": None,
    },
}


@pytest.fixture
def terry_case():
    return TERRY_CASE
```

- [ ] **Step 2: Write integration test**

Create `tests/test_osint/test_him_agent.py`:

```python
"""Tests for HIM Agent orchestrator."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.osint.him_agent import HIMAgent
from jarvis.osint.models import HIMRequest, GDPRViolationError


@pytest.mark.asyncio
async def test_full_investigation_flow(terry_case):
    mock_mcp = AsyncMock()
    mock_result = MagicMock()
    mock_result.is_error = False
    mock_result.content = "Terry is building Agent Nexus, an A2A protocol layer."
    mock_mcp.call_tool = AsyncMock(return_value=mock_result)

    agent = HIMAgent(mcp_client=mock_mcp)
    request = HIMRequest(**terry_case["request"])

    with patch(
        "jarvis.osint.collectors.github.GitHubCollector._fetch_with_retry",
        new_callable=AsyncMock,
    ) as mock_gh:
        mock_gh.side_effect = [
            terry_case["mock_github_response"],
            [{"name": "agent-nexus", "description": "A2A protocol", "stargazers_count": 8,
              "updated_at": "2025-12-01T00:00:00Z", "html_url": "https://github.com/x/y",
              "fork": False, "language": "Python"}],
            [],  # orgs
        ]
        report = await agent.run(request)

    assert report is not None
    assert 0 <= report.trust_score.total <= 100
    assert report.report_signature


@pytest.mark.asyncio
async def test_gdpr_blocked_investigation():
    agent = HIMAgent()
    request = HIMRequest(
        target_name="Private Person",
        requester_justification="short",
    )
    with pytest.raises(GDPRViolationError):
        await agent.run(request)


@pytest.mark.asyncio
async def test_all_collectors_fail_graceful():
    agent = HIMAgent(mcp_client=None)
    request = HIMRequest(
        target_name="Nobody",
        requester_justification="Testing graceful degradation",
    )
    report = await agent.run(request)
    assert report.trust_score.total == 0
    assert "No data" in report.summary or report.raw_evidence_count == 0
```

- [ ] **Step 3: Write `src/jarvis/osint/him_agent.py`**

```python
"""HIM Agent — main orchestrator for OSINT investigations."""
from __future__ import annotations

import asyncio
from typing import Any

from jarvis.osint.collectors.arxiv import ArxivCollector
from jarvis.osint.collectors.crunchbase import CrunchbaseCollector
from jarvis.osint.collectors.github import GitHubCollector
from jarvis.osint.collectors.linkedin import LinkedInCollector
from jarvis.osint.collectors.scholar import ScholarCollector
from jarvis.osint.collectors.social import SocialCollector
from jarvis.osint.collectors.web import WebCollector
from jarvis.osint.evidence_aggregator import EvidenceAggregator
from jarvis.osint.gdpr_gatekeeper import GDPRGatekeeper
from jarvis.osint.him_reporter import HIMReporter
from jarvis.osint.models import (
    Evidence,
    Finding,
    GDPRViolationError,
    HIMReport,
    HIMRequest,
    VerificationStatus,
)
from jarvis.osint.trust_scorer import TrustScorer
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class HIMAgent:
    """Orchestrate OSINT investigations: GDPR -> Collect -> Aggregate -> Score -> Report."""

    def __init__(
        self,
        mcp_client: Any = None,
        github_token: str | None = None,
        collector_timeout: int = 30,
    ) -> None:
        self._mcp = mcp_client
        self._gdpr = GDPRGatekeeper()
        self._aggregator = EvidenceAggregator()
        self._scorer = TrustScorer()
        self._reporter = HIMReporter()
        self._collector_timeout = collector_timeout

        self._collectors = {
            "github": GitHubCollector(token=github_token),
            "web": WebCollector(mcp_client=mcp_client),
            "arxiv": ArxivCollector(),
            "scholar": ScholarCollector(),
            "linkedin": LinkedInCollector(),
            "crunchbase": CrunchbaseCollector(),
            "social": SocialCollector(),
        }

    async def run(self, request: HIMRequest) -> HIMReport:
        """Execute full investigation pipeline."""
        log.info("him_investigation_start", target=request.target_name[:30], depth=request.depth)

        # 1. GDPR check
        target_handle = request.target_github or request.target_name
        github_followers = 0
        try:
            gh = self._collectors["github"]
            if gh.is_available() and request.target_github:
                profile = await gh._fetch_with_retry(
                    f"{gh.BASE_URL}/users/{request.target_github}",
                    headers=gh._headers(),
                )
                github_followers = profile.get("followers", 0)
        except Exception:
            pass

        scope = self._gdpr.check(request, github_followers=github_followers)

        # 2. Collect evidence in parallel
        all_evidence: list[Evidence] = []
        tasks = []
        for name, collector in self._collectors.items():
            if name not in scope.allowed_collectors:
                continue
            if not collector.is_available():
                continue
            tasks.append(self._collect_with_timeout(collector, target_handle, request.claims))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_evidence.extend(result)
                elif isinstance(result, Exception):
                    log.debug("him_collector_exception", error=str(result)[:100])

        # 3. Aggregate + cross-verify
        claim_results = self._aggregator.aggregate(all_evidence, request.claims)

        # 4. Score
        trust_score = self._scorer.score(claim_results, all_evidence)

        # 5. Build findings and red flags
        findings: list[Finding] = []
        red_flags: list[str] = []

        for cr in claim_results:
            if cr.status == VerificationStatus.CONTRADICTED:
                red_flags.append(f"{cr.claim}: {cr.explanation}")
                findings.append(Finding(
                    title=f"Contradicted: {cr.claim[:50]}",
                    description=cr.explanation,
                    severity="red_flag",
                    source=", ".join(cr.sources_used),
                ))
            elif cr.status == VerificationStatus.PARTIAL:
                findings.append(Finding(
                    title=f"Partially verified: {cr.claim[:50]}",
                    description=cr.explanation,
                    severity="warning",
                    source=", ".join(cr.sources_used),
                ))
            elif cr.status == VerificationStatus.UNVERIFIED:
                red_flags.append(f"{cr.claim}: not verified by any source")

        # 6. Generate summary
        summary = self._generate_summary(request, trust_score, claim_results, all_evidence)
        recommendation = self._generate_recommendation(trust_score)

        # 7. Build report
        report = HIMReport(
            target=request.target_name,
            target_type=request.target_type,
            depth=request.depth,
            trust_score=trust_score,
            claims=claim_results,
            key_findings=findings,
            red_flags=red_flags,
            summary=summary,
            recommendation=recommendation,
            raw_evidence_count=len(all_evidence),
        )

        # 8. Sign report
        md_content = self._reporter.render_markdown(report)
        report.report_signature = self._reporter.sign_report(md_content)

        # 9. Save to vault
        await self._save_to_vault(report, md_content)

        log.info(
            "him_investigation_complete",
            target=request.target_name[:30],
            score=trust_score.total,
            label=trust_score.label,
            evidence=len(all_evidence),
        )
        return report

    async def _collect_with_timeout(
        self, collector: Any, target: str, claims: list[str]
    ) -> list[Evidence]:
        try:
            return await asyncio.wait_for(
                collector.collect(target, claims),
                timeout=self._collector_timeout,
            )
        except asyncio.TimeoutError:
            log.warning("him_collector_timeout", source=collector.source_name)
            return []

    async def _save_to_vault(self, report: HIMReport, md_content: str) -> None:
        if not self._mcp:
            return
        try:
            await self._mcp.call_tool("vault_save", {
                "title": f"HIM Report: {report.target}",
                "content": md_content,
                "tags": "osint, him, investigation",
                "folder": "recherchen/osint",
                "sources": "",
            })
        except Exception:
            log.debug("him_vault_save_failed", exc_info=True)

    def _generate_summary(self, request, trust_score, claims, evidence) -> str:
        confirmed = sum(1 for c in claims if c.status == VerificationStatus.CONFIRMED)
        contradicted = sum(1 for c in claims if c.status == VerificationStatus.CONTRADICTED)
        if not evidence:
            return f"No data available for {request.target_name}."
        return (
            f"Investigation of '{request.target_name}' ({request.target_type}) "
            f"based on {len(evidence)} evidence items from {len(set(e.source_type for e in evidence))} sources. "
            f"Trust Score: {trust_score.total}/100 ({trust_score.label}). "
            f"{confirmed}/{len(claims)} claims confirmed"
            + (f", {contradicted} contradicted" if contradicted else "")
            + "."
        )

    def _generate_recommendation(self, trust_score) -> str:
        if trust_score.label == "high":
            return "Credentials appear credible. Proceed with normal engagement."
        if trust_score.label == "mixed":
            return "Some claims could not be fully verified. Request additional evidence before deep engagement."
        return "Significant credibility concerns. Verify claims independently before any commitment."
```

- [ ] **Step 4: Run tests and commit**

```bash
python -m pytest tests/test_osint/ -v
git add src/jarvis/osint/him_agent.py tests/test_osint/conftest.py tests/test_osint/test_him_agent.py
git commit -m "feat(osint): HIM Agent orchestrator with full investigation pipeline"
```

---

### Task 11: MCP Integration + Config + Gatekeeper

**Files:**
- Create: `src/jarvis/mcp/osint_tools.py`
- Modify: `src/jarvis/config.py` — Add OsintConfig
- Modify: `src/jarvis/core/gatekeeper.py` — Add investigate_* to ORANGE
- Modify: `src/jarvis/gateway/phases/tools.py` — Register osint tools

- [ ] **Step 1: Write `src/jarvis/mcp/osint_tools.py`**

```python
"""MCP bridge for HIM OSINT tools."""
from __future__ import annotations

import json
from typing import Any

from jarvis.osint.him_agent import HIMAgent
from jarvis.osint.models import HIMRequest, GDPRViolationError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class OsintTools:
    def __init__(self, mcp_client: Any, config: Any = None) -> None:
        osint_cfg = getattr(config, "osint", None)
        self._agent = HIMAgent(
            mcp_client=mcp_client,
            github_token=getattr(osint_cfg, "github_token", "") if osint_cfg else "",
            collector_timeout=getattr(osint_cfg, "collector_timeout", 30) if osint_cfg else 30,
        )

    async def investigate_person(
        self, target_name: str, target_github: str = "", claims: str = "",
        depth: str = "standard", justification: str = "",
    ) -> str:
        return await self._run(
            target_name=target_name,
            target_github=target_github or None,
            claims=claims,
            target_type="person",
            depth=depth,
            justification=justification,
        )

    async def investigate_project(
        self, target_name: str, target_github: str = "", claims: str = "",
        justification: str = "",
    ) -> str:
        return await self._run(
            target_name=target_name,
            target_github=target_github or None,
            claims=claims,
            target_type="project",
            depth="standard",
            justification=justification,
        )

    async def investigate_org(
        self, target_name: str, claims: str = "", justification: str = "",
    ) -> str:
        return await self._run(
            target_name=target_name,
            target_github=None,
            claims=claims,
            target_type="org",
            depth="standard",
            justification=justification,
        )

    async def _run(self, *, target_name, target_github, claims, target_type, depth, justification) -> str:
        try:
            claims_list = [c.strip() for c in claims.split(",") if c.strip()] if claims else []
            request = HIMRequest(
                target_name=target_name,
                target_github=target_github,
                claims=claims_list,
                target_type=target_type,
                depth=depth,
                requester_justification=justification or "MCP tool invocation",
            )
            report = await self._agent.run(request)
            from jarvis.osint.him_reporter import HIMReporter
            return HIMReporter().render_quick(report) + "\n\n" + HIMReporter().render_markdown(report)
        except GDPRViolationError as e:
            return f"GDPR VIOLATION: {e}"
        except Exception as e:
            log.debug("osint_tool_error", exc_info=True)
            return f"Investigation failed: {e}"


def register_osint_tools(mcp_client: Any, config: Any = None) -> OsintTools:
    """Register OSINT investigation MCP tools."""
    osint_cfg = getattr(config, "osint", None)
    if osint_cfg and not getattr(osint_cfg, "enabled", True):
        log.info("osint_tools_disabled")
        return None

    tools = OsintTools(mcp_client, config)

    mcp_client.register_builtin_handler(
        "investigate_person",
        tools.investigate_person,
        description="OSINT investigation of a person: collects evidence from GitHub, web, arXiv, cross-verifies claims, computes Trust Score (0-100)",
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Name or GitHub username of the person"},
                "target_github": {"type": "string", "description": "GitHub username (optional)", "default": ""},
                "claims": {"type": "string", "description": "Comma-separated claims to verify", "default": ""},
                "depth": {"type": "string", "enum": ["quick", "standard", "deep"], "default": "standard"},
                "justification": {"type": "string", "description": "Why this investigation is needed (GDPR)", "default": ""},
            },
            "required": ["target_name"],
        },
    )

    mcp_client.register_builtin_handler(
        "investigate_project",
        tools.investigate_project,
        description="OSINT investigation of a project: checks GitHub repos, web mentions, funding claims",
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Project name"},
                "target_github": {"type": "string", "description": "GitHub repo (optional)", "default": ""},
                "claims": {"type": "string", "description": "Comma-separated claims to verify", "default": ""},
                "justification": {"type": "string", "description": "Why this investigation is needed", "default": ""},
            },
            "required": ["target_name"],
        },
    )

    mcp_client.register_builtin_handler(
        "investigate_org",
        tools.investigate_org,
        description="OSINT investigation of an organization: checks web presence, funding, team",
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Organization name"},
                "claims": {"type": "string", "description": "Comma-separated claims to verify", "default": ""},
                "justification": {"type": "string", "description": "Why this investigation is needed", "default": ""},
            },
            "required": ["target_name"],
        },
    )

    log.info("osint_tools_registered", tools=["investigate_person", "investigate_project", "investigate_org"])
    return tools
```

- [ ] **Step 2: Add OsintConfig to `src/jarvis/config.py`**

Find the section where config classes are defined (near WebConfig, VaultConfig) and add:

```python
class OsintConfig(BaseModel):
    """OSINT / Human Investigation Module configuration."""
    enabled: bool = True
    github_token: str = ""
    default_depth: str = "standard"
    collector_timeout: int = Field(default=30, ge=5, le=120)
    report_ttl_days: int = Field(default=30, ge=1, le=365)
    vault_folder: str = "recherchen/osint"
```

Add `osint: OsintConfig = Field(default_factory=OsintConfig)` to the main `JarvisConfig` class.

- [ ] **Step 3: Add investigate_* tools to gatekeeper ORANGE set**

In `src/jarvis/core/gatekeeper.py`, find the `orange_tools` set inside `_classify_risk()` and add:

```python
"investigate_person", "investigate_project", "investigate_org",
```

- [ ] **Step 4: Register in gateway phases**

In `src/jarvis/gateway/phases/tools.py`, add import and registration:

At top of file:
```python
from jarvis.mcp.osint_tools import register_osint_tools
```

In `init_tools()` function, after vault tools registration:
```python
register_osint_tools(mcp_client, config)
```

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/mcp/osint_tools.py src/jarvis/config.py src/jarvis/core/gatekeeper.py src/jarvis/gateway/phases/tools.py
git commit -m "feat(osint): MCP tool registration, config, and gatekeeper integration"
```

---

### Task 12: Planner Skill

**Files:**
- Create: `data/procedures/human-investigation.md`

- [ ] **Step 1: Write skill file**

Create `data/procedures/human-investigation.md`:

```markdown
---
name: human-investigation
description: OSINT-Recherche und Verifikation von Personen, Projekten und Organisationen
trigger_keywords:
  - Investigation
  - OSINT
  - Background
  - Due Diligence
  - Recherche
  - wer ist
  - Hintergrund
  - Verifikation
  - trust score
  - Person recherchieren
  - Unternehmen prüfen
tools_required:
  - investigate_person
  - investigate_project
  - investigate_org
category: research
priority: 5
enabled: true
---

# Human Investigation Module (HIM)

## Wann anwenden
Wenn der Nutzer eine Person, ein Projekt oder eine Organisation recherchieren, verifizieren oder einen Background-Check durchfuehren moechte.

## Vorgehensweise

1. **Ziel identifizieren**: Name, GitHub-Username, Claims die geprueft werden sollen
2. **Tool waehlen**:
   - `investigate_person` fuer Personen
   - `investigate_project` fuer Projekte/Repos
   - `investigate_org` fuer Organisationen
3. **Claims als komma-separierte Liste angeben**
4. **Justification angeben** (DSGVO-Pflicht): Warum wird recherchiert?
5. **Report auswerten**: Trust Score (0-100), Claim-Status, Red Flags

## Beispiel

```
investigate_person(
  target_name="Terry Zhang",
  target_github="dinnar1407-code",
  claims="works at Anthropic, built Agent Nexus",
  justification="Collaboration request received, verifying credentials"
)
```

## Trust Score Interpretation
- 75-100 (HIGH): Glaubwuerdig, normale Zusammenarbeit moeglich
- 40-74 (MIXED): Teilweise verifiziert, zusaetzliche Belege anfordern
- 0-39 (LOW): Erhebliche Bedenken, Claims unabhaengig pruefen
```

- [ ] **Step 2: Commit**

```bash
git add data/procedures/human-investigation.md
git commit -m "feat(osint): add human-investigation Planner skill"
```

---

### Task 13: Final Integration Test + Update Exports

**Files:**
- Modify: `src/jarvis/osint/__init__.py`

- [ ] **Step 1: Update `src/jarvis/osint/__init__.py` with exports**

```python
"""Human Investigation Module (HIM) — OSINT research and trust scoring."""

from jarvis.osint.him_agent import HIMAgent
from jarvis.osint.models import (
    ClaimType,
    ClaimResult,
    Evidence,
    Finding,
    GDPRScope,
    GDPRViolationError,
    HIMReport,
    HIMRequest,
    TrustScore,
    VerificationStatus,
)

__all__ = [
    "HIMAgent",
    "ClaimType",
    "ClaimResult",
    "Evidence",
    "Finding",
    "GDPRScope",
    "GDPRViolationError",
    "HIMReport",
    "HIMRequest",
    "TrustScore",
    "VerificationStatus",
]
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/test_osint/ -v --tb=short
```
Expected: ALL PASS (15+ tests)

- [ ] **Step 3: Run existing tests to verify no regressions**

```bash
python -m pytest tests/ -x --timeout=60 -q 2>/dev/null | tail -5
```

- [ ] **Step 4: Final commit**

```bash
git add src/jarvis/osint/__init__.py
git commit -m "feat(osint): finalize HIM module exports and verify integration"
```

---

## Definition of Done

- [ ] `HIMAgent.run(HIMRequest(...))` completes without error
- [ ] GitHub collector returns real Evidence objects (mocked in tests)
- [ ] Trust Score for terry_case lands in 30-70 range
- [ ] GDPR Gatekeeper blocks private persons without justification
- [ ] Markdown report generated correctly with Trust Score table
- [ ] SHA-256 signature in report
- [ ] All tests pass (minimum 15)
- [ ] 3 MCP tools registered (investigate_person/project/org)
- [ ] Tools classified as ORANGE in gatekeeper
- [ ] Planner skill matches on "wer ist", "Investigation", etc.
