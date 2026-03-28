# Human Investigation Module (HIM) — Design Spec

> **For agentic workers:** This spec defines the HIM OSINT module for Cognithor. Use superpowers:writing-plans to create the implementation plan from this spec.

**Goal:** Autonomous OSINT module that researches persons, projects and organizations, cross-verifies claims, and computes a weighted Trust Score (0-100).

**Date:** 2026-03-28
**Status:** Approved
**License:** Apache 2.0

---

## 1. Architecture Overview

HIM is a self-contained module at `src/jarvis/osint/` with its own internal orchestrator. It exposes 3 MCP tools (`investigate_person`, `investigate_project`, `investigate_org`) that the Planner invokes. HIM handles everything internally: GDPR check, parallel collection, cross-verification, scoring, and report generation.

### Integration Points

- **MCP Tools:** Registered via `src/jarvis/mcp/osint_tools.py` following the existing `register_*_tools()` pattern
- **Skill:** `data/procedures/human-investigation.md` for Planner keyword matching
- **Vault:** Reports stored via existing `vault_save` MCP tool with TTL=30 days
- **Gatekeeper:** All 3 tools classified as ORANGE in `_classify_risk()`
- **Config:** New `OsintConfig` section in `JarvisConfig`
- **Web Search:** WebCollector uses existing `search_and_read` / `web_search` MCP tools

### No Jinja2, No Hashline

- Reports use Python f-string formatting (no Jinja2 dependency)
- Report integrity via SHA-256 hash of content (not Hashline Guard, which is for file-edit integrity)

---

## 2. File Structure

```
src/jarvis/osint/
  __init__.py
  him_agent.py            # Orchestrator: run(HIMRequest) -> HIMReport
  gdpr_gatekeeper.py      # DSGVO compliance check + scope determination
  evidence_aggregator.py   # Dedup, cross-verification, claim assignment
  trust_scorer.py          # 5-dimension weighted Trust Score (0-100)
  him_reporter.py          # Markdown/JSON/Quick report rendering (f-strings)
  models.py                # All Pydantic v2 models

  collectors/
    __init__.py
    base.py                # BaseCollector ABC + exponential backoff retry
    github.py              # FULL — GitHub REST API v3
    web.py                 # FULL — uses existing search_and_read MCP tool
    arxiv.py               # FULL — official arXiv API (no key needed)
    scholar.py             # STUB — is_available() returns False
    linkedin.py            # STUB
    crunchbase.py          # STUB
    social.py              # STUB

src/jarvis/mcp/osint_tools.py    # MCP bridge: registers investigate_* tools
data/procedures/human-investigation.md  # Skill for Planner matching

tests/test_osint/
  __init__.py
  conftest.py              # Fixtures: terry_case, mock API responses
  test_him_agent.py
  test_trust_scorer.py
  test_evidence_aggregator.py
  test_gdpr_gatekeeper.py
  test_collectors.py
```

---

## 3. Data Models (Pydantic v2)

All models in `src/jarvis/osint/models.py`.

### HIMRequest

```python
class HIMRequest(BaseModel):
    target_name: str
    target_github: str | None = None
    target_email: str | None = None
    target_linkedin: str | None = None
    target_twitter: str | None = None
    claims: list[str] = Field(default_factory=list)
    target_type: Literal["person", "project", "org"] = "person"
    depth: Literal["quick", "standard", "deep"] = "standard"
    requester_justification: str  # GDPR mandatory, min 10 chars
    language: str = "en"
```

### Enums

```python
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
```

### Evidence

```python
class Evidence(BaseModel):
    source: str              # URL or source identifier
    source_type: str         # "github" | "web" | "arxiv" etc.
    content: str             # Relevant text excerpt
    confidence: float        # 0.0-1.0
    collected_at: datetime
    url: str | None = None
```

### ClaimResult

```python
class ClaimResult(BaseModel):
    claim: str
    claim_type: ClaimType
    status: VerificationStatus
    confidence: float        # 0.0-1.0 weighted average
    evidence: list[Evidence]
    sources_used: list[str]
    explanation: str         # Short justification of status
```

### Finding

```python
class Finding(BaseModel):
    title: str
    description: str
    severity: Literal["info", "warning", "red_flag"]
    source: str
```

### TrustScore

```python
class TrustScore(BaseModel):
    total: int               # 0-100
    label: Literal["high", "mixed", "low"]
    claim_accuracy: float    # 0-100, weight 35%
    source_diversity: float  # 0-100, weight 20%
    technical_substance: float  # 0-100, weight 25%
    transparency: float      # 0-100, weight 10%
    activity_recency: float  # 0-100, weight 10%
```

### HIMReport

```python
class HIMReport(BaseModel):
    report_id: str           # UUID
    target: str
    target_type: str
    generated_at: datetime
    depth: str
    trust_score: TrustScore
    claims: list[ClaimResult]
    key_findings: list[Finding]
    red_flags: list[str]
    summary: str
    recommendation: str
    report_signature: str    # SHA-256 of report content
    raw_evidence_count: int
```

---

## 4. Data Flow

```
User/Planner calls investigate_person
        |
        v
    HIMAgent.run(HIMRequest)
        |
        v
    GDPRGatekeeper.check(request)
        | -> GDPRViolationError if not allowed
        | -> GDPRScope(allowed_collectors, restrictions)
        v
    Collectors in parallel (asyncio.gather, 30s timeout each)
        | GitHub ---+
        | Web ------+---> list[Evidence]
        | arXiv ----+
        v
    EvidenceAggregator.aggregate(all_evidence, claims)
        | -> Deduplication
        | -> Claim classification (Employment/Education/Technical/...)
        | -> Cross-verification (does source A confirm source B?)
        | -> Contradiction detection
        | -> ClaimResult per claim with confidence + status
        v
    TrustScorer.score(claim_results, evidence)
        | -> 5 weighted dimensions
        | -> TrustScore(total=0-100, label=high/mixed/low)
        v
    HIMReporter.render(report_data)
        | -> Markdown report (f-strings)
        | -> SHA-256 signature
        v
    vault_save(report) with TTL=30 days
        |
        v
    Return HIMReport
```

---

## 5. Collectors

### BaseCollector (ABC)

```python
class BaseCollector(ABC):
    source_name: str = "base"
    max_requests_per_minute: int = 30

    @abstractmethod
    async def collect(self, target: str, claims: list[str]) -> list[Evidence]: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    async def _fetch_with_retry(self, url: str, **kwargs) -> dict:
        """Exponential backoff, max 3 retries. Raises CollectorError on exhaustion."""
```

### GitHubCollector (Phase 1 — Full)

Uses GitHub REST API v3. Reads `GITHUB_TOKEN` from env (optional, 60 req/h without).

Collects:
- User profile (bio, company, location, created_at, followers)
- Public repositories (names, descriptions, stars, last activity)
- Organization memberships (public)
- Claim-specific verification:
  - "works at X" → check org membership
  - "built Y" → check if repo Y exists with commits
  - "contributes to Z" → check contribution history

### WebCollector (Phase 1 — Full)

Uses Cognithor's existing `search_and_read` MCP tool (hybrid search: SearXNG + Brave + DDG).

Collects:
- Public mentions of the target
- Conference talks, interviews, blog posts
- News mentions
- Contradictory information (actively searched)

Search queries are claim-specific:
- "{target} works at {org}" for employment claims
- "{target} PhD {university}" for education claims
- General "{target} {claim_keywords}" for other claims

### arXivCollector (Phase 1 — Full)

Uses official arXiv API (`http://export.arxiv.org/api/query`), no key required.

Collects:
- Publications with target as author
- Research focus areas from abstracts
- Co-authors and their institutions
- Submission dates for timeline

### Stub Collectors (Phase 1 — Interface Only)

Scholar, LinkedIn, Crunchbase, Social: implement `BaseCollector` interface, `is_available()` returns `False`, `collect()` returns empty list. Ready for Phase 2 activation.

---

## 6. Evidence Aggregator

The core intelligence engine.

### Claim Classification

Automatic from claim text via keyword matching:

| Keywords | ClaimType |
|----------|-----------|
| "works at", "employed", "position", "role at", "engineer at" | EMPLOYMENT |
| "phd", "doctorate", "studied", "graduate", "degree", "university" | EDUCATION |
| "built", "created", "developed", "authored", "implemented" | TECHNICAL |
| "backed", "funded", "raised", "seed", "series", "investor" | FUNDING |
| "member", "affiliated", "associated", "collaborator", "advisor" | AFFILIATION |
| "award", "publication", "published", "prize" | ACHIEVEMENT |

### Source Priority Weights

```python
SOURCE_PRIORITY = {
    "github": 0.9,
    "arxiv": 0.85,
    "scholar": 0.8,
    "crunchbase": 0.75,
    "web": 0.6,
    "linkedin": 0.4,
    "social": 0.3,
}
```

### Confidence Calculation per Claim

- 1 independent source: max 0.6
- 2 independent sources: max 0.8
- 3+ independent sources: max 0.95
- Contradiction detected: -0.3
- Self-report only (LinkedIn without external confirmation): max 0.4

### Status Logic

- confidence >= 0.7 AND no contradictions → CONFIRMED
- confidence 0.4-0.69 OR partial truth → PARTIAL
- confidence < 0.4 AND no contradiction → UNVERIFIED
- Active contradiction in evidence → CONTRADICTED

---

## 7. Trust Scorer

5 weighted dimensions producing a score from 0-100:

### Claim Accuracy (35%)

Weighted average of claim statuses:
- CONFIRMED = 100 points
- PARTIAL = 50 points
- UNVERIFIED = 20 points
- CONTRADICTED = -50 points (penalty)

### Source Diversity (20%)

Number of independent source types with positive evidence:
- 0 sources: 0
- 1 source: 30
- 2 sources: 55
- 3 sources: 75
- 4+ sources: 90-100

### Technical Substance (25%)

Real technical evidence vs. claims only:
- Commits on relevant repo: +30
- Published papers: +25
- Active OSS contributions: +20
- Only LinkedIn/bio statements: +5
- No technical evidence: 0

### Transparency (10%)

- All claims clear and direct: 100
- Implicit name-dropping detected: 30
- Actively misleading formulations: 0

Detection: If claim says "associated with X" but X contradicts or doesn't mention the person → PARTIAL + transparency penalty.

### Activity Recency (10%)

- Activity within last 3 months: 100
- Activity within last 12 months: 70
- Activity within last 3 years: 40
- Older or no activity: 10

### Label

- 75-100: "high"
- 40-74: "mixed"
- 0-39: "low"

---

## 8. GDPR Gatekeeper

Mandatory check before every investigation.

### Public Figure Detection

A person is public if ANY of:
- GitHub profile is public AND has 50+ followers
- Published papers found (arXiv/Scholar)
- Public conference appearances found (web evidence)

### Scope Matrix

| target_type | Public Figure | Allowed Collectors |
|-------------|--------------|-------------------|
| person | yes | All available |
| person | no | GitHub, Scholar, arXiv, Web (no Social, no LinkedIn) |
| person | no + depth=deep | BLOCKED → GDPRViolationError |
| project | n/a | GitHub, Crunchbase, Web, arXiv |
| org | n/a | Crunchbase, Web, GitHub |

### Mandatory Fields

- `requester_justification` must be filled (min 10 characters), else GDPRViolationError

### Data Minimisation

- Never collect: emails, phone numbers, addresses, images, biometric data
- Reports stored with TTL=30 days in vault
- Vault folder: `recherchen/osint/`

### GDPRScope Model

```python
class GDPRScope(BaseModel):
    is_public_figure: bool
    allowed_collectors: list[str]
    restrictions: list[str]
    ttl_days: int = 30
```

---

## 9. MCP Integration

### Tools Registration (`src/jarvis/mcp/osint_tools.py`)

| Tool | Description |
|------|-------------|
| `investigate_person` | Full OSINT investigation of a person |
| `investigate_project` | Investigation of a project/repository |
| `investigate_org` | Investigation of an organization |

All tools accept JSON matching `HIMRequest` schema and return `HIMReport` as JSON.

### Gatekeeper Classification

All 3 tools classified as ORANGE in `src/jarvis/core/gatekeeper.py` `_classify_risk()`.

### Skill

`data/procedures/human-investigation.md`:
- trigger_keywords: `[Investigation, OSINT, Background, "Due Diligence", Recherche, "wer ist", Hintergrund, Verifikation, "trust score"]`
- tools_required: `[investigate_person, investigate_project, investigate_org]`
- category: research
- priority: 5

---

## 10. Configuration

New section in `src/jarvis/config.py`:

```python
class OsintConfig(BaseModel):
    enabled: bool = True
    github_token: str = ""          # or GITHUB_TOKEN env var
    default_depth: str = "standard"
    collector_timeout: int = 30     # seconds per collector
    report_ttl_days: int = 30
    vault_folder: str = "recherchen/osint"
```

Env var override: `JARVIS_OSINT_ENABLED`, `JARVIS_OSINT_GITHUB_TOKEN`, etc.

---

## 11. Error Handling

- **No API key:** Collector returns `is_available() == False`, gets skipped silently
- **All collectors fail:** Report generated with trust_score=0, summary="No data available"
- **Collector timeout:** 30s per collector via `asyncio.wait_for`, timeout → skip that collector
- **GDPR violation:** `GDPRViolationError` raised, caught by MCP tool handler, returned as error message
- **Invalid request:** Pydantic validation error, returned as structured error

---

## 12. Testing

Minimum 15 tests across 5 test files.

### test_him_agent.py
- `test_full_investigation_flow` — mock all collectors, verify complete pipeline
- `test_gdpr_blocked_investigation` — missing justification → error
- `test_all_collectors_fail_graceful` — returns report with score=0

### test_trust_scorer.py
- `test_perfect_trust_score` — all CONFIRMED, many sources → score >= 85
- `test_contradicted_claim_penalty` — CONTRADICTED pulls score down
- `test_terry_case_score_range` — terry fixture → score 40-65
- `test_score_label_mapping` — 75+ → high, 40-74 → mixed, <40 → low
- `test_transparency_namedropping_penalty`

### test_evidence_aggregator.py
- `test_employment_claim_classification` — "works at Anthropic" → EMPLOYMENT
- `test_education_claim_classification` — "PhD from Stanford" → EDUCATION
- `test_cross_verification_boosts_confidence`
- `test_contradiction_detection`
- `test_self_report_confidence_cap` — LinkedIn only → max 0.4

### test_gdpr_gatekeeper.py
- `test_public_figure_allowed` — 100+ followers → all collectors
- `test_private_person_social_blocked` — no Social/LinkedIn
- `test_missing_justification_raises` — GDPRViolationError
- `test_private_person_deep_blocked` — depth=deep → GDPRViolationError

### test_collectors.py
- `test_github_collector_profile` — mock API, verify Evidence objects
- `test_github_collector_no_token_available` — is_available still True (60 req/h)
- `test_stub_collectors_return_empty` — Scholar/LinkedIn/etc. return []
- `test_arxiv_collector_paper_search` — mock API response

### Test Fixture: terry_case

```json
{
  "request": {
    "target_name": "dinnar1407-code",
    "target_github": "dinnar1407-code",
    "claims": ["works at Anthropic", "most robust A2A implementation", "pre-alpha refinement phase"],
    "target_type": "person",
    "depth": "standard",
    "requester_justification": "Received collaboration request via GitHub, verifying credentials"
  },
  "expected": {
    "trust_score_range": [40, 65],
    "trust_label": "mixed",
    "must_have_red_flags": true
  },
  "mock_github_response": {
    "login": "dinnar1407-code",
    "name": "Terry",
    "company": null,
    "bio": "Building Agent Nexus",
    "public_repos": 3,
    "followers": 12,
    "created_at": "2024-01-15T00:00:00Z",
    "organizations": []
  }
}
```

---

## 13. Dependencies

No new dependencies required. Uses:
- `httpx` (already in deps) — async HTTP for GitHub/arXiv APIs
- `pydantic` (already in deps) — data models
- `hashlib` (stdlib) — SHA-256 report signatures
- Existing MCP tools — `search_and_read`, `vault_save`, `web_search`

---

*HIM Spec v1.0 | Apache 2.0*
