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
