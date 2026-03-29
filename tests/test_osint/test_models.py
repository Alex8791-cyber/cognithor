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
