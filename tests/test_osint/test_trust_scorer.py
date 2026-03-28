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
