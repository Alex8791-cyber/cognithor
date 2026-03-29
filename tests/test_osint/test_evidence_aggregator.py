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
