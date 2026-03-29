"""Tests for KnowledgeValidator — claims extraction, cross-referencing, confidence tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from jarvis.evolution.knowledge_validator import KnowledgeClaim, KnowledgeValidator


@dataclass
class _MockToolResult:
    content: str = ""
    is_error: bool = False


# -- LLM response fixtures --------------------------------------------------

_LLM_EXTRACT_JSON = json.dumps(
    {
        "claims": [
            {
                "claim": "Die Widerrufsfrist nach Paragraph 7 VVG betraegt 14 Tage.",
                "category": "law",
                "importance": "high",
            },
            {
                "claim": "Versicherungsnehmer koennen den Vertrag ohne Angabe von Gruenden widerrufen.",
                "category": "law",
                "importance": "medium",
            },
        ]
    }
)

_LLM_VERIFY_CONFIRMED = json.dumps(
    {"verdict": "confirmed", "evidence": "Quelle bestaetigt 14-Tage-Frist.", "confidence": 0.8}
)

_LLM_VERIFY_CONTRADICTED = json.dumps(
    {
        "verdict": "contradicted",
        "evidence": "Quelle nennt 30 Tage, nicht 14.",
        "confidence": 0.3,
    }
)


# -- Tests -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_claims(tmp_path):
    """Mock LLM returns claims JSON -> list of KnowledgeClaim with correct fields."""
    llm = AsyncMock(return_value=_LLM_EXTRACT_JSON)
    v = KnowledgeValidator(db_path=tmp_path / "kv.db", llm_fn=llm)

    claims = await v.extract_claims(
        text="Langer Text ueber VVG Widerrufsfrist...",
        source_url="https://example.com/vvg",
        goal_slug="insurance-law",
    )

    assert len(claims) == 2
    assert claims[0].claim.startswith("Die Widerrufsfrist")
    assert claims[0].category == "law"
    assert claims[0].importance == "high"
    assert claims[0].first_source == "https://example.com/vvg"
    assert claims[0].goal_slug == "insurance-law"
    assert claims[0].status == "unverified"
    assert claims[0].confidence == 0.5
    assert claims[0].sources_checked == 1
    v.close()


@pytest.mark.asyncio
async def test_extract_claims_dedup(tmp_path):
    """Same claim extracted twice -> sources_confirmed incremented, not duplicated."""
    llm = AsyncMock(return_value=_LLM_EXTRACT_JSON)
    v = KnowledgeValidator(db_path=tmp_path / "kv.db", llm_fn=llm)

    claims1 = await v.extract_claims(
        text="Erster Text...", source_url="https://a.com", goal_slug="ins"
    )
    claims2 = await v.extract_claims(
        text="Zweiter Text...", source_url="https://b.com", goal_slug="ins"
    )

    # Second extraction should find existing claims and boost them
    assert len(claims2) == 2
    # The claim should have been confirmed (sources_confirmed incremented)
    boosted = claims2[0]
    assert boosted.sources_confirmed >= 1
    assert boosted.sources_checked >= 2
    assert boosted.confidence > 0.5  # boosted from 0.5

    # Only 2 distinct claims in DB, not 4
    all_claims = v.get_claims(goal_slug="ins")
    assert len(all_claims) == 2
    v.close()


@pytest.mark.asyncio
async def test_verify_claim_confirmed(tmp_path):
    """LLM says 'confirmed' -> confidence goes UP, sources_confirmed++."""
    llm = AsyncMock(return_value=_LLM_VERIFY_CONFIRMED)
    v = KnowledgeValidator(db_path=tmp_path / "kv.db", llm_fn=llm)

    claim = KnowledgeClaim(
        id="test001",
        claim="Die Widerrufsfrist betraegt 14 Tage.",
        category="law",
        goal_slug="ins",
        confidence=0.5,
        sources_checked=1,
    )
    v._save_claim(claim)

    updated = await v.verify_claim(claim, source_text="Die Frist ist 14 Tage lang.")

    assert updated.sources_confirmed == 1
    assert updated.sources_checked == 2
    assert updated.confidence > 0.5
    assert updated.evidence_for != ""
    v.close()


@pytest.mark.asyncio
async def test_verify_claim_contradicted(tmp_path):
    """LLM says 'contradicted' -> confidence goes DOWN, status changes."""
    llm = AsyncMock(return_value=_LLM_VERIFY_CONTRADICTED)
    v = KnowledgeValidator(db_path=tmp_path / "kv.db", llm_fn=llm)

    claim = KnowledgeClaim(
        id="test002",
        claim="Die Widerrufsfrist betraegt 14 Tage.",
        category="law",
        goal_slug="ins",
        confidence=0.5,
        sources_checked=1,
    )
    v._save_claim(claim)

    updated = await v.verify_claim(claim, source_text="Tatsaechlich sind es 30 Tage.")

    assert updated.sources_contradicted == 1
    assert updated.sources_checked == 2
    assert updated.confidence < 0.5
    assert updated.evidence_against != ""
    v.close()


@pytest.mark.asyncio
async def test_verify_claim_debunked(tmp_path):
    """Confidence drops below 0.2 -> status = 'debunked'."""
    llm = AsyncMock(return_value=_LLM_VERIFY_CONTRADICTED)
    v = KnowledgeValidator(db_path=tmp_path / "kv.db", llm_fn=llm)

    claim = KnowledgeClaim(
        id="test003",
        claim="Falsche Behauptung ueber irgendwas Wichtiges.",
        category="fact",
        goal_slug="ins",
        confidence=0.2,  # already low
        sources_checked=2,
        sources_contradicted=1,
        status="disputed",
    )
    v._save_claim(claim)

    updated = await v.verify_claim(claim, source_text="Das stimmt nicht.")

    # 0.2 - 0.25 = clamped to 0.0
    assert updated.confidence < 0.2
    assert updated.status == "debunked"
    assert updated.sources_contradicted == 2
    v.close()


def test_get_claims_summary(tmp_path):
    """Insert claims -> summary returns correct counts."""
    v = KnowledgeValidator(db_path=tmp_path / "kv.db")

    claims_data = [
        KnowledgeClaim(
            id="s1",
            claim="Claim A ist korrekt und verifiziert.",
            goal_slug="g1",
            status="verified",
            confidence=0.9,
        ),
        KnowledgeClaim(
            id="s2",
            claim="Claim B ist umstritten und disputed.",
            goal_slug="g1",
            status="disputed",
            confidence=0.4,
        ),
        KnowledgeClaim(
            id="s3",
            claim="Claim C ist widerlegt und debunked.",
            goal_slug="g1",
            status="debunked",
            confidence=0.1,
        ),
        KnowledgeClaim(
            id="s4",
            claim="Claim D ist noch unverified und neu.",
            goal_slug="g1",
            status="unverified",
            confidence=0.5,
        ),
        KnowledgeClaim(
            id="s5",
            claim="Claim E gehoert zu einem anderen Goal.",
            goal_slug="g2",
            status="verified",
            confidence=0.95,
        ),
    ]
    for c in claims_data:
        v._save_claim(c)

    summary = v.get_claims_summary(goal_slug="g1")

    assert summary["total_claims"] == 4
    assert summary["verified"] == 1
    assert summary["disputed"] == 1
    assert summary["debunked"] == 1
    assert summary["unverified"] == 1
    assert 0.0 < summary["avg_confidence"] < 1.0
    v.close()


@pytest.mark.asyncio
async def test_challenge_weak_claims(tmp_path):
    """Mock MCP returns text, LLM verifies -> claim confidence updated."""
    llm = AsyncMock(return_value=_LLM_VERIFY_CONFIRMED)
    mcp = AsyncMock()
    mcp.call_tool.return_value = _MockToolResult(
        content="Bestaetigung: Die Frist betraegt tatsaechlich 14 Tage.", is_error=False
    )
    v = KnowledgeValidator(db_path=tmp_path / "kv.db", llm_fn=llm, mcp_client=mcp)

    weak_claim = KnowledgeClaim(
        id="w1",
        claim="Die Widerrufsfrist betraegt 14 Tage laut VVG.",
        category="law",
        goal_slug="ins",
        confidence=0.4,
        sources_checked=1,
        status="unverified",
    )
    v._save_claim(weak_claim)

    challenged = await v.challenge_weak_claims(goal_slug="ins", max_challenges=3)

    assert len(challenged) == 1
    assert challenged[0].confidence > 0.4
    assert challenged[0].sources_confirmed >= 1
    mcp.call_tool.assert_called_once()
    v.close()


def test_persistence(tmp_path):
    """Claims survive close/reopen of DB."""
    db_path = tmp_path / "persist.db"
    v1 = KnowledgeValidator(db_path=db_path)
    v1._save_claim(
        KnowledgeClaim(
            id="p1",
            claim="Persistierter Claim der ueber Neustart hinweg bestehen bleibt.",
            category="fact",
            goal_slug="persist",
            confidence=0.75,
            status="verified",
        )
    )
    v1.close()

    # Reopen
    v2 = KnowledgeValidator(db_path=db_path)
    claims = v2.get_claims(goal_slug="persist")
    assert len(claims) == 1
    assert claims[0].id == "p1"
    assert claims[0].confidence == 0.75
    assert claims[0].status == "verified"
    assert "Persistierter Claim" in claims[0].claim
    v2.close()
