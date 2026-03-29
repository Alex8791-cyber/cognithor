"""Tests for HorizonScanner — LLM exploration + graph gap discovery."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.evolution.horizon_scanner import HorizonScanner
from jarvis.evolution.models import LearningPlan, SubGoal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(**overrides) -> LearningPlan:
    defaults = dict(
        goal="Learn Rust",
        sub_goals=[
            SubGoal(title="Ownership basics", description="Understand ownership"),
        ],
    )
    defaults.update(overrides)
    return LearningPlan(**defaults)


def _entity(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _mock_memory(entities: list[SimpleNamespace] | None = None, search_hits: int = 0):
    """Return a MagicMock that behaves like MemoryManager."""
    mm = MagicMock()
    mm.semantic.list_entities = MagicMock(return_value=entities or [])
    mm.search_memory_sync = MagicMock(
        return_value=[{"text": f"hit{i}"} for i in range(search_hits)],
    )
    return mm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_exploration():
    """LLM returns expansion suggestions -> list with titles/reasons."""

    async def fake_llm(prompt: str) -> str:
        return json.dumps(
            {
                "expansions": [
                    {"title": "Concurrency in Rust", "reason": "Extends ownership"},
                    {"title": "Unsafe Rust", "reason": "Advanced topic"},
                ],
            }
        )

    scanner = HorizonScanner(llm_fn=fake_llm, memory_manager=_mock_memory())
    plan = _make_plan()
    results = await scanner.explore_via_llm(plan)

    assert len(results) == 2
    assert results[0]["title"] == "Concurrency in Rust"
    assert results[0]["reason"] == "Extends ownership"
    assert results[0]["source"] == "llm"
    assert results[1]["source"] == "llm"


@pytest.mark.asyncio
async def test_graph_discovery():
    """Entities with few memory chunks are flagged as graph gaps."""
    entities = [_entity("Borrow Checker"), _entity("Lifetimes")]
    memory = _mock_memory(entities=entities, search_hits=1)

    scanner = HorizonScanner(llm_fn=AsyncMock(return_value="{}"), memory_manager=memory)
    results = await scanner.discover_graph_gaps("learn-rust")

    assert len(results) == 2
    assert all(r["source"] == "graph" for r in results)
    assert "Borrow Checker" in results[0]["title"]
    assert results[0]["reason"]  # non-empty reason


@pytest.mark.asyncio
async def test_full_scan():
    """scan() combines LLM exploration + graph gap discovery."""

    async def fake_llm(prompt: str) -> str:
        return json.dumps(
            {
                "expansions": [
                    {"title": "Async Rust", "reason": "Modern pattern"},
                ],
            }
        )

    entities = [_entity("Tokio")]
    memory = _mock_memory(entities=entities, search_hits=0)

    scanner = HorizonScanner(llm_fn=fake_llm, memory_manager=memory)
    plan = _make_plan()
    results = await scanner.scan(plan)

    titles = [r["title"] for r in results]
    assert "Async Rust" in titles
    assert any("Tokio" in t for t in titles)
    assert len(results) >= 2


@pytest.mark.asyncio
async def test_deduplicates_existing_subgoals():
    """SubGoal already exists with same title -> not in results."""

    async def fake_llm(prompt: str) -> str:
        return json.dumps(
            {
                "expansions": [
                    {"title": "Ownership basics", "reason": "duplicate"},
                    {"title": "Pattern matching", "reason": "new topic"},
                ],
            }
        )

    scanner = HorizonScanner(llm_fn=fake_llm, memory_manager=_mock_memory())
    plan = _make_plan()  # already has SubGoal titled "Ownership basics"
    results = await scanner.scan(plan)

    titles = [r["title"] for r in results]
    assert "Ownership basics" not in titles
    assert "Pattern matching" in titles


@pytest.mark.asyncio
async def test_llm_failure_graceful():
    """LLM returns non-JSON -> empty list, no crash."""

    async def bad_llm(prompt: str) -> str:
        return "I'm not sure what you mean."

    scanner = HorizonScanner(llm_fn=bad_llm, memory_manager=_mock_memory())
    plan = _make_plan()
    results = await scanner.explore_via_llm(plan)

    assert results == []
