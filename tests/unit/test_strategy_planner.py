"""Tests for StrategyPlanner — LLM-based goal decomposition."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from jarvis.evolution.models import LearningPlan, SeedSource, SubGoal
from jarvis.evolution.strategy_planner import StrategyPlanner

_VALID_LLM_RESPONSE = json.dumps(
    {
        "sub_goals": [
            {
                "title": "VVG Grundlagen",
                "description": "Lerne den Gesetzestext VVG",
                "priority": 10,
            },
            {
                "title": "Aktuelle Rechtsprechung",
                "description": "BGH-Urteile zum VVG",
                "priority": 5,
            },
        ],
        "sources": [
            {
                "url": "https://www.gesetze-im-internet.de/vvg/",
                "source_type": "law",
                "title": "VVG Gesetzestext",
                "fetch_strategy": "sitemap_crawl",
                "update_frequency": "once",
            }
        ],
        "schedules": [
            {
                "name": "versicherungsrecht_news",
                "cron_expression": "0 6 * * *",
                "source_url": "https://versicherungsbote.de",
                "action": "fetch_and_index",
                "description": "Taegliche News",
            }
        ],
    }
)


@pytest.fixture()
def mock_llm():
    """Mock LLM that returns valid JSON."""
    return AsyncMock(return_value=_VALID_LLM_RESPONSE)


@pytest.fixture()
def planner(mock_llm):
    return StrategyPlanner(llm_fn=mock_llm, max_retries=3)


# ── test_create_plan ─────────────────────────────────────────────────────
async def test_create_plan(planner: StrategyPlanner, mock_llm):
    plan = await planner.create_plan("Werde Experte fuer Versicherungsrecht")

    assert isinstance(plan, LearningPlan)
    assert plan.status == "active"
    assert len(plan.sub_goals) == 2
    # sorted by priority descending → VVG Grundlagen (10) first
    assert plan.sub_goals[0].title == "VVG Grundlagen"
    assert plan.sub_goals[0].priority == 10
    assert plan.sub_goals[1].title == "Aktuelle Rechtsprechung"
    assert len(plan.sources) == 1
    assert plan.sources[0].url == "https://www.gesetze-im-internet.de/vvg/"
    assert len(plan.schedules) == 1
    assert plan.schedules[0].name == "versicherungsrecht_news"
    mock_llm.assert_awaited()


# ── test_create_plan_with_seeds ──────────────────────────────────────────
async def test_create_plan_with_seeds(planner: StrategyPlanner):
    seeds = [
        SeedSource(content_type="url", value="https://example.com/seed", title="Seed"),
    ]
    plan = await planner.create_plan("Werde Experte fuer Versicherungsrecht", seed_sources=seeds)
    assert len(plan.seed_sources) == 1
    assert plan.seed_sources[0].value == "https://example.com/seed"


# ── test_create_plan_invalid_json_retries ────────────────────────────────
async def test_create_plan_invalid_json_retries():
    bad_llm = AsyncMock(return_value="This is not JSON at all!")
    sp = StrategyPlanner(llm_fn=bad_llm, max_retries=2)
    plan = await sp.create_plan("Lerne Kochen")

    assert plan.status == "error"
    assert len(plan.sub_goals) == 0
    # called once initially + 2 retries = up to max_retries calls
    assert bad_llm.await_count >= 2


# ── test_create_plan_partial_json ────────────────────────────────────────
async def test_create_plan_partial_json():
    partial = json.dumps({"sub_goals": [{"title": "Basics", "description": "Do basics"}]})
    llm = AsyncMock(return_value=partial)
    sp = StrategyPlanner(llm_fn=llm)
    plan = await sp.create_plan("Lerne etwas Einfaches")

    assert plan.status == "active"
    assert len(plan.sub_goals) == 1
    assert plan.sub_goals[0].title == "Basics"
    # missing sources/schedules → empty lists
    assert plan.sources == []
    assert plan.schedules == []


# ── test_replan_adds_subgoals ────────────────────────────────────────────
async def test_replan_adds_subgoals(planner: StrategyPlanner, mock_llm):
    # Create initial plan
    plan = await planner.create_plan("Werde Experte fuer Versicherungsrecht")
    assert len(plan.sub_goals) == 2

    # Replan with new sub_goals
    replan_response = json.dumps(
        {
            "sub_goals": [
                {
                    "title": "Haftpflichtrecht",
                    "description": "Vertiefte Haftpflicht-Analyse",
                    "priority": 8,
                },
                # Duplicate title — should NOT be added again
                {
                    "title": "VVG Grundlagen",
                    "description": "Duplicate",
                    "priority": 10,
                },
            ],
            "sources": [],
            "schedules": [],
        }
    )
    mock_llm.return_value = replan_response
    updated = await planner.replan(plan, new_context="Neue BGH-Entscheidung gefunden")

    assert len(updated.sub_goals) == 3  # 2 original + 1 new (duplicate skipped)
    titles = [sg.title for sg in updated.sub_goals]
    assert "Haftpflichtrecht" in titles


# ── test_prompt_includes_seed_sources ────────────────────────────────────
async def test_prompt_includes_seed_sources():
    captured_prompt = None

    async def tracking_llm(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return _VALID_LLM_RESPONSE

    sp = StrategyPlanner(llm_fn=tracking_llm)
    seeds = [
        SeedSource(
            content_type="url",
            value="https://example.com/vvg-guide",
            title="VVG Guide",
        )
    ]
    await sp.create_plan("Lerne VVG", seed_sources=seeds)

    assert captured_prompt is not None
    assert "https://example.com/vvg-guide" in captured_prompt


# ── test_is_complex_goal ─────────────────────────────────────────────────
async def test_is_complex_goal(planner: StrategyPlanner):
    assert planner.is_complex_goal("Werde Experte fuer Versicherungsrecht") is True
    assert planner.is_complex_goal("Master quantum computing from scratch") is True
    assert planner.is_complex_goal("Umfassend alles ueber Kochen lernen") is True
    assert planner.is_complex_goal("Python list comprehensions") is False
    assert planner.is_complex_goal("Was ist 2+2?") is False
