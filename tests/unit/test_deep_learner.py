"""Tests for DeepLearner orchestrator — plan CRUD and subgoal management."""

from __future__ import annotations

import json
import os

import pytest

from jarvis.evolution.deep_learner import DeepLearner
from jarvis.evolution.models import LearningPlan, SeedSource, SubGoal


@pytest.fixture()
def plans_dir(tmp_path):
    """Return a temporary plans directory."""
    d = tmp_path / "plans"
    d.mkdir()
    return d


@pytest.fixture()
def mock_llm():
    """Async function returning valid JSON with 1 sub_goal, 1 source, 0 schedules."""

    async def _llm(prompt: str) -> str:
        return json.dumps(
            {
                "sub_goals": [
                    {
                        "title": "Basics of topic",
                        "description": "Learn the fundamentals",
                        "priority": 1,
                    }
                ],
                "sources": [
                    {
                        "url": "https://example.com/docs",
                        "source_type": "documentation",
                        "title": "Official Docs",
                    }
                ],
                "schedules": [],
            }
        )

    return _llm


@pytest.fixture()
def learner(plans_dir, mock_llm):
    """DeepLearner instance configured with tmp plans dir and mock LLM."""
    return DeepLearner(llm_fn=mock_llm, plans_dir=str(plans_dir))


class TestDeepLearner:
    async def test_create_plan(self, learner, plans_dir):
        plan = await learner.create_plan("Learn quantum computing")
        assert isinstance(plan, LearningPlan)
        assert plan.status == "active"
        assert len(plan.sub_goals) >= 1
        # Verify persisted on disk
        plan_json = plans_dir / plan.id / "plan.json"
        assert plan_json.exists()
        with open(plan_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["goal"] == "Learn quantum computing"

    async def test_list_plans_empty(self, learner):
        plans = learner.list_plans()
        assert plans == []

    async def test_list_plans_after_create(self, learner):
        await learner.create_plan("Plan A")
        await learner.create_plan("Plan B")
        plans = learner.list_plans()
        assert len(plans) == 2

    async def test_get_plan(self, learner):
        plan = await learner.create_plan("Get me")
        loaded = learner.get_plan(plan.id)
        assert loaded is not None
        assert loaded.id == plan.id
        assert loaded.goal == "Get me"

    async def test_get_plan_not_found(self, learner):
        result = learner.get_plan("nonexistent_id_12345678")
        assert result is None

    async def test_pause_plan(self, learner, plans_dir):
        plan = await learner.create_plan("Pause me")
        ok = learner.update_plan_status(plan.id, "paused")
        assert ok is True
        # Verify persisted
        loaded = learner.get_plan(plan.id)
        assert loaded is not None
        assert loaded.status == "paused"

    async def test_delete_plan(self, learner, plans_dir):
        plan = await learner.create_plan("Delete me")
        plan_dir = plans_dir / plan.id
        assert plan_dir.exists()
        ok = learner.delete_plan(plan.id)
        assert ok is True
        assert not plan_dir.exists()

    async def test_get_next_subgoal(self, learner):
        plan = await learner.create_plan("Subgoal test")
        sg = learner.get_next_subgoal(plan.id)
        assert isinstance(sg, SubGoal)
        assert sg.status == "pending"

    async def test_get_next_subgoal_all_done(self, learner, plans_dir):
        plan = await learner.create_plan("All done test")
        # Mark all sub_goals as passed
        for sg in plan.sub_goals:
            sg.status = "passed"
        plan.save(str(plans_dir))
        sg = learner.get_next_subgoal(plan.id)
        assert sg is None

    async def test_has_active_plans_false(self, learner):
        assert learner.has_active_plans() is False

    async def test_has_active_plans_true(self, learner):
        await learner.create_plan("Active plan")
        assert learner.has_active_plans() is True

    async def test_create_plan_with_seeds(self, learner, plans_dir):
        seeds = [
            SeedSource(content_type="url", value="https://example.com/intro"),
            SeedSource(content_type="hint", value="Focus on practical examples"),
        ]
        plan = await learner.create_plan("Seeded plan", seed_sources=seeds)
        assert len(plan.seed_sources) == 2
        assert plan.seed_sources[0].value == "https://example.com/intro"
        # Verify persisted
        loaded = learner.get_plan(plan.id)
        assert loaded is not None
        assert len(loaded.seed_sources) == 2
