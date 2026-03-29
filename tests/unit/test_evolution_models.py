"""Tests for evolution Phase 5 data models."""

from __future__ import annotations

import json
import os

import pytest

from jarvis.evolution.models import (
    LearningPlan,
    QualityQuestion,
    ScheduleSpec,
    SeedSource,
    SourceSpec,
    SubGoal,
)


class TestSubGoal:
    def test_create_minimal(self):
        sg = SubGoal(title="Learn Python basics", description="Cover syntax and types")
        assert sg.title == "Learn Python basics"
        assert sg.description == "Cover syntax and types"
        assert sg.status == "pending"
        assert len(sg.id) == 16
        assert sg.sources_fetched == 0
        assert sg.chunks_created == 0
        assert sg.quality_questions == []

    def test_to_dict_roundtrip(self):
        qq = QualityQuestion(
            question="What is a list?",
            expected_answer="An ordered collection",
        )
        sg = SubGoal(
            title="Collections",
            description="Learn about collections",
            priority=2,
            quality_questions=[qq],
        )
        d = sg.to_dict()
        restored = SubGoal.from_dict(d)
        assert restored.title == sg.title
        assert restored.id == sg.id
        assert restored.priority == sg.priority
        assert len(restored.quality_questions) == 1
        assert restored.quality_questions[0].question == "What is a list?"


class TestSourceSpec:
    def test_create(self):
        s = SourceSpec(url="https://example.com", source_type="web", title="Example")
        assert s.url == "https://example.com"
        assert s.source_type == "web"
        assert s.priority == 5
        assert s.status == "pending"

    def test_to_dict(self):
        s = SourceSpec(url="https://example.com", source_type="web", title="Ex")
        d = s.to_dict()
        assert d["url"] == "https://example.com"
        restored = SourceSpec.from_dict(d)
        assert restored.url == s.url
        assert restored.title == s.title


class TestScheduleSpec:
    def test_create(self):
        s = ScheduleSpec(
            name="daily-refresh",
            cron_expression="0 3 * * *",
            source_url="https://example.com/feed",
            action="fetch",
            goal_id="abc123",
        )
        assert s.name == "daily-refresh"
        assert s.cron_expression == "0 3 * * *"
        assert s.action == "fetch"


class TestSeedSource:
    def test_url_seed(self):
        s = SeedSource(content_type="url", value="https://docs.python.org")
        assert s.content_type == "url"
        assert s.processed is False

    def test_file_seed(self):
        s = SeedSource(content_type="file", value="/tmp/notes.txt", title="My notes")
        assert s.content_type == "file"
        assert s.title == "My notes"

    def test_hint_seed(self):
        s = SeedSource(content_type="hint", value="Focus on async patterns")
        d = s.to_dict()
        restored = SeedSource.from_dict(d)
        assert restored.content_type == "hint"
        assert restored.value == "Focus on async patterns"


class TestQualityQuestion:
    def test_create(self):
        qq = QualityQuestion(
            question="What is polymorphism?",
            expected_answer="The ability of objects to take many forms",
        )
        assert qq.question == "What is polymorphism?"
        assert qq.score is None
        assert qq.passed is False


class TestLearningPlan:
    def test_create_empty(self):
        lp = LearningPlan(goal="Master Rust programming")
        assert lp.goal == "Master Rust programming"
        assert lp.status == "planning"
        assert lp.sub_goals == []
        assert lp.sources == []
        assert lp.expansions == 0
        assert len(lp.id) == 32  # full uuid hex

    def test_goal_slug_generation(self):
        lp = LearningPlan(goal="Learn Python 3.12 & Async/Await!")
        assert lp.goal_slug == "learn-python-312-asyncawait"
        # slug should be lowercase, no special chars
        assert lp.goal_slug == lp.goal_slug.lower()
        assert " " not in lp.goal_slug

    def test_save_load_roundtrip(self, tmp_path):
        qq = QualityQuestion(question="Q?", expected_answer="A")
        sg = SubGoal(title="Sub1", description="Desc", quality_questions=[qq])
        src = SourceSpec(url="https://ex.com", source_type="web", title="Ex")
        seed = SeedSource(content_type="hint", value="focus on X")
        sched = ScheduleSpec(
            name="nightly",
            cron_expression="0 0 * * *",
            source_url="https://ex.com",
            action="fetch",
            goal_id="g1",
        )
        lp = LearningPlan(
            goal="Test goal",
            sub_goals=[sg],
            sources=[src],
            seed_sources=[seed],
            schedules=[sched],
        )
        lp.save(str(tmp_path))

        plan_dir = os.path.join(str(tmp_path), lp.id)
        loaded = LearningPlan.load(plan_dir)
        assert loaded.id == lp.id
        assert loaded.goal == lp.goal
        assert len(loaded.sub_goals) == 1
        assert loaded.sub_goals[0].title == "Sub1"
        assert len(loaded.sub_goals[0].quality_questions) == 1
        assert len(loaded.sources) == 1
        assert len(loaded.seed_sources) == 1
        assert len(loaded.schedules) == 1

    def test_save_creates_directory_structure(self, tmp_path):
        lp = LearningPlan(goal="Structure test")
        lp.save(str(tmp_path))
        plan_dir = os.path.join(str(tmp_path), lp.id)
        assert os.path.isfile(os.path.join(plan_dir, "plan.json"))
        assert os.path.isdir(os.path.join(plan_dir, "subgoals"))
        assert os.path.isdir(os.path.join(plan_dir, "quality"))
        assert os.path.isdir(os.path.join(plan_dir, "uploads"))
        assert os.path.isdir(os.path.join(plan_dir, "checkpoints"))

    def test_to_summary_dict(self):
        sg1 = SubGoal(title="A", description="a", status="passed")
        sg2 = SubGoal(title="B", description="b", status="pending")
        sg3 = SubGoal(title="C", description="c", status="passed")
        lp = LearningPlan(
            goal="Summary test",
            sub_goals=[sg1, sg2, sg3],
            coverage_score=0.8,
            quality_score=0.9,
        )
        summary = lp.to_summary_dict()
        assert summary["goal"] == "Summary test"
        assert summary["sub_goals_total"] == 3
        assert summary["sub_goals_done"] == 2
        assert summary["coverage_score"] == 0.8
        assert summary["quality_score"] == 0.9
        assert "id" in summary
        assert "created_at" in summary
        assert "updated_at" in summary
