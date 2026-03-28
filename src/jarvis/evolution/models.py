"""Phase 5 data models — LearningPlan, SubGoal, SourceSpec and related types."""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


def _new_id() -> str:
    """Return a 16-char hex identifier."""
    return uuid.uuid4().hex[:16]


def _slugify(text: str) -> str:
    """Lowercase, strip non-alphanum, replace spaces with dashes, max 60 chars."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text).strip("-")
    return text[:60]


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QualityQuestion:
    question: str
    expected_answer: str
    actual_answer: Optional[str] = None
    score: Optional[float] = None
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "actual_answer": self.actual_answer,
            "score": self.score,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QualityQuestion:
        return cls(
            question=d["question"],
            expected_answer=d["expected_answer"],
            actual_answer=d.get("actual_answer"),
            score=d.get("score"),
            passed=d.get("passed", False),
        )


@dataclass
class SeedSource:
    content_type: str  # "url" | "file" | "hint"
    value: str
    title: Optional[str] = None
    processed: bool = False

    def to_dict(self) -> dict:
        return {
            "content_type": self.content_type,
            "value": self.value,
            "title": self.title,
            "processed": self.processed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SeedSource:
        return cls(
            content_type=d["content_type"],
            value=d["value"],
            title=d.get("title"),
            processed=d.get("processed", False),
        )


@dataclass
class SourceSpec:
    url: str
    source_type: str
    title: Optional[str] = None
    fetch_strategy: Optional[str] = None
    update_frequency: Optional[str] = None
    priority: int = 5
    max_pages: Optional[int] = None
    last_fetched: Optional[str] = None
    pages_fetched: int = 0
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "source_type": self.source_type,
            "title": self.title,
            "fetch_strategy": self.fetch_strategy,
            "update_frequency": self.update_frequency,
            "priority": self.priority,
            "max_pages": self.max_pages,
            "last_fetched": self.last_fetched,
            "pages_fetched": self.pages_fetched,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SourceSpec:
        return cls(
            url=d["url"],
            source_type=d["source_type"],
            title=d.get("title"),
            fetch_strategy=d.get("fetch_strategy"),
            update_frequency=d.get("update_frequency"),
            priority=d.get("priority", 5),
            max_pages=d.get("max_pages"),
            last_fetched=d.get("last_fetched"),
            pages_fetched=d.get("pages_fetched", 0),
            status=d.get("status", "pending"),
        )


@dataclass
class ScheduleSpec:
    name: str
    cron_expression: str
    source_url: Optional[str] = None
    action: str = "fetch"
    goal_id: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cron_expression": self.cron_expression,
            "source_url": self.source_url,
            "action": self.action,
            "goal_id": self.goal_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScheduleSpec:
        return cls(
            name=d["name"],
            cron_expression=d["cron_expression"],
            source_url=d.get("source_url"),
            action=d.get("action", "fetch"),
            goal_id=d.get("goal_id"),
            description=d.get("description"),
        )


@dataclass
class SubGoal:
    title: str
    description: str
    id: str = field(default_factory=_new_id)
    status: str = "pending"
    priority: int = 5
    parent_goal_id: Optional[str] = None
    sources_fetched: int = 0
    chunks_created: int = 0
    entities_created: int = 0
    vault_entries: int = 0
    skills_generated: int = 0
    cron_jobs_created: int = 0
    coverage_score: Optional[float] = None
    quality_score: Optional[float] = None
    quality_questions: List[QualityQuestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "parent_goal_id": self.parent_goal_id,
            "sources_fetched": self.sources_fetched,
            "chunks_created": self.chunks_created,
            "entities_created": self.entities_created,
            "vault_entries": self.vault_entries,
            "skills_generated": self.skills_generated,
            "cron_jobs_created": self.cron_jobs_created,
            "coverage_score": self.coverage_score,
            "quality_score": self.quality_score,
            "quality_questions": [qq.to_dict() for qq in self.quality_questions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SubGoal:
        return cls(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            status=d.get("status", "pending"),
            priority=d.get("priority", 5),
            parent_goal_id=d.get("parent_goal_id"),
            sources_fetched=d.get("sources_fetched", 0),
            chunks_created=d.get("chunks_created", 0),
            entities_created=d.get("entities_created", 0),
            vault_entries=d.get("vault_entries", 0),
            skills_generated=d.get("skills_generated", 0),
            cron_jobs_created=d.get("cron_jobs_created", 0),
            coverage_score=d.get("coverage_score"),
            quality_score=d.get("quality_score"),
            quality_questions=[
                QualityQuestion.from_dict(q)
                for q in d.get("quality_questions", [])
            ],
        )


@dataclass
class LearningPlan:
    goal: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    goal_slug: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    status: str = "planning"
    sub_goals: List[SubGoal] = field(default_factory=list)
    sources: List[SourceSpec] = field(default_factory=list)
    schedules: List[ScheduleSpec] = field(default_factory=list)
    seed_sources: List[SeedSource] = field(default_factory=list)
    coverage_score: Optional[float] = None
    quality_score: Optional[float] = None
    total_chunks_indexed: int = 0
    total_entities_created: int = 0
    total_vault_entries: int = 0
    expansions: int = 0

    def __post_init__(self) -> None:
        if not self.goal_slug:
            self.goal_slug = _slugify(self.goal)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "goal_slug": self.goal_slug,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
            "sources": [s.to_dict() for s in self.sources],
            "schedules": [s.to_dict() for s in self.schedules],
            "seed_sources": [s.to_dict() for s in self.seed_sources],
            "coverage_score": self.coverage_score,
            "quality_score": self.quality_score,
            "total_chunks_indexed": self.total_chunks_indexed,
            "total_entities_created": self.total_entities_created,
            "total_vault_entries": self.total_vault_entries,
            "expansions": self.expansions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LearningPlan:
        return cls(
            id=d["id"],
            goal=d["goal"],
            goal_slug=d.get("goal_slug", ""),
            created_at=d.get("created_at", _now_iso()),
            updated_at=d.get("updated_at", _now_iso()),
            status=d.get("status", "planning"),
            sub_goals=[SubGoal.from_dict(sg) for sg in d.get("sub_goals", [])],
            sources=[SourceSpec.from_dict(s) for s in d.get("sources", [])],
            schedules=[ScheduleSpec.from_dict(s) for s in d.get("schedules", [])],
            seed_sources=[SeedSource.from_dict(s) for s in d.get("seed_sources", [])],
            coverage_score=d.get("coverage_score"),
            quality_score=d.get("quality_score"),
            total_chunks_indexed=d.get("total_chunks_indexed", 0),
            total_entities_created=d.get("total_entities_created", 0),
            total_vault_entries=d.get("total_vault_entries", 0),
            expansions=d.get("expansions", 0),
        )

    def to_summary_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "goal_slug": self.goal_slug,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sub_goals_total": len(self.sub_goals),
            "sub_goals_done": sum(1 for sg in self.sub_goals if sg.status == "passed"),
            "coverage_score": self.coverage_score,
            "quality_score": self.quality_score,
        }

    def save(self, base_dir: str) -> str:
        """Save plan to {base_dir}/{id}/plan.json, create subdirectories."""
        plan_dir = os.path.join(base_dir, self.id)
        os.makedirs(plan_dir, exist_ok=True)
        for subdir in ("subgoals", "quality", "uploads", "checkpoints"):
            os.makedirs(os.path.join(plan_dir, subdir), exist_ok=True)
        self.updated_at = _now_iso()
        plan_path = os.path.join(plan_dir, "plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return plan_dir

    @classmethod
    def load(cls, plan_dir: str) -> LearningPlan:
        """Load a plan from {plan_dir}/plan.json."""
        plan_path = os.path.join(plan_dir, "plan.json")
        with open(plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def list_plans(cls, base_dir: str) -> List[LearningPlan]:
        """Iterate base_dir/*/plan.json and return all found plans."""
        plans: List[LearningPlan] = []
        if not os.path.isdir(base_dir):
            return plans
        for entry in os.listdir(base_dir):
            plan_path = os.path.join(base_dir, entry, "plan.json")
            if os.path.isfile(plan_path):
                plans.append(cls.load(os.path.join(base_dir, entry)))
        return plans
