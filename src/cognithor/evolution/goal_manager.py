"""ATL GoalManager — structured goal tracking with YAML persistence."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — used at runtime in __init__
from typing import Any

import yaml  # type: ignore[import-untyped]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Goal:
    """A single learning / evolution goal."""

    title: str = ""
    description: str = ""
    priority: int = 3
    source: str = "user"  # user | self | evolution | reflection
    id: str = ""
    status: str = "active"  # active | paused | completed | abandoned
    created_at: str = ""
    updated_at: str = ""
    deadline: str | None = None
    progress: float = 0.0
    sub_goals: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at


class GoalManager:
    """Manages goals with YAML persistence."""

    def __init__(self, goals_path: Path) -> None:
        self._path = goals_path
        self._goals: dict[str, Goal] = {}
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
        except Exception:
            return  # Corrupted YAML — start fresh
        if not data or not data.get("goals"):
            return
        _fields = set(Goal.__dataclass_fields__)
        for entry in data["goals"]:
            try:
                cleaned = {k: v for k, v in entry.items() if k in _fields}
                goal = Goal(**cleaned)
                self._goals[goal.id] = goal
            except Exception:
                pass  # Skip malformed entries

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"goals": [asdict(g) for g in self._goals.values()]}
        self._path.write_text(
            yaml.safe_dump(payload, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    # -- queries --------------------------------------------------------------

    def active_goals(self) -> list[Goal]:
        """Return active goals sorted by priority (ascending = highest first)."""
        return sorted(
            (g for g in self._goals.values() if g.status == "active"),
            key=lambda g: g.priority,
        )

    def get_goal(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    # -- mutations ------------------------------------------------------------

    def add_goal(self, goal: Goal) -> None:
        if not goal.id:
            goal.id = f"g_{uuid.uuid4().hex[:8]}"
        if goal.id in self._goals:
            raise ValueError(f"Goal '{goal.id}' already exists")
        self._goals[goal.id] = goal
        self._save()

    def update_progress(self, goal_id: str, delta: float, note: str) -> None:
        goal = self._goals[goal_id]
        goal.progress = max(0.0, min(1.0, goal.progress + delta))
        goal.updated_at = _now_iso()
        self._save()

    def complete_goal(self, goal_id: str) -> None:
        goal = self._goals[goal_id]
        goal.status = "completed"
        goal.progress = 1.0
        goal.updated_at = _now_iso()
        self._save()

    def pause_goal(self, goal_id: str) -> None:
        goal = self._goals[goal_id]
        goal.status = "paused"
        goal.updated_at = _now_iso()
        self._save()

    def resume_goal(self, goal_id: str) -> None:
        goal = self._goals[goal_id]
        goal.status = "active"
        goal.updated_at = _now_iso()
        self._save()

    def migrate_learning_goals(self, old_goals: list[Any]) -> None:
        """Convert plain string or dict goals into structured Goal objects."""
        for entry in old_goals:
            if isinstance(entry, dict):
                self.add_goal(
                    Goal(
                        id=entry.get("id", ""),
                        title=str(entry.get("title", "")),
                        description=str(entry.get("description", "")),
                        priority=entry.get("priority", 3),
                        status=entry.get("status", "active"),
                        progress=entry.get("progress", 0.0),
                        source="user",
                    )
                )
            elif isinstance(entry, str):
                self.add_goal(
                    Goal(
                        title=entry,
                        description=entry,
                        priority=3,
                        source="user",
                    )
                )
