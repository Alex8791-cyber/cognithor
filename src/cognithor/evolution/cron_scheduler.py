"""EvolutionScheduler -- scheduled evolution tasks with JSON persistence.

Provides cron-like scheduling for recurring learning tasks such as
periodic web-search reviews, knowledge freshness checks, and RAG
re-ingestion cycles.  Persistence is file-based (JSON) so no extra
database dependencies are needed beyond what the evolution engine
already uses.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "EvolutionScheduler",
    "ScheduledTask",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ScheduledTask:
    """A recurring or one-shot evolution task."""

    name: str
    description: str
    cron_expression: str  # 5-field cron (minute hour dom month dow)
    action: str = "research"  # research | ingest | retest | scan
    task_data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    enabled: bool = True
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "cron_expression": self.cron_expression,
            "action": self.action,
            "task_data": self.task_data,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScheduledTask:
        return cls(
            id=d.get("id", _new_id()),
            name=d["name"],
            description=d.get("description", ""),
            cron_expression=d["cron_expression"],
            action=d.get("action", "research"),
            task_data=d.get("task_data", {}),
            enabled=d.get("enabled", True),
            last_run=d.get("last_run"),
            next_run=d.get("next_run"),
            run_count=d.get("run_count", 0),
            created_at=d.get("created_at", _now_iso()),
        )


# ---------------------------------------------------------------------------
# Simple cron field matcher
# ---------------------------------------------------------------------------


def _cron_field_matches(field_expr: str, value: int) -> bool:
    """Check whether a single cron field expression matches *value*.

    Supports: ``*``, single number, comma-separated, ranges (``1-5``),
    and step values (``*/10``).
    """
    if field_expr == "*":
        return True
    # Step values: */N
    if field_expr.startswith("*/"):
        try:
            step = int(field_expr[2:])
            return step > 0 and value % step == 0
        except ValueError:
            return False
    # Comma-separated
    for part in field_expr.split(","):
        part = part.strip()
        if "-" in part:
            try:
                lo, hi = part.split("-", 1)
                if int(lo) <= value <= int(hi):
                    return True
            except ValueError:
                pass
        else:
            try:
                if int(part) == value:
                    return True
            except ValueError:
                pass
    return False


def _cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Check whether *cron_expr* (5-field) matches the given *dt*."""
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return False
    minute, hour, dom, month, dow = parts
    return (
        _cron_field_matches(minute, dt.minute)
        and _cron_field_matches(hour, dt.hour)
        and _cron_field_matches(dom, dt.day)
        and _cron_field_matches(month, dt.month)
        and _cron_field_matches(dow, dt.weekday())  # 0=Mon in Python
    )


# ---------------------------------------------------------------------------
# EvolutionScheduler
# ---------------------------------------------------------------------------


class EvolutionScheduler:
    """File-backed scheduler for recurring evolution tasks.

    Persistence: ``~/.cognithor/data/evolution_schedule.json``.
    """

    def __init__(self, schedule_path: str | Path | None = None) -> None:
        if schedule_path is None:
            schedule_path = Path.home() / ".cognithor" / "data" / "evolution_schedule.json"
        self._path = Path(schedule_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, ScheduledTask] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule_task(self, task: ScheduledTask) -> None:
        """Register a new scheduled task (or update an existing one by id)."""
        self._tasks[task.id] = task
        self._persist()
        log.info(
            "evolution_task_scheduled",
            task_id=task.id[:8],
            name=task.name,
            cron=task.cron_expression,
        )

    def get_due_tasks(self, now: datetime | None = None) -> list[ScheduledTask]:
        """Return all enabled tasks whose cron expression matches *now*.

        Tasks that were already run in the current minute are excluded.
        """
        if now is None:
            now = datetime.now(UTC)

        due: list[ScheduledTask] = []
        current_minute = now.replace(second=0, microsecond=0).isoformat()

        for task in self._tasks.values():
            if not task.enabled:
                continue
            if task.last_run and task.last_run >= current_minute:
                continue  # already ran this minute
            if _cron_matches(task.cron_expression, now):
                due.append(task)

        return due

    def mark_completed(self, task_id: str) -> bool:
        """Mark a task as completed for the current run."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.last_run = _now_iso()
        task.run_count += 1
        self._persist()
        log.info(
            "evolution_task_completed",
            task_id=task_id[:8],
            run_count=task.run_count,
        )
        return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._persist()
            return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        """Return all registered tasks."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> ScheduledTask | None:
        return self._tasks.get(task_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        data = [t.to_dict() for t in self._tasks.values()]
        content = json.dumps(data, indent=2, ensure_ascii=False)
        self._path.write_text(content, encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for item in data:
                task = ScheduledTask.from_dict(item)
                self._tasks[task.id] = task
            log.info("evolution_schedule_loaded", tasks=len(self._tasks))
        except Exception:
            log.debug("evolution_schedule_load_failed", exc_info=True)
