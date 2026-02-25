"""Per-Agent-Heartbeat: Agent-spezifische Aufgabenplanung.

Jeder Agent bekommt einen eigenen Heartbeat-Kontext:
  - Eigene Tasks und Trigger
  - Eigenes Intervall und eigene Konfiguration
  - Isolation: Agent A's Tasks beeinflussen Agent B nicht
  - Dashboard-Übersicht zeigt alle Agents und ihre Tasks

Bibel-Referenz: §10 (Cron-Engine & Proaktive Autonomie)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentTask:
    """Eine Heartbeat-Aufgabe für einen spezifischen Agenten."""

    task_id: str
    agent_id: str
    name: str
    description: str = ""
    cron_expression: str = ""  # Leer = wird manuell getriggert
    enabled: bool = True
    last_run: datetime | None = None
    last_status: TaskStatus = TaskStatus.PENDING
    last_error: str = ""
    run_count: int = 0
    fail_count: int = 0
    avg_duration_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "cron_expression": self.cron_expression,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status.value,
            "last_error": self.last_error,
            "run_count": self.run_count,
            "fail_count": self.fail_count,
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "success_rate": round(
                ((self.run_count - self.fail_count) / max(self.run_count, 1)) * 100, 1,
            ),
        }


@dataclass
class AgentHeartbeatConfig:
    """Heartbeat-Konfiguration für einen Agenten."""

    agent_id: str
    enabled: bool = True
    interval_minutes: int = 30
    channel: str = "cli"
    max_concurrent_tasks: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "enabled": self.enabled,
            "interval_minutes": self.interval_minutes,
            "channel": self.channel,
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }


@dataclass
class TaskRun:
    """Protokoll einer Task-Ausführung."""

    task_id: str
    agent_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status: TaskStatus = TaskStatus.RUNNING
    error: str = ""
    duration_ms: float = 0.0

    def complete(self, *, success: bool = True, error: str = "") -> None:
        self.completed_at = datetime.now(UTC)
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.error = error
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000


class AgentHeartbeatScheduler:
    """Verwaltet Heartbeat-Tasks pro Agent.

    Jeder Agent bekommt:
      - Eigene Task-Liste
      - Eigenes Intervall
      - Eigene Ausführungs-Historie
      - Isolierte Trigger

    Dashboard-Übersicht zeigt alle Agents zentral.
    """

    def __init__(self) -> None:
        self._configs: dict[str, AgentHeartbeatConfig] = {}
        self._tasks: dict[str, dict[str, AgentTask]] = defaultdict(dict)
        self._history: dict[str, list[TaskRun]] = defaultdict(list)
        self._max_history = 100

    # ------------------------------------------------------------------
    # Konfiguration
    # ------------------------------------------------------------------

    def configure_agent(self, config: AgentHeartbeatConfig) -> None:
        """Setzt die Heartbeat-Konfiguration für einen Agenten."""
        self._configs[config.agent_id] = config
        log.info(
            "agent_heartbeat_configured",
            agent_id=config.agent_id,
            interval=config.interval_minutes,
            enabled=config.enabled,
        )

    def get_config(self, agent_id: str) -> AgentHeartbeatConfig | None:
        return self._configs.get(agent_id)

    # ------------------------------------------------------------------
    # Task-Verwaltung
    # ------------------------------------------------------------------

    def add_task(self, task: AgentTask) -> None:
        """Fügt eine Task für einen Agenten hinzu."""
        self._tasks[task.agent_id][task.task_id] = task
        log.info("agent_task_added", agent_id=task.agent_id, task_id=task.task_id)

    def remove_task(self, agent_id: str, task_id: str) -> bool:
        """Entfernt eine Task."""
        tasks = self._tasks.get(agent_id, {})
        if task_id in tasks:
            del tasks[task_id]
            return True
        return False

    def get_task(self, agent_id: str, task_id: str) -> AgentTask | None:
        return self._tasks.get(agent_id, {}).get(task_id)

    def agent_tasks(self, agent_id: str) -> list[AgentTask]:
        """Alle Tasks eines Agenten."""
        return list(self._tasks.get(agent_id, {}).values())

    def enabled_tasks(self, agent_id: str) -> list[AgentTask]:
        """Nur aktivierte Tasks eines Agenten."""
        return [t for t in self.agent_tasks(agent_id) if t.enabled]

    # ------------------------------------------------------------------
    # Ausführung
    # ------------------------------------------------------------------

    def start_task(self, agent_id: str, task_id: str) -> TaskRun | None:
        """Startet eine Task-Ausführung."""
        task = self.get_task(agent_id, task_id)
        if not task:
            return None
        if not task.enabled:
            return None

        run = TaskRun(task_id=task_id, agent_id=agent_id)
        task.last_status = TaskStatus.RUNNING
        return run

    def complete_task(self, run: TaskRun, *, success: bool = True, error: str = "") -> None:
        """Beendet eine Task-Ausführung."""
        run.complete(success=success, error=error)

        task = self.get_task(run.agent_id, run.task_id)
        if task:
            task.last_run = run.completed_at
            task.last_status = run.status
            task.last_error = run.error
            task.run_count += 1
            if not success:
                task.fail_count += 1
            # Rolling Average
            total = task.avg_duration_ms * (task.run_count - 1) + run.duration_ms
            task.avg_duration_ms = total / task.run_count

        # Historie
        self._history[run.agent_id].append(run)
        if len(self._history[run.agent_id]) > self._max_history:
            self._history[run.agent_id] = self._history[run.agent_id][-self._max_history:]

    # ------------------------------------------------------------------
    # Dashboard-Übersicht
    # ------------------------------------------------------------------

    def agent_summary(self, agent_id: str) -> dict[str, Any]:
        """Zusammenfassung für einen Agenten."""
        config = self._configs.get(agent_id)
        tasks = self.agent_tasks(agent_id)
        history = self._history.get(agent_id, [])

        return {
            "agent_id": agent_id,
            "config": config.to_dict() if config else None,
            "task_count": len(tasks),
            "enabled_tasks": sum(1 for t in tasks if t.enabled),
            "tasks": [t.to_dict() for t in tasks],
            "recent_runs": len(history),
            "total_runs": sum(t.run_count for t in tasks),
            "total_fails": sum(t.fail_count for t in tasks),
        }

    def global_dashboard(self) -> dict[str, Any]:
        """Globale Übersicht aller Agenten und Tasks."""
        all_agents = set(list(self._configs.keys()) + list(self._tasks.keys()))
        summaries = [self.agent_summary(aid) for aid in sorted(all_agents)]

        total_tasks = sum(s["task_count"] for s in summaries)
        total_enabled = sum(s["enabled_tasks"] for s in summaries)
        total_runs = sum(s["total_runs"] for s in summaries)
        total_fails = sum(s["total_fails"] for s in summaries)

        return {
            "agent_count": len(all_agents),
            "total_tasks": total_tasks,
            "enabled_tasks": total_enabled,
            "total_runs": total_runs,
            "total_fails": total_fails,
            "success_rate": round(
                ((total_runs - total_fails) / max(total_runs, 1)) * 100, 1,
            ),
            "agents": summaries,
        }

    @property
    def agent_count(self) -> int:
        return len(set(list(self._configs.keys()) + list(self._tasks.keys())))
