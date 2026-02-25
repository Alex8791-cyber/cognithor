"""Live-Monitoring: Echtzeit-Überwachung des Jarvis-Systems.

Stellt bereit:
  - EventBus: Publish/Subscribe für System-Events
  - MetricCollector: Zeitreihen-basierte Metriken (CPU, RAM, Tokens, Latenz)
  - AuditTrailViewer: Durchsuchbarer Audit-Log mit Retention
  - HeartbeatMonitor: Status-Historie der Heartbeat-Ausführungen
  - SSE-Stream: Server-Sent-Events für Live-Dashboard

Bibel-Referenz: §15 (Monitoring & Observability)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Event-System
# ============================================================================


class EventType(Enum):
    """Kategorien von System-Events."""

    # Gateway
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_RESPONDED = "message_responded"
    CHANNEL_CONNECTED = "channel_connected"
    CHANNEL_DISCONNECTED = "channel_disconnected"

    # Agent
    AGENT_SELECTED = "agent_selected"
    AGENT_DELEGATED = "agent_delegated"
    TOOL_EXECUTED = "tool_executed"
    TOOL_BLOCKED = "tool_blocked"

    # Heartbeat
    HEARTBEAT_STARTED = "heartbeat_started"
    HEARTBEAT_COMPLETED = "heartbeat_completed"
    HEARTBEAT_FAILED = "heartbeat_failed"

    # Security
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    CREDENTIAL_ACCESSED = "credential_accessed"

    # Skill
    SKILL_INSTALLED = "skill_installed"
    SKILL_FAILED = "skill_failed"
    SKILL_PUBLISHED = "skill_published"

    # System
    CONFIG_CHANGED = "config_changed"
    ERROR = "error"
    WARNING = "warning"
    METRIC = "metric"


@dataclass
class SystemEvent:
    """Ein System-Event mit Zeitstempel und Kontext."""

    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = ""  # Modul/Component
    agent_id: str = ""
    user_id: str = ""
    session_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data": self.data,
            "severity": self.severity,
        }

    def to_sse(self) -> str:
        """Formatiert als Server-Sent-Event."""
        import json

        data_str = json.dumps(self.to_dict(), ensure_ascii=False)
        return f"event: {self.event_type.value}\ndata: {data_str}\n\n"


EventHandler = Callable[[SystemEvent], None]


class EventBus:
    """Publish/Subscribe Event-Bus für System-Events.

    Ermöglicht Echtzeit-Benachrichtigungen für das Monitoring-Dashboard.
    Unterstützt synchrone und asynchrone Handler.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._handlers: dict[EventType | None, list[EventHandler]] = {}
        self._async_handlers: dict[EventType | None, list[Any]] = {}
        self._history: deque[SystemEvent] = deque(maxlen=max_history)
        self._sse_queues: list[asyncio.Queue[SystemEvent]] = []

    def subscribe(
        self,
        event_type: EventType | None = None,
        handler: EventHandler | None = None,
    ) -> None:
        """Registriert einen Handler. event_type=None → alle Events."""
        if handler:
            self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_async(
        self,
        event_type: EventType | None = None,
        handler: Any | None = None,
    ) -> None:
        """Registriert einen async Handler."""
        if handler:
            self._async_handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: SystemEvent) -> None:
        """Publiziert ein Event an alle passenden Handler."""
        self._history.append(event)

        # Sync-Handler
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as exc:
                log.warning("event_handler_error", error=str(exc))

        # Wildcard-Handler (None = alle)
        for handler in self._handlers.get(None, []):
            try:
                handler(event)
            except Exception as exc:
                log.warning("event_handler_error", error=str(exc))

        # Async-Handler
        for handler in self._async_handlers.get(event.event_type, []):
            try:
                asyncio.ensure_future(handler(event))
            except Exception as exc:
                log.warning("async_event_handler_error", error=str(exc))
        for handler in self._async_handlers.get(None, []):
            try:
                asyncio.ensure_future(handler(event))
            except Exception as exc:
                log.warning("async_event_handler_error", error=str(exc))

        # SSE-Queues füttern -- volle Queues als tot betrachten und entfernen
        dead_queues: list[asyncio.Queue[SystemEvent]] = []
        for queue in self._sse_queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead_queues.append(queue)
                log.debug("sse_queue_removed_full", event_type=event.event_type)
        for q in dead_queues:
            self._sse_queues.remove(q)

    def create_sse_stream(self) -> asyncio.Queue[SystemEvent]:
        """Erstellt eine SSE-Queue für Live-Streaming."""
        queue: asyncio.Queue[SystemEvent] = asyncio.Queue(maxsize=100)
        self._sse_queues.append(queue)
        return queue

    def remove_sse_stream(self, queue: asyncio.Queue[SystemEvent]) -> None:
        """Entfernt eine SSE-Queue."""
        if queue in self._sse_queues:
            self._sse_queues.remove(queue)

    def recent_events(
        self,
        n: int = 50,
        event_type: EventType | None = None,
        severity: str = "",
    ) -> list[SystemEvent]:
        """Gibt die letzten Events zurück, optional gefiltert."""
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]
        return events[-n:]

    @property
    def event_count(self) -> int:
        return len(self._history)

    @property
    def sse_consumer_count(self) -> int:
        return len(self._sse_queues)


# ============================================================================
# Metriken
# ============================================================================


@dataclass
class MetricPoint:
    """Ein einzelner Metrik-Datenpunkt."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Sammelt Zeitreihen-Metriken für das Dashboard.

    Unterstützt:
      - Gauge (aktueller Wert, z.B. RAM-Nutzung)
      - Counter (kumulative Werte, z.B. Nachrichten)
      - Histogram (Verteilung, z.B. Latenz)
    """

    def __init__(self, max_points_per_metric: int = 500) -> None:
        self._gauges: dict[str, float] = {}
        self._counters: dict[str, float] = {}
        self._history: dict[str, deque[MetricPoint]] = {}
        self._max_points = max_points_per_metric

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Setzt einen Gauge-Wert."""
        self._gauges[name] = value
        self._record(name, value, labels)

    def increment(self, name: str, delta: float = 1.0, **labels: str) -> None:
        """Inkrementiert einen Counter."""
        current = self._counters.get(name, 0.0)
        self._counters[name] = current + delta
        self._record(name, self._counters[name], labels)

    def _record(self, name: str, value: float, labels: dict[str, str]) -> None:
        if name not in self._history:
            self._history[name] = deque(maxlen=self._max_points)
        self._history[name].append(MetricPoint(name=name, value=value, labels=labels))

    def get_gauge(self, name: str) -> float:
        return self._gauges.get(name, 0.0)

    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def get_history(self, name: str, last_n: int = 60) -> list[dict[str, Any]]:
        """Gibt Zeitreihe einer Metrik zurück."""
        points = list(self._history.get(name, []))
        return [
            {"value": p.value, "timestamp": p.timestamp, "labels": p.labels}
            for p in points[-last_n:]
        ]

    def snapshot(self) -> dict[str, Any]:
        """Aktueller Zustand aller Metriken."""
        return {
            "gauges": dict(self._gauges),
            "counters": dict(self._counters),
            "series_count": len(self._history),
            "total_points": sum(len(v) for v in self._history.values()),
        }

    def all_metric_names(self) -> list[str]:
        return sorted(set(list(self._gauges.keys()) + list(self._counters.keys())))


# ============================================================================
# Audit-Trail
# ============================================================================


@dataclass
class AuditEntry:
    """Ein Eintrag im Audit-Trail."""

    timestamp: datetime
    action: str
    actor: str  # user_id oder agent_id oder "system"
    target: str  # Was wurde bearbeitet
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "actor": self.actor,
            "target": self.target,
            "details": self.details,
            "severity": self.severity,
            "session_id": self.session_id,
        }


class AuditTrailViewer:
    """Durchsuchbarer Audit-Trail mit Retention.

    Speichert Security-relevante Aktionen und ermöglicht
    Suche nach Zeitraum, Aktor, Aktion und Schweregrad.
    """

    def __init__(self, max_entries: int = 5000) -> None:
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)

    def record(
        self,
        action: str,
        actor: str,
        target: str,
        details: dict[str, Any] | None = None,
        severity: str = "info",
        session_id: str = "",
    ) -> AuditEntry:
        """Zeichnet eine Audit-Aktion auf."""
        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            action=action,
            actor=actor,
            target=target,
            details=details or {},
            severity=severity,
            session_id=session_id,
        )
        self._entries.append(entry)
        return entry

    def search(
        self,
        *,
        action: str = "",
        actor: str = "",
        severity: str = "",
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Durchsucht den Audit-Trail."""
        results = list(self._entries)

        if action:
            results = [e for e in results if action.lower() in e.action.lower()]
        if actor:
            results = [e for e in results if actor.lower() in e.actor.lower()]
        if severity:
            results = [e for e in results if e.severity == severity]
        if since:
            results = [e for e in results if e.timestamp >= since]
        if until:
            results = [e for e in results if e.timestamp <= until]

        return results[-limit:]

    def recent(self, n: int = 50) -> list[AuditEntry]:
        return list(self._entries)[-n:]

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def severity_counts(self) -> dict[str, int]:
        """Zählt Einträge pro Schweregrad."""
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.severity] = counts.get(entry.severity, 0) + 1
        return counts


# ============================================================================
# Heartbeat-Monitor
# ============================================================================


@dataclass
class HeartbeatRun:
    """Ergebnis einer Heartbeat-Ausführung."""

    run_id: int
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    channel: str = ""
    tasks_found: int = 0
    tasks_executed: int = 0
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "channel": self.channel,
            "tasks_found": self.tasks_found,
            "tasks_executed": self.tasks_executed,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class HeartbeatMonitor:
    """Überwacht Heartbeat-Ausführungen mit Historie.

    Tracks:
      - Letzte N Heartbeat-Runs
      - Erfolgs-/Fehlerrate
      - Durchschnittliche Dauer
      - Nächster geplanter Run
    """

    def __init__(self, max_history: int = 200) -> None:
        self._runs: deque[HeartbeatRun] = deque(maxlen=max_history)
        self._run_counter = 0
        self._next_scheduled: datetime | None = None
        self._enabled = False
        self._interval_minutes = 30

    def start_run(self, channel: str = "") -> HeartbeatRun:
        """Markiert den Start eines Heartbeat-Runs."""
        self._run_counter += 1
        run = HeartbeatRun(
            run_id=self._run_counter,
            started_at=datetime.now(UTC),
            channel=channel,
        )
        self._runs.append(run)
        return run

    def complete_run(
        self,
        run: HeartbeatRun,
        *,
        success: bool = True,
        tasks_found: int = 0,
        tasks_executed: int = 0,
        error: str = "",
    ) -> None:
        """Markiert das Ende eines Heartbeat-Runs."""
        run.completed_at = datetime.now(UTC)
        run.success = success
        run.tasks_found = tasks_found
        run.tasks_executed = tasks_executed
        run.error = error
        run.duration_ms = int(
            (run.completed_at - run.started_at).total_seconds() * 1000
        )

    def set_schedule(self, enabled: bool, interval_minutes: int = 30) -> None:
        self._enabled = enabled
        self._interval_minutes = interval_minutes

    def set_next_scheduled(self, next_run: datetime) -> None:
        self._next_scheduled = next_run

    def last_run(self) -> HeartbeatRun | None:
        return self._runs[-1] if self._runs else None

    def recent_runs(self, n: int = 20) -> list[HeartbeatRun]:
        return list(self._runs)[-n:]

    def stats(self) -> dict[str, Any]:
        """Heartbeat-Statistiken."""
        runs = list(self._runs)
        if not runs:
            return {
                "total_runs": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0,
                "enabled": self._enabled,
                "interval_minutes": self._interval_minutes,
                "next_scheduled": None,
                "last_run": None,
            }

        successes = sum(1 for r in runs if r.success)
        durations = [r.duration_ms for r in runs if r.completed_at]

        return {
            "total_runs": len(runs),
            "success_rate": round(successes / len(runs) * 100, 1),
            "avg_duration_ms": round(sum(durations) / len(durations)) if durations else 0,
            "enabled": self._enabled,
            "interval_minutes": self._interval_minutes,
            "next_scheduled": self._next_scheduled.isoformat() if self._next_scheduled else None,
            "last_run": runs[-1].to_dict() if runs else None,
        }


# ============================================================================
# MonitoringHub: Zentrale Instanz
# ============================================================================


class MonitoringHub:
    """Zentrale Monitoring-Instanz die alle Subsysteme verbindet.

    Stellt eine einzige Anlaufstelle für das Dashboard bereit.
    Wird einmal erstellt und system-weit genutzt.
    """

    def __init__(self) -> None:
        self.events = EventBus()
        self.metrics = MetricCollector()
        self.audit = AuditTrailViewer()
        self.heartbeat = HeartbeatMonitor()

    def emit(
        self,
        event_type: EventType,
        source: str = "",
        agent_id: str = "",
        user_id: str = "",
        session_id: str = "",
        severity: str = "info",
        **data: Any,
    ) -> SystemEvent:
        """Kurzform: Event publizieren + Metrik-Counter inkrementieren."""
        event = SystemEvent(
            event_type=event_type,
            source=source,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            data=data,
            severity=severity,
        )
        self.events.publish(event)
        self.metrics.increment(f"events.{event_type.value}")

        # Severity-Counter
        if severity in ("warning", "error", "critical"):
            self.metrics.increment(f"severity.{severity}")

        return event

    def dashboard_snapshot(self) -> dict[str, Any]:
        """Komplett-Snapshot für das Dashboard."""
        return {
            "events": {
                "total": self.events.event_count,
                "recent": [e.to_dict() for e in self.events.recent_events(20)],
                "sse_consumers": self.events.sse_consumer_count,
            },
            "metrics": self.metrics.snapshot(),
            "audit": {
                "total": self.audit.entry_count,
                "recent": [e.to_dict() for e in self.audit.recent(10)],
                "severity_counts": self.audit.severity_counts(),
            },
            "heartbeat": self.heartbeat.stats(),
        }
