"""Proaktiver Heartbeat: Ereignisgesteuertes Task-Scheduling.

Jarvis wartet nicht nur auf User-Input, sondern handelt proaktiv:
  - E-Mail-Triage (neue Mails scannen, priorisieren, Zusammenfassung)
  - Tagesbriefings (morgens/abends automatisch zusammenstellen)
  - Kalender-Abgleich (bevorstehende Termine vorbereiten)
  - To-Do-Erinnerungen (fällige Aufgaben nachfassen)
  - Skill-Updates (neue P2P-Skills prüfen)
  - Memory-Pflege (stale Entitäten prunen, Compressor laufen lassen)

Architektur:
  EventSource → HeartbeatScheduler → TaskQueue → AgentRouter
                                                     ↓
                                                  Gatekeeper
                                                     ↓
                                                  Executor

Jede proaktive Aktion durchläuft den Gatekeeper.
Der User kann pro EventType:
  - ENABLED/DISABLED setzen
  - Intervall konfigurieren
  - Priorität festlegen
  - Genehmigungsmodus (auto/ask) wählen

Bibel-Referenz: §7.1 (Proaktive Automation)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger("jarvis.proactive.heartbeat")


# ============================================================================
# Event Types
# ============================================================================


class EventType(Enum):
    """Arten proaktiver Ereignisse."""

    EMAIL_TRIAGE = "email_triage"  # Neue E-Mails scannen
    DAILY_BRIEFING = "daily_briefing"  # Tagesbriefing erstellen
    CALENDAR_PREP = "calendar_prep"  # Bevorstehende Termine vorbereiten
    TODO_REMINDER = "todo_reminder"  # Fällige To-Dos nachfassen
    SKILL_UPDATE_CHECK = "skill_update_check"  # Neue Skills im Netzwerk
    MEMORY_MAINTENANCE = "memory_maintenance"  # Stale Entities prunen
    CUSTOM = "custom"  # Benutzerdefiniert


class ApprovalMode(Enum):
    """Genehmigungsmodus für proaktive Aktionen."""

    AUTO = "auto"  # Automatisch ausführen (nur informieren)
    ASK = "ask"  # User um Genehmigung bitten
    SILENT = "silent"  # Ausführen ohne Benachrichtigung


class TaskStatus(Enum):
    """Status einer proaktiven Aufgabe."""

    PENDING = "pending"  # In der Queue
    RUNNING = "running"  # Wird ausgeführt
    COMPLETED = "completed"  # Erfolgreich abgeschlossen
    FAILED = "failed"  # Fehlgeschlagen
    SKIPPED = "skipped"  # Übersprungen (Gatekeeper blockiert)
    AWAITING_APPROVAL = "awaiting_approval"  # Wartet auf User


# ============================================================================
# Konfiguration
# ============================================================================


@dataclass
class EventConfig:
    """Konfiguration für einen Event-Typ.

    Steuert Intervall, Priorität und Genehmigungsmodus.
    """

    event_type: EventType
    enabled: bool = True
    interval_seconds: int = 3600  # Default: 1 Stunde
    priority: int = 5  # 1 (niedrig) bis 10 (hoch)
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    agent_name: str = ""  # Ziel-Agent (leer = Standard)
    description: str = ""  # Beschreibung für User
    max_retries: int = 2
    quiet_hours_start: int = -1  # -1 = keine Ruhezeiten
    quiet_hours_end: int = -1

    @property
    def is_in_quiet_hours(self) -> bool:
        """Prüft ob aktuell Ruhezeit ist."""
        if self.quiet_hours_start < 0 or self.quiet_hours_end < 0:
            return False
        hour = datetime.now(timezone.utc).hour
        if self.quiet_hours_start <= self.quiet_hours_end:
            return self.quiet_hours_start <= hour < self.quiet_hours_end
        # Über Mitternacht (z.B. 22-06)
        return hour >= self.quiet_hours_start or hour < self.quiet_hours_end


# ============================================================================
# Task
# ============================================================================


@dataclass
class ProactiveTask:
    """Eine einzelne proaktive Aufgabe."""

    task_id: str
    event_type: EventType
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    started_at: str = ""
    completed_at: str = ""
    result: str = ""
    error: str = ""
    agent_name: str = ""
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    priority: int = 5
    retries: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Task ist in einem finalen Status."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        )

    @property
    def duration_seconds(self) -> float:
        """Dauer der Ausführung."""
        if not self.started_at or not self.completed_at:
            return 0.0
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            return (end - start).total_seconds()
        except (ValueError, TypeError):
            return 0.0


# ============================================================================
# Event Source
# ============================================================================


@dataclass
class EventTrigger:
    """Ein ausgelöstes Ereignis von einer EventSource."""

    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # Welche EventSource
    triggered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class EventSource:
    """Erkennt Ereignisse die proaktive Aktionen auslösen.

    Prüft verschiedene Quellen (simuliert für Offline-Betrieb):
      - E-Mail-Eingang (IMAP-Check)
      - Kalender-Änderungen
      - Fällige To-Dos
      - P2P-Skill-Updates
      - Zeitbasierte Trigger (Briefings)
    """

    def __init__(self) -> None:
        self._last_check: dict[EventType, float] = {}
        self._manual_triggers: deque[EventTrigger] = deque(maxlen=100)

    def check(
        self,
        configs: dict[EventType, EventConfig],
    ) -> list[EventTrigger]:
        """Prüft alle konfigurierten Quellen auf neue Ereignisse.

        Args:
            configs: Aktive Event-Konfigurationen.

        Returns:
            Liste ausgelöster Trigger.
        """
        triggers: list[EventTrigger] = []
        now = time.monotonic()

        for event_type, config in configs.items():
            if not config.enabled:
                continue
            if config.is_in_quiet_hours:
                continue

            last = self._last_check.get(event_type, float("-inf"))
            if now - last < config.interval_seconds:
                continue

            self._last_check[event_type] = now
            trigger = EventTrigger(
                event_type=event_type,
                source="heartbeat_scheduler",
            )
            triggers.append(trigger)

        # Manuelle Trigger einmischen
        while self._manual_triggers:
            triggers.append(self._manual_triggers.popleft())

        return triggers

    def inject_trigger(self, trigger: EventTrigger) -> None:
        """Injiziert ein manuelles Trigger-Ereignis."""
        self._manual_triggers.append(trigger)

    def reset_timer(self, event_type: EventType) -> None:
        """Setzt den Timer für einen Event-Typ zurück."""
        self._last_check.pop(event_type, None)


# ============================================================================
# Task Queue
# ============================================================================


class TaskQueue:
    """Priorisierte Queue für proaktive Tasks.

    Tasks werden nach Priorität (hoch zuerst) und Erstellzeit
    (ältere zuerst) sortiert abgearbeitet.
    """

    def __init__(self, max_size: int = 200) -> None:
        self._queue: deque[ProactiveTask] = deque(maxlen=max_size)
        self._counter = 0

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self._queue if t.status == TaskStatus.PENDING)

    def enqueue(self, task: ProactiveTask) -> None:
        """Fügt einen Task hinzu."""
        self._queue.append(task)

    def dequeue(self) -> ProactiveTask | None:
        """Gibt den höchst-priorisierten pending Task zurück.

        Returns:
            Nächster Task oder None.
        """
        best: ProactiveTask | None = None
        best_idx: int = -1

        for i, task in enumerate(self._queue):
            if task.status != TaskStatus.PENDING:
                continue
            if best is None or task.priority > best.priority:
                best = task
                best_idx = i
            elif task.priority == best.priority and task.created_at < best.created_at:
                best = task
                best_idx = i

        if best is not None:
            best.status = TaskStatus.RUNNING
            best.started_at = datetime.now(timezone.utc).isoformat()

        return best

    def complete(
        self,
        task_id: str,
        *,
        success: bool = True,
        result: str = "",
        error: str = "",
    ) -> bool:
        """Markiert einen Task als abgeschlossen.

        Returns:
            True wenn Task gefunden.
        """
        for task in self._queue:
            if task.task_id == task_id:
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task.result = result
                task.error = error
                return True
        return False

    def skip(self, task_id: str, reason: str = "") -> bool:
        """Markiert einen Task als übersprungen."""
        for task in self._queue:
            if task.task_id == task_id:
                task.status = TaskStatus.SKIPPED
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task.result = reason or "Vom Gatekeeper blockiert"
                return True
        return False

    def get(self, task_id: str) -> ProactiveTask | None:
        for task in self._queue:
            if task.task_id == task_id:
                return task
        return None

    def list_pending(self) -> list[ProactiveTask]:
        return [t for t in self._queue if t.status == TaskStatus.PENDING]

    def list_history(self, limit: int = 50) -> list[ProactiveTask]:
        """Abgeschlossene Tasks (neueste zuerst)."""
        terminal = [t for t in self._queue if t.is_terminal]
        terminal.sort(key=lambda t: t.completed_at, reverse=True)
        return terminal[:limit]

    def cleanup_completed(self, keep: int = 100) -> int:
        """Entfernt alte abgeschlossene Tasks.

        Returns:
            Anzahl entfernter Tasks.
        """
        terminal = [t for t in self._queue if t.is_terminal]
        if len(terminal) <= keep:
            return 0

        # Älteste entfernen
        terminal.sort(key=lambda t: t.completed_at)
        to_remove = set()
        for t in terminal[: len(terminal) - keep]:
            to_remove.add(t.task_id)

        before = len(self._queue)
        self._queue = deque(
            (t for t in self._queue if t.task_id not in to_remove),
            maxlen=self._queue.maxlen,
        )
        return before - len(self._queue)


# ============================================================================
# Heartbeat Scheduler
# ============================================================================


# Type für Task-Handler (async Callable)
TaskHandler = Callable[[ProactiveTask], Coroutine[Any, Any, str]]


class HeartbeatScheduler:
    """Orchestriert proaktive Aufgaben.

    Der Heartbeat prüft regelmäßig auf Ereignisse, erstellt
    Tasks und delegiert sie an die registrierten Handler.

    Usage:
        scheduler = HeartbeatScheduler()

        # Konfigurieren
        scheduler.configure(EventType.EMAIL_TRIAGE, enabled=True, interval=300)
        scheduler.configure(EventType.DAILY_BRIEFING, enabled=True, interval=86400)

        # Handler registrieren
        scheduler.register_handler(EventType.EMAIL_TRIAGE, email_triage_handler)

        # Heartbeat-Tick (wird vom CronEngine aufgerufen)
        await scheduler.tick()
    """

    def __init__(self) -> None:
        self._configs: dict[EventType, EventConfig] = {}
        self._handlers: dict[EventType, TaskHandler] = {}
        self._event_source = EventSource()
        self._queue = TaskQueue()
        self._task_counter = 0
        self._total_ticks = 0
        self._total_tasks_created = 0
        self._total_tasks_completed = 0
        self._total_tasks_failed = 0

        # Default-Konfigurationen
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Setzt Standard-Konfigurationen für alle Event-Typen."""
        defaults = [
            EventConfig(
                EventType.EMAIL_TRIAGE,
                interval_seconds=300,
                priority=7,
                description="Neue E-Mails scannen und priorisieren",
            ),
            EventConfig(
                EventType.DAILY_BRIEFING,
                interval_seconds=86400,
                priority=6,
                description="Tagesbriefing zusammenstellen",
                quiet_hours_start=22,
                quiet_hours_end=6,
            ),
            EventConfig(
                EventType.CALENDAR_PREP,
                interval_seconds=1800,
                priority=8,
                description="Bevorstehende Termine vorbereiten",
            ),
            EventConfig(
                EventType.TODO_REMINDER,
                interval_seconds=3600,
                priority=5,
                description="Fällige To-Dos nachfassen",
            ),
            EventConfig(
                EventType.SKILL_UPDATE_CHECK,
                interval_seconds=43200,
                priority=3,
                description="Neue Skills im P2P-Netzwerk prüfen",
            ),
            EventConfig(
                EventType.MEMORY_MAINTENANCE,
                interval_seconds=86400,
                priority=2,
                description="Stale Memory-Einträge aufräumen",
            ),
        ]
        for config in defaults:
            config.enabled = False  # Default: Alle deaktiviert
            self._configs[config.event_type] = config

    # ── Konfiguration ────────────────────────────────────────────

    def configure(
        self,
        event_type: EventType,
        *,
        enabled: bool | None = None,
        interval_seconds: int | None = None,
        priority: int | None = None,
        approval_mode: ApprovalMode | None = None,
        agent_name: str | None = None,
        quiet_hours: tuple[int, int] | None = None,
    ) -> EventConfig:
        """Konfiguriert einen Event-Typ.

        Args:
            event_type: Zu konfigurierender Event.
            enabled: Aktivieren/Deaktivieren.
            interval_seconds: Intervall in Sekunden.
            priority: Priorität (1-10).
            approval_mode: Genehmigungsmodus.
            agent_name: Ziel-Agent.
            quiet_hours: Ruhezeiten (start_hour, end_hour).

        Returns:
            Aktualisierte Konfiguration.
        """
        config = self._configs.setdefault(
            event_type, EventConfig(event_type=event_type),
        )

        if enabled is not None:
            config.enabled = enabled
        if interval_seconds is not None:
            config.interval_seconds = max(30, interval_seconds)
        if priority is not None:
            config.priority = max(1, min(10, priority))
        if approval_mode is not None:
            config.approval_mode = approval_mode
        if agent_name is not None:
            config.agent_name = agent_name
        if quiet_hours is not None:
            config.quiet_hours_start, config.quiet_hours_end = quiet_hours

        return config

    def get_config(self, event_type: EventType) -> EventConfig | None:
        return self._configs.get(event_type)

    def list_configs(self) -> list[EventConfig]:
        return list(self._configs.values())

    def enabled_configs(self) -> list[EventConfig]:
        return [c for c in self._configs.values() if c.enabled]

    # ── Handler ──────────────────────────────────────────────────

    def register_handler(
        self,
        event_type: EventType,
        handler: TaskHandler,
    ) -> None:
        """Registriert einen async Handler für einen Event-Typ.

        Handler-Signatur: async def handler(task: ProactiveTask) -> str
        """
        self._handlers[event_type] = handler

    def has_handler(self, event_type: EventType) -> bool:
        return event_type in self._handlers

    # ── Heartbeat Tick ───────────────────────────────────────────

    async def tick(self) -> list[ProactiveTask]:
        """Ein Heartbeat-Zyklus.

        1. EventSource prüfen → Trigger sammeln
        2. Trigger → Tasks erstellen
        3. Tasks aus Queue abarbeiten

        Returns:
            Liste verarbeiteter Tasks.
        """
        self._total_ticks += 1
        processed: list[ProactiveTask] = []

        # 1. Ereignisse prüfen
        triggers = self._event_source.check(self._configs)

        # 2. Tasks erstellen
        for trigger in triggers:
            config = self._configs.get(trigger.event_type)
            if not config or not config.enabled:
                continue

            self._task_counter += 1
            task = ProactiveTask(
                task_id=f"ht_{self._task_counter}",
                event_type=trigger.event_type,
                priority=config.priority,
                agent_name=config.agent_name,
                approval_mode=config.approval_mode,
                context=trigger.payload,
            )
            self._queue.enqueue(task)
            self._total_tasks_created += 1

        # 3. Tasks abarbeiten
        while True:
            task = self._queue.dequeue()
            if task is None:
                break

            # Handler vorhanden?
            handler = self._handlers.get(task.event_type)
            if not handler:
                self._queue.skip(task.task_id, "Kein Handler registriert")
                processed.append(task)
                continue

            # Approval prüfen
            if task.approval_mode == ApprovalMode.ASK:
                task.status = TaskStatus.AWAITING_APPROVAL
                processed.append(task)
                continue

            # Ausführen
            try:
                result = await handler(task)
                self._queue.complete(task.task_id, success=True, result=result)
                self._total_tasks_completed += 1
            except Exception as exc:
                task.retries += 1
                config = self._configs.get(task.event_type)
                max_retries = config.max_retries if config else 2

                if task.retries < max_retries:
                    # Zurück in Queue
                    task.status = TaskStatus.PENDING
                else:
                    self._queue.complete(
                        task.task_id, success=False, error=str(exc),
                    )
                    self._total_tasks_failed += 1

            processed.append(task)

        return processed

    # ── Manuelle Trigger ─────────────────────────────────────────

    def trigger_now(self, event_type: EventType, **payload: Any) -> str:
        """Löst ein Ereignis sofort aus (überspringt Timer).

        Returns:
            Task-ID.
        """
        # Sofort in Task umwandeln (ohne inject_trigger um Doppelausführung zu vermeiden)
        self._task_counter += 1
        config = self._configs.get(event_type, EventConfig(event_type=event_type))
        task = ProactiveTask(
            task_id=f"ht_{self._task_counter}",
            event_type=event_type,
            priority=config.priority,
            agent_name=config.agent_name,
            approval_mode=ApprovalMode.AUTO,  # Manuell = immer auto
            context=payload,
        )
        self._queue.enqueue(task)
        self._total_tasks_created += 1
        return task.task_id

    def approve_task(self, task_id: str) -> bool:
        """Genehmigt einen wartenden Task.

        Returns:
            True wenn genehmigt.
        """
        task = self._queue.get(task_id)
        if task and task.status == TaskStatus.AWAITING_APPROVAL:
            task.status = TaskStatus.PENDING
            task.approval_mode = ApprovalMode.AUTO  # Nicht erneut fragen
            return True
        return False

    def reject_task(self, task_id: str) -> bool:
        """Lehnt einen wartenden Task ab.

        Returns:
            True wenn abgelehnt.
        """
        return self._queue.skip(task_id, "Vom User abgelehnt")

    # ── Zugriff ──────────────────────────────────────────────────

    @property
    def queue(self) -> TaskQueue:
        return self._queue

    @property
    def event_source(self) -> EventSource:
        return self._event_source

    def stats(self) -> dict[str, Any]:
        return {
            "total_ticks": self._total_ticks,
            "total_tasks_created": self._total_tasks_created,
            "total_tasks_completed": self._total_tasks_completed,
            "total_tasks_failed": self._total_tasks_failed,
            "queue_size": self._queue.size,
            "pending": self._queue.pending_count,
            "enabled_events": len(self.enabled_configs()),
            "registered_handlers": len(self._handlers),
        }
