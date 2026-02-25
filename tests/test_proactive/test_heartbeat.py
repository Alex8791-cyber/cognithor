"""Tests für Proaktiven Heartbeat.

Testet:
  - EventConfig: Konfiguration, Ruhezeiten
  - ProactiveTask: Status, Dauer
  - EventSource: Timer, manuelle Trigger
  - TaskQueue: Priorisierung, Lifecycle
  - HeartbeatScheduler: Tick-Zyklen, Handler, Approval
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from jarvis.proactive import (
    ApprovalMode,
    EventConfig,
    EventSource,
    EventTrigger,
    EventType,
    HeartbeatScheduler,
    ProactiveTask,
    TaskQueue,
    TaskStatus,
)


# ============================================================================
# EventConfig
# ============================================================================


class TestEventConfig:
    def test_default_values(self) -> None:
        config = EventConfig(event_type=EventType.EMAIL_TRIAGE)
        assert config.enabled is True
        assert config.interval_seconds == 3600
        assert config.priority == 5

    def test_quiet_hours_active(self) -> None:
        config = EventConfig(
            event_type=EventType.DAILY_BRIEFING,
            quiet_hours_start=0,
            quiet_hours_end=24,  # Immer Ruhezeit
        )
        assert config.is_in_quiet_hours is True

    def test_quiet_hours_inactive(self) -> None:
        config = EventConfig(
            event_type=EventType.DAILY_BRIEFING,
            quiet_hours_start=-1,
            quiet_hours_end=-1,
        )
        assert config.is_in_quiet_hours is False


# ============================================================================
# ProactiveTask
# ============================================================================


class TestProactiveTask:
    def test_initial_status(self) -> None:
        task = ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE)
        assert task.status == TaskStatus.PENDING
        assert not task.is_terminal

    def test_terminal_states(self) -> None:
        for status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]:
            task = ProactiveTask(task_id="t1", event_type=EventType.TODO_REMINDER)
            task.status = status
            assert task.is_terminal

    def test_duration(self) -> None:
        now = datetime.now(timezone.utc)
        task = ProactiveTask(
            task_id="t1", event_type=EventType.CALENDAR_PREP,
            started_at=now.isoformat(),
            completed_at=(now + timedelta(seconds=5)).isoformat(),
        )
        assert 4.9 < task.duration_seconds < 5.1

    def test_duration_no_times(self) -> None:
        task = ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE)
        assert task.duration_seconds == 0.0


# ============================================================================
# EventSource
# ============================================================================


class TestEventSource:
    def test_initial_check_fires(self) -> None:
        source = EventSource()
        configs = {
            EventType.EMAIL_TRIAGE: EventConfig(
                event_type=EventType.EMAIL_TRIAGE,
                interval_seconds=0,  # Sofort feuern
            ),
        }
        triggers = source.check(configs)
        assert len(triggers) == 1
        assert triggers[0].event_type == EventType.EMAIL_TRIAGE

    def test_interval_respected(self) -> None:
        source = EventSource()
        configs = {
            EventType.EMAIL_TRIAGE: EventConfig(
                event_type=EventType.EMAIL_TRIAGE,
                interval_seconds=9999,
            ),
        }
        # Erster Check feuert
        source.check(configs)
        # Zweiter Check innerhalb des Intervalls → nichts
        triggers = source.check(configs)
        assert len(triggers) == 0

    def test_disabled_not_fired(self) -> None:
        source = EventSource()
        configs = {
            EventType.EMAIL_TRIAGE: EventConfig(
                event_type=EventType.EMAIL_TRIAGE,
                enabled=False,
            ),
        }
        assert len(source.check(configs)) == 0

    def test_manual_trigger(self) -> None:
        source = EventSource()
        source.inject_trigger(EventTrigger(
            event_type=EventType.CUSTOM,
            payload={"key": "value"},
            source="test",
        ))
        triggers = source.check({})  # Keine configs nötig
        assert len(triggers) == 1
        assert triggers[0].event_type == EventType.CUSTOM
        assert triggers[0].payload == {"key": "value"}

    def test_reset_timer(self) -> None:
        source = EventSource()
        configs = {
            EventType.EMAIL_TRIAGE: EventConfig(
                event_type=EventType.EMAIL_TRIAGE,
                interval_seconds=99999,  # Sehr lang
            ),
        }
        # Erster Check feuert (last=0, Differenz groß)
        triggers = source.check(configs)
        assert len(triggers) == 1
        # Zweiter Check: Timer gerade gesetzt → sollte NICHT feuern
        assert len(source.check(configs)) == 0
        # Reset → Timer gelöscht → last=0 → Differenz wieder groß → feuert
        source.reset_timer(EventType.EMAIL_TRIAGE)
        triggers = source.check(configs)
        assert len(triggers) == 1


# ============================================================================
# TaskQueue
# ============================================================================


class TestTaskQueue:
    def test_enqueue_dequeue(self) -> None:
        q = TaskQueue()
        task = ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE)
        q.enqueue(task)
        assert q.size == 1

        dequeued = q.dequeue()
        assert dequeued is not None
        assert dequeued.task_id == "t1"
        assert dequeued.status == TaskStatus.RUNNING

    def test_priority_ordering(self) -> None:
        q = TaskQueue()
        q.enqueue(ProactiveTask(task_id="low", event_type=EventType.TODO_REMINDER, priority=1))
        q.enqueue(ProactiveTask(task_id="high", event_type=EventType.CALENDAR_PREP, priority=10))
        q.enqueue(ProactiveTask(task_id="mid", event_type=EventType.EMAIL_TRIAGE, priority=5))

        first = q.dequeue()
        assert first.task_id == "high"

    def test_empty_dequeue(self) -> None:
        q = TaskQueue()
        assert q.dequeue() is None

    def test_complete_success(self) -> None:
        q = TaskQueue()
        task = ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE)
        q.enqueue(task)
        q.dequeue()  # RUNNING

        assert q.complete("t1", success=True, result="3 neue Mails")
        assert task.status == TaskStatus.COMPLETED

    def test_complete_failure(self) -> None:
        q = TaskQueue()
        task = ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE)
        q.enqueue(task)
        q.dequeue()

        assert q.complete("t1", success=False, error="IMAP-Fehler")
        assert task.status == TaskStatus.FAILED

    def test_skip(self) -> None:
        q = TaskQueue()
        task = ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE)
        q.enqueue(task)
        assert q.skip("t1", "Gatekeeper blockiert")
        assert task.status == TaskStatus.SKIPPED

    def test_pending_count(self) -> None:
        q = TaskQueue()
        q.enqueue(ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE))
        q.enqueue(ProactiveTask(task_id="t2", event_type=EventType.TODO_REMINDER))
        assert q.pending_count == 2
        q.dequeue()
        assert q.pending_count == 1

    def test_list_history(self) -> None:
        q = TaskQueue()
        task = ProactiveTask(task_id="t1", event_type=EventType.EMAIL_TRIAGE)
        q.enqueue(task)
        q.dequeue()
        q.complete("t1", success=True)

        history = q.list_history()
        assert len(history) == 1

    def test_cleanup(self) -> None:
        q = TaskQueue()
        for i in range(10):
            t = ProactiveTask(task_id=f"t{i}", event_type=EventType.EMAIL_TRIAGE)
            t.status = TaskStatus.COMPLETED
            t.completed_at = datetime.now(timezone.utc).isoformat()
            q.enqueue(t)

        removed = q.cleanup_completed(keep=3)
        assert removed == 7


# ============================================================================
# HeartbeatScheduler
# ============================================================================


class TestHeartbeatScheduler:
    @pytest.fixture
    def scheduler(self) -> HeartbeatScheduler:
        return HeartbeatScheduler()

    def test_default_configs(self, scheduler: HeartbeatScheduler) -> None:
        configs = scheduler.list_configs()
        assert len(configs) >= 6
        # Alle deaktiviert per Default
        for c in configs:
            assert c.enabled is False

    def test_configure(self, scheduler: HeartbeatScheduler) -> None:
        config = scheduler.configure(
            EventType.EMAIL_TRIAGE,
            enabled=True,
            interval_seconds=120,
            priority=9,
        )
        assert config.enabled is True
        assert config.interval_seconds == 120
        assert config.priority == 9

    def test_configure_clamps(self, scheduler: HeartbeatScheduler) -> None:
        config = scheduler.configure(
            EventType.EMAIL_TRIAGE,
            interval_seconds=5,  # < 30 → wird auf 30 gesetzt
            priority=20,  # > 10 → wird auf 10 gesetzt
        )
        assert config.interval_seconds == 30
        assert config.priority == 10

    def test_register_handler(self, scheduler: HeartbeatScheduler) -> None:
        async def handler(task: ProactiveTask) -> str:
            return "done"

        scheduler.register_handler(EventType.EMAIL_TRIAGE, handler)
        assert scheduler.has_handler(EventType.EMAIL_TRIAGE)
        assert not scheduler.has_handler(EventType.DAILY_BRIEFING)

    @pytest.mark.asyncio
    async def test_tick_creates_and_processes_tasks(
        self, scheduler: HeartbeatScheduler,
    ) -> None:
        scheduler.configure(
            EventType.EMAIL_TRIAGE,
            enabled=True,
            interval_seconds=0,
        )

        call_count = 0

        async def handler(task: ProactiveTask) -> str:
            nonlocal call_count
            call_count += 1
            return f"Verarbeitet #{call_count}"

        scheduler.register_handler(EventType.EMAIL_TRIAGE, handler)

        processed = await scheduler.tick()
        assert len(processed) >= 1
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_tick_no_handler_skips(
        self, scheduler: HeartbeatScheduler,
    ) -> None:
        scheduler.configure(
            EventType.EMAIL_TRIAGE,
            enabled=True,
            interval_seconds=0,
        )
        # Kein Handler registriert

        processed = await scheduler.tick()
        assert len(processed) >= 1
        assert processed[0].status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_tick_approval_mode(
        self, scheduler: HeartbeatScheduler,
    ) -> None:
        scheduler.configure(
            EventType.EMAIL_TRIAGE,
            enabled=True,
            interval_seconds=0,
            approval_mode=ApprovalMode.ASK,
        )

        async def handler(task: ProactiveTask) -> str:
            return "done"

        scheduler.register_handler(EventType.EMAIL_TRIAGE, handler)

        processed = await scheduler.tick()
        assert len(processed) >= 1
        assert processed[0].status == TaskStatus.AWAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_handler_error_retries(
        self, scheduler: HeartbeatScheduler,
    ) -> None:
        scheduler.configure(
            EventType.EMAIL_TRIAGE,
            enabled=True,
            interval_seconds=0,
        )

        call_count = 0

        async def failing_handler(task: ProactiveTask) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("IMAP down")

        scheduler.register_handler(EventType.EMAIL_TRIAGE, failing_handler)

        # Erster Tick: Fehler, aber Retry möglich
        processed = await scheduler.tick()
        assert call_count >= 1

    def test_trigger_now(self, scheduler: HeartbeatScheduler) -> None:
        task_id = scheduler.trigger_now(EventType.EMAIL_TRIAGE, query="inbox")
        assert task_id.startswith("ht_")

        task = scheduler.queue.get(task_id)
        assert task is not None
        assert task.event_type == EventType.EMAIL_TRIAGE

    def test_approve_and_reject(self, scheduler: HeartbeatScheduler) -> None:
        task = ProactiveTask(
            task_id="t1",
            event_type=EventType.EMAIL_TRIAGE,
            status=TaskStatus.AWAITING_APPROVAL,
        )
        scheduler.queue.enqueue(task)

        assert scheduler.approve_task("t1") is True
        assert task.status == TaskStatus.PENDING

        task2 = ProactiveTask(
            task_id="t2",
            event_type=EventType.TODO_REMINDER,
            status=TaskStatus.AWAITING_APPROVAL,
        )
        scheduler.queue.enqueue(task2)

        assert scheduler.reject_task("t2") is True
        assert task2.status == TaskStatus.SKIPPED

    def test_stats(self, scheduler: HeartbeatScheduler) -> None:
        stats = scheduler.stats()
        assert "total_ticks" in stats
        assert "total_tasks_created" in stats
        assert "queue_size" in stats
