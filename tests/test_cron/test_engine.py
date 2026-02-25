"""Tests für die Cron-Engine und den JobStore.

Testet Job-Verwaltung, Scheduling, Cron-Parsing und Runtime-API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
import yaml

from jarvis.cron.engine import CronEngine, _parse_cron_fields
from jarvis.cron.jobs import DEFAULT_JOBS, JobStore
from jarvis.models import CronJob, IncomingMessage

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# _parse_cron_fields
# ============================================================================


class TestParseCronFields:
    """Tests für die Cron-Ausdruck-Parser-Funktion."""

    def test_standard_five_fields(self) -> None:
        result = _parse_cron_fields("0 7 * * 1-5")
        assert result == {
            "minute": "0",
            "hour": "7",
            "day": "*",
            "month": "*",
            "day_of_week": "1-5",
        }

    def test_all_stars(self) -> None:
        result = _parse_cron_fields("* * * * *")
        assert result["minute"] == "*"
        assert result["day_of_week"] == "*"

    def test_complex_expression(self) -> None:
        result = _parse_cron_fields("*/15 9-17 1,15 1-6 MON-FRI")
        assert result["minute"] == "*/15"
        assert result["hour"] == "9-17"
        assert result["day"] == "1,15"
        assert result["month"] == "1-6"
        assert result["day_of_week"] == "MON-FRI"

    def test_too_few_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="5 Felder"):
            _parse_cron_fields("0 7 *")

    def test_too_many_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="5 Felder"):
            _parse_cron_fields("0 7 * * 1-5 extra")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="5 Felder"):
            _parse_cron_fields("")

    def test_whitespace_handling(self) -> None:
        result = _parse_cron_fields("  30   12   *   *   0  ")
        assert result["minute"] == "30"
        assert result["hour"] == "12"
        assert result["day_of_week"] == "0"


# ============================================================================
# JobStore
# ============================================================================


class TestJobStore:
    """Tests für den JobStore (YAML-Persistenz)."""

    def test_load_creates_defaults_if_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "cron" / "jobs.yaml"
        store = JobStore(path)
        jobs = store.load()

        assert path.exists()
        assert len(jobs) == len(DEFAULT_JOBS)
        assert "morning_briefing" in jobs
        assert "weekly_review" in jobs
        assert "memory_maintenance" in jobs

    def test_load_dict_format(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "test_job": {
                    "schedule": "0 8 * * *",
                    "prompt": "Test prompt",
                    "channel": "cli",
                    "enabled": True,
                }
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        store = JobStore(path)
        jobs = store.load()

        assert "test_job" in jobs
        assert jobs["test_job"].schedule == "0 8 * * *"
        assert jobs["test_job"].prompt == "Test prompt"
        assert jobs["test_job"].channel == "cli"

    def test_load_list_format(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": [
                {
                    "name": "list_job",
                    "schedule": "0 9 * * *",
                    "prompt": "List test",
                }
            ]
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        store = JobStore(path)
        jobs = store.load()

        assert "list_job" in jobs
        assert jobs["list_job"].schedule == "0 9 * * *"

    def test_load_invalid_yaml_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        path.write_text("{{invalid yaml:", encoding="utf-8")

        store = JobStore(path)
        jobs = store.load()
        assert jobs == {}

    def test_load_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        path.write_text("", encoding="utf-8")

        store = JobStore(path)
        jobs = store.load()
        assert jobs == {}

    def test_get_enabled(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "enabled_job": {
                    "schedule": "0 8 * * *",
                    "prompt": "Enabled",
                    "enabled": True,
                },
                "disabled_job": {
                    "schedule": "0 9 * * *",
                    "prompt": "Disabled",
                    "enabled": False,
                },
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        store = JobStore(path)
        store.load()
        enabled = store.get_enabled()

        assert len(enabled) == 1
        assert enabled[0].name == "enabled_job"

    def test_add_job(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        store = JobStore(path)
        store.load()  # Creates defaults

        new_job = CronJob(
            name="custom_job",
            schedule="*/30 * * * *",
            prompt="Custom prompt",
            channel="cli",
        )
        store.add_job(new_job)

        assert "custom_job" in store.jobs

        # Verify persistence
        store2 = JobStore(path)
        store2.load()
        assert "custom_job" in store2.jobs

    def test_remove_job(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        store = JobStore(path)
        store.load()

        assert "morning_briefing" in store.jobs
        result = store.remove_job("morning_briefing")
        assert result is True
        assert "morning_briefing" not in store.jobs

    def test_remove_nonexistent_job(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        store = JobStore(path)
        store.load()

        result = store.remove_job("does_not_exist")
        assert result is False

    def test_toggle_job(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        store = JobStore(path)
        store.load()

        assert store.jobs["morning_briefing"].enabled is False
        result = store.toggle_job("morning_briefing", enabled=True)
        assert result is True
        assert store.jobs["morning_briefing"].enabled is True

    def test_toggle_nonexistent_job(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        store = JobStore(path)
        store.load()

        result = store.toggle_job("ghost_job", enabled=True)
        assert result is False

    def test_skip_invalid_jobs_in_dict_format(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "good_job": {
                    "schedule": "0 8 * * *",
                    "prompt": "Good",
                },
                "bad_job": "not a dict",
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        store = JobStore(path)
        jobs = store.load()

        assert "good_job" in jobs
        assert "bad_job" not in jobs

    def test_skip_jobs_without_name_in_list_format(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": [
                {"name": "valid", "schedule": "0 8 * * *", "prompt": "OK"},
                {"schedule": "0 9 * * *", "prompt": "Missing name"},
            ]
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        store = JobStore(path)
        jobs = store.load()

        assert "valid" in jobs
        assert len(jobs) == 1


# ============================================================================
# CronEngine
# ============================================================================


class TestCronEngine:
    """Tests für die CronEngine."""

    def test_init_without_path(self) -> None:
        engine = CronEngine()
        assert engine.job_store is None
        assert engine.running is False

    def test_init_with_path(self, tmp_path: Path) -> None:
        engine = CronEngine(jobs_path=tmp_path / "cron" / "jobs.yaml")
        assert engine.job_store is not None
        assert engine.running is False

    def test_set_handler(self) -> None:
        engine = CronEngine()
        handler = AsyncMock()
        engine.set_handler(handler)
        assert engine._handler is handler

    @pytest.mark.asyncio
    async def test_start_and_stop(self, tmp_path: Path) -> None:
        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml")
        await engine.start()

        assert engine.running is True
        assert engine._scheduler is not None

        await engine.stop()
        assert engine.running is False
        assert engine._scheduler is None

    @pytest.mark.asyncio
    async def test_start_loads_enabled_jobs(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "active_job": {
                    "schedule": "0 8 * * *",
                    "prompt": "Active",
                    "enabled": True,
                },
                "inactive_job": {
                    "schedule": "0 9 * * *",
                    "prompt": "Inactive",
                    "enabled": False,
                },
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        engine = CronEngine(jobs_path=path)
        await engine.start()

        assert "active_job" in engine._active_jobs
        assert "inactive_job" not in engine._active_jobs

        await engine.stop()

    @pytest.mark.asyncio
    async def test_double_start_warns(self, tmp_path: Path) -> None:
        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml")
        await engine.start()
        await engine.start()  # Should warn but not crash
        assert engine.running is True
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        engine = CronEngine()
        await engine.stop()  # Should not crash
        assert engine.running is False

    @pytest.mark.asyncio
    async def test_execute_job_calls_handler(self) -> None:
        handler = AsyncMock()
        engine = CronEngine()
        engine.set_handler(handler)

        job = CronJob(name="test", schedule="0 8 * * *", prompt="Hello Jarvis")
        await engine._execute_job(job)

        handler.assert_called_once()
        msg = handler.call_args[0][0]
        assert isinstance(msg, IncomingMessage)
        assert msg.channel == "telegram"
        assert msg.user_id == "cron"
        assert "[CRON:test]" in msg.text
        assert "Hello Jarvis" in msg.text

    @pytest.mark.asyncio
    async def test_execute_job_without_handler(self) -> None:
        engine = CronEngine()
        job = CronJob(name="test", schedule="0 8 * * *", prompt="No handler")
        # Should not crash, just log warning
        await engine._execute_job(job)

    @pytest.mark.asyncio
    async def test_execute_job_handler_exception(self) -> None:
        handler = AsyncMock(side_effect=RuntimeError("boom"))
        engine = CronEngine()
        engine.set_handler(handler)

        job = CronJob(name="test", schedule="0 8 * * *", prompt="Boom")
        # Should not propagate exception
        await engine._execute_job(job)

    @pytest.mark.asyncio
    async def test_trigger_now(self, tmp_path: Path) -> None:
        handler = AsyncMock()
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "manual_job": {
                    "schedule": "0 8 * * *",
                    "prompt": "Triggered manually",
                    "enabled": False,  # Even disabled jobs can be triggered
                },
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        engine = CronEngine(jobs_path=path, handler=handler)
        engine.job_store.load()

        result = await engine.trigger_now("manual_job")
        assert result is True
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_now_nonexistent(self, tmp_path: Path) -> None:
        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml")
        engine.job_store.load()

        result = await engine.trigger_now("ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_now_without_store(self) -> None:
        engine = CronEngine()
        result = await engine.trigger_now("anything")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_runtime_job(self, tmp_path: Path) -> None:
        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml")
        await engine.start()

        job = CronJob(
            name="dynamic",
            schedule="*/5 * * * *",
            prompt="Dynamic job",
            enabled=True,
        )
        result = engine.add_runtime_job(job)
        assert result is True
        assert "dynamic" in engine._active_jobs
        assert "dynamic" in engine.job_store.jobs

        await engine.stop()

    @pytest.mark.asyncio
    async def test_remove_runtime_job(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "removable": {
                    "schedule": "0 8 * * *",
                    "prompt": "Remove me",
                    "enabled": True,
                },
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        engine = CronEngine(jobs_path=path)
        await engine.start()

        assert "removable" in engine._active_jobs
        result = engine.remove_runtime_job("removable")
        assert result is True
        assert "removable" not in engine._active_jobs

        await engine.stop()

    def test_list_jobs(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        engine = CronEngine(jobs_path=path)
        engine.job_store.load()

        jobs = engine.list_jobs()
        assert len(jobs) == len(DEFAULT_JOBS)

    def test_list_jobs_without_store(self) -> None:
        engine = CronEngine()
        assert engine.list_jobs() == []

    @pytest.mark.asyncio
    async def test_get_next_run_times(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "scheduled_job": {
                    "schedule": "0 8 * * *",
                    "prompt": "Next run?",
                    "enabled": True,
                },
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        engine = CronEngine(jobs_path=path)
        await engine.start()

        times = engine.get_next_run_times()
        assert "scheduled_job" in times
        assert times["scheduled_job"] is not None

        await engine.stop()

    def test_get_next_run_times_without_scheduler(self) -> None:
        engine = CronEngine()
        assert engine.get_next_run_times() == {}

    @pytest.mark.asyncio
    async def test_schedule_job_invalid_cron(self, tmp_path: Path) -> None:
        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml")
        await engine.start()

        job = CronJob(
            name="bad_cron",
            schedule="not a cron",
            prompt="Bad",
            enabled=True,
        )
        result = engine._schedule_job(job)
        assert result is False

        await engine.stop()

    def test_schedule_job_without_scheduler(self) -> None:
        engine = CronEngine()
        job = CronJob(name="test", schedule="0 8 * * *", prompt="No scheduler")
        result = engine._schedule_job(job)
        assert result is False


class TestCronGracefulShutdown:
    """Tests fuer graceful shutdown der CronEngine."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_no_running_jobs(self, tmp_path: Path) -> None:
        """Engine kann sauber heruntergefahren werden ohne laufende Jobs."""
        path = tmp_path / "jobs.yaml"
        data = {
            "jobs": {
                "job_a": {
                    "schedule": "0 8 * * *",
                    "prompt": "Job A",
                    "enabled": True,
                },
                "job_b": {
                    "schedule": "0 9 * * *",
                    "prompt": "Job B",
                    "enabled": True,
                },
            }
        }
        path.write_text(yaml.dump(data), encoding="utf-8")

        engine = CronEngine(jobs_path=path)
        await engine.start()

        assert engine.running is True
        assert len(engine._active_jobs) == 2

        await engine.stop()

        assert engine.running is False
        assert engine._scheduler is None
        # Active jobs dict should be cleared after stop
        assert len(engine._active_jobs) == 0

    @pytest.mark.asyncio
    async def test_graceful_shutdown_idempotent(self, tmp_path: Path) -> None:
        """Mehrfaches stop() darf nicht crashen."""
        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml")
        await engine.start()
        assert engine.running is True

        await engine.stop()
        assert engine.running is False

        # Second stop should be a no-op
        await engine.stop()
        assert engine.running is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown_without_start(self) -> None:
        """stop() ohne vorheriges start() darf nicht crashen."""
        engine = CronEngine()
        assert engine.running is False
        await engine.stop()
        assert engine.running is False

    @pytest.mark.asyncio
    async def test_shutdown_cancels_scheduler(self, tmp_path: Path) -> None:
        """Scheduler wird bei stop() korrekt beendet."""
        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml")
        await engine.start()

        # Scheduler should exist while running
        assert engine._scheduler is not None
        scheduler_ref = engine._scheduler

        await engine.stop()

        # Scheduler should be cleared
        assert engine._scheduler is None


class TestCronAgentTargeting:
    """CronJobs können gezielt an bestimmte Agenten gerichtet werden."""

    @pytest.mark.asyncio
    async def test_cron_job_passes_agent_metadata(self, tmp_path: Path) -> None:
        captured_messages: list = []

        async def mock_handler(msg):
            captured_messages.append(msg)

        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml", handler=mock_handler)
        await engine.start()

        job = CronJob(
            name="agent_test",
            schedule="0 8 * * *",
            prompt="Morgen-Briefing erstellen",
            agent="organizer",
        )
        await engine._execute_job(job)

        assert len(captured_messages) == 1
        msg = captured_messages[0]
        assert msg.metadata.get("target_agent") == "organizer"
        assert msg.metadata.get("cron_job") == "agent_test"

        await engine.stop()

    @pytest.mark.asyncio
    async def test_cron_job_without_agent_no_target(self, tmp_path: Path) -> None:
        captured_messages: list = []

        async def mock_handler(msg):
            captured_messages.append(msg)

        engine = CronEngine(jobs_path=tmp_path / "jobs.yaml", handler=mock_handler)
        await engine.start()

        job = CronJob(
            name="no_agent",
            schedule="0 8 * * *",
            prompt="Test",
        )
        await engine._execute_job(job)

        msg = captured_messages[0]
        assert "target_agent" not in msg.metadata
        assert msg.metadata.get("cron_job") == "no_agent"

        await engine.stop()
