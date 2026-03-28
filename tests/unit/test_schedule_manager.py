"""Tests for ScheduleManager — cron jobs for recurring source updates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis.evolution.models import LearningPlan, ScheduleSpec
from jarvis.evolution.schedule_manager import ScheduleManager, _normalize_cron


def _make_plan(
    schedules: list[ScheduleSpec] | None = None,
    goal: str = "Learn Rust",
) -> LearningPlan:
    plan = LearningPlan(goal=goal)
    if schedules:
        plan.schedules = schedules
    return plan


def _make_cron_engine() -> MagicMock:
    engine = MagicMock()
    engine.add_runtime_job = MagicMock(return_value=True)
    return engine


@pytest.mark.asyncio
async def test_create_schedules():
    """1 ScheduleSpec -> add_runtime_job called once, returns 1."""
    engine = _make_cron_engine()
    mgr = ScheduleManager(cron_engine=engine)

    plan = _make_plan(schedules=[
        ScheduleSpec(name="daily-rust", cron_expression="0 8 * * *", source_url="https://doc.rust-lang.org"),
    ])

    count = await mgr.create_schedules(plan)

    assert count == 1
    engine.add_runtime_job.assert_called_once()


@pytest.mark.asyncio
async def test_create_multiple_schedules():
    """2 ScheduleSpecs -> 2 jobs, returns 2."""
    engine = _make_cron_engine()
    mgr = ScheduleManager(cron_engine=engine)

    plan = _make_plan(schedules=[
        ScheduleSpec(name="daily-rust", cron_expression="0 8 * * *", source_url="https://doc.rust-lang.org"),
        ScheduleSpec(name="weekly-crates", cron_expression="0 9 * * 1", source_url="https://crates.io"),
    ])

    count = await mgr.create_schedules(plan)

    assert count == 2
    assert engine.add_runtime_job.call_count == 2


@pytest.mark.asyncio
async def test_skip_empty_schedules():
    """No schedules -> returns 0, no cron calls."""
    engine = _make_cron_engine()
    mgr = ScheduleManager(cron_engine=engine)

    plan = _make_plan(schedules=[])

    count = await mgr.create_schedules(plan)

    assert count == 0
    engine.add_runtime_job.assert_not_called()


@pytest.mark.asyncio
async def test_cron_job_name_prefixed():
    """Job name includes 'evolution_' prefix and is sanitized."""
    engine = _make_cron_engine()
    mgr = ScheduleManager(cron_engine=engine)

    plan = _make_plan(schedules=[
        ScheduleSpec(name="daily-rust", cron_expression="0 8 * * *", source_url="https://doc.rust-lang.org"),
    ])

    await mgr.create_schedules(plan)

    cron_job = engine.add_runtime_job.call_args[0][0]
    assert cron_job.name.startswith("evolution_")
    assert " " not in cron_job.name  # sanitized


@pytest.mark.asyncio
async def test_no_cron_engine():
    """cron_engine=None -> returns 0, no crash."""
    mgr = ScheduleManager(cron_engine=None)

    plan = _make_plan(schedules=[
        ScheduleSpec(name="daily-rust", cron_expression="0 8 * * *", source_url="https://doc.rust-lang.org"),
    ])

    count = await mgr.create_schedules(plan)

    assert count == 0


class TestNormalizeCron:
    def test_5_field_passthrough(self):
        assert _normalize_cron("0 8 * * 1") == "0 8 * * 1"

    def test_6_field_drops_seconds(self):
        assert _normalize_cron("0 0 8 * * MON") == "0 8 * * 1"

    def test_named_days(self):
        assert _normalize_cron("0 8 * * MON") == "0 8 * * 1"
        assert _normalize_cron("0 8 * * FRI") == "0 8 * * 5"

    def test_6_field_with_named_day(self):
        assert _normalize_cron("0 0 10 1 */3 SAT") == "0 10 1 */3 6"
