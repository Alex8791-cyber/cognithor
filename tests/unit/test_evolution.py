"""Tests for Evolution Engine — IdleDetector + EvolutionLoop."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestIdleDetector:
    def test_not_idle_initially(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=5)
        assert d.is_idle is False

    def test_idle_after_threshold(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        # Threshold 0 = immediately idle
        assert d.is_idle is True

    def test_activity_resets_idle(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        d._last_activity = time.time() - 100  # Force old
        assert d.is_idle is True
        d.notify_activity()
        # After activity with threshold 0, need to wait
        # But with very small threshold it should flip back quickly
        assert d.idle_seconds < 1

    def test_idle_seconds(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=10)
        d._last_activity = time.time() - 30
        assert d.idle_seconds >= 29


class TestEvolutionLoop:
    @pytest.fixture
    def idle_detector(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        d._last_activity = time.time() - 100  # Force idle
        return d

    @pytest.fixture
    def loop(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop

        return EvolutionLoop(idle_detector=idle_detector)

    @pytest.mark.asyncio
    async def test_cycle_skips_when_not_idle(self):
        from jarvis.evolution.idle_detector import IdleDetector
        from jarvis.evolution.loop import EvolutionLoop

        d = IdleDetector(idle_threshold_seconds=9999)
        loop = EvolutionLoop(idle_detector=d)
        result = await loop.run_cycle()
        assert result.skipped is True
        assert result.reason == "not_idle"

    @pytest.mark.asyncio
    async def test_cycle_runs_when_idle(self, loop):
        result = await loop.run_cycle()
        # Without curiosity engine, should skip with no_gaps
        assert result.skipped is True
        assert result.reason == "no_gaps"
        assert "scout" in result.steps_completed

    @pytest.mark.asyncio
    async def test_daily_limit(self, loop):
        import time

        loop._cycles_today = 100
        loop._last_cycle_day = time.strftime("%Y-%m-%d")
        assert loop._can_run_cycle() is False

    def test_stats(self, loop):
        stats = loop.stats()
        assert "running" in stats
        assert "total_cycles" in stats
        assert "is_idle" in stats

    @pytest.mark.asyncio
    async def test_start_stop(self, loop):
        await loop.start()
        assert loop._running is True
        loop.stop()
        assert loop._running is False
        await asyncio.sleep(0.1)  # Let cancellation propagate


class TestCooperativeScheduling:
    """Tests fuer ResourceMonitor + Budget Integration im EvolutionLoop."""

    @pytest.fixture
    def idle_detector(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        d._last_activity = time.time() - 100
        return d

    @pytest.mark.asyncio
    async def test_cycle_skips_when_system_busy(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.system.resource_monitor import ResourceMonitor, ResourceSnapshot

        monitor = ResourceMonitor()
        # Inject a busy snapshot
        monitor._last_snapshot = ResourceSnapshot(cpu_percent=95.0, is_busy=True)
        monitor._last_sample_time = time.monotonic()

        loop = EvolutionLoop(idle_detector=idle_detector, resource_monitor=monitor)
        result = await loop.run_cycle()
        assert result.skipped is True
        assert result.reason == "system_busy"

    @pytest.mark.asyncio
    async def test_cycle_proceeds_when_resources_ok(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.system.resource_monitor import ResourceMonitor, ResourceSnapshot

        monitor = ResourceMonitor()
        monitor._last_snapshot = ResourceSnapshot(cpu_percent=30.0, is_busy=False)
        monitor._last_sample_time = time.monotonic()

        loop = EvolutionLoop(idle_detector=idle_detector, resource_monitor=monitor)
        result = await loop.run_cycle()
        # Should pass resource check but skip at no_gaps (no curiosity engine)
        assert result.reason == "no_gaps"
        assert "scout" in result.steps_completed

    @pytest.mark.asyncio
    async def test_cycle_skips_when_budget_exhausted(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.models import AgentBudgetStatus

        mock_tracker = MagicMock()
        mock_tracker.check_agent_budget.return_value = AgentBudgetStatus(
            agent_name="scout",
            daily_cost_usd=1.0,
            daily_limit_usd=0.5,
            ok=False,
            warning="Agent 'scout' Tageslimit erreicht",
        )
        mock_config = MagicMock()
        mock_config.agent_budgets = {"scout": 0.5}
        mock_config.max_cycles_per_day = 10
        mock_config.cycle_cooldown_seconds = 300

        loop = EvolutionLoop(
            idle_detector=idle_detector,
            config=mock_config,
            cost_tracker=mock_tracker,
        )
        result = await loop.run_cycle()
        assert result.skipped is True
        assert result.reason == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_cycle_ok_when_budget_available(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.models import AgentBudgetStatus

        mock_tracker = MagicMock()
        mock_tracker.check_agent_budget.return_value = AgentBudgetStatus(
            agent_name="scout",
            daily_cost_usd=0.1,
            daily_limit_usd=0.5,
            ok=True,
        )
        mock_config = MagicMock()
        mock_config.agent_budgets = {"scout": 0.5}
        mock_config.max_cycles_per_day = 10

        loop = EvolutionLoop(
            idle_detector=idle_detector,
            config=mock_config,
            cost_tracker=mock_tracker,
        )
        result = await loop.run_cycle()
        # Passes budget check, skips at no_gaps
        assert result.reason == "no_gaps"

    def test_stats_includes_resources(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.system.resource_monitor import ResourceMonitor, ResourceSnapshot

        monitor = ResourceMonitor()
        monitor._last_snapshot = ResourceSnapshot(
            cpu_percent=45.0, ram_percent=62.0, gpu_util_percent=30.0
        )
        loop = EvolutionLoop(idle_detector=idle_detector, resource_monitor=monitor)
        stats = loop.stats()
        assert "resources" in stats
        assert stats["resources"]["cpu_percent"] == 45.0
        assert stats["resources"]["available"] is True

    def test_stats_resources_without_monitor(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop

        loop = EvolutionLoop(idle_detector=idle_detector)
        stats = loop.stats()
        assert stats["resources"]["available"] is True
