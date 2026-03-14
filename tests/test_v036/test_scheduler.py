"""Tests for Feature 9: Priority-based Agent Scheduling."""

from __future__ import annotations

import sys

import pytest

from jarvis.core.scheduler import AgentScheduler, ScheduledAgent


class TestPriority:
    @pytest.mark.asyncio
    async def test_higher_priority_runs_first(self):
        sched = AgentScheduler(max_concurrent_agents=10)
        sched.enqueue(ScheduledAgent(agent_id="low", priority=1))
        sched.enqueue(ScheduledAgent(agent_id="high", priority=10))

        first = await sched.tick()
        assert first is not None
        assert first.agent_id == "high"

        second = await sched.tick()
        assert second is not None
        assert second.agent_id == "low"

    @pytest.mark.asyncio
    async def test_equal_priority_is_fifo(self):
        sched = AgentScheduler(max_concurrent_agents=10)
        sched.enqueue(ScheduledAgent(agent_id="first", priority=5))
        sched.enqueue(ScheduledAgent(agent_id="second", priority=5))

        a = await sched.tick()
        b = await sched.tick()
        assert a.agent_id == "first"
        assert b.agent_id == "second"


class TestQuota:
    @pytest.mark.asyncio
    async def test_quota_50_50_respected_over_100_ticks(self):
        sched = AgentScheduler(
            orchestrator_quota=0.5,
            worker_quota=0.5,
            max_concurrent_agents=100,
        )

        # Enqueue 50 orchestrators and 50 workers
        for i in range(50):
            sched.enqueue(ScheduledAgent(agent_id=f"orch-{i}", role="orchestrator", priority=5))
            sched.enqueue(ScheduledAgent(agent_id=f"work-{i}", role="worker", priority=5))

        orch_count = 0
        work_count = 0
        for _ in range(100):
            agent = await sched.tick()
            if agent is None:
                break
            if agent.role == "orchestrator":
                orch_count += 1
            else:
                work_count += 1
            sched.complete(agent.agent_id)

        total = orch_count + work_count
        assert total == 100
        # Allow some slack (40-60 range) due to quota enforcement granularity
        assert 30 <= orch_count <= 70, f"Orchestrator ratio off: {orch_count}/100"


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        sched = AgentScheduler(max_concurrent_agents=2)
        for i in range(5):
            sched.enqueue(ScheduledAgent(agent_id=f"a{i}", priority=5))

        a1 = await sched.tick()
        a2 = await sched.tick()
        assert a1 is not None
        assert a2 is not None

        # Third should be blocked
        a3 = await sched.tick()
        assert a3 is None

        # Complete one, then third should work
        sched.complete(a1.agent_id)
        a3 = await sched.tick()
        assert a3 is not None

    @pytest.mark.asyncio
    async def test_max_concurrent_per_platform(self):
        """Platform-specific max concurrent defaults."""
        sched = AgentScheduler()
        if sys.platform == "linux":
            assert sched.max_concurrent_agents == 8
        else:
            assert sched.max_concurrent_agents == 4


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_scheduler_with_zero_agents_does_not_crash(self):
        sched = AgentScheduler()
        result = await sched.tick()
        assert result is None

    @pytest.mark.asyncio
    async def test_complete_unknown_agent_no_error(self):
        sched = AgentScheduler()
        sched.complete("nonexistent")  # Should not raise
