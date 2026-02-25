"""Tests für core/orchestrator.py – Multi-Agent Orchestrator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from jarvis.core.orchestrator import Orchestrator
from jarvis.models import (
    AgentHandle,
    AgentResult,
    AgentType,
    SubAgentConfig,
)
from jarvis.security.audit import AuditTrail
from jarvis.security.policies import PolicyEngine, PolicyViolation, ResourceQuota

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# Helpers
# ============================================================================


async def _mock_runner(config: SubAgentConfig, session_id: str) -> AgentResult:
    """Simuliert einen Agent-Runner."""
    await asyncio.sleep(0.01)
    return AgentResult(
        response=f"Done: {config.task}",
        success=True,
        total_iterations=2,
        model_used=config.model,
    )


async def _slow_runner(config: SubAgentConfig, session_id: str) -> AgentResult:
    """Simuliert einen langsamen Agent (für Timeout-Tests)."""
    await asyncio.sleep(10)
    return AgentResult(response="should not reach", success=True)


async def _failing_runner(config: SubAgentConfig, session_id: str) -> AgentResult:
    """Simuliert einen fehlschlagenden Agent."""
    raise RuntimeError("Agent crashed")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def orchestrator() -> Orchestrator:
    orch = Orchestrator(
        policy_engine=PolicyEngine(),
    )
    orch.set_runner(_mock_runner)
    return orch


@pytest.fixture
def audited_orchestrator(tmp_path: Path) -> Orchestrator:
    audit = AuditTrail(log_dir=tmp_path / "audit")
    orch = Orchestrator(
        policy_engine=PolicyEngine(),
        audit_trail=audit,
    )
    orch.set_runner(_mock_runner)
    return orch


@pytest.fixture
def limited_orchestrator() -> Orchestrator:
    orch = Orchestrator(
        policy_engine=PolicyEngine(
            default_quota=ResourceQuota(
                max_sub_agents=2,
                max_parallel=2,
            )
        ),
    )
    orch.set_runner(_mock_runner)
    return orch


def _config(task: str = "test_task", **kwargs) -> SubAgentConfig:
    return SubAgentConfig(task=task, **kwargs)


# ============================================================================
# Spawn
# ============================================================================


class TestSpawn:
    @pytest.mark.asyncio
    async def test_spawn_returns_handle(self, orchestrator: Orchestrator):
        result = await orchestrator.spawn_agent(_config("do something"), "sess_1")
        assert isinstance(result, AgentHandle)
        assert result.status == "pending"
        assert result.agent_id.startswith("agent_")

    @pytest.mark.asyncio
    async def test_spawn_rejected_by_policy(self, orchestrator: Orchestrator):
        # Worker cannot spawn
        result = await orchestrator.spawn_agent(
            _config("test"), "sess_1", parent_type=AgentType.WORKER
        )
        assert isinstance(result, PolicyViolation)

    @pytest.mark.asyncio
    async def test_spawn_limit(self, limited_orchestrator: Orchestrator):
        r1 = await limited_orchestrator.spawn_agent(_config("t1"), "sess_1")
        r2 = await limited_orchestrator.spawn_agent(_config("t2"), "sess_1")
        assert isinstance(r1, AgentHandle)
        assert isinstance(r2, AgentHandle)

        # Third should be rejected (max_sub_agents=2)
        r3 = await limited_orchestrator.spawn_agent(_config("t3"), "sess_1")
        assert isinstance(r3, PolicyViolation)


# ============================================================================
# Run
# ============================================================================


class TestRun:
    @pytest.mark.asyncio
    async def test_run_agent(self, orchestrator: Orchestrator):
        handle = await orchestrator.spawn_agent(_config("task1"), "sess_1")
        assert isinstance(handle, AgentHandle)

        result = await orchestrator.run_agent(handle.agent_id, "sess_1")
        assert result.success is True
        assert "Done: task1" in result.response

    @pytest.mark.asyncio
    async def test_run_unknown_agent(self, orchestrator: Orchestrator):
        result = await orchestrator.run_agent("nonexistent", "sess_1")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_without_runner(self):
        orch = Orchestrator()
        handle = await orch.spawn_agent(_config("task"), "sess_1")
        assert isinstance(handle, AgentHandle)

        result = await orch.run_agent(handle.agent_id, "sess_1")
        assert result.success is False
        assert "Runner" in result.error

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        orch = Orchestrator(policy_engine=PolicyEngine())
        orch.set_runner(_slow_runner)

        config = SubAgentConfig(task="slow", timeout_seconds=1)
        handle = await orch.spawn_agent(config, "sess_1")
        assert isinstance(handle, AgentHandle)

        result = await orch.run_agent(handle.agent_id, "sess_1")
        assert result.success is False
        assert "Timeout" in result.error

        # Handle should be marked as timeout
        h = orch.get_handle(handle.agent_id)
        assert h.status == "timeout"

    @pytest.mark.asyncio
    async def test_run_failure(self):
        orch = Orchestrator(policy_engine=PolicyEngine())
        orch.set_runner(_failing_runner)

        handle = await orch.spawn_agent(_config("crash"), "sess_1")
        assert isinstance(handle, AgentHandle)

        result = await orch.run_agent(handle.agent_id, "sess_1")
        assert result.success is False
        assert "crashed" in result.error

        h = orch.get_handle(handle.agent_id)
        assert h.status == "failed"


# ============================================================================
# Parallel Execution
# ============================================================================


class TestParallel:
    @pytest.mark.asyncio
    async def test_run_parallel(self, orchestrator: Orchestrator):
        h1 = await orchestrator.spawn_agent(_config("task1"), "sess_1")
        h2 = await orchestrator.spawn_agent(_config("task2"), "sess_1")
        assert isinstance(h1, AgentHandle)
        assert isinstance(h2, AgentHandle)

        results = await orchestrator.run_parallel([h1.agent_id, h2.agent_id], "sess_1")
        assert len(results) == 2
        assert all(r.success for r in results.values())

    @pytest.mark.asyncio
    async def test_run_parallel_empty(self, orchestrator: Orchestrator):
        results = await orchestrator.run_parallel([], "sess_1")
        assert results == {}


# ============================================================================
# Convenience Methods
# ============================================================================


class TestConvenience:
    @pytest.mark.asyncio
    async def test_spawn_and_run(self, orchestrator: Orchestrator):
        result = await orchestrator.spawn_and_run(_config("one_shot"), "sess_1")
        assert result.success is True
        assert "Done: one_shot" in result.response

    @pytest.mark.asyncio
    async def test_spawn_and_run_rejected(self, orchestrator: Orchestrator):
        result = await orchestrator.spawn_and_run(
            _config("test"), "sess_1", parent_type=AgentType.WORKER
        )
        assert result.success is False
        assert "Policy" in result.error

    @pytest.mark.asyncio
    async def test_spawn_and_run_parallel(self, orchestrator: Orchestrator):
        configs = [
            _config("task1"),
            _config("task2"),
            _config("task3"),
        ]
        results = await orchestrator.spawn_and_run_parallel(configs, "sess_1")
        assert len(results) == 3
        assert all(r.success for r in results)


# ============================================================================
# State Management
# ============================================================================


class TestState:
    @pytest.mark.asyncio
    async def test_get_handle(self, orchestrator: Orchestrator):
        handle = await orchestrator.spawn_agent(_config("t"), "sess_1")
        assert isinstance(handle, AgentHandle)

        retrieved = orchestrator.get_handle(handle.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == handle.agent_id

    @pytest.mark.asyncio
    async def test_get_result(self, orchestrator: Orchestrator):
        await orchestrator.spawn_and_run(_config("t"), "sess_1")
        # There should be at least one result
        handles = orchestrator.get_all_handles()
        assert len(handles) >= 1

        agent_id = handles[0].agent_id
        stored = orchestrator.get_result(agent_id)
        assert stored is not None
        assert stored.success is True

    @pytest.mark.asyncio
    async def test_agent_count(self, orchestrator: Orchestrator):
        assert orchestrator.agent_count == 0
        await orchestrator.spawn_agent(_config("t1"), "sess_1")
        await orchestrator.spawn_agent(_config("t2"), "sess_1")
        assert orchestrator.agent_count == 2

    @pytest.mark.asyncio
    async def test_active_agents(self, orchestrator: Orchestrator):
        h = await orchestrator.spawn_agent(_config("t"), "sess_1")
        assert isinstance(h, AgentHandle)
        assert orchestrator.active_count == 1

        await orchestrator.run_agent(h.agent_id, "sess_1")
        assert orchestrator.active_count == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, orchestrator: Orchestrator):
        await orchestrator.spawn_and_run(_config("t1"), "sess_1")
        await orchestrator.spawn_and_run(_config("t2"), "sess_1")
        assert orchestrator.agent_count == 2

        removed = orchestrator.cleanup_session("sess_1")
        assert removed == 2
        assert orchestrator.agent_count == 0


# ============================================================================
# Audit Integration
# ============================================================================


class TestAudit:
    @pytest.mark.asyncio
    async def test_audit_spawn(self, audited_orchestrator: Orchestrator):
        await audited_orchestrator.spawn_agent(_config("t"), "sess_1")
        # Audit trail should have entries
        # (We just verify it doesn't crash)

    @pytest.mark.asyncio
    async def test_audit_completion(self, audited_orchestrator: Orchestrator):
        await audited_orchestrator.spawn_and_run(_config("t"), "sess_1")
        # Should not crash with audit enabled

    @pytest.mark.asyncio
    async def test_audit_rejection(self, audited_orchestrator: Orchestrator):
        await audited_orchestrator.spawn_agent(_config("t"), "sess_1", parent_type=AgentType.WORKER)
        # Rejection should also be audited


# ============================================================================
# Depth Tests (v22: Max Depth 3)
# ============================================================================


class TestDepth:
    @pytest.mark.asyncio
    async def test_depth_0_can_spawn(self, orchestrator: Orchestrator):
        """Top-Level (depth=0) darf spawnen."""
        result = await orchestrator.spawn_agent(
            _config("task_d0"), "sess_d", depth=0
        )
        assert isinstance(result, AgentHandle)
        assert result.depth == 1

    @pytest.mark.asyncio
    async def test_depth_1_can_spawn(self, orchestrator: Orchestrator):
        """Depth 1 darf weiterspawnen."""
        result = await orchestrator.spawn_agent(
            _config("task_d1"), "sess_d", depth=1
        )
        assert isinstance(result, AgentHandle)
        assert result.depth == 2

    @pytest.mark.asyncio
    async def test_depth_2_can_spawn(self, orchestrator: Orchestrator):
        """Depth 2 darf noch spawnen (max_depth=3)."""
        result = await orchestrator.spawn_agent(
            _config("task_d2"), "sess_d", depth=2
        )
        assert isinstance(result, AgentHandle)
        assert result.depth == 3

    @pytest.mark.asyncio
    async def test_depth_3_cannot_spawn(self, orchestrator: Orchestrator):
        """Depth 3 wird blockiert (max_depth=3 → depth 3 nicht erlaubt)."""
        result = await orchestrator.spawn_agent(
            _config("task_d3"), "sess_d", depth=3
        )
        assert isinstance(result, PolicyViolation)
        assert result.rule == "max_depth_exceeded"

    @pytest.mark.asyncio
    async def test_max_depth_configurable(self):
        """Custom max_depth wird respektiert."""
        orch = Orchestrator(
            policy_engine=PolicyEngine(
                default_quota=ResourceQuota(max_depth=1)
            ),
        )
        orch.set_runner(_mock_runner)

        # Depth 0 → OK
        r0 = await orch.spawn_agent(_config("t0"), "sess_c", depth=0)
        assert isinstance(r0, AgentHandle)

        # Depth 1 → Blockiert (max_depth=1)
        r1 = await orch.spawn_agent(_config("t1"), "sess_c", depth=1)
        assert isinstance(r1, PolicyViolation)
        assert r1.rule == "max_depth_exceeded"

    @pytest.mark.asyncio
    async def test_spawn_and_run_with_depth(self, orchestrator: Orchestrator):
        """spawn_and_run propagiert depth korrekt."""
        result = await orchestrator.spawn_and_run(
            _config("deep_task"), "sess_d", depth=1
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_spawn_and_run_parallel_with_depth(self, orchestrator: Orchestrator):
        """spawn_and_run_parallel propagiert depth korrekt."""
        configs = [_config("t1"), _config("t2")]
        results = await orchestrator.spawn_and_run_parallel(
            configs, "sess_d", depth=2
        )
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_depth_default_backward_compatible(self, orchestrator: Orchestrator):
        """Ohne depth-Parameter funktioniert alles wie bisher (depth=0)."""
        result = await orchestrator.spawn_agent(_config("legacy"), "sess_compat")
        assert isinstance(result, AgentHandle)
        assert result.depth == 1  # child of depth=0

    @pytest.mark.asyncio
    async def test_researcher_can_spawn_at_depth(self, orchestrator: Orchestrator):
        """RESEARCHER darf seit v22 Sub-Agents spawnen (bei depth < max_depth)."""
        result = await orchestrator.spawn_agent(
            _config("research_sub"), "sess_r",
            parent_type=AgentType.RESEARCHER, depth=1,
        )
        assert isinstance(result, AgentHandle)
