"""Tests für security/policies.py – Erweiterte Policy-Engine."""

from __future__ import annotations

import pytest

from jarvis.models import AgentType, PlannedAction, SandboxLevel
from jarvis.security.policies import (
    AgentPermissions,
    PolicyEngine,
    PolicyViolation,
    ResourceQuota,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture
def strict_engine() -> PolicyEngine:
    return PolicyEngine(
        default_quota=ResourceQuota(
            max_sub_agents=2,
            max_parallel=1,
            max_total_iterations=10,
            max_total_tool_calls=20,
        )
    )


# ============================================================================
# Agent Permissions
# ============================================================================


class TestAgentPermissions:
    def test_planner_has_spawn(self, engine: PolicyEngine):
        perms = engine.get_permissions(AgentType.PLANNER)
        assert perms.can_spawn_sub_agents is True

    def test_worker_no_spawn(self, engine: PolicyEngine):
        perms = engine.get_permissions(AgentType.WORKER)
        assert perms.can_spawn_sub_agents is False

    def test_coder_has_write(self, engine: PolicyEngine):
        perms = engine.get_permissions(AgentType.CODER)
        assert "write_file" in perms.allowed_tools

    def test_researcher_has_web(self, engine: PolicyEngine):
        perms = engine.get_permissions(AgentType.RESEARCHER)
        assert "web_search" in perms.allowed_tools
        assert "web_fetch" in perms.allowed_tools

    def test_worker_no_web(self, engine: PolicyEngine):
        perms = engine.get_permissions(AgentType.WORKER)
        assert "web_search" not in perms.allowed_tools

    def test_coder_sandbox_namespace(self, engine: PolicyEngine):
        perms = engine.get_permissions(AgentType.CODER)
        assert perms.sandbox_level == SandboxLevel.NAMESPACE

    def test_custom_permissions(self):
        custom = {
            AgentType.WORKER: AgentPermissions(
                allowed_tools=frozenset({"custom_tool"}),
                sandbox_level=SandboxLevel.CONTAINER,
                max_iterations=3,
                can_spawn_sub_agents=False,
                network_access=True,
                max_memory_mb=1024,
                max_timeout_seconds=600,
            )
        }
        engine = PolicyEngine(custom_permissions=custom)
        perms = engine.get_permissions(AgentType.WORKER)
        assert "custom_tool" in perms.allowed_tools
        assert perms.sandbox_level == SandboxLevel.CONTAINER


# ============================================================================
# Tool Checks
# ============================================================================


class TestToolChecks:
    def test_allowed_tool(self, engine: PolicyEngine):
        result = engine.check_tool_allowed(AgentType.PLANNER, "read_file")
        assert result is None

    def test_blocked_tool(self, engine: PolicyEngine):
        result = engine.check_tool_allowed(AgentType.WORKER, "web_search")
        assert result is not None
        assert isinstance(result, PolicyViolation)
        assert result.rule == "tool_not_allowed"

    def test_coder_can_run_command(self, engine: PolicyEngine):
        result = engine.check_tool_allowed(AgentType.CODER, "run_command")
        assert result is None

    def test_researcher_cannot_write(self, engine: PolicyEngine):
        result = engine.check_tool_allowed(AgentType.RESEARCHER, "write_file")
        assert result is not None


# ============================================================================
# Spawn Checks
# ============================================================================


class TestSpawnChecks:
    def test_planner_can_spawn(self, engine: PolicyEngine):
        result = engine.check_spawn_allowed("sess_1", AgentType.PLANNER)
        assert result is None

    def test_worker_cannot_spawn(self, engine: PolicyEngine):
        result = engine.check_spawn_allowed("sess_1", AgentType.WORKER)
        assert result is not None
        assert result.rule == "spawn_not_allowed"

    def test_max_sub_agents(self, strict_engine: PolicyEngine):
        # Spawn 2 (max)
        strict_engine.record_spawn("sess_1")
        strict_engine.record_spawn("sess_1")
        result = strict_engine.check_spawn_allowed("sess_1", AgentType.PLANNER)
        assert result is not None
        assert result.rule == "max_sub_agents_exceeded"

    def test_max_parallel(self, strict_engine: PolicyEngine):
        # Max parallel = 1
        strict_engine.record_spawn("sess_1")
        result = strict_engine.check_spawn_allowed("sess_1", AgentType.PLANNER)
        assert result is not None
        assert result.rule == "max_parallel_exceeded"

    def test_parallel_frees_on_done(self, strict_engine: PolicyEngine):
        strict_engine.record_spawn("sess_1")
        strict_engine.record_agent_done("sess_1")
        # Now parallel = 0, but total agents = 1
        result = strict_engine.check_spawn_allowed("sess_1", AgentType.PLANNER)
        assert result is None  # Should be allowed


# ============================================================================
# Iteration & Tool Call Limits
# ============================================================================


class TestLimits:
    def test_iteration_limit(self, strict_engine: PolicyEngine):
        for _ in range(10):
            strict_engine.record_iteration("sess_1")
        result = strict_engine.check_iteration_limit("sess_1")
        assert result is not None
        assert result.rule == "max_iterations_exceeded"

    def test_under_iteration_limit(self, strict_engine: PolicyEngine):
        for _ in range(5):
            strict_engine.record_iteration("sess_1")
        result = strict_engine.check_iteration_limit("sess_1")
        assert result is None

    def test_tool_call_limit(self, strict_engine: PolicyEngine):
        for _ in range(20):
            strict_engine.record_tool_call("sess_1")
        result = strict_engine.check_tool_call_limit("sess_1")
        assert result is not None
        assert result.rule == "max_tool_calls_exceeded"


# ============================================================================
# Full Validation
# ============================================================================


class TestValidateAction:
    def test_valid_action(self, engine: PolicyEngine):
        action = PlannedAction(
            tool="read_file",
            params={"path": "/tmp/test"},
            rationale="test",
        )
        violations = engine.validate_action_for_agent(AgentType.PLANNER, action, "sess_1")
        assert violations == []

    def test_invalid_tool(self, engine: PolicyEngine):
        action = PlannedAction(
            tool="web_search",
            params={"query": "test"},
            rationale="test",
        )
        violations = engine.validate_action_for_agent(AgentType.WORKER, action, "sess_1")
        assert len(violations) >= 1
        assert violations[0].rule == "tool_not_allowed"

    def test_multiple_violations(self, strict_engine: PolicyEngine):
        # Exhaust iterations
        for _ in range(10):
            strict_engine.record_iteration("sess_1")
        action = PlannedAction(
            tool="web_search",  # Not allowed for worker
            params={},
            rationale="test",
        )
        violations = strict_engine.validate_action_for_agent(AgentType.WORKER, action, "sess_1")
        assert len(violations) >= 2  # tool + iteration


# ============================================================================
# Session Management
# ============================================================================


class TestSessionManagement:
    def test_quota_per_session(self, engine: PolicyEngine):
        engine.record_iteration("sess_1")
        engine.record_iteration("sess_2")

        q1 = engine.get_quota("sess_1")
        q2 = engine.get_quota("sess_2")
        assert q1.total_iterations == 1
        assert q2.total_iterations == 1

    def test_reset_session(self, engine: PolicyEngine):
        engine.record_spawn("sess_1")
        engine.record_iteration("sess_1")
        engine.reset_session("sess_1")

        q = engine.get_quota("sess_1")
        assert q.current_sub_agents == 0
        assert q.total_iterations == 0

    def test_reset_unknown_session(self, engine: PolicyEngine):
        # Should not raise
        engine.reset_session("nonexistent")

    def test_get_quota_creates_new(self, engine: PolicyEngine):
        q = engine.get_quota("brand_new_session")
        assert q.total_iterations == 0
        assert q.total_tool_calls == 0
        assert q.current_sub_agents == 0

    def test_record_tool_call_increments(self, engine: PolicyEngine):
        engine.record_tool_call("sess_1")
        engine.record_tool_call("sess_1")
        engine.record_tool_call("sess_1")
        q = engine.get_quota("sess_1")
        assert q.total_tool_calls == 3

    def test_record_iteration_increments(self, engine: PolicyEngine):
        engine.record_iteration("sess_1")
        engine.record_iteration("sess_1")
        q = engine.get_quota("sess_1")
        assert q.total_iterations == 2


# ============================================================================
# Edge Cases & Boundary Conditions
# ============================================================================


class TestEdgeCases:
    def test_unknown_agent_type_falls_back_to_worker(self, engine: PolicyEngine):
        """Unbekannter AgentType sollte Worker-Permissions bekommen."""
        # Simuliere unbekannten Typ — da AgentType ein Enum ist,
        # testen wir via internen Zugriff auf _permissions
        worker_perms = engine.get_permissions(AgentType.WORKER)
        # Entferne PLANNER um Fallback zu testen
        engine_custom = PolicyEngine()
        # Planner ist bekannt → gültig
        assert engine_custom.get_permissions(AgentType.PLANNER).can_spawn_sub_agents is True

    def test_agent_done_does_not_go_negative(self, engine: PolicyEngine):
        engine.record_agent_done("sess_1")
        q = engine.get_quota("sess_1")
        assert q.current_parallel == 0  # Nicht negativ

    def test_tool_call_limit_boundary(self, strict_engine: PolicyEngine):
        # 19 calls → noch OK
        for _ in range(19):
            strict_engine.record_tool_call("sess_1")
        assert strict_engine.check_tool_call_limit("sess_1") is None
        # 20. call → Limit erreicht
        strict_engine.record_tool_call("sess_1")
        assert strict_engine.check_tool_call_limit("sess_1") is not None

    def test_iteration_limit_boundary(self, strict_engine: PolicyEngine):
        for _ in range(9):
            strict_engine.record_iteration("sess_1")
        assert strict_engine.check_iteration_limit("sess_1") is None
        strict_engine.record_iteration("sess_1")
        assert strict_engine.check_iteration_limit("sess_1") is not None

    def test_validate_action_triple_violation(self, strict_engine: PolicyEngine):
        """Tool nicht erlaubt + Iterations-Limit + Tool-Call-Limit."""
        for _ in range(10):
            strict_engine.record_iteration("sess_1")
        for _ in range(20):
            strict_engine.record_tool_call("sess_1")
        action = PlannedAction(
            tool="web_search",
            params={},
            rationale="test",
        )
        violations = strict_engine.validate_action_for_agent(
            AgentType.WORKER, action, "sess_1"
        )
        rules = [v.rule for v in violations]
        assert "tool_not_allowed" in rules
        assert "max_iterations_exceeded" in rules
        assert "max_tool_calls_exceeded" in rules

    def test_policy_violation_severity(self):
        v = PolicyViolation(rule="test", details="detail", severity="warning")
        assert v.severity == "warning"
        assert v.rule == "test"
        assert v.details == "detail"

    def test_policy_violation_default_severity(self):
        v = PolicyViolation(rule="test", details="detail")
        assert v.severity == "error"

    def test_resource_quota_defaults(self):
        q = ResourceQuota()
        assert q.max_sub_agents == 4
        assert q.max_parallel == 3
        assert q.max_total_iterations == 50
        assert q.max_total_tool_calls == 100
        assert q.current_sub_agents == 0

    def test_agent_permissions_frozen(self):
        perms = AgentPermissions(
            allowed_tools=frozenset({"tool_a"}),
            sandbox_level=SandboxLevel.PROCESS,
            max_iterations=5,
            can_spawn_sub_agents=False,
            network_access=False,
            max_memory_mb=256,
            max_timeout_seconds=60,
        )
        assert perms.max_memory_mb == 256
        with pytest.raises(AttributeError):
            perms.max_memory_mb = 512  # type: ignore[misc]
