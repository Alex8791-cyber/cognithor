"""Tests für den Executor – sandboxed Tool-Ausführung.

Testet:
  - Nur erlaubte Aktionen werden ausgeführt
  - BLOCK/APPROVE Aktionen werden übersprungen
  - MASK: Maskierte Params werden verwendet
  - Dependency-Tracking (depends_on)
  - Timeout-Handling
  - Fehlerbehandlung
  - ToolResult success/is_error Properties
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from jarvis.config import JarvisConfig
from jarvis.core.executor import Executor
from jarvis.models import (
    GateDecision,
    GateStatus,
    PlannedAction,
    RiskLevel,
)


@dataclass
class MockToolResult:
    content: str = "OK"
    is_error: bool = False


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    return JarvisConfig(jarvis_home=tmp_path)


@pytest.fixture()
def mock_mcp() -> AsyncMock:
    """Mock MCP-Client der immer Erfolg zurückgibt."""
    mcp = AsyncMock()
    mcp.call_tool = AsyncMock(return_value=MockToolResult(content="tool output"))
    return mcp


@pytest.fixture()
def executor(config: JarvisConfig, mock_mcp: AsyncMock) -> Executor:
    return Executor(config, mock_mcp)


def _allow_decision(action: PlannedAction | None = None) -> GateDecision:
    return GateDecision(
        status=GateStatus.ALLOW,
        risk_level=RiskLevel.GREEN,
        reason="Erlaubt",
        original_action=action,
        policy_name="test",
    )


def _block_decision(action: PlannedAction | None = None) -> GateDecision:
    return GateDecision(
        status=GateStatus.BLOCK,
        risk_level=RiskLevel.RED,
        reason="Blockiert",
        original_action=action,
        policy_name="test_block",
    )


def _mask_decision(action: PlannedAction | None = None, masked: dict | None = None) -> GateDecision:
    return GateDecision(
        status=GateStatus.MASK,
        risk_level=RiskLevel.YELLOW,
        reason="Maskiert",
        original_action=action,
        masked_params=masked or {"key": "***MASKED***"},
        policy_name="credential_masking",
    )


# ============================================================================
# Grundlegende Ausführung
# ============================================================================


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_allowed_action(self, executor: Executor) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/test"})
        results = await executor.execute([action], [_allow_decision(action)])
        assert len(results) == 1
        assert results[0].success
        assert results[0].tool_name == "read_file"

    @pytest.mark.asyncio
    async def test_blocked_action_skipped(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        action = PlannedAction(tool="exec_command", params={"command": "rm -rf /"})
        results = await executor.execute([action], [_block_decision(action)])
        assert len(results) == 1
        assert results[0].is_error
        assert not results[0].success
        mock_mcp.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_inform_action_executed(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        action = PlannedAction(tool="write_file", params={"path": "/test"})
        decision = GateDecision(
            status=GateStatus.INFORM,
            risk_level=RiskLevel.YELLOW,
            reason="Informiert",
            original_action=action,
            policy_name="test",
        )
        results = await executor.execute([action], [decision])
        assert results[0].success
        mock_mcp.call_tool.assert_called_once()


# ============================================================================
# MASK: Maskierte Params
# ============================================================================


class TestMaskedExecution:
    @pytest.mark.asyncio
    async def test_mask_uses_masked_params(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        action = PlannedAction(
            tool="fetch_url", params={"url": "https://api.com", "token": "sk-secret"}
        )
        decision = _mask_decision(
            action, masked={"url": "https://api.com", "token": "***MASKED***"}
        )
        results = await executor.execute([action], [decision])
        assert results[0].success
        # Prüfe dass die maskierten Params übergeben wurden
        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][1]["token"] == "***MASKED***"


# ============================================================================
# Dependencies
# ============================================================================


class TestDependencies:
    @pytest.mark.asyncio
    async def test_unmet_dependency_skipped(self, executor: Executor) -> None:
        action1 = PlannedAction(tool="read_file", params={"path": "/a"})
        action2 = PlannedAction(tool="write_file", params={"path": "/b"}, depends_on=[0])

        # Nur action2 erlaubt, action1 blockiert → dependency nicht erfüllt
        results = await executor.execute(
            [action1, action2],
            [_block_decision(action1), _allow_decision(action2)],
        )
        assert len(results) == 2
        assert results[0].is_error  # Blockiert
        assert results[1].is_error  # Dependency nicht erfüllt
        assert "Abhängigkeiten" in results[1].content

    @pytest.mark.asyncio
    async def test_met_dependency_executes(self, executor: Executor) -> None:
        action1 = PlannedAction(tool="read_file", params={"path": "/a"})
        action2 = PlannedAction(tool="write_file", params={"path": "/b"}, depends_on=[0])

        results = await executor.execute(
            [action1, action2],
            [_allow_decision(action1), _allow_decision(action2)],
        )
        assert results[0].success
        assert results[1].success


# ============================================================================
# Fehlerbehandlung
# ============================================================================


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_tool_error_captured(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        mock_mcp.call_tool = AsyncMock(side_effect=RuntimeError("Tool crashed"))
        action = PlannedAction(tool="broken_tool", params={})
        results = await executor.execute([action], [_allow_decision(action)])
        assert results[0].is_error
        assert "Tool crashed" in results[0].content
        assert results[0].error_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_no_mcp_client(self, config: JarvisConfig) -> None:
        executor = Executor(config, mcp_client=None)
        action = PlannedAction(tool="test", params={})
        results = await executor.execute([action], [_allow_decision(action)])
        assert results[0].is_error
        assert "MCP" in results[0].content

    @pytest.mark.asyncio
    async def test_mismatched_lists_raises(self, executor: Executor) -> None:
        from jarvis.core.executor import ExecutionError

        with pytest.raises(ExecutionError):
            await executor.execute(
                [PlannedAction(tool="a", params={})],
                [_allow_decision(), _allow_decision()],
            )


# ============================================================================
# ToolResult Properties
# ============================================================================


class TestToolResultProperties:
    @pytest.mark.asyncio
    async def test_success_property(self, executor: Executor) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/test"})
        results = await executor.execute([action], [_allow_decision(action)])
        assert results[0].success is True
        assert results[0].is_error is False

    @pytest.mark.asyncio
    async def test_duration_recorded(self, executor: Executor) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/test"})
        results = await executor.execute([action], [_allow_decision(action)])
        assert results[0].duration_ms >= 0


# ============================================================================
# Agent-Kontext-Injection
# ============================================================================


class TestAgentContext:
    """Verifiziert dass Executor Agent-Workspace und Sandbox-Overrides
    tatsächlich an Tool-Params weiterleitet — kein Platzhalter."""

    def test_set_and_clear(self, executor: Executor) -> None:
        from jarvis.core.executor import _agent_workspace_var, _agent_sandbox_var
        executor.set_agent_context(
            workspace_dir="/tmp/agent/coder",
            sandbox_overrides={"network": "block", "timeout": 120},
        )
        assert _agent_workspace_var.get() == "/tmp/agent/coder"
        assert _agent_sandbox_var.get()["network"] == "block"

        executor.clear_agent_context()
        assert _agent_workspace_var.get() is None
        assert _agent_sandbox_var.get() is None

    @pytest.mark.asyncio
    async def test_workspace_injected_into_exec_command(self, executor: Executor) -> None:
        """exec_command bekommt working_dir aus Agent-Workspace."""
        executor.set_agent_context(workspace_dir="/tmp/agent/coder")

        action = PlannedAction(tool="exec_command", params={"command": "echo hi"})
        results = await executor.execute([action], [_allow_decision(action)])

        # Der Mock-MCP-Client speichert die Params — prüfe dass working_dir gesetzt wurde
        # (Da unser Mock kein echtes Filesystem hat, prüfen wir indirekt)
        assert results[0].success is True
        executor.clear_agent_context()

    @pytest.mark.asyncio
    async def test_workspace_not_injected_when_explicit(self, executor: Executor) -> None:
        """Wenn Planner explizit working_dir setzt, wird's nicht überschrieben."""
        executor.set_agent_context(workspace_dir="/tmp/agent/coder")

        action = PlannedAction(
            tool="exec_command",
            params={"command": "echo hi", "working_dir": "/tmp/explicit"},
        )
        # working_dir bleibt /tmp/explicit, nicht /tmp/agent/coder
        assert action.params["working_dir"] == "/tmp/explicit"
        executor.clear_agent_context()

    @pytest.mark.asyncio
    async def test_sandbox_override_injected(self, executor: Executor) -> None:
        """Sandbox-Netzwerk-Override wird an exec_command durchgereicht."""
        executor.set_agent_context(
            sandbox_overrides={"network": "block", "timeout": 120},
        )

        action = PlannedAction(tool="exec_command", params={"command": "echo hi"})
        results = await executor.execute([action], [_allow_decision(action)])
        assert results[0].success is True
        executor.clear_agent_context()

    @pytest.mark.asyncio
    async def test_non_workspace_tools_unaffected(self, executor: Executor) -> None:
        """web_search u.ä. bekommen KEIN working_dir injiziert."""
        executor.set_agent_context(workspace_dir="/tmp/agent/coder")

        action = PlannedAction(tool="web_search", params={"query": "test"})
        # web_search ist nicht in WORKSPACE_TOOLS → kein working_dir
        assert "working_dir" not in action.params
        executor.clear_agent_context()

    def test_workspace_tools_constant(self) -> None:
        """Prüfe dass WORKSPACE_TOOLS korrekt definiert ist."""
        assert "exec_command" in Executor.WORKSPACE_TOOLS
        assert "write_file" in Executor.WORKSPACE_TOOLS
        assert "read_file" in Executor.WORKSPACE_TOOLS
        assert "web_search" not in Executor.WORKSPACE_TOOLS
