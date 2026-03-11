"""Coverage-Tests fuer executor.py -- fehlende Zeilen (retry, backoff, edge cases)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import JarvisConfig
from jarvis.core.executor import Executor, ExecutionError
from jarvis.models import GateDecision, GateStatus, PlannedAction, RiskLevel, ToolResult


@dataclass
class MockToolResult:
    content: str = "OK"
    is_error: bool = False


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    return JarvisConfig(jarvis_home=tmp_path)


@pytest.fixture()
def mock_mcp() -> AsyncMock:
    mcp = AsyncMock()
    mcp.call_tool = AsyncMock(return_value=MockToolResult(content="tool output"))
    return mcp


@pytest.fixture()
def executor(config: JarvisConfig, mock_mcp: AsyncMock) -> Executor:
    return Executor(config, mock_mcp)


def _allow(action=None):
    return GateDecision(
        status=GateStatus.ALLOW,
        risk_level=RiskLevel.GREEN,
        reason="OK",
        original_action=action,
        policy_name="test",
    )


def _block(action=None):
    return GateDecision(
        status=GateStatus.BLOCK,
        risk_level=RiskLevel.RED,
        reason="Blocked",
        original_action=action,
        policy_name="test",
    )


def _approve(action=None):
    return GateDecision(
        status=GateStatus.APPROVE,
        risk_level=RiskLevel.ORANGE,
        reason="Needs approval",
        original_action=action,
        policy_name="test",
    )


# ============================================================================
# Basic execution
# ============================================================================


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_allow_action_executed(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        action = PlannedAction(tool="list_directory", params={"path": "/tmp"})
        results = await executor.execute([action], [_allow(action)])
        assert results[0].success
        mock_mcp.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_block_action_skipped(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        """BLOCK status should skip the action."""
        action = PlannedAction(tool="dangerous_tool", params={})
        results = await executor.execute([action], [_block(action)])
        assert results[0].is_error
        assert "GatekeeperBlock" in (results[0].error_type or "")
        mock_mcp.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_approve_action_skipped(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        """APPROVE status is NOT in the allowed set, so it's skipped (needs pre-approval)."""
        action = PlannedAction(tool="exec_command", params={"command": "ls"})
        results = await executor.execute([action], [_approve(action)])
        assert results[0].is_error
        mock_mcp.call_tool.assert_not_called()


# ============================================================================
# Multiple actions with dependencies
# ============================================================================


class TestMultipleActionsWithDeps:
    @pytest.mark.asyncio
    async def test_three_chained_actions(self, executor: Executor) -> None:
        a1 = PlannedAction(tool="read_file", params={"path": "/a"})
        a2 = PlannedAction(tool="read_file", params={"path": "/b"}, depends_on=[0])
        a3 = PlannedAction(tool="write_file", params={"path": "/c"}, depends_on=[1])

        results = await executor.execute(
            [a1, a2, a3],
            [_allow(a1), _allow(a2), _allow(a3)],
        )
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_unmet_dependency_skipped(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        """If dependency fails, dependent action is skipped."""
        mock_mcp.call_tool = AsyncMock(side_effect=ValueError("fail"))
        a1 = PlannedAction(tool="read_file", params={"path": "/a"})
        a2 = PlannedAction(tool="write_file", params={"path": "/b"}, depends_on=[0])

        results = await executor.execute(
            [a1, a2],
            [_allow(a1), _allow(a2)],
        )
        assert len(results) == 2
        # First action failed, second was skipped due to unmet dependency
        assert results[0].is_error
        assert results[1].is_error
        assert "DependencyError" in (results[1].error_type or "")


# ============================================================================
# Mismatched actions/decisions
# ============================================================================


class TestMismatchedInputs:
    @pytest.mark.asyncio
    async def test_mismatched_lengths(self, executor: Executor) -> None:
        a1 = PlannedAction(tool="test", params={})
        with pytest.raises(ExecutionError):
            await executor.execute([a1], [])


# ============================================================================
# Tool result with is_error from MCP
# ============================================================================


class TestMCPErrorResult:
    @pytest.mark.asyncio
    async def test_mcp_returns_error_result(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        mock_mcp.call_tool = AsyncMock(
            return_value=MockToolResult(content="Permission denied", is_error=True)
        )
        action = PlannedAction(tool="exec_command", params={"command": "rm -rf /"})
        results = await executor.execute([action], [_allow(action)])
        assert results[0].is_error


# ============================================================================
# Timeout behavior
# ============================================================================


class TestTimeoutBehavior:
    @pytest.mark.asyncio
    async def test_timeout_action(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        import asyncio

        mock_mcp.call_tool = AsyncMock(side_effect=asyncio.TimeoutError("Tool timed out"))
        action = PlannedAction(tool="slow_tool", params={})
        results = await executor.execute([action], [_allow(action)])
        assert results[0].is_error


# ============================================================================
# Error type recording
# ============================================================================


class TestErrorTypeRecording:
    @pytest.mark.asyncio
    async def test_error_type_recorded(self, executor: Executor, mock_mcp: AsyncMock) -> None:
        mock_mcp.call_tool = AsyncMock(side_effect=ValueError("bad input"))
        action = PlannedAction(tool="bad_tool", params={})
        results = await executor.execute([action], [_allow(action)])
        assert results[0].error_type == "ValueError"
        assert "bad input" in results[0].content


# ============================================================================
# Agent context
# ============================================================================


class TestAgentContext:
    def test_set_and_clear_agent_context(self, executor: Executor) -> None:
        executor.set_agent_context(
            workspace_dir=str(Path(tempfile.gettempdir()) / "agent"),
            sandbox_overrides={"network": False},
            agent_name="test-agent",
            session_id="sess-123",
        )
        executor.clear_agent_context()

    def test_set_mcp_client(self, config: JarvisConfig) -> None:
        executor = Executor(config)
        mock_mcp = AsyncMock()
        executor.set_mcp_client(mock_mcp)
        assert executor._mcp_client is mock_mcp


# ============================================================================
# Helpers for MASK / INFORM decisions
# ============================================================================


def _mask(action=None, masked_params=None):
    return GateDecision(
        status=GateStatus.MASK,
        risk_level=RiskLevel.YELLOW,
        reason="Masked",
        original_action=action,
        policy_name="test",
        masked_params=masked_params,
    )


def _inform(action=None):
    return GateDecision(
        status=GateStatus.INFORM,
        risk_level=RiskLevel.YELLOW,
        reason="Info",
        original_action=action,
        policy_name="test",
    )


# ============================================================================
# Retry / Backoff behaviour
# ============================================================================


class TestRetryBehavior:
    """Tests for retry logic with retryable and non-retryable errors."""

    @pytest.mark.asyncio
    async def test_retryable_error_retries(
        self,
        executor: Executor,
        mock_mcp: AsyncMock,
    ) -> None:
        """ConnectionError is retryable -- should retry and eventually succeed."""
        executor._max_retries = 3
        executor._base_delay = 0.001

        mock_mcp.call_tool = AsyncMock(
            side_effect=[
                ConnectionError("fail1"),
                ConnectionError("fail2"),
                MockToolResult(content="OK"),
            ],
        )
        action = PlannedAction(tool="read_file", params={"path": "/a"})
        results = await executor.execute([action], [_allow(action)])

        assert len(results) == 1
        assert results[0].success
        assert results[0].content == "OK"
        assert mock_mcp.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(
        self,
        executor: Executor,
        mock_mcp: AsyncMock,
    ) -> None:
        """ValueError is NOT retryable -- should fail immediately without retry."""
        executor._max_retries = 3
        executor._base_delay = 0.001

        mock_mcp.call_tool = AsyncMock(side_effect=ValueError("bad input"))
        action = PlannedAction(tool="bad_tool", params={})
        results = await executor.execute([action], [_allow(action)])

        assert len(results) == 1
        assert results[0].is_error
        assert results[0].error_type == "ValueError"
        # Non-retryable: only ONE call, no retries
        assert mock_mcp.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(
        self,
        executor: Executor,
        mock_mcp: AsyncMock,
    ) -> None:
        """All attempts raise ConnectionError -> final error result."""
        executor._max_retries = 3
        executor._base_delay = 0.001

        mock_mcp.call_tool = AsyncMock(
            side_effect=ConnectionError("persistent failure"),
        )
        action = PlannedAction(tool="flaky_tool", params={})
        results = await executor.execute([action], [_allow(action)])

        assert len(results) == 1
        assert results[0].is_error
        assert results[0].error_type == "ConnectionError"
        assert "persistent failure" in results[0].content
        # All 3 retries attempted
        assert mock_mcp.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_error_retried(
        self,
        executor: Executor,
        mock_mcp: AsyncMock,
    ) -> None:
        """asyncio.TimeoutError is retryable -- should retry and succeed."""
        import asyncio as _asyncio

        executor._max_retries = 3
        executor._base_delay = 0.001

        mock_mcp.call_tool = AsyncMock(
            side_effect=[
                _asyncio.TimeoutError("slow"),
                MockToolResult(content="recovered"),
            ],
        )
        action = PlannedAction(tool="slow_tool", params={})
        results = await executor.execute([action], [_allow(action)])

        assert len(results) == 1
        assert results[0].success
        assert results[0].content == "recovered"
        assert mock_mcp.call_tool.call_count == 2


# ============================================================================
# Output truncation
# ============================================================================


class TestOutputTruncation:
    @pytest.mark.asyncio
    async def test_long_output_truncated(
        self,
        executor: Executor,
        mock_mcp: AsyncMock,
    ) -> None:
        """Tool output exceeding max_output_chars is truncated."""
        big_content = "X" * 50_000
        mock_mcp.call_tool = AsyncMock(
            return_value=MockToolResult(content=big_content),
        )
        # Ensure max_output is smaller than the content
        executor._max_output = 10_000

        action = PlannedAction(tool="big_tool", params={})
        results = await executor.execute([action], [_allow(action)])

        assert len(results) == 1
        assert results[0].success
        assert results[0].truncated is True
        assert len(results[0].content) == 10_000


# ============================================================================
# MASK status
# ============================================================================


class TestMaskStatus:
    @pytest.mark.asyncio
    async def test_mask_uses_masked_params(
        self,
        executor: Executor,
        mock_mcp: AsyncMock,
    ) -> None:
        """MASK decision should pass masked_params to the tool instead of original."""
        original_params = {"api_key": "SECRET_123", "query": "hello"}
        masked_params = {"api_key": "***MASKED***", "query": "hello"}
        action = PlannedAction(tool="web_search", params=original_params)
        decision = _mask(action=action, masked_params=masked_params)

        results = await executor.execute([action], [decision])

        assert len(results) == 1
        assert results[0].success
        # Verify the MCP client was called with the masked params, not the original
        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][1]["api_key"] == "***MASKED***"
        assert call_args[0][1]["query"] == "hello"


# ============================================================================
# INFORM status
# ============================================================================


class TestInformStatus:
    @pytest.mark.asyncio
    async def test_inform_status_allowed(
        self,
        executor: Executor,
        mock_mcp: AsyncMock,
    ) -> None:
        """INFORM decision allows execution just like ALLOW."""
        action = PlannedAction(tool="write_file", params={"path": "/a", "content": "x"})
        decision = _inform(action=action)

        results = await executor.execute([action], [decision])

        assert len(results) == 1
        assert results[0].success
        mock_mcp.call_tool.assert_called_once()


# ============================================================================
# No MCP client
# ============================================================================


class TestNoMCPClient:
    @pytest.mark.asyncio
    async def test_no_mcp_client_error(self, config: JarvisConfig) -> None:
        """Executor without mcp_client should return an error result."""
        exec_no_mcp = Executor(config, mcp_client=None)
        action = PlannedAction(tool="any_tool", params={})
        results = await exec_no_mcp.execute([action], [_allow(action)])

        assert len(results) == 1
        assert results[0].is_error
        assert results[0].error_type == "NoMCPClient"


# ============================================================================
# Runtime monitor block
# ============================================================================


class TestRuntimeMonitorBlock:
    @pytest.mark.asyncio
    async def test_runtime_monitor_blocks(
        self,
        config: JarvisConfig,
        mock_mcp: AsyncMock,
    ) -> None:
        """RuntimeMonitor blocking a tool should produce a SecurityBlock error."""
        monitor = MagicMock()
        security_event = MagicMock()
        security_event.is_blocked = True
        security_event.description = "Rate limit exceeded"
        monitor.check_tool_call = MagicMock(return_value=security_event)

        exec_with_monitor = Executor(
            config,
            mcp_client=mock_mcp,
            runtime_monitor=monitor,
        )
        action = PlannedAction(tool="exec_command", params={"command": "rm -rf /"})
        results = await exec_with_monitor.execute([action], [_allow(action)])

        assert len(results) == 1
        assert results[0].is_error
        assert results[0].error_type == "SecurityBlock"
        assert "Rate limit exceeded" in results[0].content
        # The actual MCP call should never have been made
        mock_mcp.call_tool.assert_not_called()


# ============================================================================
# Audit logger
# ============================================================================


class TestAuditLogger:
    @pytest.mark.asyncio
    async def test_audit_logger_called_on_success(
        self,
        config: JarvisConfig,
        mock_mcp: AsyncMock,
    ) -> None:
        """On successful tool execution, audit_logger.log_tool_call is called with success=True."""
        audit = MagicMock()
        exec_with_audit = Executor(
            config,
            mcp_client=mock_mcp,
            audit_logger=audit,
        )
        action = PlannedAction(tool="list_directory", params={"path": "/tmp"})
        results = await exec_with_audit.execute([action], [_allow(action)])

        assert results[0].success
        audit.log_tool_call.assert_called_once()
        call_kwargs = audit.log_tool_call.call_args
        # success=True should be among the keyword arguments
        assert call_kwargs.kwargs.get("success") is True or call_kwargs[1].get("success") is True

    @pytest.mark.asyncio
    async def test_audit_logger_called_on_failure(
        self,
        config: JarvisConfig,
        mock_mcp: AsyncMock,
    ) -> None:
        """On failed tool execution, audit_logger.log_tool_call is called with success=False."""
        audit = MagicMock()
        mock_mcp_fail = AsyncMock()
        mock_mcp_fail.call_tool = AsyncMock(side_effect=ValueError("bad"))
        exec_with_audit = Executor(
            config,
            mcp_client=mock_mcp_fail,
            audit_logger=audit,
        )
        action = PlannedAction(tool="bad_tool", params={})
        results = await exec_with_audit.execute([action], [_allow(action)])

        assert results[0].is_error
        audit.log_tool_call.assert_called_once()
        call_kwargs = audit.log_tool_call.call_args
        assert call_kwargs.kwargs.get("success") is False or call_kwargs[1].get("success") is False


# ============================================================================
# Gap detector
# ============================================================================


class TestGapDetector:
    @pytest.mark.asyncio
    async def test_gap_detector_unknown_tool(
        self,
        config: JarvisConfig,
        mock_mcp: AsyncMock,
    ) -> None:
        """Non-retryable error -> gap_detector.report_unknown_tool is called."""
        gap = MagicMock()
        mock_mcp_fail = AsyncMock()
        mock_mcp_fail.call_tool = AsyncMock(side_effect=ValueError("not found"))
        exec_with_gap = Executor(
            config,
            mcp_client=mock_mcp_fail,
            gap_detector=gap,
        )
        action = PlannedAction(tool="unknown_tool", params={})
        results = await exec_with_gap.execute([action], [_allow(action)])

        assert results[0].is_error
        gap.report_unknown_tool.assert_called_once()
        assert gap.report_unknown_tool.call_args[0][0] == "unknown_tool"

    @pytest.mark.asyncio
    async def test_gap_detector_repeated_failure(
        self,
        config: JarvisConfig,
        mock_mcp: AsyncMock,
    ) -> None:
        """All retries exhausted -> gap_detector.report_repeated_failure is called."""
        gap = MagicMock()
        mock_mcp_fail = AsyncMock()
        mock_mcp_fail.call_tool = AsyncMock(
            side_effect=ConnectionError("keep failing"),
        )
        exec_with_gap = Executor(
            config,
            mcp_client=mock_mcp_fail,
            gap_detector=gap,
        )
        exec_with_gap._max_retries = 3
        exec_with_gap._base_delay = 0.001

        action = PlannedAction(tool="flaky_tool", params={})
        results = await exec_with_gap.execute([action], [_allow(action)])

        assert results[0].is_error
        gap.report_repeated_failure.assert_called_once()
        assert gap.report_repeated_failure.call_args[0][0] == "flaky_tool"


# ============================================================================
# Workspace injection (Agent context)
# ============================================================================


class TestWorkspaceInjection:
    @pytest.mark.asyncio
    async def test_workspace_tool_gets_working_dir(
        self,
        config: JarvisConfig,
        mock_mcp: AsyncMock,
    ) -> None:
        """exec_command with agent workspace context -> working_dir injected."""
        exec_ws = Executor(config, mcp_client=mock_mcp)
        exec_ws.set_agent_context(
            workspace_dir=str(Path(tempfile.gettempdir()) / "agent_workspace"),
            agent_name="test-agent",
            session_id="sess-001",
        )
        try:
            action = PlannedAction(tool="exec_command", params={"command": "ls"})
            results = await exec_ws.execute([action], [_allow(action)])

            assert results[0].success
            # Verify the working_dir was injected into the call params
            call_args = mock_mcp.call_tool.call_args
            passed_params = call_args[0][1]
            assert passed_params.get("working_dir") == str(Path(tempfile.gettempdir()) / "agent_workspace")
        finally:
            exec_ws.clear_agent_context()
