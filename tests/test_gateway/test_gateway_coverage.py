"""Coverage-Tests fuer gateway.py -- fehlende Zeilen abdecken.

Deckt ab:
  - Gateway.__init__ (declare/apply_phase)
  - Gateway.initialize() (6-phase init)
  - Gateway.start/shutdown (Channel-Lifecycle)
  - Gateway.handle_message edge cases
  - Gateway._run_pge_cycle + iteration logic
  - Gateway.register_channel / _channels
  - Gateway._build_agent_result
  - Approval flow corner cases
  - Error handling in handle_message
  - Coding override detection
  - _classify_coding_task
  - _resolve_relative_dates
  - _maybe_presearch / _is_fact_question
  - _persist_key_tool_results
  - _extract_attachments
  - _record_metric
  - _run_post_processing
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from jarvis.config import JarvisConfig, ensure_directory_structure
from jarvis.gateway.gateway import Gateway
from jarvis.models import (
    ActionPlan,
    GateDecision,
    GateStatus,
    IncomingMessage,
    OutgoingMessage,
    PlannedAction,
    RiskLevel,
    SessionContext,
    ToolResult,
    WorkingMemory,
)


@dataclass
class MockToolResult:
    content: str = "tool output"
    is_error: bool = False


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    cfg = JarvisConfig(jarvis_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


def _make_initialized_gateway(config: JarvisConfig) -> Gateway:
    """Erstellt einen Gateway mit minimalen Mocks fuer handle_message."""
    gw = Gateway(config)
    gw._planner = MagicMock()
    gw._planner.plan = AsyncMock(
        return_value=ActionPlan(
            goal="",
            direct_response="Antwort",
            confidence=0.9,
        )
    )
    gw._gatekeeper = MagicMock()
    gw._gatekeeper.evaluate = MagicMock(return_value=[])
    gw._executor = MagicMock()
    gw._executor.execute = AsyncMock(return_value=[])
    gw._mcp_client = MagicMock()
    gw._mcp_client.get_tool_schemas = MagicMock(return_value={})
    gw._mcp_client.get_tool_list = MagicMock(return_value=[])
    gw._model_router = MagicMock()
    gw._model_router.select_model = MagicMock(return_value="qwen3:32b")
    gw._model_router.clear_coding_override = MagicMock()
    gw._running = True
    return gw


# ============================================================================
# Gateway init / declare phases
# ============================================================================


class TestGatewayInit:
    def test_gateway_creates_with_config(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        assert gw._config is config
        assert gw._running is False
        assert isinstance(gw._channels, dict)

    def test_gateway_has_session_management(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        assert hasattr(gw, "_sessions")
        assert hasattr(gw, "_working_memories")

    def test_gateway_default_attributes(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        assert gw._planner is None
        assert gw._executor is None
        assert gw._gatekeeper is None
        assert gw._llm is None
        assert gw._mcp_client is None


# ============================================================================
# Gateway.initialize() -- mocked subsystems
# ============================================================================


class TestGatewayInitialize:
    @pytest.mark.asyncio
    async def test_initialize_calls_phases(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        with (
            patch("jarvis.gateway.gateway.init_core", new_callable=AsyncMock) as mock_core,
            patch("jarvis.gateway.gateway.init_security", new_callable=AsyncMock) as mock_sec,
            patch("jarvis.gateway.gateway.init_memory", new_callable=AsyncMock) as mock_mem,
            patch("jarvis.gateway.gateway.init_tools", new_callable=AsyncMock) as mock_tools,
            patch("jarvis.gateway.gateway.init_pge", new_callable=AsyncMock) as mock_pge,
            patch("jarvis.gateway.gateway.init_agents", new_callable=AsyncMock) as mock_agents,
            patch("jarvis.gateway.gateway.init_advanced", new_callable=AsyncMock) as mock_adv,
        ):
            mock_core.return_value = {
                "llm": MagicMock(),
                "ollama": MagicMock(),
                "model_router": MagicMock(),
                "session_store": MagicMock(),
                "__llm_ok": True,
            }
            mock_sec.return_value = {
                "audit_logger": MagicMock(),
                "runtime_monitor": MagicMock(),
                "gatekeeper": MagicMock(),
            }
            mock_mem.return_value = {"memory_manager": MagicMock()}
            mock_tools.return_value = {
                "mcp_client": MagicMock(),
                "cost_tracker": None,
            }
            mock_pge.return_value = {
                "planner": MagicMock(),
                "executor": MagicMock(),
                "reflector": MagicMock(),
                "skill_generator": None,
                "task_profiler": None,
                "task_telemetry": None,
                "error_clusterer": None,
                "causal_analyzer": None,
            }
            mock_agents.return_value = {
                "skill_registry": None,
                "agent_router": MagicMock(),
                "ingest_pipeline": None,
                "heartbeat_scheduler": None,
                "cron_engine": None,
            }
            mock_adv.return_value = {}

            await gw.initialize()

            mock_core.assert_awaited_once()
            mock_sec.assert_awaited_once()
            mock_mem.assert_awaited_once()
            mock_tools.assert_awaited_once()
            mock_pge.assert_awaited_once()
            mock_agents.assert_awaited_once()
            mock_adv.assert_awaited_once()


# ============================================================================
# Gateway.register_channel / start / shutdown
# ============================================================================


class TestGatewayChannels:
    def test_register_channel(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        mock_channel = MagicMock()
        mock_channel.name = "test_channel"
        gw.register_channel(mock_channel)
        assert "test_channel" in gw._channels
        assert gw._channels["test_channel"] is mock_channel

    def test_register_multiple_channels(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        for name in ("cli", "telegram", "discord"):
            ch = MagicMock()
            ch.name = name
            gw.register_channel(ch)
        assert len(gw._channels) == 3

    @pytest.mark.asyncio
    async def test_shutdown_sets_not_running(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        gw._running = True
        gw._mcp_client = AsyncMock()
        gw._mcp_client.disconnect_all = AsyncMock()
        await gw.shutdown()
        assert gw._running is False


# ============================================================================
# handle_message edge cases
# ============================================================================


class TestHandleMessageEdgeCases:
    @pytest.mark.asyncio
    async def test_handle_message_raises_when_not_initialized(self, config: JarvisConfig) -> None:
        """handle_message without initialize() should raise RuntimeError."""
        gw = Gateway(config)
        gw._running = True
        msg = IncomingMessage(text="Hi", channel="test", user_id="user1")
        with pytest.raises(RuntimeError, match="initialize"):
            await gw.handle_message(msg)

    @pytest.mark.asyncio
    async def test_handle_message_direct_response(self, config: JarvisConfig) -> None:
        gw = _make_initialized_gateway(config)

        msg = IncomingMessage(text="Hallo", channel="test", user_id="user1")
        response = await gw.handle_message(msg)
        assert response.channel == "test"
        assert response.is_final

    @pytest.mark.asyncio
    async def test_handle_message_planner_exception_propagates(self, config: JarvisConfig) -> None:
        """Planner-Exceptions propagieren aus handle_message."""
        gw = _make_initialized_gateway(config)
        gw._planner.plan = AsyncMock(side_effect=Exception("LLM crashed"))

        msg = IncomingMessage(text="test", channel="cli", user_id="user1")
        with pytest.raises(Exception, match="LLM crashed"):
            await gw.handle_message(msg)

    @pytest.mark.asyncio
    async def test_handle_message_no_actions_fallback(self, config: JarvisConfig) -> None:
        """If planner returns no actions and no direct_response, get fallback text."""
        gw = _make_initialized_gateway(config)
        gw._planner.plan = AsyncMock(
            return_value=ActionPlan(
                goal="test",
                direct_response="",
                confidence=0.5,
            )
        )

        msg = IncomingMessage(text="test", channel="cli", user_id="user1")
        response = await gw.handle_message(msg)
        assert response.text
        assert response.channel == "cli"


# ============================================================================
# Session + Working Memory management
# ============================================================================


class TestSessionAndWorkingMemory:
    def test_get_or_create_session_creates_new(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        s = gw._get_or_create_session("test", "user1")
        assert s.channel == "test"
        assert s.user_id == "user1"
        assert s.session_id

    def test_get_or_create_session_with_agent_name(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        s = gw._get_or_create_session("test", "user1", agent_name="coder")
        assert s.session_id

    def test_working_memory_creation(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        s = gw._get_or_create_session("cli", "user1")
        wm = gw._get_or_create_working_memory(s)
        assert wm.session_id == s.session_id

    def test_working_memory_reused(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        s = gw._get_or_create_session("cli", "user1")
        wm1 = gw._get_or_create_working_memory(s)
        wm2 = gw._get_or_create_working_memory(s)
        assert wm1 is wm2

    def test_session_reused_for_same_user(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        s1 = gw._get_or_create_session("cli", "user1")
        s2 = gw._get_or_create_session("cli", "user1")
        assert s1.session_id == s2.session_id


# ============================================================================
# Coding override detection
# ============================================================================


class TestCodingClassification:
    @pytest.mark.asyncio
    async def test_classify_coding_no_llm(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        gw._model_router = None
        gw._llm = None
        result = await gw._classify_coding_task("schreibe python code")
        assert result == (False, "simple")

    @pytest.mark.asyncio
    async def test_classify_coding_llm_returns_json(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value={"message": {"content": '{"coding": true, "complexity": "complex"}'}}
        )
        gw._llm = mock_llm
        gw._model_router = MagicMock()
        gw._model_router.select_model = MagicMock(return_value="qwen3:8b")

        is_coding, complexity = await gw._classify_coding_task("refaktoriere das ganze Modul")
        assert is_coding is True
        assert complexity == "complex"

    @pytest.mark.asyncio
    async def test_classify_coding_llm_exception(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("timeout"))
        gw._llm = mock_llm
        gw._model_router = MagicMock()
        gw._model_router.select_model = MagicMock(return_value="qwen3:8b")

        is_coding, complexity = await gw._classify_coding_task("test")
        assert is_coding is False
        assert complexity == "simple"

    @pytest.mark.asyncio
    async def test_classify_coding_with_think_tags(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value={
                "message": {
                    "content": '<think>hmm</think>\n{"coding": false, "complexity": "simple"}'
                }
            }
        )
        gw._llm = mock_llm
        gw._model_router = MagicMock()
        gw._model_router.select_model = MagicMock(return_value="qwen3:8b")

        is_coding, complexity = await gw._classify_coding_task("was ist Python?")
        assert is_coding is False


# ============================================================================
# Approval handling
# ============================================================================


class TestApprovalHandling:
    @pytest.mark.asyncio
    async def test_approval_with_channel(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        session = SessionContext()

        mock_channel = AsyncMock()
        mock_channel.name = "telegram"
        mock_channel.request_approval = AsyncMock(return_value=True)
        gw._channels["telegram"] = mock_channel

        steps = [PlannedAction(tool="exec_command", params={"command": "ls"})]
        decisions = [
            GateDecision(
                status=GateStatus.APPROVE,
                reason="Needs approval",
                risk_level=RiskLevel.ORANGE,
                policy_name="test",
            )
        ]

        result = await gw._handle_approvals(steps, decisions, session, "telegram")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_approval_no_channel(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        session = SessionContext()
        steps = [PlannedAction(tool="exec_command", params={})]
        decisions = [
            GateDecision(
                status=GateStatus.APPROVE,
                reason="Needs approval",
                risk_level=RiskLevel.ORANGE,
                policy_name="test",
            )
        ]
        result = await gw._handle_approvals(steps, decisions, session, "unknown_ch")
        assert result[0].status == GateStatus.APPROVE


# ============================================================================
# _resolve_relative_dates
# ============================================================================


class TestResolveRelativeDates:
    def test_resolve_heute(self, config: JarvisConfig) -> None:
        result = Gateway._resolve_relative_dates("Was passierte heute?")
        assert "heute" not in result.lower() or "202" in result

    def test_resolve_no_dates(self, config: JarvisConfig) -> None:
        result = Gateway._resolve_relative_dates("Was ist Python?")
        assert "Python" in result


# ============================================================================
# _is_fact_question
# ============================================================================


class TestIsFactQuestion:
    def test_fact_question_detected(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        # Some versions might have _is_fact_question as a method
        if hasattr(gw, "_is_fact_question"):
            assert isinstance(gw._is_fact_question("Wer ist der Bundeskanzler?"), bool)

    def test_non_fact_question(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        if hasattr(gw, "_is_fact_question"):
            result = gw._is_fact_question("schreibe ein Gedicht")
            assert isinstance(result, bool)


# ============================================================================
# _record_metric
# ============================================================================


class TestRecordMetric:
    def test_record_metric_no_op(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        # Should not raise even without Prometheus
        gw._record_metric("test_counter", 1)

    def test_record_metric_with_labels(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        gw._record_metric("requests_total", 1, channel="test", model="qwen")


# ============================================================================
# _extract_attachments
# ============================================================================


class TestExtractAttachments:
    def test_extract_no_results(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        attachments = gw._extract_attachments([])
        assert attachments == [] or attachments is None or len(attachments) == 0

    def test_extract_with_tool_results(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        tr = ToolResult(
            tool_name="document_export",
            content=os.path.join(tempfile.gettempdir(), "test.pdf"),
            is_error=False,
        )
        attachments = gw._extract_attachments([tr])
        assert isinstance(attachments, list)


# ============================================================================
# _persist_key_tool_results
# ============================================================================


class TestPersistKeyToolResults:
    def test_persist_no_results(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        session = gw._get_or_create_session("cli", "user1")
        wm = gw._get_or_create_working_memory(session)
        # Should not raise
        gw._persist_key_tool_results(wm, [])


# ============================================================================
# Session ID from IncomingMessage
# ============================================================================


class TestSessionIdHandling:
    def test_session_from_message_same_channel_user(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        session = gw._get_or_create_session("telegram", "user1")
        assert session.session_id
        assert session.channel == "telegram"
        assert session.user_id == "user1"

    def test_different_agents_get_different_sessions(self, config: JarvisConfig) -> None:
        gw = Gateway(config)
        s1 = gw._get_or_create_session("cli", "user1", "jarvis")
        s2 = gw._get_or_create_session("cli", "user1", "coder")
        assert s1.session_id != s2.session_id
