"""Integration-Tests für den Gateway – vollständiger PGE-Zyklus.

Testet:
  - Vollständiger Agent-Loop: Message → Planner → Gatekeeper → Executor → Response
  - Direkte Antwort (kein Tool nötig)
  - Tool-Ausführung mit Gatekeeper-Prüfung
  - Session-Management
  - Working Memory
  - Blockierung und Eskalation
  - Mehrere Iterationen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.config import JarvisConfig, ensure_directory_structure
from jarvis.gateway.gateway import Gateway
from jarvis.models import IncomingMessage

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class MockToolResult:
    content: str = "tool output"
    is_error: bool = False


@pytest.fixture()
def config(tmp_path: Path) -> JarvisConfig:
    cfg = JarvisConfig(jarvis_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


def _mock_ollama_chat(response_text: str = "Hallo!", tool_calls: list | None = None):
    """Erstellt einen Mock für ollama.chat()."""
    result = {
        "message": {
            "role": "assistant",
            "content": response_text,
        },
    }
    if tool_calls:
        result["message"]["tool_calls"] = tool_calls
    return result


# ============================================================================
# Vollständiger PGE-Zyklus
# ============================================================================


class TestFullPGECycle:
    """Ende-zu-Ende Tests für den Agent-Loop."""

    @pytest.mark.asyncio
    async def test_direct_response_no_tools(self, config: JarvisConfig) -> None:
        """Einfache Frage → direkte Antwort ohne Tool-Calls."""
        gateway = Gateway(config)

        with (
            patch.object(gateway, "_ollama") as mock_ollama,
            patch.object(gateway, "_model_router") as mock_router,
            patch.object(gateway, "_mcp_client") as mock_mcp,
        ):
            # Setup mocks
            mock_ollama.chat = AsyncMock(return_value=_mock_ollama_chat("Guten Morgen!"))
            mock_ollama.is_available = AsyncMock(return_value=True)
            mock_router.select_model.return_value = "qwen3:32b"
            mock_router.get_model_config.return_value = {
                "temperature": 0.7,
                "top_p": 0.9,
                "context_window": 32768,
            }
            mock_router.initialize = AsyncMock()
            mock_mcp.get_tool_schemas.return_value = {}
            mock_mcp.get_tool_list.return_value = []
            mock_mcp.disconnect_all = AsyncMock()

            # Initialize with mocks already set
            from jarvis.core.executor import Executor
            from jarvis.core.gatekeeper import Gatekeeper
            from jarvis.core.planner import Planner

            gateway._planner = Planner(config, mock_ollama, mock_router)
            gateway._gatekeeper = Gatekeeper(config)
            gateway._gatekeeper.initialize()
            gateway._executor = Executor(config, mock_mcp)
            gateway._running = True  # handle_message benötigt _running=True
            gateway._running = True  # Ohne start() manuell setzen

            msg = IncomingMessage(text="Guten Morgen!", channel="test", user_id="alex")
            response = await gateway.handle_message(msg)

            assert response.text
            assert "User" in response.text or "Morgen" in response.text
            assert response.channel == "test"
            assert response.session_id
            assert response.is_final

    @pytest.mark.asyncio
    async def test_tool_execution_cycle(self, config: JarvisConfig) -> None:
        """User will Datei lesen → Planner plant → Gatekeeper prüft → Executor führt aus."""
        gateway = Gateway(config)

        # Planner gibt JSON-Plan zurück, dann formuliert Antwort
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Erster Aufruf: Plan erstellen
                return _mock_ollama_chat("""```json
{
  "goal": "Datei lesen",
  "reasoning": "User will Datei sehen",
  "steps": [{"tool": "read_file", "params": {"path": "/home/test.txt"}, "rationale": "Lesen"}],
  "confidence": 0.9
}
```""")
            else:
                # Zweiter Aufruf: Antwort formulieren
                return _mock_ollama_chat("Die Datei enthält: Hello World")

        mock_ollama = AsyncMock()
        mock_ollama.chat = mock_chat
        mock_ollama.is_available = AsyncMock(return_value=True)

        mock_router = MagicMock()
        mock_router.select_model.return_value = "qwen3:32b"
        mock_router.get_model_config.return_value = {
            "temperature": 0.7,
            "top_p": 0.9,
            "context_window": 32768,
        }

        mock_mcp = MagicMock()
        mock_mcp.get_tool_schemas.return_value = {"read_file": {"description": "Reads a file"}}
        mock_mcp.get_tool_list.return_value = ["read_file"]
        mock_mcp.call_tool = AsyncMock(return_value=MockToolResult(content="Hello World"))
        mock_mcp.disconnect_all = AsyncMock()

        from jarvis.core.executor import Executor
        from jarvis.core.gatekeeper import Gatekeeper
        from jarvis.core.planner import Planner

        gateway._planner = Planner(config, mock_ollama, mock_router)
        gateway._gatekeeper = Gatekeeper(config)
        gateway._gatekeeper.initialize()
        gateway._executor = Executor(config, mock_mcp)
        gateway._mcp_client = mock_mcp
        gateway._running = True  # handle_message benötigt _running=True
        gateway._running = True

        msg = IncomingMessage(text="Lies /home/test.txt", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert response.text
        assert response.is_final


# ============================================================================
# Session-Management
# ============================================================================


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_session_created(self, config: JarvisConfig) -> None:
        gateway = Gateway(config)
        session = gateway._get_or_create_session("cli", "alex")
        assert session.channel == "cli"
        assert session.user_id == "alex"
        assert session.session_id

    @pytest.mark.asyncio
    async def test_session_reused(self, config: JarvisConfig) -> None:
        gateway = Gateway(config)
        s1 = gateway._get_or_create_session("cli", "alex")
        s2 = gateway._get_or_create_session("cli", "alex")
        assert s1.session_id == s2.session_id

    @pytest.mark.asyncio
    async def test_different_channels_different_sessions(self, config: JarvisConfig) -> None:
        gateway = Gateway(config)
        s1 = gateway._get_or_create_session("cli", "alex")
        s2 = gateway._get_or_create_session("telegram", "alex")
        assert s1.session_id != s2.session_id


# ============================================================================
# Working Memory
# ============================================================================


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_working_memory_created(self, config: JarvisConfig) -> None:
        gateway = Gateway(config)
        session = gateway._get_or_create_session("cli", "alex")
        wm = gateway._get_or_create_working_memory(session)
        assert wm.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_core_memory_loaded(self, config: JarvisConfig) -> None:
        """Core Memory wird aus CORE.md geladen."""
        # Schreibe CORE.md
        config.core_memory_file.write_text("Ich bin Jarvis.", encoding="utf-8")

        gateway = Gateway(config)
        session = gateway._get_or_create_session("cli", "alex")
        wm = gateway._get_or_create_working_memory(session)
        assert wm.core_memory_text == "Ich bin Jarvis."


# ============================================================================
# SessionContext erweiterte Features
# ============================================================================


class TestSessionContextFeatures:
    def test_iteration_tracking(self) -> None:
        from jarvis.models import SessionContext

        session = SessionContext(max_iterations=5)
        assert not session.iterations_exhausted
        assert session.iteration_count == 0

        session.iteration_count = 4
        assert not session.iterations_exhausted

        session.iteration_count = 5
        assert session.iterations_exhausted

    def test_reset_iteration(self) -> None:
        from jarvis.models import SessionContext

        session = SessionContext()
        session.iteration_count = 7
        session.record_block("test_tool")
        session.reset_iteration()
        assert session.iteration_count == 0

    def test_record_block(self) -> None:
        from jarvis.models import SessionContext

        session = SessionContext()
        assert session.record_block("exec_command") == 1
        assert session.record_block("exec_command") == 2
        assert session.record_block("exec_command") == 3
        assert session.record_block("other_tool") == 1


# ============================================================================
# Approval-Handling
# ============================================================================


class TestApprovalHandling:
    @pytest.mark.asyncio
    async def test_no_channel_returns_original_decisions(self, config: JarvisConfig) -> None:
        from jarvis.models import GateDecision, GateStatus, PlannedAction, RiskLevel, SessionContext

        gateway = Gateway(config)
        session = SessionContext()

        steps = [PlannedAction(tool="email_send", params={})]
        decisions = [
            GateDecision(
                status=GateStatus.APPROVE,
                reason="E-Mail erfordert Bestätigung",
                risk_level=RiskLevel.ORANGE,
                policy_name="email_requires_approval",
            )
        ]

        result = await gateway._handle_approvals(steps, decisions, session, "nonexistent_channel")
        # Kein Channel → Entscheidungen unverändert
        assert result[0].status == GateStatus.APPROVE
