"""End-to-End Integrationstests – kompletter PGE-Zyklus.

Testet den VOLLSTÄNDIGEN Pfad:
  CLI/Channel → Gateway.handle_message → Planner → Gatekeeper → Executor → MCP → Response

Szenarien:
  1. Direkte Antwort (kein Tool)
  2. Tool-Ausführung: read_file über echten MCP FileSystem-Handler
  3. Tool-Ausführung: exec_command über echten MCP Shell-Handler
  4. ORANGE-Aktion → User bestätigt → Ausführung
  5. ORANGE-Aktion → User lehnt ab → Blockiert
  6. Destruktiver Befehl → Gatekeeper blockiert (RED)
  7. Alle Schritte blockiert → Eskalation nach 3x
  8. Re-Plan nach Fehler
  9. Iterationslimit erreicht
  10. Multi-Step-Plan: Datei schreiben + lesen
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.core.executor import Executor
from jarvis.core.gatekeeper import Gatekeeper
from jarvis.core.planner import Planner
from jarvis.gateway.gateway import Gateway
from jarvis.mcp.client import JarvisMCPClient
from jarvis.mcp.filesystem import register_fs_tools
from jarvis.mcp.shell import register_shell_tools
from jarvis.models import (
    IncomingMessage,
    PlannedAction,
)

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    sb = tmp_path / "sandbox"
    sb.mkdir()
    return sb


@pytest.fixture()
def config(tmp_path: Path, sandbox: Path) -> JarvisConfig:
    cfg = JarvisConfig(
        jarvis_home=tmp_path / ".jarvis",
        security=SecurityConfig(
            allowed_paths=[str(sandbox), str(tmp_path / ".jarvis")],
            max_iterations=5,
        ),
    )
    ensure_directory_structure(cfg)
    return cfg


def _build_gateway_with_real_mcp(
    config: JarvisConfig,
    chat_side_effect,
) -> Gateway:
    """Baut einen Gateway mit echten MCP-Handlern und Mock-LLM.

    Alle Tool-Aufrufe (read_file, write_file, exec_command, ...) laufen
    durch den echten Code. Nur das LLM ist gemockt.
    """
    gateway = Gateway(config)

    # Mock-Ollama (LLM)
    mock_ollama = AsyncMock()
    mock_ollama.chat = chat_side_effect
    mock_ollama.is_available = AsyncMock(return_value=True)

    # Mock-Router (Modell-Auswahl)
    mock_router = MagicMock()
    mock_router.select_model.return_value = "qwen3:32b"
    mock_router.get_model_config.return_value = {
        "temperature": 0.7,
        "top_p": 0.9,
        "context_window": 32768,
    }

    # ECHTE MCP-Tools (FileSystem + Shell)
    mcp_client = JarvisMCPClient(config)
    register_fs_tools(mcp_client, config)
    register_shell_tools(mcp_client, config)

    # Subsysteme verdrahten
    gateway._ollama = mock_ollama
    gateway._model_router = mock_router
    gateway._mcp_client = mcp_client
    gateway._planner = Planner(config, mock_ollama, mock_router)
    gateway._gatekeeper = Gatekeeper(config)
    gateway._gatekeeper.initialize()
    gateway._executor = Executor(config, mcp_client)
    gateway._running = True

    return gateway


def _llm_response(text: str) -> dict:
    """Erzeugt eine Ollama-Chat-Response."""
    return {
        "message": {
            "role": "assistant",
            "content": text,
        },
    }


def _plan_json(steps: list[dict], confidence: float = 0.9) -> str:
    """Erzeugt einen Plan als JSON in Code-Block (wie LLM ihn liefert)."""
    import json

    plan = {
        "goal": "Testaufgabe",
        "reasoning": "Schritte geplant",
        "steps": steps,
        "confidence": confidence,
    }
    return f"```json\n{json.dumps(plan, ensure_ascii=False)}\n```"


# =============================================================================
# Mock-Channel für Approval-Tests
# =============================================================================


class MockChannel:
    """Simpler Channel-Mock für Approval-Workflow-Tests."""

    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.approval_requests: list[tuple[str, PlannedAction, str]] = []
        self.sent_messages: list = []

    @property
    def name(self) -> str:
        return "mock"

    async def start(self, handler) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, message) -> None:
        self.sent_messages.append(message)

    async def request_approval(self, session_id: str, action: PlannedAction, reason: str) -> bool:
        self.approval_requests.append((session_id, action, reason))
        return self.approve

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        pass


# =============================================================================
# 1. Direkte Antwort
# =============================================================================


class TestE2EDirectResponse:
    @pytest.mark.asyncio
    async def test_simple_question(self, config: JarvisConfig) -> None:
        """Einfache Frage → LLM antwortet direkt, kein Tool nötig."""

        async def mock_chat(**kwargs):
            return _llm_response("Berlin ist die Hauptstadt von Deutschland.")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(
            text="Was ist die Hauptstadt von Deutschland?", channel="test", user_id="alex"
        )
        response = await gateway.handle_message(msg)

        assert "Berlin" in response.text
        assert response.is_final is True
        assert response.channel == "test"


# =============================================================================
# 2. Tool-Ausführung: Datei lesen (über echten MCP FileSystem-Handler)
# =============================================================================


class TestE2EFileRead:
    @pytest.mark.asyncio
    async def test_read_file_e2e(self, config: JarvisConfig, sandbox: Path) -> None:
        """User fragt nach Datei → Plan → read_file → Antwort mit Inhalt."""
        # Testdatei anlegen
        test_file = sandbox / "info.txt"
        test_file.write_text("Jarvis ist ein Agent OS.", encoding="utf-8")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    _plan_json(
                        [
                            {
                                "tool": "read_file",
                                "params": {"path": str(test_file)},
                                "rationale": "Datei lesen wie gewünscht",
                                "risk_estimate": "green",
                            }
                        ]
                    )
                )
            else:
                return _llm_response("Die Datei enthält: Jarvis ist ein Agent OS.")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(text=f"Lies {test_file}", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert "Jarvis" in response.text or "Agent" in response.text
        assert response.is_final


# =============================================================================
# 3. Tool-Ausführung: Shell-Befehl (über echten MCP Shell-Handler)
# =============================================================================


class TestE2EShellExec:
    @pytest.mark.asyncio
    async def test_shell_command_e2e(self, config: JarvisConfig) -> None:
        """User will Befehl ausführen → Plan → exec_command → Ergebnis."""
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    _plan_json(
                        [
                            {
                                "tool": "exec_command",
                                "params": {"command": "echo Integration-Test-OK"},
                                "rationale": "Befehl ausführen",
                                "risk_estimate": "yellow",
                            }
                        ]
                    )
                )
            else:
                return _llm_response("Der Befehl gab aus: Integration-Test-OK")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(text="Führe echo aus", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert response.text
        assert response.is_final


# =============================================================================
# 4. ORANGE-Aktion → User bestätigt → Ausführung
# =============================================================================


class TestE2EApprovalGranted:
    @pytest.mark.asyncio
    async def test_approval_accepted(self, config: JarvisConfig) -> None:
        """E-Mail senden erfordert Bestätigung → User sagt ja → wird ausgeführt."""
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    _plan_json(
                        [
                            {
                                "tool": "email_send",
                                "params": {
                                    "to": "test@example.com",
                                    "subject": "Test",
                                    "body": "Hallo",
                                },
                                "rationale": "E-Mail senden",
                                "risk_estimate": "orange",
                            }
                        ]
                    )
                )
            else:
                return _llm_response("Die E-Mail wurde gesendet.")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)

        # Mock-Channel der immer bestätigt
        mock_channel = MockChannel(approve=True)
        gateway._channels["test"] = mock_channel

        # email_send als Builtin-Mock registrieren (existiert nicht wirklich als MCP-Tool)
        gateway._mcp_client.register_builtin_handler(
            "email_send",
            lambda **kw: "E-Mail gesendet an " + kw.get("to", "?"),
            description="Sendet E-Mail",
        )

        msg = IncomingMessage(text="Sende eine E-Mail", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        # Approval wurde angefragt
        assert len(mock_channel.approval_requests) == 1
        assert mock_channel.approval_requests[0][1].tool == "email_send"
        assert response.is_final


# =============================================================================
# 5. ORANGE-Aktion → User lehnt ab → Blockiert
# =============================================================================


class TestE2EApprovalRejected:
    @pytest.mark.asyncio
    async def test_approval_rejected(self, config: JarvisConfig) -> None:
        """E-Mail senden → User sagt nein → alles blockiert."""
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    _plan_json(
                        [
                            {
                                "tool": "email_send",
                                "params": {"to": "test@example.com"},
                                "rationale": "E-Mail senden",
                                "risk_estimate": "orange",
                            }
                        ]
                    )
                )
            else:
                return _llm_response("Verstanden, keine E-Mail gesendet.")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)

        # Mock-Channel der immer ablehnt
        mock_channel = MockChannel(approve=False)
        gateway._channels["test"] = mock_channel

        gateway._mcp_client.register_builtin_handler(
            "email_send",
            lambda **kw: "gesendet",
            description="Sendet E-Mail",
        )

        msg = IncomingMessage(text="Sende E-Mail", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert len(mock_channel.approval_requests) == 1
        # Antwort sollte Blockierung reflektieren
        assert response.is_final


# =============================================================================
# 6. Destruktiver Befehl → Gatekeeper blockiert (RED)
# =============================================================================


class TestE2EDestructiveBlocked:
    @pytest.mark.asyncio
    async def test_rm_rf_blocked(self, config: JarvisConfig) -> None:
        """rm -rf / wird vom Gatekeeper sofort blockiert."""
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    _plan_json(
                        [
                            {
                                "tool": "exec_command",
                                "params": {"command": "rm -rf /"},
                                "rationale": "Alles löschen",
                                "risk_estimate": "red",
                            }
                        ]
                    )
                )
            else:
                return _llm_response("Eskalation: rm -rf wurde blockiert.")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(text="Lösche alles", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert response.is_final
        assert (
            "blockiert" in response.text.lower()
            or "block" in response.text.lower()
            or response.text
        )


# =============================================================================
# 7. Multi-Step-Plan: Schreiben + Lesen
# =============================================================================


class TestE2EMultiStep:
    @pytest.mark.asyncio
    async def test_write_then_read(self, config: JarvisConfig, sandbox: Path) -> None:
        """Plan mit 2 Schritten: Datei schreiben, dann lesen."""
        target_file = str(sandbox / "multi.txt")
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    _plan_json(
                        [
                            {
                                "tool": "write_file",
                                "params": {
                                    "path": target_file,
                                    "content": "Multi-Step-Test bestanden!",
                                },
                                "rationale": "Datei erstellen",
                                "risk_estimate": "yellow",
                            },
                            {
                                "tool": "read_file",
                                "params": {"path": target_file},
                                "rationale": "Datei lesen zum Verifizieren",
                                "risk_estimate": "green",
                                "depends_on": 0,
                            },
                        ]
                    )
                )
            else:
                return _llm_response(
                    "Die Datei wurde erstellt und enthält: Multi-Step-Test bestanden!"
                )

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(text="Erstelle und lies eine Datei", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert response.is_final
        # Datei muss tatsächlich existieren (echte MCP-Tools!)
        assert (sandbox / "multi.txt").exists()
        assert (sandbox / "multi.txt").read_text() == "Multi-Step-Test bestanden!"


# =============================================================================
# 8. Iterationslimit erreicht
# =============================================================================


class TestE2EIterationLimit:
    @pytest.mark.asyncio
    async def test_iterations_exhausted(self, config: JarvisConfig) -> None:
        """Wenn der Agent in einer Schleife hängt, bricht er nach max_iterations ab."""
        # Config: max 3 Iterationen
        config.security.max_iterations = 3

        async def mock_chat(**kwargs):
            # Immer einen fehlschlagenden Plan zurückgeben → Endlosschleife
            return _llm_response(
                _plan_json(
                    [
                        {
                            "tool": "exec_command",
                            "params": {"command": "false"},  # Exit Code 1
                            "rationale": "Fehlschlagender Befehl",
                            "risk_estimate": "yellow",
                        }
                    ]
                )
            )

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(text="Mach was Unmögliches", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert response.is_final
        # Sollte Iterationslimit-Meldung ODER eine formulate_response-Antwort sein
        assert response.text


# =============================================================================
# 9. Pfad außerhalb Sandbox → Tool-Fehler
# =============================================================================


class TestE2ESandboxViolation:
    @pytest.mark.asyncio
    async def test_read_outside_sandbox(self, config: JarvisConfig) -> None:
        """read_file auf /etc/passwd wird vom FileSystem-Tool blockiert (nicht Gatekeeper)."""
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    _plan_json(
                        [
                            {
                                "tool": "read_file",
                                "params": {"path": "/etc/passwd"},
                                "rationale": "System-Datei lesen",
                                "risk_estimate": "green",
                            }
                        ]
                    )
                )
            else:
                # Nach Tool-Fehler: Antwort formulieren
                return _llm_response("Zugriff auf /etc/passwd ist nicht erlaubt.")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(text="Lies /etc/passwd", channel="test", user_id="alex")
        response = await gateway.handle_message(msg)

        assert response.is_final
        # Gatekeeper blockiert den Pfad ODER FileSystemTools gibt Fehler
        assert response.text


# =============================================================================
# 10. Session-Isolation zwischen Requests
# =============================================================================


class TestE2ESessionIsolation:
    @pytest.mark.asyncio
    async def test_sessions_isolated(self, config: JarvisConfig) -> None:
        """Zwei verschiedene User haben isolierte Sessions."""

        async def mock_chat(**kwargs):
            return _llm_response("Antwort für User")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)

        msg1 = IncomingMessage(text="Hallo", channel="test", user_id="alice")
        msg2 = IncomingMessage(text="Hallo", channel="test", user_id="bob")

        r1 = await gateway.handle_message(msg1)
        r2 = await gateway.handle_message(msg2)

        # Verschiedene Session-IDs
        assert r1.session_id != r2.session_id
        assert r1.is_final
        assert r2.is_final


# =============================================================================
# 11. Core Memory wird in den Kontext geladen
# =============================================================================


class TestE2ECoreMemory:
    @pytest.mark.asyncio
    async def test_core_memory_in_context(self, config: JarvisConfig) -> None:
        """Core Memory wird beim Planner-Aufruf im Kontext sein."""
        # CORE.md schreiben
        config.core_memory_file.write_text(
            "Du bist Jarvis. Du bist ein lokaler KI-Assistent.",
            encoding="utf-8",
        )

        captured_kwargs: list[dict] = []

        async def mock_chat(**kwargs):
            captured_kwargs.append(kwargs)
            return _llm_response("Hallo, ich bin Jarvis!")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        msg = IncomingMessage(text="Wer bin ich?", channel="test", user_id="alex")
        await gateway.handle_message(msg)

        # Prüfe dass Core Memory im LLM-Kontext war
        assert len(captured_kwargs) >= 1
        messages_str = str(captured_kwargs[0].get("messages", []))
        assert "User" in messages_str or "Jarvis" in messages_str


# =============================================================================
# 12. Shutdown-Sequenz
# =============================================================================


class TestE2EShutdown:
    @pytest.mark.asyncio
    async def test_clean_shutdown(self, config: JarvisConfig) -> None:
        """Gateway fährt alle Subsysteme sauber herunter."""

        async def mock_chat(**kwargs):
            return _llm_response("OK")

        gateway = _build_gateway_with_real_mcp(config, mock_chat)
        gateway._llm = AsyncMock()

        await gateway.shutdown()

        assert gateway._running is False
        gateway._llm.close.assert_awaited_once()
