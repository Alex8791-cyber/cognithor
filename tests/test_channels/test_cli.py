"""Tests für CliChannel – Terminal-REPL.

Testet:
  - Channel-Eigenschaften (name, version)
  - Slash-Commands: /quit, /exit, /q, /help, /status, /clear, /version, unbekannt
  - send(): leere Nachricht, normale Nachricht
  - request_approval(): ja/nein/ungültig/Ctrl+C
  - REPL-Loop: Nachricht senden, leere Eingabe, /quit, EOF, KeyboardInterrupt
  - send_streaming_token
  - Banner-Anzeige
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.cli import BANNER, CliChannel
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def cli() -> CliChannel:
    return CliChannel(version="1.2.3")


@pytest.fixture()
def cli_with_console(cli: CliChannel) -> CliChannel:
    """CLI mit gemockter Console für Output-Capture."""
    cli._console = MagicMock()
    return cli


# =============================================================================
# Channel-Eigenschaften
# =============================================================================


class TestChannelProperties:
    def test_name(self, cli: CliChannel) -> None:
        assert cli.name == "cli"

    def test_version_stored(self, cli: CliChannel) -> None:
        assert cli._version == "1.2.3"

    def test_default_version(self) -> None:
        cli = CliChannel()
        assert cli._version == "0.1.0"

    def test_initial_state(self, cli: CliChannel) -> None:
        assert cli._running is False
        assert cli._handler is None
        assert cli._session_id == "cli-session"


# =============================================================================
# Slash-Commands
# =============================================================================


class TestSlashCommands:
    @pytest.mark.asyncio
    async def test_quit_returns_false(self, cli_with_console: CliChannel) -> None:
        """'/quit' beendet die REPL."""
        result = await cli_with_console._handle_command("/quit")
        assert result is False

    @pytest.mark.asyncio
    async def test_exit_returns_false(self, cli_with_console: CliChannel) -> None:
        result = await cli_with_console._handle_command("/exit")
        assert result is False

    @pytest.mark.asyncio
    async def test_q_returns_false(self, cli_with_console: CliChannel) -> None:
        result = await cli_with_console._handle_command("/q")
        assert result is False

    @pytest.mark.asyncio
    async def test_help_returns_true(self, cli_with_console: CliChannel) -> None:
        """'/help' zeigt Hilfe und setzt REPL fort."""
        result = await cli_with_console._handle_command("/help")
        assert result is True
        # Console.print wurde mit einem Panel aufgerufen
        cli_with_console._console.print.assert_called_once()
        from rich.panel import Panel

        call_arg = cli_with_console._console.print.call_args[0][0]
        assert isinstance(call_arg, Panel)

    @pytest.mark.asyncio
    async def test_version_returns_true(self, cli_with_console: CliChannel) -> None:
        """'/version' zeigt Version und setzt REPL fort."""
        result = await cli_with_console._handle_command("/version")
        assert result is True
        call_args_str = str(cli_with_console._console.print.call_args)
        assert "1.2.3" in call_args_str

    @pytest.mark.asyncio
    async def test_status_returns_true(self, cli_with_console: CliChannel) -> None:
        """'/status' zeigt Status und setzt REPL fort."""
        result = await cli_with_console._handle_command("/status")
        assert result is True
        call_args_str = str(cli_with_console._console.print.call_args)
        assert "Aktiv" in call_args_str
        assert "cli-session" in call_args_str

    @pytest.mark.asyncio
    async def test_clear_returns_true(self, cli_with_console: CliChannel) -> None:
        """'/clear' leert Bildschirm und setzt REPL fort."""
        result = await cli_with_console._handle_command("/clear")
        assert result is True
        cli_with_console._console.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_command_returns_true(self, cli_with_console: CliChannel) -> None:
        """Unbekannter Command zeigt Hinweis und setzt REPL fort."""
        result = await cli_with_console._handle_command("/foobar")
        assert result is True
        call_args_str = str(cli_with_console._console.print.call_args)
        assert "Unbekannter Befehl" in call_args_str

    @pytest.mark.asyncio
    async def test_case_insensitive(self, cli_with_console: CliChannel) -> None:
        """Commands sind case-insensitiv."""
        result = await cli_with_console._handle_command("/QUIT")
        assert result is False

    @pytest.mark.asyncio
    async def test_command_with_whitespace(self, cli_with_console: CliChannel) -> None:
        """Commands mit Leerzeichen funktionieren."""
        result = await cli_with_console._handle_command("  /quit  ")
        assert result is False


# =============================================================================
# send()
# =============================================================================


class TestSend:
    @pytest.mark.asyncio
    async def test_send_normal_message(self, cli_with_console: CliChannel) -> None:
        """Normale Nachricht wird ausgegeben."""
        msg = OutgoingMessage(text="Hallo!", channel="cli", session_id="s1")
        await cli_with_console.send(msg)
        # Console.print wurde mindestens einmal mit Text aufgerufen
        assert cli_with_console._console.print.call_count >= 1

    @pytest.mark.asyncio
    async def test_send_empty_message_skipped(self, cli_with_console: CliChannel) -> None:
        """Leere Nachricht wird nicht ausgegeben."""
        msg = OutgoingMessage(text="", channel="cli", session_id="s1")
        await cli_with_console.send(msg)
        cli_with_console._console.print.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_whitespace_only_skipped(self, cli_with_console: CliChannel) -> None:
        """Nachricht mit nur Whitespace: send() gibt trotzdem aus (kein Crash)."""
        msg = OutgoingMessage(text="   ", channel="cli", session_id="s1")
        await cli_with_console.send(msg)
        # Whitespace ist truthy, wird daher ausgegeben
        assert cli_with_console._console.print.call_count >= 1

    @pytest.mark.asyncio
    async def test_send_shows_jarvis_prefix(self, cli_with_console: CliChannel) -> None:
        """Ausgabe zeigt 'Jarvis:' Prefix."""
        msg = OutgoingMessage(text="Test", channel="cli", session_id="s1")
        await cli_with_console.send(msg)
        all_calls = str(cli_with_console._console.print.call_args_list)
        assert "Jarvis" in all_calls


# =============================================================================
# request_approval()
# =============================================================================


class TestRequestApproval:
    @pytest.mark.asyncio
    async def test_approval_accepted_j(self, cli_with_console: CliChannel) -> None:
        """User antwortet 'j' → True."""
        action = PlannedAction(tool="email_send", params={"to": "test@test.de"})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="j"
        ):
            result = await cli_with_console.request_approval("s1", action, "E-Mail senden")
        assert result is True

    @pytest.mark.asyncio
    async def test_approval_accepted_ja(self, cli_with_console: CliChannel) -> None:
        action = PlannedAction(tool="email_send", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="ja"
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is True

    @pytest.mark.asyncio
    async def test_approval_accepted_yes(self, cli_with_console: CliChannel) -> None:
        action = PlannedAction(tool="email_send", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="yes"
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is True

    @pytest.mark.asyncio
    async def test_approval_accepted_y(self, cli_with_console: CliChannel) -> None:
        action = PlannedAction(tool="email_send", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="y"
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is True

    @pytest.mark.asyncio
    async def test_approval_rejected_n(self, cli_with_console: CliChannel) -> None:
        """User antwortet 'n' → False."""
        action = PlannedAction(tool="email_send", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="n"
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_rejected_nein(self, cli_with_console: CliChannel) -> None:
        action = PlannedAction(tool="email_send", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="nein"
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_rejected_no(self, cli_with_console: CliChannel) -> None:
        action = PlannedAction(tool="email_send", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="no"
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_invalid_then_valid(self, cli_with_console: CliChannel) -> None:
        """Ungültige Eingabe → erneute Abfrage → dann gültig."""
        action = PlannedAction(tool="test_tool", params={})
        call_count = 0

        async def mock_input(prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "vielleicht"
            return "j"

        with patch.object(cli_with_console, "_read_input", side_effect=mock_input):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_approval_eof_returns_false(self, cli_with_console: CliChannel) -> None:
        """EOF bei Approval → False (sicher blockieren)."""
        action = PlannedAction(tool="test_tool", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value=None
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_keyboard_interrupt_returns_false(
        self, cli_with_console: CliChannel
    ) -> None:
        """Ctrl+C bei Approval → False."""
        action = PlannedAction(tool="test_tool", params={})
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, side_effect=KeyboardInterrupt
        ):
            result = await cli_with_console.request_approval("s1", action, "Grund")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_panel_shows_tool_info(self, cli_with_console: CliChannel) -> None:
        """Approval-Panel zeigt Tool-Name und Parameter."""
        action = PlannedAction(
            tool="exec_command",
            params={"command": "rm important.txt"},
            rationale="Datei löschen",
        )
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="n"
        ):
            await cli_with_console.request_approval("s1", action, "Destruktive Aktion")

        # Finde den Panel-Aufruf
        from rich.panel import Panel

        panel_found = False
        for call in cli_with_console._console.print.call_args_list:
            if call.args and isinstance(call.args[0], Panel):
                panel = call.args[0]
                # Panel.renderable enthält den formatierten String
                panel_text = str(panel.renderable)
                assert "exec_command" in panel_text
                assert "Gatekeeper" in str(panel.title) or "Bestätigung" in panel_text
                panel_found = True
                break
        assert panel_found, "Kein Panel in Console-Ausgabe gefunden"


# =============================================================================
# send_streaming_token()
# =============================================================================


class TestStreamingToken:
    @pytest.mark.asyncio
    async def test_token_printed_without_newline(self, cli_with_console: CliChannel) -> None:
        """Token wird ohne Newline ausgegeben."""
        await cli_with_console.send_streaming_token("s1", "Hallo")
        cli_with_console._console.print.assert_called_once_with(
            "Hallo",
            end="",
            highlight=False,
        )

    @pytest.mark.asyncio
    async def test_multiple_tokens(self, cli_with_console: CliChannel) -> None:
        """Mehrere Tokens werden sequenziell ausgegeben."""
        await cli_with_console.send_streaming_token("s1", "A")
        await cli_with_console.send_streaming_token("s1", "B")
        await cli_with_console.send_streaming_token("s1", "C")
        assert cli_with_console._console.print.call_count == 3


# =============================================================================
# stop()
# =============================================================================


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, cli: CliChannel) -> None:
        cli._running = True
        await cli.stop()
        assert cli._running is False


# =============================================================================
# REPL-Loop (start)
# =============================================================================


class TestREPLLoop:
    @pytest.mark.asyncio
    async def test_quit_exits_loop(self, cli_with_console: CliChannel) -> None:
        """'/quit' beendet die REPL-Schleife."""
        handler = AsyncMock()
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="/quit"
        ):
            await cli_with_console.start(handler)
        # /quit bricht den Loop, Handler wird nie aufgerufen
        handler.assert_not_called()
        # _running bleibt True (nur stop() setzt False) — Loop endet durch break
        assert cli_with_console._handler is handler

    @pytest.mark.asyncio
    async def test_eof_exits_loop(self, cli_with_console: CliChannel) -> None:
        """EOF (Ctrl+D) beendet die Schleife."""
        handler = AsyncMock()
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, side_effect=EOFError
        ):
            await cli_with_console.start(handler)
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_exits_loop(self, cli_with_console: CliChannel) -> None:
        """Ctrl+C beendet die Schleife."""
        handler = AsyncMock()
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, side_effect=KeyboardInterrupt
        ):
            await cli_with_console.start(handler)
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_input_skipped(self, cli_with_console: CliChannel) -> None:
        """Leere Eingabe wird übersprungen, REPL läuft weiter."""
        call_count = 0

        async def mock_input(prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return ""  # Leere Eingaben
            return "/quit"

        handler = AsyncMock()
        with patch.object(cli_with_console, "_read_input", side_effect=mock_input):
            await cli_with_console.start(handler)
        handler.assert_not_called()  # Leere Eingaben werden nicht an Handler geschickt
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_message_sent_to_handler(self, cli_with_console: CliChannel) -> None:
        """Normale Eingabe wird an Handler gesendet."""
        response = OutgoingMessage(text="Antwort", channel="cli", session_id="s1")
        handler = AsyncMock(return_value=response)

        call_count = 0

        async def mock_input(prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Was ist 2+2?"
            return "/quit"

        with patch.object(cli_with_console, "_read_input", side_effect=mock_input):
            await cli_with_console.start(handler)

        handler.assert_called_once()
        received_msg = handler.call_args[0][0]
        assert isinstance(received_msg, IncomingMessage)
        assert received_msg.text == "Was ist 2+2?"
        assert received_msg.channel == "cli"
        assert received_msg.user_id == "local"

    @pytest.mark.asyncio
    async def test_handler_error_displayed(self, cli_with_console: CliChannel) -> None:
        """Handler-Fehler werden im Terminal angezeigt."""
        handler = AsyncMock(side_effect=RuntimeError("LLM nicht erreichbar"))

        call_count = 0

        async def mock_input(prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Test"
            return "/quit"

        with patch.object(cli_with_console, "_read_input", side_effect=mock_input):
            await cli_with_console.start(handler)

        all_calls = str(cli_with_console._console.print.call_args_list)
        assert "Fehler" in all_calls or "LLM" in all_calls

    @pytest.mark.asyncio
    async def test_banner_displayed(self, cli_with_console: CliChannel) -> None:
        """Banner wird beim Start angezeigt."""
        handler = AsyncMock()
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="/quit"
        ):
            await cli_with_console.start(handler)

        # Panel mit Banner muss der erste Console-Aufruf sein
        first_call = cli_with_console._console.print.call_args_list[0]
        first_call_str = str(first_call)
        assert "JARVIS" in first_call_str or "Panel" in first_call_str

    @pytest.mark.asyncio
    async def test_handler_registered(self, cli_with_console: CliChannel) -> None:
        """Handler wird in start() registriert."""
        handler = AsyncMock()
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value="/quit"
        ):
            await cli_with_console.start(handler)
        assert cli_with_console._handler is handler

    @pytest.mark.asyncio
    async def test_multiple_messages(self, cli_with_console: CliChannel) -> None:
        """Mehrere Nachrichten werden verarbeitet."""
        response = OutgoingMessage(text="OK", channel="cli", session_id="s1")
        handler = AsyncMock(return_value=response)

        call_count = 0

        async def mock_input(prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Erste Nachricht"
            if call_count == 2:
                return "Zweite Nachricht"
            return "/quit"

        with patch.object(cli_with_console, "_read_input", side_effect=mock_input):
            await cli_with_console.start(handler)

        assert handler.call_count == 2

    @pytest.mark.asyncio
    async def test_none_input_exits(self, cli_with_console: CliChannel) -> None:
        """None von _read_input beendet die Schleife."""
        handler = AsyncMock()
        with patch.object(
            cli_with_console, "_read_input", new_callable=AsyncMock, return_value=None
        ):
            await cli_with_console.start(handler)
        handler.assert_not_called()


# =============================================================================
# Banner
# =============================================================================


class TestBanner:
    def test_banner_contains_version_placeholder(self) -> None:
        """Banner enthält {version} Platzhalter."""
        assert "{version}" in BANNER

    def test_banner_format(self) -> None:
        """Banner kann mit Version formatiert werden."""
        rendered = BANNER.format(version="1.2.3")
        assert "1.2.3" in rendered
        assert "{version}" not in rendered

    def test_banner_contains_jarvis(self) -> None:
        """Banner enthält JARVIS ASCII-Art."""
        assert "JARVIS" in BANNER or "██" in BANNER
