"""Tests für die user-friendly Error-Messages.

Testet:
  - classify_error_for_user: Timeout, Connection, Permission, etc.
  - gatekeeper_block_message: Kontext + Vorschlag
  - retry_exhausted_message: Tool-Name + Attempts + Error
  - all_actions_blocked_message: Pro Aktion eine Begründung
  - _friendly_tool_name: Tool-Name-Mapping

Tests run in both DE and EN locale to validate i18n integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.i18n import set_locale
from jarvis.utils.error_messages import (
    _friendly_tool_name,
    all_actions_blocked_message,
    classify_error_for_user,
    gatekeeper_block_message,
    retry_exhausted_message,
)


class TestClassifyErrorForUser:
    """Tests für die Error-Klassifizierung."""

    def test_timeout_error(self) -> None:
        exc = TimeoutError("Operation timed out")
        msg = classify_error_for_user(exc)
        assert "Zeitlimit" in msg

    def test_connection_error(self) -> None:
        exc = ConnectionError("Connection refused")
        msg = classify_error_for_user(exc)
        assert "Verbindungsproblem" in msg

    def test_permission_error(self) -> None:
        exc = PermissionError("Access denied")
        msg = classify_error_for_user(exc)
        assert "Berechtigung" in msg

    def test_file_not_found_error(self) -> None:
        exc = FileNotFoundError("No such file")
        msg = classify_error_for_user(exc)
        assert "nicht gefunden" in msg

    def test_rate_limit_error(self) -> None:
        exc = Exception("429 Too Many Requests - rate limit exceeded")
        msg = classify_error_for_user(exc)
        assert "überlastet" in msg

    def test_memory_error(self) -> None:
        exc = MemoryError("Out of memory")
        msg = classify_error_for_user(exc)
        assert "Speicherproblem" in msg

    def test_generic_error(self) -> None:
        exc = ValueError("Something went wrong")
        msg = classify_error_for_user(exc)
        assert "Fehler" in msg
        assert "ValueError" in msg

    def test_os_error_with_connection_keyword(self) -> None:
        exc = OSError("Connection reset by peer")
        msg = classify_error_for_user(exc)
        assert "Verbindungsproblem" in msg


class TestGatekeeperBlockMessage:
    """Tests für Gatekeeper-Block-Nachrichten."""

    def test_known_tool(self) -> None:
        msg = gatekeeper_block_message("exec_command", "Gefährlicher Befehl")
        assert "Shell-Befehl" in msg
        assert "Gefährlicher Befehl" in msg
        assert "Sicherheitsgründen" in msg

    def test_unknown_tool(self) -> None:
        msg = gatekeeper_block_message("custom_tool", "Policy blockiert")
        assert "custom_tool" in msg
        assert "Sicherheitsgründen" in msg or "blockiert" in msg

    def test_contains_suggestion(self) -> None:
        msg = gatekeeper_block_message("write_file", "Keine Erlaubnis")
        assert "blockiert" in msg or "Sicherheitsgründen" in msg


class TestRetryExhaustedMessage:
    """Tests für Retry-Exhausted-Nachrichten."""

    def test_timeout_error(self) -> None:
        msg = retry_exhausted_message("web_search", 3, "Timeout nach 30 Sekunden")
        assert "Web-Suche" in msg or "web_search" in msg
        assert "3" in msg
        assert "nicht rechtzeitig" in msg

    def test_connection_error(self) -> None:
        msg = retry_exhausted_message("web_fetch", 3, "Connection refused")
        assert "Verbindung" in msg

    def test_rate_limit_error(self) -> None:
        msg = retry_exhausted_message("web_search", 3, "429 rate limit")
        assert "überlastet" in msg

    def test_generic_error(self) -> None:
        msg = retry_exhausted_message("run_python", 3, "SyntaxError: invalid syntax")
        assert "Technischer Fehler" in msg


class TestAllActionsBlockedMessage:
    """Tests für die All-Blocked-Nachricht."""

    @dataclass
    class MockStep:
        tool: str

    @dataclass
    class MockDecision:
        reason: str

    def test_single_action(self) -> None:
        steps = [self.MockStep(tool="exec_command")]
        decisions = [self.MockDecision(reason="Root-Befehl")]
        msg = all_actions_blocked_message(steps, decisions)
        assert "Shell-Befehl" in msg
        assert "Root-Befehl" in msg
        assert "blockiert" in msg or "Gatekeeper" in msg

    def test_multiple_actions(self) -> None:
        steps = [
            self.MockStep(tool="exec_command"),
            self.MockStep(tool="write_file"),
        ]
        decisions = [
            self.MockDecision(reason="Gefährlich"),
            self.MockDecision(reason="Kein Zugriff"),
        ]
        msg = all_actions_blocked_message(steps, decisions)
        assert "Shell-Befehl" in msg
        assert "Datei schreiben" in msg


class TestFriendlyToolName:
    """Tests für Tool-Name-Mapping."""

    def test_known_tools(self) -> None:
        assert _friendly_tool_name("exec_command") == "Shell-Befehl"
        assert _friendly_tool_name("web_search") == "Web-Suche"
        assert _friendly_tool_name("document_export") == "Dokument erstellen"
        assert _friendly_tool_name("read_file") == "Datei lesen"

    def test_unknown_tool_returns_name(self) -> None:
        assert _friendly_tool_name("custom_tool") == "custom_tool"


class TestEnglishLocale:
    """Verify error messages work in English locale."""

    def test_timeout_en(self) -> None:
        set_locale("en")
        msg = classify_error_for_user(TimeoutError("timeout"))
        assert "timed out" in msg
        set_locale("de")

    def test_connection_en(self) -> None:
        set_locale("en")
        msg = classify_error_for_user(ConnectionError("refused"))
        assert "connection" in msg.lower()
        set_locale("de")

    def test_friendly_tool_name_en(self) -> None:
        set_locale("en")
        assert _friendly_tool_name("exec_command") == "Shell command"
        assert _friendly_tool_name("web_search") == "Web search"
        set_locale("de")

    def test_gatekeeper_en(self) -> None:
        set_locale("en")
        msg = gatekeeper_block_message("exec_command", "dangerous")
        assert "Shell command" in msg
        assert "security" in msg.lower()
        set_locale("de")
