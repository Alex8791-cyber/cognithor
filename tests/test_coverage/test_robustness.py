"""Tests für Session-Persistenz, Executor-Retry und Gateway-Integrationen.

Deckt ab:
  - SessionStore: CRUD, Chat-History, Cleanup
  - Executor Retry: Backoff, Retryable vs Non-Retryable Errors
  - Gateway: Session-Restore, Memory-Init, Cron-Wiring
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.core.executor import Executor
from jarvis.gateway.session_store import SessionStore
from jarvis.models import (
    GateDecision,
    GateStatus,
    Message,
    MessageRole,
    PlannedAction,
    SessionContext,
)

# ============================================================================
# SessionStore Tests
# ============================================================================


class TestSessionStore:
    """Tests für SQLite-basierte Session-Persistenz."""

    def setup_method(self) -> None:
        """Erstellt einen temporären SessionStore pro Test."""
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "test_sessions.db"
        self.store = SessionStore(db_path)

    def teardown_method(self) -> None:
        """Räumt auf."""
        self.store.close()
        self._tmp.cleanup()

    def test_save_and_load_session(self) -> None:
        """Session speichern und wieder laden."""
        session = SessionContext(
            user_id="alexander",
            channel="cli",
            message_count=5,
        )
        self.store.save_session(session)

        loaded = self.store.load_session("cli", "alexander")
        assert loaded is not None
        assert loaded.session_id == session.session_id
        assert loaded.user_id == "alexander"
        assert loaded.channel == "cli"
        assert loaded.message_count == 5
        assert loaded.active is True

    def test_load_returns_none_for_unknown(self) -> None:
        """Nicht-existierende Session gibt None zurück."""
        assert self.store.load_session("cli", "unknown") is None

    def test_save_updates_existing(self) -> None:
        """Wiederholtes Speichern aktualisiert die Session."""
        session = SessionContext(user_id="alex", channel="cli")
        self.store.save_session(session)

        session.message_count = 10
        session.touch()
        self.store.save_session(session)

        loaded = self.store.load_session("cli", "alex")
        assert loaded is not None
        assert loaded.message_count == 11  # touch() inkrementiert

    def test_deactivate_session(self) -> None:
        """Deaktivierte Sessions werden nicht mehr geladen."""
        session = SessionContext(user_id="alex", channel="cli")
        self.store.save_session(session)
        self.store.deactivate_session(session.session_id)

        assert self.store.load_session("cli", "alex") is None

    def test_save_and_load_chat_history(self) -> None:
        """Chat-History speichern und laden."""
        session = SessionContext(user_id="alex", channel="cli")
        self.store.save_session(session)

        messages = [
            Message(role=MessageRole.USER, content="Hallo Jarvis"),
            Message(role=MessageRole.ASSISTANT, content="Hallo!"),
            Message(role=MessageRole.USER, content="Was ist eine BU?"),
        ]
        count = self.store.save_chat_history(session.session_id, messages)
        assert count == 3

        loaded = self.store.load_chat_history(session.session_id)
        assert len(loaded) == 3
        assert loaded[0].role == MessageRole.USER
        assert loaded[0].content == "Hallo Jarvis"
        assert loaded[1].role == MessageRole.ASSISTANT
        assert loaded[2].content == "Was ist eine BU?"

    def test_chat_history_limit(self) -> None:
        """Limit begrenzt geladene Messages."""
        session = SessionContext(user_id="alex", channel="cli")
        self.store.save_session(session)

        messages = [Message(role=MessageRole.USER, content=f"Nachricht {i}") for i in range(10)]
        self.store.save_chat_history(session.session_id, messages)

        loaded = self.store.load_chat_history(session.session_id, limit=3)
        assert len(loaded) == 3
        # Neueste 3, chronologisch sortiert
        assert loaded[0].content == "Nachricht 7"
        assert loaded[2].content == "Nachricht 9"

    def test_chat_history_replace_on_save(self) -> None:
        """Wiederholtes Speichern ersetzt die History."""
        session = SessionContext(user_id="alex", channel="cli")
        self.store.save_session(session)

        self.store.save_chat_history(
            session.session_id,
            [Message(role=MessageRole.USER, content="Alt")],
        )
        self.store.save_chat_history(
            session.session_id,
            [Message(role=MessageRole.USER, content="Neu")],
        )

        loaded = self.store.load_chat_history(session.session_id)
        assert len(loaded) == 1
        assert loaded[0].content == "Neu"

    def test_list_sessions(self) -> None:
        """Sessions auflisten."""
        for i in range(3):
            self.store.save_session(SessionContext(user_id=f"user{i}", channel="cli"))
        sessions = self.store.list_sessions()
        assert len(sessions) == 3

    def test_count_sessions(self) -> None:
        """Sessions zählen."""
        assert self.store.count_sessions() == 0
        self.store.save_session(SessionContext(user_id="alex", channel="cli"))
        assert self.store.count_sessions() == 1

    def test_cleanup_old_sessions(self) -> None:
        """Alte Sessions deaktivieren."""
        old_session = SessionContext(user_id="old", channel="cli")
        # Manuell altes Datum setzen
        old_session.last_activity = datetime(2020, 1, 1, tzinfo=UTC)
        self.store.save_session(old_session)

        new_session = SessionContext(user_id="new", channel="cli")
        self.store.save_session(new_session)

        cleaned = self.store.cleanup_old_sessions(max_age_days=30)
        assert cleaned == 1
        assert self.store.load_session("cli", "old") is None
        assert self.store.load_session("cli", "new") is not None

    def test_multiple_channels_separate(self) -> None:
        """Gleicher User auf verschiedenen Channels hat separate Sessions."""
        self.store.save_session(SessionContext(user_id="alex", channel="cli"))
        self.store.save_session(SessionContext(user_id="alex", channel="telegram"))

        cli = self.store.load_session("cli", "alex")
        tg = self.store.load_session("telegram", "alex")
        assert cli is not None
        assert tg is not None
        assert cli.session_id != tg.session_id

    def test_empty_chat_history(self) -> None:
        """Leere Chat-History laden."""
        session = SessionContext(user_id="alex", channel="cli")
        self.store.save_session(session)
        loaded = self.store.load_chat_history(session.session_id)
        assert loaded == []


# ============================================================================
# Executor Retry Tests
# ============================================================================


class TestExecutorRetry:
    """Tests für Executor Retry-Logik mit Backoff."""

    def setup_method(self) -> None:
        """Erstellt Executor mit Mock-MCP-Client."""
        self.config = MagicMock()
        self.mcp = MagicMock()
        self.executor = Executor(self.config, self.mcp)
        # Schnelle Tests — kein echtes Warten
        self.executor._base_delay = 0.01

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        """Erfolg beim ersten Versuch — kein Retry."""
        mock_result = MagicMock(content="OK", is_error=False)
        self.mcp.call_tool = AsyncMock(return_value=mock_result)

        result = await self.executor._execute_single("test_tool", {})
        assert result.success
        assert result.content == "OK"
        assert self.mcp.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self) -> None:
        """Timeout wird retried, dann Erfolg."""
        mock_result = MagicMock(content="Delayed OK", is_error=False)
        self.mcp.call_tool = AsyncMock(
            side_effect=[
                TimeoutError("timeout"),
                mock_result,
            ]
        )

        result = await self.executor._execute_single("test_tool", {})
        assert result.success
        assert result.content == "Delayed OK"
        assert self.mcp.call_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        """Alle Retries erschöpft → Fehler."""
        self.mcp.call_tool = AsyncMock(side_effect=TimeoutError("immer timeout"))

        result = await self.executor._execute_single("test_tool", {})
        assert result.is_error
        assert "3 Versuchen" in result.content
        assert self.mcp.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self) -> None:
        """Nicht-retryable Fehler werden sofort gemeldet."""
        self.mcp.call_tool = AsyncMock(side_effect=PermissionError("verboten"))

        result = await self.executor._execute_single("test_tool", {})
        assert result.is_error
        assert result.error_type == "PermissionError"
        # Nur 1 Versuch — kein Retry
        assert self.mcp.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self) -> None:
        """ConnectionError ist retryable."""
        mock_result = MagicMock(content="OK", is_error=False)
        self.mcp.call_tool = AsyncMock(
            side_effect=[
                ConnectionError("refused"),
                ConnectionError("refused"),
                mock_result,
            ]
        )

        result = await self.executor._execute_single("test_tool", {})
        assert result.success
        assert self.mcp.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_no_mcp_client(self) -> None:
        """Ohne MCP-Client → sofort Fehler."""
        executor = Executor(self.config, None)
        result = await executor._execute_single("test_tool", {})
        assert result.is_error
        assert result.error_type == "NoMCPClient"

    @pytest.mark.asyncio
    async def test_output_truncation(self) -> None:
        """Zu lange Ausgabe wird gekürzt."""
        long_content = "x" * 20000
        mock_result = MagicMock(content=long_content, is_error=False)
        self.mcp.call_tool = AsyncMock(return_value=mock_result)

        result = await self.executor._execute_single("test_tool", {})
        assert result.success
        assert result.truncated
        assert len(result.content) == 10000

    @pytest.mark.asyncio
    async def test_execute_skips_blocked(self) -> None:
        """Blockierte Aktionen werden übersprungen."""
        actions = [PlannedAction(tool="danger", params={}, rationale="test")]
        decisions = [
            GateDecision(
                status=GateStatus.BLOCK,
                reason="gefährlich",
                risk_level="red",
            )
        ]

        results = await self.executor.execute(actions, decisions)
        assert len(results) == 1
        assert results[0].is_error
        assert "GatekeeperBlock" in results[0].error_type


# ============================================================================
# CronEngine Properties Tests
# ============================================================================


class TestCronEngineProperties:
    """Tests für CronEngine job_count und has_enabled_jobs."""

    def test_job_count_no_store(self) -> None:
        """Ohne JobStore ist job_count 0."""
        from jarvis.cron.engine import CronEngine

        engine = CronEngine(jobs_path=None)
        assert engine.job_count == 0
        assert engine.has_enabled_jobs is False

    def test_job_count_with_disabled_jobs(self) -> None:
        """Alle Jobs deaktiviert → 0."""
        from jarvis.cron.engine import CronEngine

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("jobs:\n  test_job:\n    schedule: '0 7 * * *'\n")
            f.write("    prompt: test\n    enabled: false\n")
            f.flush()
            engine = CronEngine(jobs_path=f.name)
            assert engine.has_enabled_jobs is False


# ============================================================================
# Gateway Integration Tests
# ============================================================================


class TestGatewaySessionPersistence:
    """Tests für Session-Persistenz im Gateway."""

    def test_session_store_schema_creation(self) -> None:
        """SessionStore erstellt Schema beim ersten Zugriff."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sessions.db"
            store = SessionStore(db_path)
            assert store.count_sessions() == 0
            assert db_path.exists()
            store.close()

    def test_session_roundtrip(self) -> None:
        """Voller Roundtrip: erstellen → speichern → laden → History."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sessions.db"
            store = SessionStore(db_path)

            # Erstellen + Speichern
            session = SessionContext(
                user_id="alexander",
                channel="cli",
            )
            session.touch()
            session.touch()
            store.save_session(session)

            # History speichern
            messages = [
                Message(role=MessageRole.USER, content="Hallo"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Hi!",
                ),
            ]
            store.save_chat_history(session.session_id, messages)

            # Neuer Store (simuliert Neustart)
            store.close()
            store2 = SessionStore(db_path)

            loaded = store2.load_session("cli", "alexander")
            assert loaded is not None
            assert loaded.message_count == 2

            history = store2.load_chat_history(loaded.session_id)
            assert len(history) == 2
            assert history[0].content == "Hallo"
            assert history[1].content == "Hi!"
            store2.close()
