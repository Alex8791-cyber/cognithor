"""Tests für Blocking-I/O-Fixes.

Validiert:
- UserPreferenceStore verwendet persistente Connection statt pro-Aufruf connect
- Gatekeeper _write_audit verwendet Buffered Writes
- MCP Server führt sync-Handler in Thread-Pool aus
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from jarvis.core.user_preferences import UserPreferenceStore

# ── UserPreferenceStore: Persistente Connection ──────────────────────────


class TestUserPreferencePersistentConn:
    """Testet dass UserPreferenceStore eine persistente Connection nutzt."""

    def test_has_persistent_connection(self, tmp_path: Path) -> None:
        """Store hat ein _conn Attribut."""
        store = UserPreferenceStore(tmp_path / "prefs.db")
        assert hasattr(store, "_conn")
        # encrypted_connect may return sqlite3.Connection or sqlcipher Connection
        assert hasattr(store._conn, "execute")
        store.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """WAL-Modus ist aktiviert."""
        store = UserPreferenceStore(tmp_path / "prefs.db")
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        # row may be a tuple (sqlite3) or dict (sqlcipher with row_factory)
        mode = row[0] if isinstance(row, (tuple, list)) else row.get("journal_mode", row.get(0))
        assert mode == "wal"
        store.close()

    def test_get_or_create_works(self, tmp_path: Path) -> None:
        """get_or_create funktioniert mit persistenter Connection."""
        store = UserPreferenceStore(tmp_path / "prefs.db")
        pref = store.get_or_create("alex")
        assert pref.user_id == "alex"
        assert pref.verbosity == "normal"
        store.close()

    def test_update_persists(self, tmp_path: Path) -> None:
        """Update wird gespeichert."""
        store = UserPreferenceStore(tmp_path / "prefs.db")
        pref = store.get_or_create("alex")
        pref.verbosity = "verbose"
        store.update(pref)

        loaded = store.get_or_create("alex")
        assert loaded.verbosity == "verbose"
        store.close()

    def test_multiple_users(self, tmp_path: Path) -> None:
        """Mehrere User können angelegt werden."""
        store = UserPreferenceStore(tmp_path / "prefs.db")
        for uid in ("alice", "bob", "charlie"):
            pref = store.get_or_create(uid)
            assert pref.user_id == uid
        store.close()

    def test_close_is_safe(self, tmp_path: Path) -> None:
        """Close kann mehrfach aufgerufen werden."""
        store = UserPreferenceStore(tmp_path / "prefs.db")
        store.close()
        store.close()  # Zweiter Aufruf darf nicht crashen


# ── Gatekeeper: Buffered Audit Writes ────────────────────────────────────


class TestGatekeeperAuditBuffer:
    """Testet den Audit-Buffer im Gatekeeper."""

    def test_has_audit_buffer(self) -> None:
        """Gatekeeper hat einen _audit_buffer."""
        from jarvis.core.gatekeeper import Gatekeeper

        config = MagicMock()
        config.logs_dir = Path(tempfile.gettempdir()) / "test_logs"
        config.security.policies_dir = None
        config.security.blocked_commands = []
        config.security.credential_patterns = []
        config.security.allowed_paths = []
        config.jarvis_home = Path(tempfile.gettempdir())

        gk = Gatekeeper(config)
        assert hasattr(gk, "_audit_buffer")
        assert isinstance(gk._audit_buffer, list)

    def test_flush_writes_all_entries(self, tmp_path: Path) -> None:
        """_flush_audit_buffer schreibt alle Einträge auf Disk."""
        from jarvis.core.gatekeeper import Gatekeeper

        config = MagicMock()
        config.logs_dir = tmp_path
        config.security.policies_dir = None
        config.security.blocked_commands = []
        config.security.credential_patterns = []
        config.security.allowed_paths = []
        config.jarvis_home = tmp_path

        gk = Gatekeeper(config)
        gk._audit_buffer = ["entry1\n", "entry2\n", "entry3\n"]
        gk._flush_audit_buffer()

        content = (tmp_path / "gatekeeper.jsonl").read_text(encoding="utf-8")
        assert "entry1" in content
        assert "entry2" in content
        assert "entry3" in content
        assert gk._audit_buffer == []

    def test_flush_empty_buffer_noop(self, tmp_path: Path) -> None:
        """Leerer Buffer → keine Datei geschrieben."""
        from jarvis.core.gatekeeper import Gatekeeper

        config = MagicMock()
        config.logs_dir = tmp_path
        config.security.policies_dir = None
        config.security.blocked_commands = []
        config.security.credential_patterns = []
        config.security.allowed_paths = []
        config.jarvis_home = tmp_path

        gk = Gatekeeper(config)
        gk._flush_audit_buffer()  # Kein Crash bei leerem Buffer
        assert not (tmp_path / "gatekeeper.jsonl").exists()


# ── MCP Server: Sync Handler in Executor ─────────────────────────────────


class TestMCPServerSyncHandlerExecutor:
    """Testet dass sync-Handler via run_in_executor aufgerufen werden."""

    def test_code_uses_run_in_executor(self) -> None:
        """Der MCP-Server-Code enthält run_in_executor für sync-Handler."""
        import inspect

        from jarvis.mcp.server import JarvisMCPServer

        source = inspect.getsource(JarvisMCPServer.handle_tools_call)
        # Verifiziere dass run_in_executor im else-Zweig (sync handler) steht
        assert "run_in_executor" in source, (
            "handle_tools_call muss run_in_executor für sync-Handler verwenden"
        )

    async def test_sync_function_runs_in_executor(self) -> None:
        """Demonstriert dass run_in_executor einen sync-Handler non-blocking macht."""
        import asyncio
        import threading

        main_thread = threading.current_thread().name
        call_threads = []

        def blocking_handler() -> str:
            call_threads.append(threading.current_thread().name)
            return "done"

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, blocking_handler)
        assert result == "done"
        # Handler lief in einem ANDEREN Thread (ThreadPool)
        assert call_threads[0] != main_thread
