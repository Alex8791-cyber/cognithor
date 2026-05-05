"""Tests für Session-Lock in gateway.py — Concurrent-Access-Schutz.

Validiert dass _session_lock korrekt verwendet wird:
- Concurrent _get_or_create_session liefert gleiche Session
- _cleanup_stale_sessions modifiziert Dicts sicher
- _get_or_create_working_memory mit Double-Check-Pattern
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from cognithor.gateway.gateway import Gateway

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def gateway(tmp_path: Path) -> Gateway:
    """Minimal-Gateway für Session-Tests."""
    config = MagicMock()
    config.cognithor_home = tmp_path
    config.core_memory_path = tmp_path / "CORE.md"
    config.workspace_dir = tmp_path / "workspace"
    config.security.max_iterations = 10
    config.models.planner.context_window = 16000
    config.log_level = "WARNING"
    config.api_port = 0
    config.persistence = MagicMock()
    config.persistence.enabled = False

    gw = Gateway.__new__(Gateway)
    gw._config = config
    gw._sessions = {}
    gw._working_memories = {}
    gw._session_last_accessed = {}
    gw._last_session_cleanup = time.monotonic()
    gw._session_lock = threading.Lock()
    gw._wm_creation_locks = {}
    gw._session_store = None
    gw._SESSION_TTL_SECONDS = 3600
    gw._CLEANUP_INTERVAL_SECONDS = 300
    return gw


# ── _session_lock ist threading.Lock ─────────────────────────────────────


class TestSessionLockType:
    """Verifiziert dass _session_lock ein threading.Lock ist."""

    def test_lock_is_threading_lock(self, gateway: Gateway) -> None:
        assert isinstance(gateway._session_lock, type(threading.Lock()))


# ── Concurrent _get_or_create_session ────────────────────────────────────


class TestConcurrentSessionCreation:
    """Testet Thread-Sicherheit bei parallelem Session-Zugriff."""

    def test_same_session_from_multiple_threads(self, gateway: Gateway) -> None:
        """Parallele Aufrufe mit gleichen Parametern liefern gleiche Session."""
        results = []

        def create_session():
            session = gateway._get_or_create_session("cli", "alex", "jarvis")
            results.append(session.session_id)

        threads = [threading.Thread(target=create_session) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Alle Threads müssen die GLEICHE Session-ID bekommen
        assert len(set(results)) == 1, f"Divergent sessions: {set(results)}"

    def test_different_channels_get_different_sessions(self, gateway: Gateway) -> None:
        """Unterschiedliche Channel/User-Kombinationen → verschiedene Sessions."""
        s1 = gateway._get_or_create_session("cli", "alex")
        s2 = gateway._get_or_create_session("telegram", "alex")
        s3 = gateway._get_or_create_session("cli", "bob")
        assert s1.session_id != s2.session_id
        assert s1.session_id != s3.session_id

    def test_concurrent_different_users(self, gateway: Gateway) -> None:
        """Parallele Session-Erstellung für verschiedene User ist sicher."""
        results: dict[str, str] = {}

        def create_for_user(user_id: str):
            session = gateway._get_or_create_session("cli", user_id)
            results[user_id] = session.session_id

        threads = [threading.Thread(target=create_for_user, args=(f"user_{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 20 verschiedene Sessions
        assert len(results) == 20
        assert len(set(results.values())) == 20


# ── _cleanup_stale_sessions mit Lock ─────────────────────────────────────


class TestCleanupStaleSessionsLock:
    """Testet dass Cleanup unter Lock sicher läuft."""

    def test_cleanup_removes_stale(self, gateway: Gateway) -> None:
        """Alte Sessions werden entfernt."""
        gateway._get_or_create_session("cli", "stale_user")
        key = "cli:stale_user:jarvis"

        # Timestamp auf "1 Stunde ago" setzen (älter als TTL)
        gateway._session_last_accessed[key] = time.monotonic() - 7200

        gateway._cleanup_stale_sessions()

        assert key not in gateway._sessions
        assert key not in gateway._session_last_accessed

    def test_cleanup_preserves_active(self, gateway: Gateway) -> None:
        """Aktive Sessions bleiben erhalten."""
        gateway._get_or_create_session("cli", "active_user")
        key = "cli:active_user:jarvis"

        gateway._cleanup_stale_sessions()

        assert key in gateway._sessions

    def test_concurrent_cleanup_and_creation(self, gateway: Gateway) -> None:
        """Cleanup und Session-Creation parallel sind sicher."""
        # Erstelle viele Sessions
        for i in range(50):
            gateway._get_or_create_session("cli", f"user_{i}")

        # Markiere die Hälfte als stale
        for i in range(25):
            key = f"cli:user_{i}:jarvis"
            gateway._session_last_accessed[key] = time.monotonic() - 7200

        errors = []

        def cleanup():
            try:
                gateway._cleanup_stale_sessions()
            except Exception as e:
                errors.append(e)

        def create_new():
            try:
                for i in range(50, 70):
                    gateway._get_or_create_session("cli", f"user_{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=cleanup)
        t2 = threading.Thread(target=create_new)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Errors during concurrent ops: {errors}"


# ── _get_or_create_working_memory Double-Check ───────────────────────────


class TestWorkingMemoryDoubleCheck:
    """Testet das Double-Check-Locking-Pattern für Working Memory."""

    def test_same_wm_for_same_session(self, gateway: Gateway) -> None:
        """Gleiche Session liefert gleiche WorkingMemory."""
        session = gateway._get_or_create_session("cli", "alex")
        wm1 = gateway._get_or_create_working_memory(session)
        wm2 = gateway._get_or_create_working_memory(session)
        assert wm1 is wm2

    def test_concurrent_wm_creation(self, gateway: Gateway) -> None:
        """Parallele WM-Erstellung liefert gleiche Instanz."""
        session = gateway._get_or_create_session("cli", "alex")
        results = []

        def get_wm():
            wm = gateway._get_or_create_working_memory(session)
            results.append(id(wm))

        threads = [threading.Thread(target=get_wm) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Alle müssen die GLEICHE WM-Instanz sein
        assert len(set(results)) == 1, f"Divergent WMs: {set(results)}"
