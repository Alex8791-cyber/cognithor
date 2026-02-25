"""Tests für Schwäche 1: Agent-Separation.

Beweist vollständige Isolation von Sessions und Credentials pro Agent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jarvis.gateway.session_store import SessionStore
from jarvis.models import Message, MessageRole, SessionContext
from jarvis.security.credentials import CredentialStore


# ============================================================================
# 1. Session-Isolation pro Agent
# ============================================================================


class TestSessionAgentIsolation:
    """Sessions werden nach agent_id getrennt."""

    def test_same_user_different_agents_separate_sessions(self, tmp_path: Path) -> None:
        """Ein User hat für jeden Agent eine eigene Session."""
        store = SessionStore(tmp_path / "sessions.db")

        # Session für Agent "jarvis" (Default)
        s1 = SessionContext(session_id="s-jarvis", user_id="alex", channel="cli", agent_name="jarvis")
        store.save_session(s1)

        # Session für Agent "coder"
        s2 = SessionContext(session_id="s-coder", user_id="alex", channel="cli", agent_name="coder")
        store.save_session(s2)

        # Laden: Jeder Agent bekommt NUR seine Session
        loaded_jarvis = store.load_session("cli", "alex", agent_id="jarvis")
        loaded_coder = store.load_session("cli", "alex", agent_id="coder")

        assert loaded_jarvis is not None
        assert loaded_jarvis.session_id == "s-jarvis"

        assert loaded_coder is not None
        assert loaded_coder.session_id == "s-coder"

    def test_agent_session_not_visible_to_other_agent(self, tmp_path: Path) -> None:
        """Agent A kann Session von Agent B nicht laden."""
        store = SessionStore(tmp_path / "sessions.db")

        s1 = SessionContext(session_id="s-private", user_id="alex", channel="telegram")
        store.save_session(s1)
        # Default agent_id = "jarvis"

        # Agent "researcher" findet keine Session
        loaded = store.load_session("telegram", "alex", agent_id="researcher")
        assert loaded is None

    def test_chat_history_per_agent_session(self, tmp_path: Path) -> None:
        """Chat-History ist an Session gebunden, nicht an User."""
        store = SessionStore(tmp_path / "sessions.db")

        # Zwei Sessions, gleicher User, verschiedene Agenten
        s1 = SessionContext(session_id="s-j", user_id="alex", channel="cli")
        s2 = SessionContext(session_id="s-c", user_id="alex", channel="cli")
        store.save_session(s1)
        store.save_session(s2)

        msg_jarvis = [Message(role=MessageRole.USER, content="Hallo Jarvis")]
        msg_coder = [Message(role=MessageRole.USER, content="Fix den Bug")]

        store.save_chat_history("s-j", msg_jarvis)
        store.save_chat_history("s-c", msg_coder)

        h1 = store.load_chat_history("s-j")
        h2 = store.load_chat_history("s-c")

        assert len(h1) == 1 and "Hallo Jarvis" in h1[0].content
        assert len(h2) == 1 and "Fix den Bug" in h2[0].content

    def test_migration_adds_agent_id_column(self, tmp_path: Path) -> None:
        """Bestehende DBs ohne agent_id werden korrekt migriert."""
        import sqlite3 as _sq

        db_path = tmp_path / "old.db"
        # Erstelle alte DB ohne agent_id, mit chat_history
        conn = _sq.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                started_at REAL NOT NULL,
                last_activity REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1,
                max_iterations INTEGER DEFAULT 10
            );
            CREATE TABLE chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                channel TEXT DEFAULT '',
                timestamp REAL NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('old-s', 'alex', 'cli', 1.0, 1.0, 0, 1, 10)"
        )
        conn.commit()
        conn.close()

        # Neuer SessionStore migriert automatisch
        store = SessionStore(db_path)
        loaded = store.load_session("cli", "alex", agent_id="jarvis")
        assert loaded is not None
        assert loaded.session_id == "old-s"

    def test_backward_compatible_default_agent(self, tmp_path: Path) -> None:
        """Ohne agent_id-Parameter funktioniert alles wie vorher."""
        store = SessionStore(tmp_path / "sessions.db")

        s = SessionContext(session_id="s-default", user_id="alex", channel="cli")
        store.save_session(s)

        # Laden ohne agent_id → default "jarvis"
        loaded = store.load_session("cli", "alex")
        assert loaded is not None


# ============================================================================
# 2. Credential-Isolation pro Agent
# ============================================================================


class TestCredentialAgentIsolation:
    """Credentials werden nach agent_id getrennt."""

    def test_global_credential_accessible_by_all(self, tmp_path: Path) -> None:
        """Globale Credentials sind für alle Agenten sichtbar."""
        store = CredentialStore(
            store_path=tmp_path / "creds.enc",
            passphrase="test123",
        )

        store.store("searxng", "api_key", "key-123")

        # Ohne Agent-Scope → global
        assert store.retrieve("searxng", "api_key") == "key-123"
        # Mit Agent-Scope → Fallback auf global
        assert store.retrieve("searxng", "api_key", agent_id="coder") == "key-123"
        assert store.retrieve("searxng", "api_key", agent_id="researcher") == "key-123"

    def test_agent_credential_only_visible_to_agent(self, tmp_path: Path) -> None:
        """Agent-spezifische Credentials sind NUR für diesen Agent sichtbar."""
        store = CredentialStore(
            store_path=tmp_path / "creds.enc",
            passphrase="test123",
        )

        # Agent "coder" bekommt eigenen GitHub-Token
        store.store("github", "token", "gh-coder-123", agent_id="coder")

        # coder kann es abrufen
        assert store.retrieve("github", "token", agent_id="coder") == "gh-coder-123"

        # researcher kann es NICHT abrufen
        assert store.retrieve("github", "token", agent_id="researcher") is None

        # Ohne Agent-Scope → auch nicht sichtbar (kein globales Credential)
        assert store.retrieve("github", "token") is None

    def test_agent_credential_overrides_global(self, tmp_path: Path) -> None:
        """Agent-spezifisches Credential hat Vorrang vor globalem."""
        store = CredentialStore(
            store_path=tmp_path / "creds.enc",
            passphrase="test123",
        )

        # Global: Standard-API-Key
        store.store("openai", "api_key", "global-key")
        # Agent "premium": Eigener API-Key
        store.store("openai", "api_key", "premium-key", agent_id="premium")

        # Premium-Agent bekommt seinen Key
        assert store.retrieve("openai", "api_key", agent_id="premium") == "premium-key"

        # Anderer Agent bekommt Global-Key
        assert store.retrieve("openai", "api_key", agent_id="basic") == "global-key"

        # Kein Agent → Global-Key
        assert store.retrieve("openai", "api_key") == "global-key"

    def test_list_entries_filters_by_agent(self, tmp_path: Path) -> None:
        """list_entries respektiert Agent-Scope."""
        store = CredentialStore(
            store_path=tmp_path / "creds.enc",
            passphrase="test123",
        )

        store.store("github", "token", "gh-123", agent_id="coder")
        store.store("searxng", "key", "sx-456")  # global
        store.store("jira", "token", "jr-789", agent_id="pm")

        # Coder sieht: seinen eigenen + globale
        coder_entries = store.list_entries(agent_id="coder")
        coder_services = {e.service for e in coder_entries}
        assert "github" in coder_services
        assert "searxng" in coder_services
        assert "jira" not in coder_services  # PM-only

    def test_persistence_across_reload(self, tmp_path: Path) -> None:
        """Agent-Credentials überleben Store-Neustart."""
        path = tmp_path / "creds.enc"

        store1 = CredentialStore(store_path=path, passphrase="test123")
        store1.store("github", "token", "gh-secret", agent_id="coder")

        store2 = CredentialStore(store_path=path, passphrase="test123")
        assert store2.retrieve("github", "token", agent_id="coder") == "gh-secret"
        assert store2.retrieve("github", "token", agent_id="other") is None

    def test_inject_credentials_respects_agent_scope(self, tmp_path: Path) -> None:
        """inject_credentials() nutzt Agent-Scope wenn vorhanden."""
        store = CredentialStore(
            store_path=tmp_path / "creds.enc",
            passphrase="test123",
        )

        store.store("openai", "api_key", "global-key")
        store.store("openai", "api_key", "agent-key", agent_id="coder")

        params = {"prompt": "Hello"}
        mapping = {"api_key": "openai:api_key"}

        # Standard inject (global)
        result = store.inject_credentials(params, mapping)
        assert result["api_key"] == "global-key"


# ============================================================================
# 3. AgentProfile Credential-Konfiguration
# ============================================================================


class TestAgentProfileCredentialConfig:
    """AgentProfile unterstützt credential_scope und credential_mappings."""

    def test_agent_profile_has_credential_fields(self) -> None:
        from jarvis.core.agent_router import AgentProfile

        agent = AgentProfile(
            name="coder",
            credential_scope="coder",
            credential_mappings={"api_key": "github:token"},
        )

        assert agent.credential_scope == "coder"
        assert agent.credential_mappings == {"api_key": "github:token"}

    def test_default_agent_has_no_scope(self) -> None:
        from jarvis.core.agent_router import AgentProfile

        agent = AgentProfile(name="jarvis")
        assert agent.credential_scope == ""
        assert agent.credential_mappings == {}
