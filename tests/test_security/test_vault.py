"""Tests: Encrypted Vault & Session Isolation (Review-Punkt 3).

  - EncryptedVault: Verschlüsselung, Entschlüsselung, Integrität
  - VaultManager: Cross-Agent-Blockade, Agent-Isolation
  - IsolatedSessionStore: Session-Trennung, Token-Lookup
  - SessionIsolationGuard: Violation-Detection
"""

from __future__ import annotations

import pytest

from jarvis.security.vault import (
    EncryptedVault,
    VaultManager,
    IsolatedSessionStore,
    SessionIsolationGuard,
    AgentSession,
    VaultEntry,
)


# ============================================================================
# EncryptedVault
# ============================================================================


class TestEncryptedVault:
    def test_store_and_retrieve(self) -> None:
        vault = EncryptedVault("coder")
        vault.store("github", "token", "ghp_abc123xyz")
        assert vault.retrieve("github", "token") == "ghp_abc123xyz"

    def test_encrypted_value_differs_from_plain(self) -> None:
        vault = EncryptedVault("coder")
        entry = vault.store("github", "token", "ghp_abc123xyz")
        assert entry.encrypted_value != "ghp_abc123xyz"

    def test_retrieve_nonexistent(self) -> None:
        vault = EncryptedVault("coder")
        assert vault.retrieve("unknown", "key") is None

    def test_delete(self) -> None:
        vault = EncryptedVault("coder")
        vault.store("github", "token", "val")
        assert vault.delete("github", "token") is True
        assert vault.retrieve("github", "token") is None

    def test_delete_nonexistent(self) -> None:
        vault = EncryptedVault("coder")
        assert vault.delete("x", "y") is False

    def test_list_entries_no_values(self) -> None:
        vault = EncryptedVault("coder")
        vault.store("github", "token", "secret")
        entries = vault.list_entries()
        assert len(entries) == 1
        assert "secret" not in str(entries)
        assert entries[0]["service"] == "github"

    def test_has(self) -> None:
        vault = EncryptedVault("coder")
        vault.store("svc", "key", "val")
        assert vault.has("svc", "key")
        assert not vault.has("svc", "other")

    def test_clear(self) -> None:
        vault = EncryptedVault("coder")
        vault.store("a", "b", "c")
        vault.store("d", "e", "f")
        assert vault.clear() == 2
        assert vault.entry_count == 0

    def test_different_agents_different_encryption(self) -> None:
        v1 = EncryptedVault("agent-a")
        v2 = EncryptedVault("agent-b")
        e1 = v1.store("svc", "key", "same_value")
        e2 = v2.store("svc", "key", "same_value")
        assert e1.encrypted_value != e2.encrypted_value

    def test_access_count(self) -> None:
        vault = EncryptedVault("coder")
        vault.store("svc", "key", "val")
        vault.retrieve("svc", "key")
        vault.retrieve("svc", "key")
        entries = vault.list_entries()
        assert entries[0]["access_count"] == 2

    def test_tampered_token_fails(self) -> None:
        vault = EncryptedVault("coder")
        entry = vault.store("svc", "key", "secret")
        original = entry.encrypted_value
        # Ensure the replacement character is actually different
        replacement = "Y" if original[5] != "Y" else "Z"
        tampered = original[:5] + replacement + original[6:]
        entry.encrypted_value = tampered
        with pytest.raises(Exception):
            vault.retrieve("svc", "key")

    def test_stats(self) -> None:
        vault = EncryptedVault("coder")
        vault.store("github", "token", "val")
        vault.store("aws", "key", "val2")
        stats = vault.stats()
        assert stats["agent_id"] == "coder"
        assert stats["entry_count"] == 2
        assert "github" in stats["services"]


# ============================================================================
# VaultManager
# ============================================================================


class TestVaultManager:
    def test_store_and_retrieve(self) -> None:
        vm = VaultManager()
        vm.store("coder", "github", "token", "secret123")
        assert vm.retrieve("coder", "github", "token") == "secret123"

    def test_cross_agent_blocked(self) -> None:
        vm = VaultManager()
        vm.store("coder", "github", "token", "secret")
        result = vm.cross_agent_attempt("attacker", "coder", "github", "token")
        assert result is None

    def test_same_agent_allowed(self) -> None:
        vm = VaultManager()
        vm.store("coder", "github", "token", "secret")
        result = vm.cross_agent_attempt("coder", "coder", "github", "token")
        assert result == "secret"

    def test_vault_isolation(self) -> None:
        vm = VaultManager()
        vm.store("agent-a", "svc", "key", "val-a")
        vm.store("agent-b", "svc", "key", "val-b")
        assert vm.retrieve("agent-a", "svc", "key") == "val-a"
        assert vm.retrieve("agent-b", "svc", "key") == "val-b"

    def test_retrieve_unknown_agent(self) -> None:
        vm = VaultManager()
        assert vm.retrieve("unknown", "svc", "key") is None

    def test_stats(self) -> None:
        vm = VaultManager()
        vm.store("a", "s", "k", "v")
        vm.store("b", "s", "k", "v")
        stats = vm.stats()
        assert stats["total_vaults"] == 2
        assert stats["total_entries"] == 2


# ============================================================================
# IsolatedSessionStore
# ============================================================================


class TestIsolatedSessionStore:
    def test_create_and_get(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("coder", "user-1")
        retrieved = store.get_session("coder", s.session_id)
        assert retrieved is not None
        assert retrieved.user_id == "user-1"

    def test_get_by_token(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("coder", "user-1", token="my-token-123")
        retrieved = store.get_by_token("my-token-123")
        assert retrieved is not None
        assert retrieved.session_id == s.session_id

    def test_cross_agent_session_blocked(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("coder", "user-1")
        result = store.cross_agent_attempt("attacker", "coder", s.session_id)
        assert result is None

    def test_same_agent_allowed(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("coder", "user-1")
        result = store.cross_agent_attempt("coder", "coder", s.session_id)
        assert result is not None

    def test_revoke_session(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("coder", "user-1")
        assert store.revoke_session("coder", s.session_id) is True
        assert store.get_by_token(s.token) is None

    def test_agent_sessions(self) -> None:
        store = IsolatedSessionStore()
        store.create_session("coder", "u1")
        store.create_session("coder", "u2")
        store.create_session("researcher", "u3")
        assert len(store.agent_sessions("coder")) == 2
        assert len(store.agent_sessions("researcher")) == 1

    def test_stats(self) -> None:
        store = IsolatedSessionStore()
        store.create_session("coder", "u1")
        store.create_session("researcher", "u2")
        stats = store.stats()
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 2


# ============================================================================
# SessionIsolationGuard
# ============================================================================


class TestSessionIsolationGuard:
    def test_valid_credential_access(self) -> None:
        vm = VaultManager()
        vm.store("coder", "github", "token", "secret")
        ss = IsolatedSessionStore()
        guard = SessionIsolationGuard(vm, ss)
        result = guard.check_credential_access("coder", "coder", "github", "token")
        assert result == "secret"

    def test_cross_agent_credential_logged(self) -> None:
        vm = VaultManager()
        vm.store("coder", "github", "token", "secret")
        ss = IsolatedSessionStore()
        guard = SessionIsolationGuard(vm, ss)
        result = guard.check_credential_access("attacker", "coder", "github", "token")
        assert result is None
        assert guard.violation_count == 1
        assert guard.violations()[0]["type"] == "credential_cross_access"

    def test_cross_agent_session_logged(self) -> None:
        vm = VaultManager()
        ss = IsolatedSessionStore()
        s = ss.create_session("coder", "user-1")
        guard = SessionIsolationGuard(vm, ss)
        result = guard.check_session_access("attacker", "coder", s.session_id)
        assert result is None
        assert guard.violation_count == 1
        assert guard.violations()[0]["type"] == "session_cross_access"

    def test_stats(self) -> None:
        vm = VaultManager()
        ss = IsolatedSessionStore()
        guard = SessionIsolationGuard(vm, ss)
        stats = guard.stats()
        assert "violations" in stats
        assert "vault_stats" in stats
        assert "session_stats" in stats
