"""Deep coverage for cognithor.security.agent_vault.

Covers happy-path, edge cases, error paths, and inter-class
isolation invariants for ``AgentSecret``, ``AgentVault``,
``VaultRotator``, ``IsolatedSessionStore``, ``SessionFirewall``,
and ``AgentVaultManager``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from cognithor.security.agent_vault import (
    AgentSecret,
    AgentSession,
    AgentVault,
    AgentVaultManager,
    IsolatedSessionStore,
    RotationPolicy,
    SecretStatus,
    SecretType,
    SessionFirewall,
    VaultRotator,
    _load_or_create_master_secret,
)

if TYPE_CHECKING:
    from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# AgentSecret
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentSecret:
    def test_to_dict_omits_encrypted_value(self) -> None:
        secret = AgentSecret(
            secret_id="SEC-x-0001",
            agent_id="agent-x",
            name="api",
            secret_type=SecretType.API_KEY,
            _encrypted_value="cipher",
        )
        d = secret.to_dict()
        assert "secret_id" in d
        assert "_encrypted_value" not in d
        # Type field uses the StrEnum value, not the python attr name
        assert d["type"] == "api_key"
        assert d["status"] == "active"

    def test_is_active_property(self) -> None:
        active = AgentSecret(secret_id="s", agent_id="a", name="n", secret_type=SecretType.API_KEY)
        rotated = AgentSecret(
            secret_id="s",
            agent_id="a",
            name="n",
            secret_type=SecretType.API_KEY,
            status=SecretStatus.ROTATED,
        )
        assert active.is_active is True
        assert rotated.is_active is False

    @pytest.mark.parametrize(
        ("expires_at", "expected"),
        [
            ("", False),  # No expiry → never expired
            ("9999-12-31T23:59:59Z", False),  # Far future
            ("1970-01-01T00:00:00Z", True),  # Far past
        ],
    )
    def test_is_expired(self, expires_at: str, expected: bool) -> None:
        secret = AgentSecret(
            secret_id="s",
            agent_id="a",
            name="n",
            secret_type=SecretType.API_KEY,
            expires_at=expires_at,
        )
        assert secret.is_expired is expected


# ─────────────────────────────────────────────────────────────────────────────
# AgentVault — happy path + edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentVaultBasic:
    def test_store_and_retrieve_round_trip(self) -> None:
        vault = AgentVault("agent-rt", master_secret=b"deterministic")
        secret = vault.store("api", "ghp_abcd1234")
        assert vault.retrieve(secret.secret_id) == "ghp_abcd1234"

    def test_secret_id_is_namespaced_to_agent(self) -> None:
        vault = AgentVault("agent-z", master_secret=b"m")
        s1 = vault.store("a", "x")
        s2 = vault.store("b", "y")
        assert s1.secret_id.startswith("SEC-")
        # Counter increments: 0001 → 0002
        assert s1.secret_id.endswith("0001")
        assert s2.secret_id.endswith("0002")

    def test_store_encrypted_value_differs_from_plaintext(self) -> None:
        vault = AgentVault("agent-1", master_secret=b"secret")
        secret = vault.store("api", "plaintext-secret")
        # The encrypted blob is base64 Fernet — must differ from the
        # plaintext and not contain it.
        assert secret._encrypted_value != "plaintext-secret"
        assert "plaintext-secret" not in secret._encrypted_value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "x",
            "a" * 100_000,
            "über-secret-mit-Ümlauten-😀",
            "\x00\x01\x02binary-ish",
        ],
        ids=["empty", "single_char", "large_100k", "unicode_emoji", "control_chars"],
    )
    def test_round_trip_preserves_arbitrary_strings(self, value: str) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", value)
        assert vault.retrieve(s.secret_id) == value

    def test_retrieve_missing_id_returns_none_and_logs(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        assert vault.retrieve("does-not-exist") is None
        # Failed lookups still produce an access-log entry
        log = vault.access_log()
        assert any(e["action"] == "retrieve_failed" for e in log)

    def test_retrieve_after_revoke_returns_none(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v")
        assert vault.revoke(s.secret_id) is True
        # Revoked secret is removed from the store entirely
        assert vault.retrieve(s.secret_id) is None

    def test_revoke_unknown_returns_false(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        assert vault.revoke("SEC-nonexistent-9999") is False

    def test_rotate_changes_value_keeps_id_and_increments_counter(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v1")
        original_id = s.secret_id
        rotated = vault.rotate(s.secret_id, "v2")
        assert rotated is not None
        assert rotated.secret_id == original_id
        assert rotated.rotation_count == 1
        assert rotated.last_rotated  # non-empty timestamp
        assert vault.retrieve(s.secret_id) == "v2"

    def test_rotate_unknown_returns_none(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        assert vault.rotate("SEC-missing-0001", "v") is None

    def test_ttl_secret_has_expiry_set_in_future(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v", ttl_hours=24)
        assert s.expires_at != ""
        # Format is ISO Z, lexicographically sortable
        assert s.expires_at > time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def test_ttl_zero_means_no_expiry(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v", ttl_hours=0)
        assert s.expires_at == ""

    def test_decrypt_invalid_fernet_raises_value_error(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v")
        # Tamper with the cipher: replace with random Fernet-shaped bytes
        # that won't authenticate
        s._encrypted_value = "gAAAAABxxxinvalidtokendata"
        with pytest.raises(ValueError, match="Decryption failed"):
            vault.retrieve(s.secret_id)

    def test_active_and_all_secrets_diverge_after_revoke(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s1 = vault.store("a", "v1")
        s2 = vault.store("b", "v2")
        assert len(vault.active_secrets()) == 2
        vault.revoke(s1.secret_id)
        active = vault.active_secrets()
        all_ = vault.all_secrets()
        # Revoke deletes the secret entirely → both lists shrink
        assert len(active) == 1
        assert len(all_) == 1
        assert active[0].secret_id == s2.secret_id

    def test_expire_check_marks_only_overdue(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        # Long-lived secret
        live = vault.store("live", "v", ttl_hours=24)
        # Past-dated secret (manually backdate)
        past = vault.store("past", "v")
        past.expires_at = "1970-01-01T00:00:00Z"
        expired = vault.expire_check()
        assert past in expired
        assert live not in expired
        assert past.status is SecretStatus.EXPIRED
        assert live.status is SecretStatus.ACTIVE

    def test_access_log_returns_recent_first(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v")
        vault.retrieve(s.secret_id)
        vault.retrieve(s.secret_id)
        log = vault.access_log(limit=10)
        # Most recent first
        assert log[0]["action"] == "retrieve"
        # Each entry carries agent_id + timestamp
        assert all(e["agent_id"] == "agent" for e in log)
        assert all("timestamp" in e for e in log)

    def test_stats_counts_by_status(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        a = vault.store("a", "v")
        b = vault.store("b", "v")
        c = vault.store("c", "v")
        # Mark b expired manually
        b.status = SecretStatus.EXPIRED
        # Revoke c (deletes it from the dict)
        vault.revoke(c.secret_id)
        stats = vault.stats()
        assert stats["agent_id"] == "agent"
        # a is active, b is expired, c is gone (revoked)
        assert stats["total_secrets"] == 2
        assert stats["active"] == 1
        assert stats["expired"] == 1
        # access_events counts every store/retrieve/revoke logged
        assert stats["access_events"] >= 4
        # Sanity-check we exercised the unused secret
        _ = a.secret_id

    def test_two_vaults_with_same_master_secret_share_keys(self) -> None:
        # Key derivation is deterministic from agent_id + master_secret.
        v1 = AgentVault("agent-x", master_secret=b"m")
        v2 = AgentVault("agent-x", master_secret=b"m")
        s = v1.store("k", "v")
        # v2 has independent _secrets dict but the same fernet key, so
        # it can decrypt v1's ciphertext directly:
        assert v2._decrypt(s._encrypted_value) == "v"

    def test_two_vaults_with_different_master_secrets_cannot_decrypt(self) -> None:
        v1 = AgentVault("agent-x", master_secret=b"m1")
        v2 = AgentVault("agent-x", master_secret=b"m2")
        s = v1.store("k", "v")
        with pytest.raises(ValueError, match="Decryption failed"):
            v2._decrypt(s._encrypted_value)


# ─────────────────────────────────────────────────────────────────────────────
# VaultRotator
# ─────────────────────────────────────────────────────────────────────────────


class TestVaultRotator:
    def test_default_policies_loaded(self) -> None:
        r = VaultRotator()
        # Should have one policy per default entry
        assert r.policy_count == len(VaultRotator.DEFAULT_POLICIES)
        # API_KEY is one of the defaults
        assert r.get_policy(SecretType.API_KEY) is not None

    def test_no_defaults_skips_loading(self) -> None:
        r = VaultRotator(load_defaults=False)
        assert r.policy_count == 0
        assert r.get_policy(SecretType.API_KEY) is None

    def test_add_policy_overrides_existing(self) -> None:
        r = VaultRotator()
        new_policy = RotationPolicy(
            "ROT-API",  # same id as the default API_KEY policy
            SecretType.API_KEY,
            rotation_interval_hours=1,
            max_age_hours=2,
        )
        r.add_policy(new_policy)
        # Same id → replaces, count unchanged
        assert r.policy_count == len(VaultRotator.DEFAULT_POLICIES)
        # get_policy returns the new one for API_KEY
        got = r.get_policy(SecretType.API_KEY)
        assert got is not None
        assert got.rotation_interval_hours == 1

    def test_check_rotation_for_fresh_secret_returns_empty(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        vault.store("k", "v", secret_type=SecretType.API_KEY)
        r = VaultRotator()
        assert r.check_rotation_needed(vault) == []

    def test_check_rotation_for_aged_secret(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v", secret_type=SecretType.TOKEN)
        # Token policy: rotate every 24h. Backdate to 25h ago.
        s.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 25 * 3600))
        s.last_rotated = ""
        r = VaultRotator()
        needs = r.check_rotation_needed(vault)
        assert s in needs

    def test_check_rotation_skips_disabled_policy(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        vault.store("k", "v", secret_type=SecretType.API_KEY)
        r = VaultRotator(load_defaults=False)
        r.add_policy(
            RotationPolicy(
                "ROT-API-OFF",
                SecretType.API_KEY,
                rotation_interval_hours=1,
                auto_rotate=False,
            )
        )
        # auto_rotate=False → should never propose rotation
        assert r.check_rotation_needed(vault) == []

    def test_check_rotation_handles_corrupt_timestamp(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v", secret_type=SecretType.TOKEN)
        s.created_at = "not-a-valid-iso-timestamp"
        s.last_rotated = ""
        r = VaultRotator()
        # Falls back to now_ts → age == 0 → no rotation
        assert r.check_rotation_needed(vault) == []

    def test_auto_rotate_writes_log_entries(self) -> None:
        vault = AgentVault("agent", master_secret=b"m")
        s = vault.store("k", "v", secret_type=SecretType.API_KEY)
        r = VaultRotator()
        ids = r.auto_rotate(vault)
        assert s.secret_id in ids
        stats = r.stats()
        assert stats["total_rotations"] == 1
        # The new value differs from the stored "v"
        assert vault.retrieve(s.secret_id) != "v"


# ─────────────────────────────────────────────────────────────────────────────
# IsolatedSessionStore
# ─────────────────────────────────────────────────────────────────────────────


class TestIsolatedSessionStore:
    def test_create_and_get_session(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("agent-1", tenant_id="tenant-x")
        got = store.get_session("agent-1", s.session_id)
        assert got is s
        assert got.tenant_id == "tenant-x"
        assert got.is_active is True

    def test_get_session_for_unknown_agent_returns_none(self) -> None:
        store = IsolatedSessionStore()
        assert store.get_session("nobody", "SESS-x") is None

    def test_close_session_marks_inactive_keeps_record(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("agent-1")
        assert store.close_session("agent-1", s.session_id) is True
        # Still retrievable, but inactive
        got = store.get_session("agent-1", s.session_id)
        assert got is not None
        assert got.is_active is False

    @pytest.mark.parametrize(
        ("agent", "session"),
        [
            ("nobody", "any-sess"),  # unknown agent
            ("agent-1", "missing-sess"),  # known agent, unknown session
        ],
    )
    def test_close_session_unknown_returns_false(self, agent: str, session: str) -> None:
        store = IsolatedSessionStore()
        store.create_session("agent-1")
        assert store.close_session(agent, session) is False

    def test_destroy_session_removes_record(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("agent-1")
        assert store.destroy_session("agent-1", s.session_id) is True
        assert store.get_session("agent-1", s.session_id) is None
        assert store.destroy_session("agent-1", s.session_id) is False

    def test_destroy_session_unknown_agent(self) -> None:
        store = IsolatedSessionStore()
        assert store.destroy_session("nobody", "any") is False

    def test_active_vs_all_after_close(self) -> None:
        store = IsolatedSessionStore()
        s1 = store.create_session("agent-1")
        s2 = store.create_session("agent-1")
        store.close_session("agent-1", s1.session_id)
        all_sess = store.agent_sessions("agent-1")
        active_sess = store.active_sessions("agent-1")
        assert len(all_sess) == 2
        assert len(active_sess) == 1
        assert active_sess[0].session_id == s2.session_id

    def test_purge_agent_returns_count_and_clears(self) -> None:
        store = IsolatedSessionStore()
        store.create_session("a")
        store.create_session("a")
        store.create_session("b")
        n = store.purge_agent("a")
        assert n == 2
        assert store.agent_sessions("a") == []
        # Other agent untouched
        assert len(store.agent_sessions("b")) == 1

    def test_purge_unknown_agent_returns_zero(self) -> None:
        store = IsolatedSessionStore()
        assert store.purge_agent("nobody") == 0

    def test_stats_aggregate(self) -> None:
        store = IsolatedSessionStore()
        store.create_session("a")
        s2 = store.create_session("a")
        store.create_session("b")
        store.close_session("a", s2.session_id)
        stats = store.stats()
        assert stats["agent_stores"] == 2
        assert stats["total_sessions"] == 3
        assert stats["active_sessions"] == 2

    def test_session_to_dict_lists_data_keys(self) -> None:
        store = IsolatedSessionStore()
        s = store.create_session("a", data={"role": "user", "lang": "de"})
        d = s.to_dict()
        assert set(d["data_keys"]) == {"role", "lang"}
        assert d["active"] is True
        assert d["agent_id"] == "a"


# ─────────────────────────────────────────────────────────────────────────────
# SessionFirewall
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionFirewall:
    def test_same_agent_authorized(self) -> None:
        fw = SessionFirewall(IsolatedSessionStore())
        assert fw.authorize("agent-1", "agent-1", "SESS-x") is True
        assert fw.violation_count == 0

    def test_cross_agent_blocked_and_logged(self) -> None:
        fw = SessionFirewall(IsolatedSessionStore())
        assert fw.authorize("attacker", "victim", "SESS-x") is False
        assert fw.violation_count == 1
        v = fw.violations()[0]
        assert v["requester"] == "attacker"
        assert v["target"] == "victim"
        assert v["action"] == "BLOCKED"

    def test_violations_returns_recent_first(self) -> None:
        fw = SessionFirewall(IsolatedSessionStore())
        fw.authorize("att1", "victim", "S1")
        fw.authorize("att2", "victim", "S2")
        violations = fw.violations(limit=10)
        # Most recent first
        assert violations[0]["requester"] == "att2"
        assert violations[1]["requester"] == "att1"

    def test_stats_counts_unique_attackers(self) -> None:
        fw = SessionFirewall(IsolatedSessionStore())
        fw.authorize("att1", "victim", "S1")
        fw.authorize("att1", "victim", "S2")  # same attacker again
        fw.authorize("att2", "victim", "S3")
        stats = fw.stats()
        assert stats["total_violations"] == 3
        assert stats["unique_attackers"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# AgentVaultManager (uses tmp_path to avoid touching ~/.cognithor/)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def vault_manager(tmp_path: Path) -> AgentVaultManager:
    """AgentVaultManager with master-secret in tmp_path (no ~/.cognithor write)."""
    return AgentVaultManager(master_secret_path=str(tmp_path / "vault_master.key"))


class TestAgentVaultManager:
    def test_create_vault_persists_in_manager(self, vault_manager: AgentVaultManager) -> None:
        v = vault_manager.create_vault("agent-1")
        assert vault_manager.get_vault("agent-1") is v
        assert vault_manager.vault_count == 1

    def test_get_unknown_vault_returns_none(self, vault_manager: AgentVaultManager) -> None:
        assert vault_manager.get_vault("nobody") is None

    def test_destroy_vault_revokes_secrets_and_purges_sessions(
        self, vault_manager: AgentVaultManager
    ) -> None:
        v = vault_manager.create_vault("agent-1")
        v.store("k", "secret-value")
        vault_manager.sessions.create_session("agent-1")
        assert vault_manager.destroy_vault("agent-1") is True
        # Vault gone
        assert vault_manager.get_vault("agent-1") is None
        # Sessions purged
        assert vault_manager.sessions.agent_sessions("agent-1") == []

    def test_destroy_unknown_vault_returns_false(self, vault_manager: AgentVaultManager) -> None:
        assert vault_manager.destroy_vault("nobody") is False

    def test_rotate_all_returns_per_agent_results(self, vault_manager: AgentVaultManager) -> None:
        v_a = vault_manager.create_vault("agent-a")
        v_b = vault_manager.create_vault("agent-b")
        v_a.store("k", "v", secret_type=SecretType.API_KEY)
        v_b.store("k", "v", secret_type=SecretType.API_KEY)
        results = vault_manager.rotate_all()
        # Both agents had at least one secret rotated
        assert "agent-a" in results
        assert "agent-b" in results
        assert len(results["agent-a"]) == 1
        assert len(results["agent-b"]) == 1

    def test_stats_aggregates_all_subsystems(self, vault_manager: AgentVaultManager) -> None:
        v = vault_manager.create_vault("agent-1")
        v.store("k", "v")
        vault_manager.sessions.create_session("agent-1")
        # Trigger a firewall violation
        vault_manager.firewall.authorize("att", "agent-1", "S1")
        stats = vault_manager.stats()
        assert stats["total_vaults"] == 1
        assert stats["total_secrets"] == 1
        assert stats["sessions"]["total_sessions"] == 1
        assert stats["firewall"]["total_violations"] == 1
        assert stats["rotation"]["policies"] == len(VaultRotator.DEFAULT_POLICIES)


# ─────────────────────────────────────────────────────────────────────────────
# _load_or_create_master_secret (file-system contract)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadOrCreateMasterSecret:
    def test_creates_32_byte_file_when_missing(self, tmp_path: Path) -> None:
        key_file = tmp_path / "vault_master.key"
        assert not key_file.exists()
        secret = _load_or_create_master_secret(str(key_file))
        assert key_file.exists()
        assert len(secret) == 32
        # File on disk matches what we got back
        assert key_file.read_bytes()[:32] == secret

    def test_loads_existing_secret_unchanged(self, tmp_path: Path) -> None:
        key_file = tmp_path / "vault_master.key"
        original = b"x" * 32 + b"trailing-data-ignored"
        key_file.write_bytes(original)
        secret = _load_or_create_master_secret(str(key_file))
        # Returns first 32 bytes only
        assert secret == b"x" * 32
        # File content untouched
        assert key_file.read_bytes() == original

    def test_truncated_file_is_regenerated(self, tmp_path: Path) -> None:
        key_file = tmp_path / "vault_master.key"
        key_file.write_bytes(b"too-short")
        secret = _load_or_create_master_secret(str(key_file))
        # Regenerated to full 32 bytes
        assert len(secret) == 32
        assert key_file.read_bytes()[:32] == secret

    def test_uses_cognithor_home_env_when_no_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Steer the default path to tmp_path so we don't touch ~/.cognithor/
        monkeypatch.setenv("COGNITHOR_HOME", str(tmp_path))
        secret = _load_or_create_master_secret(None)
        assert (tmp_path / "vault_master.key").exists()
        assert len(secret) == 32

    def test_two_calls_with_same_path_return_same_secret(self, tmp_path: Path) -> None:
        path = str(tmp_path / "vault_master.key")
        s1 = _load_or_create_master_secret(path)
        s2 = _load_or_create_master_secret(path)
        assert s1 == s2


# ─────────────────────────────────────────────────────────────────────────────
# RotationPolicy.to_dict (small but worth one assertion)
# ─────────────────────────────────────────────────────────────────────────────


class TestRotationPolicy:
    def test_to_dict_uses_enum_value(self) -> None:
        p = RotationPolicy("ROT-X", SecretType.PASSWORD)
        d = p.to_dict()
        assert d["type"] == "password"
        assert d["policy_id"] == "ROT-X"
        assert d["auto"] is True


# ─────────────────────────────────────────────────────────────────────────────
# AgentSession dataclass — quick sanity
# ─────────────────────────────────────────────────────────────────────────────


def test_agent_session_default_data_factory_is_independent() -> None:
    """Ensure ``data: dict[str, Any] = field(default_factory=dict)``
    does not leak state between instances (classic mutable-default bug)."""
    s1 = AgentSession(session_id="A", agent_id="a")
    s2 = AgentSession(session_id="B", agent_id="a")
    s1.data["x"] = 1
    assert "x" not in s2.data
    # mypy --strict needs this dict to be Any-valued
    s2.data["y"] = "str"
    typed: dict[str, Any] = s1.data
    assert typed["x"] == 1
