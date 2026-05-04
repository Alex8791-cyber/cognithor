"""Tests for the TRUST-9 wiring into AgentVault.store (#412) and
IsolatedSessionStore.create_session (this PR)."""

from __future__ import annotations

from cognithor.security.agent_vault import (
    AgentVault,
    IsolatedSessionStore,
    SecretType,
)


class TestAgentVaultProvenance:
    """``AgentVault.store(provenance_source_type=..., provenance_source_id=...)``
    writes a tag to the canonical PROVENANCE_LEDGER keyed by the new
    ``secret_id`` so the operational-trust receipt can answer
    "where did this stored secret come from?".
    """

    def test_store_without_provenance_does_not_tag(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            vault = AgentVault("agent-1", master_secret=b"test-secret")
            secret = vault.store("api_key_no_prov", "value-1")
            assert secret.secret_id not in isolated
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_store_with_provenance_tags_ledger(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            vault = AgentVault("agent-1", master_secret=b"test-secret")
            secret = vault.store(
                "api_key_with_prov",
                "value-1",
                secret_type=SecretType.API_KEY,
                provenance_source_type="user_directive",
                provenance_source_id="owner-paste-7",
                provenance_notes="owner pasted at boot",
            )
            tag = isolated.current(secret.secret_id)
            assert tag is not None
            assert tag.source_type == SourceType.USER_DIRECTIVE
            assert tag.source_id == "owner-paste-7"
            assert tag.notes == "owner pasted at boot"
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_store_unknown_source_type_falls_back(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            vault = AgentVault("agent-1", master_secret=b"test-secret")
            secret = vault.store(
                "api_key_unknown",
                "value-1",
                provenance_source_type="not_a_real_source",
                provenance_source_id="x",
            )
            tag = isolated.current(secret.secret_id)
            assert tag is not None
            assert tag.source_type == SourceType.UNKNOWN
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_partial_provenance_args_skip_tag(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            vault = AgentVault("agent-1", master_secret=b"test-secret")
            s1 = vault.store(
                "only_type",
                "v",
                provenance_source_type="user_directive",
            )
            s2 = vault.store(
                "only_id",
                "v",
                provenance_source_id="owner-paste-7",
            )
            assert s1.secret_id not in isolated
            assert s2.secret_id not in isolated
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_empty_source_id_does_not_break_store(self) -> None:
        # ProvenanceTag construction rejects empty source_id, but the
        # vault helper must swallow that ValueError so secret storage
        # still succeeds.
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            vault = AgentVault("agent-1", master_secret=b"test-secret")
            # source_id is "" → fails the both-required check, no tag.
            secret = vault.store(
                "k",
                "v",
                provenance_source_type="user_directive",
                provenance_source_id="",
            )
            assert secret.secret_id is not None
            # Round-trip retrieval still works (secret was stored).
            assert vault.retrieve(secret.secret_id) == "v"
            assert len(isolated) == 0
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]


class TestIsolatedSessionStoreProvenance:
    """``IsolatedSessionStore.create_session(provenance_source_type=...,
    provenance_source_id=...)`` writes a tag to the canonical
    PROVENANCE_LEDGER keyed by the new session_id.
    """

    def test_create_session_without_provenance_does_not_tag(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            store = IsolatedSessionStore()
            session = store.create_session("agent-1")
            assert session.session_id not in isolated
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_create_session_with_provenance_tags_ledger(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            store = IsolatedSessionStore()
            session = store.create_session(
                "agent-1",
                tenant_id="tenant-x",
                data={"role": "user"},
                provenance_source_type="chat_utterance",
                provenance_source_id="msg-42",
                provenance_notes="onboarded via telegram",
            )
            tag = isolated.current(session.session_id)
            assert tag is not None
            assert tag.source_type == SourceType.CHAT_UTTERANCE
            assert tag.source_id == "msg-42"
            assert tag.notes == "onboarded via telegram"
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_partial_provenance_args_skip_tag(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            store = IsolatedSessionStore()
            s1 = store.create_session("agent-a", provenance_source_type="chat_utterance")
            s2 = store.create_session("agent-b", provenance_source_id="msg-7")
            assert s1.session_id not in isolated
            assert s2.session_id not in isolated
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_unknown_source_type_falls_back_to_unknown(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            store = IsolatedSessionStore()
            session = store.create_session(
                "agent-1",
                provenance_source_type="not_real",
                provenance_source_id="x",
            )
            tag = isolated.current(session.session_id)
            assert tag is not None
            assert tag.source_type == SourceType.UNKNOWN
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]
