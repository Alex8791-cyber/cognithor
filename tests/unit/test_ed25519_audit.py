"""Tests for Ed25519 asymmetric audit trail signatures."""

from __future__ import annotations

import json

import pytest


class TestEd25519Signatures:
    """Ed25519 signing and verification on audit entries."""

    @pytest.fixture
    def key_pair(self):
        """Generate Ed25519 key pair for testing."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            private_key = Ed25519PrivateKey.generate()
            return (
                private_key.private_bytes_raw(),
                private_key.public_key().public_bytes_raw(),
            )
        except ImportError:
            pytest.skip("cryptography package not installed")

    @pytest.fixture
    def audit_trail(self, tmp_path, key_pair):
        from jarvis.security.audit import AuditTrail

        private_bytes, _ = key_pair
        return AuditTrail(log_path=tmp_path / "ed25519_audit.jsonl", ed25519_key=private_bytes)

    @staticmethod
    def _make_entry():
        from jarvis.models import AuditEntry as GateAuditEntry, GateStatus, RiskLevel

        return GateAuditEntry(
            session_id="s1",
            action_tool="t1",
            action_params_hash="h1",
            decision_status=GateStatus.ALLOW,
            decision_reason="r1",
            risk_level=RiskLevel.GREEN,
            policy_name="p",
        )

    def test_record_includes_ed25519_sig(self, audit_trail, key_pair):
        entry = self._make_entry()
        audit_trail.record(entry)
        lines = audit_trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        assert "ed25519_sig" in record
        assert len(record["ed25519_sig"]) == 128  # 64 bytes hex

    def test_signature_verifiable_with_public_key(self, audit_trail, key_pair):
        from jarvis.security.audit import AuditTrail

        _, public_bytes = key_pair
        entry = self._make_entry()
        audit_trail.record(entry)
        lines = audit_trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        assert AuditTrail.verify_signature(record, public_bytes) is True

    def test_tampered_entry_fails_verification(self, audit_trail, key_pair):
        from jarvis.security.audit import AuditTrail

        _, public_bytes = key_pair
        entry = self._make_entry()
        audit_trail.record(entry)
        lines = audit_trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        record["hash"] = "tampered_hash_value"
        assert AuditTrail.verify_signature(record, public_bytes) is False

    def test_wrong_public_key_fails(self, audit_trail, key_pair):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        from jarvis.security.audit import AuditTrail

        entry = self._make_entry()
        audit_trail.record(entry)
        lines = audit_trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        wrong_pub = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        assert AuditTrail.verify_signature(record, wrong_pub) is False

    def test_no_sig_when_key_is_none(self, tmp_path):
        from jarvis.security.audit import AuditTrail

        trail = AuditTrail(log_path=tmp_path / "no_ed25519.jsonl", ed25519_key=None)
        entry = self._make_entry()
        trail.record(entry)
        lines = trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        assert "ed25519_sig" not in record

    def test_verify_signature_missing_fields(self):
        from jarvis.security.audit import AuditTrail

        assert AuditTrail.verify_signature({}, b"\x00" * 32) is False
        assert AuditTrail.verify_signature({"hash": "abc"}, b"\x00" * 32) is False
        assert AuditTrail.verify_signature({"ed25519_sig": "aa"}, b"\x00" * 32) is False

    def test_verify_chain_with_ed25519(self, audit_trail, key_pair):
        """Chain verification still works when ed25519_sig is present."""
        for i in range(3):
            entry = self._make_entry()
            audit_trail.record(entry)
        valid, total, broken_at = audit_trail.verify_chain()
        assert valid is True
        assert total == 3
        assert broken_at == -1
