"""Tests for audit compliance features: HMAC, blockchain, export, breach."""

import hashlib
import hmac
import json
import tempfile
from pathlib import Path

import pytest


class TestHMACSignatures:
    """HMAC-SHA256 signatures on audit trail entries."""

    @pytest.fixture
    def audit_trail(self, tmp_path):
        from jarvis.security.audit import AuditTrail

        return AuditTrail(
            log_path=tmp_path / "test_audit.jsonl",
            hmac_key=b"test-secret-key-32bytes-long!!!!",
        )

    def test_record_includes_hmac_field(self, audit_trail):
        from jarvis.models import AuditEntry as GateAuditEntry, GateStatus, RiskLevel

        entry = GateAuditEntry(
            session_id="test-session",
            action_tool="test_tool",
            action_params_hash="abc123",
            decision_status=GateStatus.ALLOW,
            decision_reason="test",
            risk_level=RiskLevel.GREEN,
            policy_name="default",
        )
        audit_trail.record(entry)
        # Read the JSONL and verify hmac field exists
        lines = audit_trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        assert "hmac" in record
        assert len(record["hmac"]) == 64  # SHA-256 hex digest

    def test_hmac_is_deterministic(self, audit_trail):
        from jarvis.models import AuditEntry as GateAuditEntry, GateStatus, RiskLevel

        entry = GateAuditEntry(
            session_id="s1",
            action_tool="tool1",
            action_params_hash="h1",
            decision_status=GateStatus.ALLOW,
            decision_reason="r1",
            risk_level=RiskLevel.GREEN,
            policy_name="p1",
        )
        h1 = audit_trail.record(entry)

        # Recompute HMAC manually
        lines = audit_trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        data_for_hmac = record["hash"]
        expected = hmac.new(
            b"test-secret-key-32bytes-long!!!!",
            data_for_hmac.encode(),
            hashlib.sha256,
        ).hexdigest()
        assert record["hmac"] == expected

    def test_no_hmac_when_key_is_none(self, tmp_path):
        from jarvis.security.audit import AuditTrail
        from jarvis.models import AuditEntry as GateAuditEntry, GateStatus, RiskLevel

        trail = AuditTrail(log_path=tmp_path / "no_hmac.jsonl", hmac_key=None)
        entry = GateAuditEntry(
            session_id="s1",
            action_tool="tool1",
            action_params_hash="h1",
            decision_status=GateStatus.ALLOW,
            decision_reason="r1",
            risk_level=RiskLevel.GREEN,
            policy_name="p1",
        )
        trail.record(entry)
        lines = trail._log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[-1])
        assert "hmac" not in record

    def test_verify_chain_with_hmac(self, audit_trail):
        from jarvis.models import AuditEntry as GateAuditEntry, GateStatus, RiskLevel

        for i in range(5):
            entry = GateAuditEntry(
                session_id=f"s{i}",
                action_tool=f"tool{i}",
                action_params_hash=f"h{i}",
                decision_status=GateStatus.ALLOW,
                decision_reason=f"r{i}",
                risk_level=RiskLevel.GREEN,
                policy_name="p",
            )
            audit_trail.record(entry)

        valid, total, broken = audit_trail.verify_chain()
        assert valid is True
        assert total == 5
        assert broken == -1


class TestBlockchainAnchoring:
    """Periodic hash anchoring to external store."""

    @pytest.fixture
    def audit_trail(self, tmp_path):
        from jarvis.security.audit import AuditTrail

        return AuditTrail(log_path=tmp_path / "bc_audit.jsonl")

    def test_get_anchor_returns_hash_and_count(self, audit_trail):
        from jarvis.models import AuditEntry as GateAuditEntry, GateStatus, RiskLevel

        for i in range(3):
            entry = GateAuditEntry(
                session_id=f"s{i}",
                action_tool=f"tool{i}",
                action_params_hash=f"h{i}",
                decision_status=GateStatus.ALLOW,
                decision_reason=f"r{i}",
                risk_level=RiskLevel.GREEN,
                policy_name="p",
            )
            audit_trail.record(entry)
        anchor = audit_trail.get_anchor()
        assert "hash" in anchor
        assert anchor["entry_count"] == 3
        assert len(anchor["hash"]) == 64
        assert "timestamp" in anchor

    def test_anchor_changes_after_new_entry(self, audit_trail):
        from jarvis.models import AuditEntry as GateAuditEntry, GateStatus, RiskLevel

        entry = GateAuditEntry(
            session_id="s1",
            action_tool="t1",
            action_params_hash="h1",
            decision_status=GateStatus.ALLOW,
            decision_reason="r1",
            risk_level=RiskLevel.GREEN,
            policy_name="p",
        )
        audit_trail.record(entry)
        anchor1 = audit_trail.get_anchor()
        entry2 = GateAuditEntry(
            session_id="s2",
            action_tool="t2",
            action_params_hash="h2",
            decision_status=GateStatus.ALLOW,
            decision_reason="r2",
            risk_level=RiskLevel.GREEN,
            policy_name="p",
        )
        audit_trail.record(entry2)
        anchor2 = audit_trail.get_anchor()
        assert anchor1["hash"] != anchor2["hash"]
        assert anchor2["entry_count"] == 2


class TestUserDataExport:
    """GDPR Art. 15 — user can export their audit data."""

    def test_export_filters_by_channel(self, tmp_path):
        from jarvis.audit import AuditLogger

        logger = AuditLogger(log_dir=tmp_path)
        logger.log_tool_call("tool1", agent_name="jarvis", result="ok")
        logger.log_user_input("telegram", "hello from telegram")
        logger.log_user_input("cli", "hello from cli")

        entries = logger.get_entries_for_export(channel="telegram")
        assert len(entries) >= 1

    def test_export_returns_all_without_filter(self, tmp_path):
        from jarvis.audit import AuditLogger

        logger = AuditLogger(log_dir=tmp_path)
        logger.log_tool_call("tool1", result="ok")
        logger.log_tool_call("tool2", result="ok")
        entries = logger.get_entries_for_export()
        assert len(entries) >= 2


class TestBreachDetector:
    """GDPR Art. 33 — automatic breach detection and notification."""

    @pytest.fixture
    def detector(self, tmp_path):
        from jarvis.audit.breach_detector import BreachDetector

        return BreachDetector(state_path=tmp_path / "breach_state.json", cooldown_hours=0)

    def test_no_breach_on_normal_entries(self, detector, tmp_path):
        from jarvis.audit import AuditLogger

        logger = AuditLogger(log_dir=tmp_path)
        logger.log_tool_call("read_file", result="ok")
        breaches = detector.scan(logger)
        assert len(breaches) == 0

    def test_detects_security_critical_event(self, detector, tmp_path):
        from jarvis.audit import AuditLogger, AuditSeverity

        logger = AuditLogger(log_dir=tmp_path)
        logger.log_security("Unauthorized access attempt detected", severity=AuditSeverity.CRITICAL)
        breaches = detector.scan(logger)
        assert len(breaches) >= 1
        assert breaches[0]["severity"] == "critical"

    def test_cooldown_prevents_duplicate(self, tmp_path):
        from jarvis.audit.breach_detector import BreachDetector
        from jarvis.audit import AuditLogger, AuditSeverity

        detector = BreachDetector(state_path=tmp_path / "breach_state.json", cooldown_hours=24)
        logger = AuditLogger(log_dir=tmp_path)
        logger.log_security("Breach attempt", severity=AuditSeverity.CRITICAL)
        breaches1 = detector.scan(logger)
        assert len(breaches1) >= 1
        breaches2 = detector.scan(logger)
        assert len(breaches2) == 0

    def test_breach_report_format(self, detector, tmp_path):
        from jarvis.audit import AuditLogger, AuditSeverity

        logger = AuditLogger(log_dir=tmp_path)
        logger.log_security("Data exfiltration attempt", severity=AuditSeverity.CRITICAL)
        breaches = detector.scan(logger)
        assert len(breaches) >= 1
        report = breaches[0]
        assert "severity" in report
        assert "description" in report
        assert "timestamp" in report
        assert "gdpr_article" in report
        assert report["gdpr_article"] == "Art. 33 DSGVO"
