"""Tests for Compliance Audit Log."""

from __future__ import annotations

import json
import pytest
from jarvis.security.compliance_audit import ComplianceAuditLog


@pytest.fixture
def audit_log(tmp_path):
    return ComplianceAuditLog(log_path=str(tmp_path / "audit" / "compliance.jsonl"))


def test_record_creates_entry(audit_log):
    entry = audit_log.record("consent_granted", user_id="user1", channel="telegram")
    assert entry["event"] == "consent_granted"
    assert entry["user_id"] == "user1"
    assert "hash" in entry
    assert "ts" in entry


def test_hash_chain_valid(audit_log):
    audit_log.record("consent_granted", user_id="user1")
    audit_log.record("erasure_requested", user_id="user1")
    audit_log.record("data_exported", user_id="user1")
    valid, count = audit_log.verify_chain()
    assert valid is True
    assert count == 3


def test_tamper_detection(audit_log):
    audit_log.record("consent_granted", user_id="user1")
    audit_log.record("erasure_requested", user_id="user2")
    # Tamper with the file
    with open(audit_log._path, "r") as f:
        lines = f.readlines()
    entry = json.loads(lines[0])
    entry["user_id"] = "tampered"
    lines[0] = json.dumps(entry) + "\n"
    with open(audit_log._path, "w") as f:
        f.writelines(lines)
    valid, _ = audit_log.verify_chain()
    assert valid is False


def test_get_entries_filtered(audit_log):
    audit_log.record("consent_granted", user_id="user1")
    audit_log.record("erasure_requested", user_id="user2")
    audit_log.record("consent_granted", user_id="user3")
    entries = audit_log.get_entries(event="consent_granted")
    assert len(entries) == 2


def test_pseudonymize_user(audit_log):
    audit_log.record("consent_granted", user_id="user1")
    audit_log.record("erasure_requested", user_id="user1")
    audit_log.record("consent_granted", user_id="user2")
    count = audit_log.pseudonymize_user("user1", salt="test_salt")
    assert count == 2
    # Chain should still be valid after pseudonymization
    valid, total = audit_log.verify_chain()
    assert valid is True
    assert total == 3
    # user1 should be replaced
    entries = audit_log.get_entries()
    assert all(e.get("user_id") != "user1" for e in entries)


def test_genesis_hash(audit_log):
    entry = audit_log.record("test_event")
    assert entry["prev_hash"] == "genesis"


def test_empty_log_verify(audit_log):
    valid, count = audit_log.verify_chain()
    assert valid is True
    assert count == 0
