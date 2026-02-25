"""Tests für security/audit.py – Unveränderliches Audit-Trail."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from jarvis.models import (
    AuditEntry,
    GateStatus,
    RiskLevel,
)
from jarvis.security.audit import AuditTrail, mask_credentials, mask_dict

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audit"
    d.mkdir()
    return d


@pytest.fixture
def trail(log_dir: Path) -> AuditTrail:
    return AuditTrail(log_dir=log_dir)


def _make_entry(
    tool: str = "read_file",
    status: GateStatus = GateStatus.ALLOW,
    session_id: str = "sess_123",
    params: dict | None = None,
    execution_result: str | None = None,
) -> AuditEntry:
    p = params or {"path": "/tmp/test.txt"}
    params_hash = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    return AuditEntry(
        session_id=session_id,
        action_tool=tool,
        action_params_hash=params_hash,
        decision_status=status,
        decision_reason="test",
        risk_level=RiskLevel.GREEN,
        execution_result=execution_result,
    )


# ============================================================================
# mask_credentials
# ============================================================================


class TestMaskCredentials:
    def test_masks_api_key(self):
        text = "key is sk-abc123456789012345678901234567"
        result = mask_credentials(text)
        assert "sk-abc1***" in result
        assert "012345678901234567" not in result

    def test_masks_token(self):
        result = mask_credentials("use token_abcd1234567890")
        assert "token_abcd***" in result

    def test_masks_password(self):
        result = mask_credentials("password: my_secret_pass")
        assert "my_secret_pass" not in result
        assert "password: ***" in result

    def test_masks_bearer(self):
        result = mask_credentials("Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_masks_github_token(self):
        result = mask_credentials("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
        assert "ghp_***" in result
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in result

    def test_masks_slack_token(self):
        result = mask_credentials("xoxb-123-456-abc")
        assert "xoxb-***" in result

    def test_empty_string(self):
        assert mask_credentials("") == ""

    def test_no_credentials(self):
        text = "Dies ist ein normaler Text."
        assert mask_credentials(text) == text


class TestMaskDict:
    def test_masks_nested_values(self):
        data = {
            "config": {
                "api_key": "sk-abc123456789012345678901234567",
                "name": "test",
            }
        }
        result = mask_dict(data)
        assert "sk-abc1***" in result["config"]["api_key"]
        assert result["config"]["name"] == "test"

    def test_masks_list_values(self):
        data = {"tokens": ["sk-abc123456789012345678901234567", "normal"]}
        result = mask_dict(data)
        assert "***" in result["tokens"][0]
        assert result["tokens"][1] == "normal"

    def test_depth_limit(self):
        # 12 levels deep should stop at depth 10
        data: dict = {"a": "sk-abc123456789012345678901234567"}
        current = data
        for _ in range(12):
            current["nested"] = {"a": "sk-abc123456789012345678901234567"}
            current = current["nested"]
        result = mask_dict(data)
        assert isinstance(result, dict)

    def test_empty_dict(self):
        assert mask_dict({}) == {}


# ============================================================================
# AuditTrail
# ============================================================================


class TestAuditTrailRecord:
    def test_records_entry(self, trail: AuditTrail):
        entry = _make_entry()
        h = trail.record(entry)
        assert h != ""
        assert trail.entry_count == 1

    def test_hash_chain(self, trail: AuditTrail):
        e1 = _make_entry(tool="read_file")
        e2 = _make_entry(tool="write_file")
        h1 = trail.record(e1)
        h2 = trail.record(e2)
        assert h1 != h2
        assert trail.entry_count == 2

    def test_masks_credentials_by_default(self, trail: AuditTrail):
        entry = _make_entry(execution_result="token is sk-abc123456789012345678901234567")
        trail.record(entry)
        content = trail.log_path.read_text()
        assert "sk-abc1***" in content
        assert "sk-abc123456789012345678901234567" not in content

    def test_no_mask_when_disabled(self, trail: AuditTrail):
        entry = _make_entry(execution_result="token is sk-abc123456789012345678901234567")
        trail.record(entry, mask=False)
        content = trail.log_path.read_text()
        assert "sk-abc123456789012345678901234567" in content

    def test_jsonl_format(self, trail: AuditTrail):
        trail.record(_make_entry())
        lines = trail.log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "hash" in parsed
        assert "prev_hash" in parsed
        assert parsed["prev_hash"] == "genesis"


class TestAuditTrailRecordEvent:
    def test_records_event(self, trail: AuditTrail):
        h = trail.record_event(
            session_id="sess_1",
            event_type="agent_spawn",
            details={"agent_id": "a1", "task": "test"},
        )
        assert h != ""
        assert trail.entry_count == 1

    def test_masks_event_details(self, trail: AuditTrail):
        trail.record_event(
            session_id="sess_1",
            event_type="test",
            details={"token": "sk-abc123456789012345678901234567"},
        )
        content = trail.log_path.read_text()
        assert "sk-abc123456789012345678901234567" not in content


class TestAuditTrailVerify:
    def test_verify_empty(self, trail: AuditTrail):
        valid, total, _broken = trail.verify_chain()
        assert valid is True
        assert total == 0

    def test_verify_valid_chain(self, trail: AuditTrail):
        for i in range(5):
            trail.record(_make_entry(tool=f"tool_{i}"))
        valid, total, broken = trail.verify_chain()
        assert valid is True
        assert total == 5
        assert broken == -1

    def test_verify_detects_tampering(self, trail: AuditTrail):
        trail.record(_make_entry(tool="tool_1"))
        trail.record(_make_entry(tool="tool_2"))

        # Tamper: Ändere den ersten Eintrag
        lines = trail.log_path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        entry["session_id"] = "TAMPERED"
        lines[0] = json.dumps(entry)
        trail.log_path.write_text("\n".join(lines) + "\n")

        valid, _total, broken = trail.verify_chain()
        assert valid is False
        assert broken == 0


class TestAuditTrailQuery:
    def test_query_by_session(self, trail: AuditTrail):
        trail.record(_make_entry(session_id="s1"))
        trail.record(_make_entry(session_id="s2"))
        trail.record(_make_entry(session_id="s1"))

        results = trail.query(session_id="s1")
        assert len(results) == 2

    def test_query_by_tool(self, trail: AuditTrail):
        trail.record(_make_entry(tool="read_file"))
        trail.record(_make_entry(tool="write_file"))
        trail.record(_make_entry(tool="read_file"))

        results = trail.query(tool="read_file")
        assert len(results) == 2

    def test_query_by_status(self, trail: AuditTrail):
        trail.record(_make_entry(status=GateStatus.ALLOW))
        trail.record(_make_entry(status=GateStatus.BLOCK))

        results = trail.query(status=GateStatus.BLOCK)
        assert len(results) == 1

    def test_query_limit(self, trail: AuditTrail):
        for _ in range(10):
            trail.record(_make_entry())
        results = trail.query(limit=3)
        assert len(results) == 3

    def test_query_empty_log(self, trail: AuditTrail):
        results = trail.query()
        assert results == []


class TestAuditTrailRestore:
    def test_restores_chain_on_reopen(self, log_dir: Path):
        trail1 = AuditTrail(log_dir=log_dir)
        trail1.record(_make_entry(tool="tool_1"))
        last_hash = trail1.last_hash

        # Neues AuditTrail-Objekt mit gleichem Pfad
        trail2 = AuditTrail(log_dir=log_dir)
        assert trail2.last_hash == last_hash
        assert trail2.entry_count == 1

        # Neue Einträge angehängt → Chain bleibt intakt
        trail2.record(_make_entry(tool="tool_2"))
        valid, total, _ = trail2.verify_chain()
        assert valid is True
        assert total == 2
