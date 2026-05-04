"""Tests für Audit Logger.

Testet:
  - Logging-Methoden (Tool-Calls, Datei, Netzwerk, Gatekeeper, etc.)
  - Abfragen (Filtering, Zeiträume)
  - Zusammenfassung (Summary mit Statistiken)
  - Export (JSON, CSV)
  - DSGVO-Compliance (PII-Löschung, Retention)
  - Parameter-Sanitizing (Credential-Redaction)
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cognithor.audit import (
    AuditCategory,
    AuditEntry,
    AuditLogger,
    AuditSeverity,
)

# ============================================================================
# Logging-Methoden
# ============================================================================


class TestAuditLogging:
    @pytest.fixture
    def logger(self) -> AuditLogger:
        return AuditLogger()

    def test_log_tool_call(self, logger: AuditLogger) -> None:
        entry = logger.log_tool_call(
            "memory_search",
            {"query": "BU-Tarif"},
            agent_name="researcher",
            duration_ms=42.5,
        )
        assert entry.category == AuditCategory.TOOL_CALL
        assert entry.tool_name == "memory_search"
        assert entry.agent_name == "researcher"
        assert entry.duration_ms == 42.5
        assert entry.success is True

    def test_log_tool_call_failure(self, logger: AuditLogger) -> None:
        entry = logger.log_tool_call("broken_tool", success=False)
        assert entry.severity == AuditSeverity.ERROR
        assert entry.success is False

    def test_log_file_access(self, logger: AuditLogger) -> None:
        entry = logger.log_file_access(
            str(Path(tempfile.gettempdir()) / "test.txt"), "write", agent_name="coder"
        )
        assert entry.category == AuditCategory.FILE_ACCESS
        assert "write" in entry.description

    def test_log_network(self, logger: AuditLogger) -> None:
        entry = logger.log_network("https://api.example.com", "POST", status_code=200)
        assert entry.category == AuditCategory.NETWORK
        assert "POST" in entry.description

    def test_log_agent_delegation(self, logger: AuditLogger) -> None:
        entry = logger.log_agent_delegation("planner", "coder", task="Bug fixen")
        assert entry.category == AuditCategory.AGENT_DELEGATION
        assert "planner" in entry.description

    def test_log_skill_install(self, logger: AuditLogger) -> None:
        entry = logger.log_skill_install(
            "bu_helper@1.0.0:abc",
            source="p2p",
            analysis_verdict="safe",
        )
        assert entry.category == AuditCategory.SKILL_INSTALL
        assert entry.success is True

    def test_log_gatekeeper_block(self, logger: AuditLogger) -> None:
        entry = logger.log_gatekeeper(
            "BLOCK",
            "Netzwerkzugriff verweigert",
            tool_name="http_fetch",
        )
        assert entry.category == AuditCategory.GATEKEEPER
        assert entry.success is False
        assert entry.severity == AuditSeverity.WARNING

    def test_log_gatekeeper_allow(self, logger: AuditLogger) -> None:
        entry = logger.log_gatekeeper("ALLOW", "Unbedenklich")
        assert entry.success is True

    def test_log_memory_op(self, logger: AuditLogger) -> None:
        entry = logger.log_memory_op("index", details="5 neue Chunks")
        assert entry.category == AuditCategory.MEMORY_OP

    def test_log_security(self, logger: AuditLogger) -> None:
        entry = logger.log_security(
            "Verdächtiger Zugriff auf /etc/passwd",
            severity=AuditSeverity.CRITICAL,
            blocked=True,
        )
        assert entry.category == AuditCategory.SECURITY
        assert entry.success is False

    def test_entry_count(self, logger: AuditLogger) -> None:
        logger.log_tool_call("a")
        logger.log_tool_call("b")
        logger.log_tool_call("c")
        assert logger.entry_count == 3


# ============================================================================
# Parameter-Sanitizing
# ============================================================================


class TestSanitizing:
    def test_redacts_credentials(self) -> None:
        logger = AuditLogger()
        entry = logger.log_tool_call(
            "api_call",
            {"url": "https://api.example.com", "api_key": "sk-secret123"},
        )
        assert entry.parameters["api_key"] == "***REDACTED***"
        assert entry.parameters["url"] == "https://api.example.com"

    def test_redacts_password(self) -> None:
        logger = AuditLogger()
        entry = logger.log_tool_call("login", {"password": "geheim", "user": "admin"})
        assert entry.parameters["password"] == "***REDACTED***"
        assert entry.parameters["user"] == "admin"

    def test_truncates_long_values(self) -> None:
        logger = AuditLogger()
        entry = logger.log_tool_call("tool", {"data": "x" * 5000})
        assert len(entry.parameters["data"]) < 5000
        assert "chars" in entry.parameters["data"]


# ============================================================================
# Abfragen
# ============================================================================


class TestAuditQuery:
    @pytest.fixture
    def logger_with_data(self) -> AuditLogger:
        logger = AuditLogger()
        logger.log_tool_call("memory_search", agent_name="researcher", duration_ms=10)
        logger.log_tool_call("file_write", agent_name="coder", success=False)
        logger.log_gatekeeper("BLOCK", "Verboten", tool_name="exec")
        logger.log_network("https://example.com", agent_name="researcher")
        logger.log_security("Warnung", blocked=True)
        return logger

    def test_query_all(self, logger_with_data: AuditLogger) -> None:
        entries = logger_with_data.query()
        assert len(entries) == 5

    def test_query_by_category(self, logger_with_data: AuditLogger) -> None:
        entries = logger_with_data.query(category=AuditCategory.TOOL_CALL)
        assert len(entries) == 2

    def test_query_by_agent(self, logger_with_data: AuditLogger) -> None:
        entries = logger_with_data.query(agent_name="researcher")
        assert len(entries) == 2

    def test_query_by_tool(self, logger_with_data: AuditLogger) -> None:
        entries = logger_with_data.query(tool_name="exec")
        assert len(entries) == 1

    def test_query_failures_only(self, logger_with_data: AuditLogger) -> None:
        entries = logger_with_data.query(success=False)
        assert len(entries) >= 2  # file_write failure + gatekeeper block + security

    def test_query_with_limit(self, logger_with_data: AuditLogger) -> None:
        entries = logger_with_data.query(limit=2)
        assert len(entries) == 2

    def test_query_by_severity(self, logger_with_data: AuditLogger) -> None:
        entries = logger_with_data.query(severity=AuditSeverity.WARNING)
        assert len(entries) >= 1

    def test_get_blocked_actions(self, logger_with_data: AuditLogger) -> None:
        blocked = logger_with_data.get_blocked_actions()
        assert len(blocked) >= 1


# ============================================================================
# Zusammenfassung
# ============================================================================


class TestAuditSummary:
    def test_summarize(self) -> None:
        logger = AuditLogger()
        logger.log_tool_call("a", agent_name="agent1", duration_ms=100)
        logger.log_tool_call("b", agent_name="agent1", duration_ms=200)
        logger.log_tool_call("c", agent_name="agent2", success=False)
        logger.log_gatekeeper("BLOCK", "Test")

        summary = logger.summarize(hours=1)
        assert summary.total_entries == 4
        assert summary.by_category.get("tool_call", 0) == 3
        assert summary.blocked_actions >= 1
        assert summary.errors >= 1
        assert summary.avg_duration_ms > 0

    def test_summary_to_dict(self) -> None:
        logger = AuditLogger()
        logger.log_tool_call("test", duration_ms=50)
        summary = logger.summarize(hours=1)
        d = summary.to_dict()
        assert "total_entries" in d
        assert "top_tools" in d


# ============================================================================
# Export
# ============================================================================


class TestAuditExport:
    def test_export_json(self, tmp_path: Path) -> None:
        logger = AuditLogger()
        logger.log_tool_call("tool_a")
        logger.log_tool_call("tool_b")

        path = tmp_path / "audit.json"
        count = logger.export_json(path, hours=1)
        assert count == 2
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["entry_count"] == 2
        assert len(data["entries"]) == 2

    def test_export_csv(self, tmp_path: Path) -> None:
        logger = AuditLogger()
        logger.log_tool_call("tool_a")
        logger.log_gatekeeper("BLOCK", "Test")

        path = tmp_path / "audit.csv"
        count = logger.export_csv(path, hours=1)
        assert count == 2
        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # Header + 2 Einträge
        assert "timestamp" in lines[0]

    def test_export_empty(self, tmp_path: Path) -> None:
        logger = AuditLogger()
        path = tmp_path / "empty.json"
        count = logger.export_json(path, hours=1)
        assert count == 0


# ============================================================================
# DSGVO & Retention
# ============================================================================


class TestAuditCompliance:
    def test_delete_pii_entries(self) -> None:
        logger = AuditLogger()
        logger.log_tool_call("tool_a")
        entry2 = logger.log_tool_call("tool_b")
        entry2.contains_pii = True

        removed = logger.delete_pii_entries()
        assert removed == 1
        assert logger.entry_count == 1

    def test_cleanup_old_entries(self) -> None:
        logger = AuditLogger(retention_days=30)

        # Alten Eintrag simulieren
        old_entry = AuditEntry(
            entry_id="old_1",
            timestamp=(datetime.now(UTC) - timedelta(days=60)).isoformat(),
            category=AuditCategory.TOOL_CALL,
            action="old_action",
        )
        logger._entries.append(old_entry)

        # Neuen Eintrag hinzufügen
        logger.log_tool_call("new_tool")

        removed = logger.cleanup_old_entries()
        assert removed == 1
        assert logger.entry_count == 1

    def test_entry_to_dict(self) -> None:
        entry = AuditEntry(
            entry_id="test_1",
            category=AuditCategory.TOOL_CALL,
            severity=AuditSeverity.INFO,
            action="tool:test",
            tool_name="test",
        )
        d = entry.to_dict()
        assert d["entry_id"] == "test_1"
        assert d["category"] == "tool_call"
        assert d["severity"] == "info"


# ============================================================================
# Persistenz
# ============================================================================


class TestAuditPersistence:
    def test_persist_to_file(self, tmp_path: Path) -> None:
        logger = AuditLogger(log_dir=tmp_path / "audit")
        logger.log_tool_call("tool_a")
        logger.log_tool_call("tool_b")

        # JSONL-Datei sollte existieren
        log_files = list((tmp_path / "audit").glob("audit_*.jsonl"))
        assert len(log_files) == 1

        lines = log_files[0].read_text().strip().split("\n")
        assert len(lines) == 2

        # Jede Zeile ist valides JSON
        for line in lines:
            data = json.loads(line)
            assert "entry_id" in data

    def test_stats(self, tmp_path: Path) -> None:
        logger = AuditLogger(log_dir=tmp_path / "audit", retention_days=60)
        stats = logger.stats()
        assert stats["retention_days"] == 60
        assert stats["has_persistence"] is True

    def test_no_persistence(self) -> None:
        logger = AuditLogger()
        stats = logger.stats()
        assert stats["has_persistence"] is False


# ============================================================================
# SEC-HIGH-5 — hash chain (autonomous security audit, 2026-05-04)
# ============================================================================


class TestHashChain:
    """``AuditLogger`` writes a per-file SHA-256 hash chain (``prev_hash``
    on every entry). ``validate_chain`` detects post-hoc tampering.
    """

    def _write_three_entries(self, audit_dir: Path) -> Path:
        logger = AuditLogger(log_dir=audit_dir)
        for i in range(3):
            logger.log_tool_call(f"tool_{i}", {"k": str(i)}, agent_name="test_agent", success=True)
        # Find the JSONL just produced (single date file).
        files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(files) == 1, f"expected 1 audit file, got {files}"
        return files[0]

    def test_first_entry_has_empty_prev_hash(self, tmp_path: Path) -> None:
        log_file = self._write_three_entries(tmp_path / "audit")
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        first = json.loads(lines[0])
        assert "prev_hash" in first
        assert first["prev_hash"] == ""

    def test_subsequent_entries_link_to_previous(self, tmp_path: Path) -> None:
        import hashlib

        log_file = self._write_three_entries(tmp_path / "audit")
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        for i in range(1, len(lines)):
            prev_data = json.loads(lines[i - 1])
            curr_data = json.loads(lines[i])
            canon_prev = json.dumps(prev_data, sort_keys=True, ensure_ascii=False)
            expected = hashlib.sha256(canon_prev.encode("utf-8")).hexdigest()
            assert curr_data["prev_hash"] == expected, (
                f"line {i + 1} prev_hash {curr_data['prev_hash'][:12]} "
                f"!= sha256(line {i}) {expected[:12]}"
            )

    def test_validate_chain_passes_clean_log(self, tmp_path: Path) -> None:
        audit_dir = tmp_path / "audit"
        log_file = self._write_three_entries(audit_dir)
        logger = AuditLogger(log_dir=audit_dir)
        ok, errors = logger.validate_chain(log_file)
        assert ok, f"clean chain should validate; errors: {errors}"
        assert errors == []

    def test_validate_chain_detects_mutation(self, tmp_path: Path) -> None:
        audit_dir = tmp_path / "audit"
        log_file = self._write_three_entries(audit_dir)
        # Tamper: rewrite the SECOND entry's parameters in place.
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        tampered = json.loads(lines[1])
        tampered["parameters"] = {"k": "PWNED"}
        lines[1] = json.dumps(tampered, ensure_ascii=False)
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger = AuditLogger(log_dir=audit_dir)
        ok, errors = logger.validate_chain(log_file)
        # The 3rd line's prev_hash was computed from the ORIGINAL 2nd
        # line; after mutation it no longer matches.
        assert not ok
        assert any("prev_hash mismatch" in e for e in errors), errors

    def test_validate_chain_detects_deletion(self, tmp_path: Path) -> None:
        audit_dir = tmp_path / "audit"
        log_file = self._write_three_entries(audit_dir)
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        del lines[1]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger = AuditLogger(log_dir=audit_dir)
        ok, errors = logger.validate_chain(log_file)
        assert not ok
        assert any("prev_hash mismatch" in e for e in errors), errors

    def test_validate_chain_handles_missing_file(self, tmp_path: Path) -> None:
        logger = AuditLogger(log_dir=tmp_path / "audit")
        ok, errors = logger.validate_chain(tmp_path / "audit" / "ghost.jsonl")
        assert ok
        assert errors == []

    def test_chain_survives_logger_restart(self, tmp_path: Path) -> None:
        """A fresh ``AuditLogger`` pointing at an existing log file
        must reload the prior tail's hash so its first new write
        chains correctly. Otherwise process restarts would silently
        break the chain.
        """
        audit_dir = tmp_path / "audit"
        first = AuditLogger(log_dir=audit_dir)
        for i in range(2):
            first.log_tool_call(f"a{i}", {}, agent_name="agent", success=True)

        log_file = next(iter(audit_dir.glob("audit_*.jsonl")))

        second = AuditLogger(log_dir=audit_dir)
        second.log_tool_call("a2", {}, agent_name="agent", success=True)

        ok, errors = second.validate_chain(log_file)
        assert ok, f"chain broken after restart: {errors}"


# ============================================================================
# TRUST-1 — Run-Receipts (operational-trust audit, 2026-05-04)
# ============================================================================


class TestRunReceipt:
    """``AuditLogger.run_receipt(session_id)`` aggregates every entry
    tagged with that session into a signed JSON bundle for post-mortem
    reconstruction. Reviewer asked: "If something goes wrong, can an
    operator reconstruct exactly what the agent did?" — receipts are
    the answer.
    """

    def _logger_with_two_runs(self) -> AuditLogger:
        logger = AuditLogger()
        # Run A: tool call + gatekeeper allow + delegation
        logger.log_tool_call(
            "read_file",
            {"path": "/tmp/x"},
            agent_name="planner",
            session_id="run_A",
            duration_ms=12.5,
        )
        logger.log_gatekeeper(
            "ALLOW",
            "GREEN tool",
            tool_name="read_file",
            session_id="run_A",
        )
        logger.log_agent_delegation("planner", "executor", task="read it", session_id="run_A")
        # Run B: failed tool call + security warning
        logger.log_tool_call(
            "exec_command",
            {"command": "rm -rf /"},
            agent_name="planner",
            success=False,
            session_id="run_B",
        )
        logger.log_security(
            "Destructive command blocked",
            blocked=True,
            session_id="run_B",
        )
        # Untagged entry (e.g. boot-time) — must NOT appear in either receipt
        logger.log_system("startup")
        return logger

    def test_session_id_threads_through_log_methods(self) -> None:
        logger = AuditLogger()
        entry = logger.log_tool_call("x", session_id="run_42", agent_name="a", success=True)
        assert entry.session_id == "run_42"

        gk = logger.log_gatekeeper("ALLOW", "ok", tool_name="x", session_id="run_42")
        assert gk.session_id == "run_42"

    def test_receipt_aggregates_only_matching_session(self) -> None:
        logger = self._logger_with_two_runs()
        receipt = logger.run_receipt("run_A")
        assert receipt["session_id"] == "run_A"
        assert receipt["entry_count"] == 3
        # No run_B entries leaked in.
        for entry in receipt["entries"]:
            assert entry["session_id"] == "run_A"
        # Untagged system entry must also not leak.
        actions = [e["action"] for e in receipt["entries"]]
        assert "system:startup" not in actions

    def test_receipt_aggregate_counts(self) -> None:
        logger = self._logger_with_two_runs()
        receipt = logger.run_receipt("run_A")
        agg = receipt["aggregate"]
        assert agg["success_count"] == 3
        assert agg["failure_count"] == 0
        assert agg["by_category"]["tool_call"] == 1
        assert agg["by_category"]["gatekeeper"] == 1
        # ``by_tool`` counts every entry that references the tool — both
        # the tool_call itself and the gatekeeper decision about it.
        assert agg["by_tool"]["read_file"] == 2
        assert agg["total_duration_ms"] == pytest.approx(12.5, rel=1e-6)

    def test_receipt_failure_run_marked(self) -> None:
        logger = self._logger_with_two_runs()
        receipt = logger.run_receipt("run_B")
        agg = receipt["aggregate"]
        assert agg["failure_count"] == 2  # tool failed + security blocked
        assert agg["success_count"] == 0
        assert "destructive command" in receipt["entries"][1]["description"].lower()

    def test_unknown_session_returns_empty_receipt(self) -> None:
        """Reading a session that was never logged returns a structured
        empty receipt — caller can distinguish 'no run' from 'crash'.
        """
        logger = AuditLogger()
        logger.log_tool_call("x", session_id="real")
        receipt = logger.run_receipt("ghost")
        assert receipt["entry_count"] == 0
        assert receipt["entries"] == []
        assert receipt["session_id"] == "ghost"
        assert receipt["schema_version"] == AuditLogger.RECEIPT_SCHEMA_VERSION

    def test_signed_receipt_round_trip(self) -> None:
        logger = self._logger_with_two_runs()
        key = "test-secret-key-do-not-use-in-prod"
        receipt = logger.run_receipt("run_A", signing_key=key)
        assert receipt["signature"]  # non-empty
        assert AuditLogger.verify_receipt_signature(receipt, key)

    def test_signed_receipt_rejects_tampering(self) -> None:
        logger = self._logger_with_two_runs()
        key = "k"
        receipt = logger.run_receipt("run_A", signing_key=key)
        # Tamper with one entry's parameters.
        receipt["entries"][0]["parameters"] = {"path": "/etc/shadow"}
        assert not AuditLogger.verify_receipt_signature(receipt, key)

    def test_signed_receipt_rejects_wrong_key(self) -> None:
        logger = self._logger_with_two_runs()
        receipt = logger.run_receipt("run_A", signing_key="key1")
        assert not AuditLogger.verify_receipt_signature(receipt, "key2")

    def test_unsigned_receipt_signature_empty(self) -> None:
        logger = self._logger_with_two_runs()
        receipt = logger.run_receipt("run_A")  # no signing_key
        assert receipt["signature"] == ""
        # Verifying an unsigned receipt with any key returns False.
        assert not AuditLogger.verify_receipt_signature(receipt, "anything")

    def test_receipt_entries_ordered_by_entry_id(self) -> None:
        """Entry order in the bundle is deterministic — sorted by the
        numeric tail of ``audit_<n>``. Important for repeatable
        signatures + readable receipts.
        """
        logger = self._logger_with_two_runs()
        receipt = logger.run_receipt("run_A")
        ids = [e["entry_id"] for e in receipt["entries"]]
        # Numeric-tail sort (audit_1 < audit_2 < ... < audit_10)
        nums = [int(eid.rsplit("_", 1)[-1]) for eid in ids]
        assert nums == sorted(nums)

    def test_receipt_reads_from_disk_when_in_memory_empty(self, tmp_path: Path) -> None:
        """A fresh logger pointed at an existing log_dir can still
        produce a receipt for a past run — important for post-mortem
        audits across process restarts.
        """
        audit_dir = tmp_path / "audit"
        first = AuditLogger(log_dir=audit_dir)
        first.log_tool_call("x", session_id="past_run", agent_name="a")
        first.log_tool_call("y", session_id="past_run", agent_name="a")

        # New logger (cold cache).
        second = AuditLogger(log_dir=audit_dir)
        receipt = second.run_receipt("past_run")
        assert receipt["entry_count"] == 2
        # Both entries share the session.
        assert all(e["session_id"] == "past_run" for e in receipt["entries"])

    def test_legacy_entries_without_session_id_default_to_empty(self) -> None:
        """Existing callers that didn't pass session_id keep working —
        the default is empty string, and run_receipt('') would match
        all such entries (use case: 'show me everything outside any run').
        """
        logger = AuditLogger()
        logger.log_tool_call("legacy_call")  # no session_id
        receipt = logger.run_receipt("")
        assert receipt["entry_count"] == 1
