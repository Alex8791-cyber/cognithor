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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.audit import (
    AuditCategory,
    AuditEntry,
    AuditLogger,
    AuditSeverity,
    AuditSummary,
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
        entry = logger.log_file_access("/tmp/test.txt", "write", agent_name="coder")
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
            "BLOCK", "Netzwerkzugriff verweigert",
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
        entry1 = logger.log_tool_call("tool_a")
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
            timestamp=(datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
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
