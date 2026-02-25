"""Tests: Compliance & Audit-Report Framework.

Prüft DecisionLog, ComplianceFramework, ReportExporter
und RemediationTracker.
"""

from __future__ import annotations

import pytest

from jarvis.audit.compliance import (
    ComplianceCheck,
    ComplianceFramework,
    ComplianceReport,
    ComplianceStatus,
    DecisionLog,
    DecisionRecord,
    RemediationItem,
    RemediationStatus,
    RemediationTracker,
    ReportExporter,
    RiskLevel,
)


# ============================================================================
# DecisionLog
# ============================================================================


class TestDecisionLog:
    def test_log_and_count(self) -> None:
        log = DecisionLog()
        log.log(DecisionRecord(
            decision_id="d1", agent_id="coder", timestamp="2025-01-01T00:00:00Z",
            action="execute_tool", reasoning="User requested code execution",
        ))
        assert log.count == 1

    def test_query_by_agent(self) -> None:
        log = DecisionLog()
        log.log(DecisionRecord(decision_id="d1", agent_id="coder", timestamp="2025-01-01T00:00:00Z", action="code", reasoning="r"))
        log.log(DecisionRecord(decision_id="d2", agent_id="researcher", timestamp="2025-01-01T00:00:00Z", action="search", reasoning="r"))
        results = log.query(agent_id="coder")
        assert len(results) == 1
        assert results[0].agent_id == "coder"

    def test_query_by_risk_flags(self) -> None:
        log = DecisionLog()
        log.log(DecisionRecord(decision_id="d1", agent_id="a", timestamp="t", action="a", reasoning="r", risk_flags=["high_cost"]))
        log.log(DecisionRecord(decision_id="d2", agent_id="a", timestamp="t", action="a", reasoning="r"))
        flagged = log.flagged_decisions()
        assert len(flagged) == 1

    def test_approval_rate(self) -> None:
        log = DecisionLog()
        log.log(DecisionRecord(decision_id="d1", agent_id="a", timestamp="t", action="a", reasoning="r", human_approved=True))
        log.log(DecisionRecord(decision_id="d2", agent_id="a", timestamp="t", action="a", reasoning="r", human_approved=False))
        assert log.approval_rate() == 50.0

    def test_stats(self) -> None:
        log = DecisionLog()
        for i in range(5):
            log.log(DecisionRecord(
                decision_id=f"d{i}", agent_id=f"agent_{i%2}", timestamp="t",
                action="act", reasoning="r", confidence=0.8,
            ))
        stats = log.stats()
        assert stats["total_decisions"] == 5
        assert stats["unique_agents"] == 2
        assert stats["avg_confidence"] == 0.8

    def test_max_entries(self) -> None:
        log = DecisionLog(max_entries=5)
        for i in range(10):
            log.log(DecisionRecord(decision_id=f"d{i}", agent_id="a", timestamp="t", action="a", reasoning="r"))
        assert log.count == 5


# ============================================================================
# ComplianceFramework
# ============================================================================


class TestComplianceFramework:
    def test_default_checks_loaded(self) -> None:
        fw = ComplianceFramework()
        assert fw.check_count == 10

    def test_assess(self) -> None:
        fw = ComplianceFramework()
        result = fw.assess("EUAIA-12.1", ComplianceStatus.COMPLIANT, evidence="AuditLogger aktiv")
        assert result is not None
        assert result.status == ComplianceStatus.COMPLIANT

    def test_assess_nonexistent(self) -> None:
        fw = ComplianceFramework()
        assert fw.assess("NONEXISTENT", ComplianceStatus.COMPLIANT) is None

    def test_auto_assess(self) -> None:
        fw = ComplianceFramework()
        fw.auto_assess(
            has_audit_log=True,
            has_decision_log=True,
            has_kill_switch=True,
            has_encryption=True,
            has_rbac=True,
            has_sandbox=True,
            has_approval_workflow=True,
            has_redteam=True,
        )
        score = fw.compliance_score()
        assert score >= 80.0  # Mostly compliant

    def test_auto_assess_minimal(self) -> None:
        fw = ComplianceFramework()
        fw.auto_assess()  # Alles False
        score = fw.compliance_score()
        assert score < 50.0

    def test_non_compliant_checks(self) -> None:
        fw = ComplianceFramework()
        fw.auto_assess()
        nc = fw.non_compliant_checks()
        assert len(nc) >= 1

    def test_compliance_score_empty(self) -> None:
        fw = ComplianceFramework(checks=[])
        assert fw.compliance_score() == 0.0

    def test_generate_report(self) -> None:
        fw = ComplianceFramework()
        fw.auto_assess(has_audit_log=True, has_decision_log=True)
        report = fw.generate_report()
        assert isinstance(report, ComplianceReport)
        assert report.framework_name == "EU-AI-Act + DSGVO"
        assert report.total_checks == 10

    def test_stats(self) -> None:
        fw = ComplianceFramework()
        stats = fw.stats()
        assert stats["total_checks"] == 10


# ============================================================================
# ReportExporter
# ============================================================================


class TestReportExporter:
    def _make_report(self) -> ComplianceReport:
        fw = ComplianceFramework()
        fw.auto_assess(has_audit_log=True, has_sandbox=True)
        return fw.generate_report()

    def test_to_json(self) -> None:
        report = self._make_report()
        json_str = ReportExporter.to_json(report)
        assert "compliance_score" in json_str
        assert "EUAIA" in json_str

    def test_to_csv(self) -> None:
        report = self._make_report()
        csv_str = ReportExporter.to_csv(report)
        lines = csv_str.strip().split("\n")
        assert len(lines) == 11  # Header + 10 checks
        assert "Check-ID" in lines[0]

    def test_to_markdown(self) -> None:
        report = self._make_report()
        md = ReportExporter.to_markdown(report)
        assert "# Compliance-Bericht" in md
        assert "✅" in md or "⚠️" in md or "❌" in md


# ============================================================================
# RemediationTracker
# ============================================================================


class TestRemediationTracker:
    def test_add_and_count(self) -> None:
        tracker = RemediationTracker()
        tracker.add(RemediationItem(item_id="r1", check_id="EUAIA-12.1", title="Fix Logs", description="desc"))
        assert tracker.count == 1

    def test_resolve(self) -> None:
        tracker = RemediationTracker()
        tracker.add(RemediationItem(item_id="r1", check_id="c1", title="Fix", description="d"))
        result = tracker.resolve("r1")
        assert result is not None
        assert result.status == RemediationStatus.RESOLVED

    def test_resolve_nonexistent(self) -> None:
        tracker = RemediationTracker()
        assert tracker.resolve("nope") is None

    def test_open_items(self) -> None:
        tracker = RemediationTracker()
        tracker.add(RemediationItem(item_id="r1", check_id="c1", title="Fix 1", description="d"))
        tracker.add(RemediationItem(item_id="r2", check_id="c2", title="Fix 2", description="d"))
        tracker.resolve("r1")
        assert len(tracker.open_items()) == 1

    def test_overdue_items(self) -> None:
        tracker = RemediationTracker()
        tracker.add(RemediationItem(item_id="r1", check_id="c1", title="Fix", description="d", due_date="2024-01-01T00:00:00Z"))
        overdue = tracker.overdue_items(reference_date="2025-01-01T00:00:00Z")
        assert len(overdue) == 1

    def test_stats(self) -> None:
        tracker = RemediationTracker()
        tracker.add(RemediationItem(item_id="r1", check_id="c1", title="A", description="d"))
        tracker.add(RemediationItem(item_id="r2", check_id="c2", title="B", description="d"))
        tracker.resolve("r1")
        stats = tracker.stats()
        assert stats["total"] == 2
        assert stats["resolved"] == 1
        assert stats["open"] == 1
