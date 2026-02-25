"""Tests fuer GovernanceAgent (Feature 5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis.governance.governor import GovernanceAgent
from jarvis.models import PolicyChange, PolicyProposal


def _mock_telemetry(tool_stats=None, unused_tools=None):
    """Mock TaskTelemetryCollector mit konfigurierbaren Stats."""
    mock = MagicMock()
    mock.get_tool_stats.return_value = tool_stats or {}
    mock.get_unused_tools.return_value = unused_tools or []
    return mock


def _mock_error_clusterer(clusters=None):
    mock = MagicMock()
    # Governor erwartet Liste von Dicts mit "count", "id", "pattern"
    mock.get_clusters.return_value = clusters or []
    return mock


def _mock_profiler(latency_stats=None):
    mock = MagicMock()
    # Governor ruft get_latency_stats() auf → dict[str, {"p95": float}]
    mock.get_latency_stats.return_value = latency_stats or {}
    return mock


def _mock_cost_tracker(budget_limit=10.0, budget_used=5.0):
    mock = MagicMock()
    # Governor ruft get_budget_info() auf → dict mit "limit" und "used"
    mock.get_budget_info.return_value = {
        "limit": budget_limit,
        "used": budget_used,
    }
    return mock


@pytest.fixture()
def governor(tmp_path):
    db = str(tmp_path / "gov.db")
    gov = GovernanceAgent(db_path=db)
    yield gov
    gov.close()


class TestGovernorAnalysis:
    def test_analyze_high_error_rate_proposes_timeout(self, tmp_path):
        """Tool mit 40% Fehler → Vorschlag."""
        tel = _mock_telemetry(tool_stats={
            "broken_tool": {"total": 10, "errors": 4},
        })
        gov = GovernanceAgent(task_telemetry=tel, db_path=str(tmp_path / "gov1.db"))
        proposals = gov.analyze()
        gov.close()

        tool_proposals = [p for p in proposals if "broken_tool" in p.title]
        assert len(tool_proposals) > 0
        assert tool_proposals[0].category == "error_rate"

    def test_analyze_budget_warning(self, tmp_path):
        """85% Budget → Vorschlag fuer guenstigeres Model."""
        cost = _mock_cost_tracker(budget_limit=10.0, budget_used=8.5)
        gov = GovernanceAgent(cost_tracker=cost, db_path=str(tmp_path / "gov2.db"))
        proposals = gov.analyze()
        gov.close()

        budget_proposals = [p for p in proposals if p.category == "budget"]
        assert len(budget_proposals) > 0

    def test_analyze_recurring_error_cluster(self, tmp_path):
        """6x gleicher Fehler → Policy-Regel-Vorschlag."""
        cluster_mock = _mock_error_clusterer(clusters=[
            {"id": "cluster_1", "pattern": "TimeoutError in web_fetch", "count": 6},
        ])
        gov = GovernanceAgent(error_clusterer=cluster_mock, db_path=str(tmp_path / "gov3.db"))
        proposals = gov.analyze()
        gov.close()

        cluster_proposals = [p for p in proposals if p.category == "recurring_error"]
        assert len(cluster_proposals) > 0

    def test_analyze_healthy_system_no_proposals(self, governor):
        """Alles OK → leere Liste."""
        proposals = governor.analyze()
        assert proposals == []

    def test_analyze_slow_tool(self, tmp_path):
        """p95 > 10s → Timeout-Anpassung."""
        profiler = _mock_profiler(latency_stats={
            "slow_tool": {"p95": 15.0},
        })
        gov = GovernanceAgent(task_profiler=profiler, db_path=str(tmp_path / "gov4.db"))
        proposals = gov.analyze()
        gov.close()

        slow_proposals = [p for p in proposals if "slow_tool" in p.title]
        assert len(slow_proposals) > 0
        assert slow_proposals[0].category == "tool_latency"

    def test_analyze_unused_tools(self, tmp_path):
        """0 Calls → Cleanup-Vorschlag."""
        tel = _mock_telemetry(unused_tools=["dead_tool"])
        gov = GovernanceAgent(task_telemetry=tel, db_path=str(tmp_path / "gov5.db"))
        proposals = gov.analyze()
        gov.close()

        unused_proposals = [p for p in proposals if p.category == "unused_tool"]
        assert len(unused_proposals) > 0
        assert "dead_tool" in unused_proposals[0].title


class TestGovernorProposals:
    def test_approve_proposal_creates_change(self, governor):
        """Genehmigung erzeugt PolicyChange."""
        pid = governor._create_proposal(
            category="timeout",
            title="Timeout erhoehen",
            description="Tool X hat hohe Latenz",
            evidence={"p95": 15.0},
            suggested_change={"timeout": 30},
        )
        change = governor.approve_proposal(pid)
        assert change is not None
        assert isinstance(change, PolicyChange)
        assert change.proposal_id == pid

    def test_reject_proposal(self, governor):
        """Ablehnung mit Grund."""
        pid = governor._create_proposal(
            category="access",
            title="Tool entfernen",
            description="Tool Y ungenutzt",
            evidence={"calls": 0},
            suggested_change={"action": "remove"},
        )
        governor.reject_proposal(pid, reason="Behalte fuer spaeter")

        proposals = governor.get_proposal_history()
        rejected = [p for p in proposals if p.id == pid]
        assert len(rejected) == 1
        assert rejected[0].status == "rejected"
        assert rejected[0].decision_reason == "Behalte fuer spaeter"

    def test_get_pending_proposals(self, governor):
        """Nur unentschiedene Vorschlaege."""
        pid1 = governor._create_proposal(
            category="timeout", title="P1", description="D1",
            evidence={"x": 1}, suggested_change={"a": 1},
        )
        pid2 = governor._create_proposal(
            category="access", title="P2", description="D2",
            evidence={"x": 2}, suggested_change={"b": 2},
        )
        governor.approve_proposal(pid1)

        pending = governor.get_pending_proposals()
        assert len(pending) == 1
        assert pending[0].id == pid2

    def test_proposals_persisted(self, tmp_path):
        """Daten ueberleben close/reopen."""
        db = str(tmp_path / "persist_gov.db")
        gov1 = GovernanceAgent(db_path=db)
        gov1._create_proposal(
            category="test", title="Persist", description="D",
            evidence={"k": "v"}, suggested_change={"action": "test"},
        )
        gov1.close()

        gov2 = GovernanceAgent(db_path=db)
        proposals = gov2.get_pending_proposals()
        gov2.close()
        assert len(proposals) == 1
        assert proposals[0].title == "Persist"
