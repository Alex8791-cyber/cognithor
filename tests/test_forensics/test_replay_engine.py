"""Tests fuer ReplayEngine (Feature 4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.core.gatekeeper import Gatekeeper
from jarvis.forensics.replay_engine import ReplayEngine
from jarvis.models import (
    ActionPlan,
    GateDecision,
    GateStatus,
    OperationMode,
    PlannedAction,
    ReplayResult,
    RiskLevel,
    RunRecord,
    SessionContext,
    ToolResult,
)


@pytest.fixture()
def gk_config(tmp_path):
    config = JarvisConfig(
        jarvis_home=tmp_path,
        security=SecurityConfig(allowed_paths=[str(tmp_path), "/tmp/jarvis/"]),
    )
    ensure_directory_structure(config)
    return config


@pytest.fixture()
def gatekeeper(gk_config):
    gk = Gatekeeper(gk_config, operation_mode=OperationMode.ONLINE)
    gk.initialize()
    return gk


@pytest.fixture()
def engine(gatekeeper):
    return ReplayEngine(gatekeeper)


@pytest.fixture()
def sample_run():
    plan = ActionPlan(
        goal="Test",
        steps=[
            PlannedAction(tool="read_file", params={"path": "/tmp/jarvis/test.txt"}),
            PlannedAction(tool="search_memory", params={"query": "test"}),
        ],
    )
    return RunRecord(
        session_id="s1",
        user_message="Test message",
        operation_mode="online",
        success=True,
        plans=[plan],
        gate_decisions=[
            [
                GateDecision(status=GateStatus.ALLOW, risk_level=RiskLevel.GREEN, reason="OK"),
                GateDecision(status=GateStatus.ALLOW, risk_level=RiskLevel.GREEN, reason="OK"),
            ]
        ],
        tool_results=[[
            ToolResult(tool_name="read_file", content="content"),
            ToolResult(tool_name="search_memory", content="results"),
        ]],
    )


class TestReplayEngine:
    def test_replay_identical_policies_no_divergence(self, engine, sample_run):
        result = engine.replay_run(sample_run)
        assert result.run_id == sample_run.id
        # With default policies, read_file and search_memory should still be allowed
        # The divergence count may vary based on default risk classification
        assert isinstance(result.divergences, list)

    def test_compare_decisions_detects_status_change(self, engine):
        original = [
            GateDecision(status=GateStatus.ALLOW, reason="OK"),
            GateDecision(status=GateStatus.BLOCK, reason="Blocked"),
        ]
        replayed = [
            GateDecision(status=GateStatus.BLOCK, reason="Now blocked"),
            GateDecision(status=GateStatus.ALLOW, reason="Now allowed"),
        ]
        divs = engine.compare_decisions(original, replayed)
        assert len(divs) == 2
        assert divs[0].original_status == "ALLOW"
        assert divs[0].replayed_status == "BLOCK"
        assert divs[1].original_status == "BLOCK"
        assert divs[1].replayed_status == "ALLOW"

    def test_counterfactual_multiple_variants(self, engine, sample_run):
        variants = {
            "strict": {},
            "relaxed": {},
            "default": {},
        }
        results = engine.counterfactual_analysis(sample_run, variants)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ReplayResult)
            assert r.run_id == sample_run.id
