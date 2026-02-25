"""Tests fuer RunRecorder (Feature 4)."""

from __future__ import annotations

import pytest

from jarvis.models import (
    ActionPlan,
    GateDecision,
    GateStatus,
    PlannedAction,
    ReflectionResult,
    RiskLevel,
    SessionSummary,
    ToolResult,
)
from jarvis.forensics.run_recorder import RunRecorder


@pytest.fixture()
def recorder(tmp_path):
    db = str(tmp_path / "runs.db")
    rec = RunRecorder(db)
    yield rec
    rec.close()


@pytest.fixture()
def sample_plan():
    return ActionPlan(
        goal="Test goal",
        reasoning="Test reasoning",
        steps=[
            PlannedAction(tool="read_file", params={"path": "/tmp/test.txt"}),
            PlannedAction(tool="write_file", params={"path": "/tmp/out.txt", "content": "hello"}),
        ],
        confidence=0.9,
    )


@pytest.fixture()
def sample_decisions():
    return [
        GateDecision(status=GateStatus.ALLOW, risk_level=RiskLevel.GREEN, reason="OK"),
        GateDecision(status=GateStatus.INFORM, risk_level=RiskLevel.YELLOW, reason="File write"),
    ]


@pytest.fixture()
def sample_results():
    return [
        ToolResult(tool_name="read_file", content="file content", is_error=False),
        ToolResult(tool_name="write_file", content="written", is_error=False),
    ]


class TestRunRecorder:
    def test_start_and_finish_run(self, recorder):
        run_id = recorder.start_run(session_id="s1", user_message="Hello", operation_mode="offline")
        assert run_id
        recorder.finish_run(run_id, success=True, final_response="Done")
        run = recorder.get_run(run_id)
        assert run is not None
        assert run.id == run_id
        assert run.session_id == "s1"
        assert run.success is True
        assert run.final_response == "Done"

    def test_record_all_phases(self, recorder, sample_plan, sample_decisions, sample_results):
        run_id = recorder.start_run(session_id="s1", user_message="Test", operation_mode="online")
        recorder.record_plan(run_id, sample_plan)
        recorder.record_gate_decisions(run_id, sample_decisions)
        recorder.record_tool_results(run_id, sample_results)

        reflection = ReflectionResult(
            session_id="s1",
            success_score=0.9,
            evaluation="Good",
            session_summary=SessionSummary(goal="Test goal", outcome="Success"),
        )
        recorder.record_reflection(run_id, reflection)
        recorder.record_policy_snapshot(run_id, {"rule1": "allow"})
        recorder.finish_run(run_id, success=True, final_response="OK")

        run = recorder.get_run(run_id)
        assert len(run.plans) == 1
        assert len(run.gate_decisions) == 1
        assert len(run.tool_results) == 1
        assert run.reflection is not None
        assert run.policy_snapshot == {"rule1": "allow"}

    def test_get_run_restores_complete_record(self, recorder, sample_plan, sample_decisions, sample_results):
        run_id = recorder.start_run(session_id="s2", user_message="Full test", operation_mode="offline")
        recorder.record_plan(run_id, sample_plan)
        recorder.record_gate_decisions(run_id, sample_decisions)
        recorder.record_tool_results(run_id, sample_results)
        recorder.finish_run(run_id, success=True, final_response="Complete")

        run = recorder.get_run(run_id)
        assert run.plans[0].goal == "Test goal"
        assert run.plans[0].steps[0].tool == "read_file"
        assert run.user_message == "Full test"

    def test_list_runs_by_session(self, recorder):
        recorder.start_run(session_id="s1", user_message="A", operation_mode="offline")
        recorder.start_run(session_id="s2", user_message="B", operation_mode="offline")
        recorder.start_run(session_id="s1", user_message="C", operation_mode="offline")

        runs = recorder.list_runs(session_id="s1")
        assert len(runs) == 2
        for r in runs:
            assert r.session_id == "s1"

    def test_list_runs_limit(self, recorder):
        for i in range(5):
            recorder.start_run(session_id="s1", user_message=f"Msg {i}", operation_mode="offline")

        runs = recorder.list_runs(limit=3)
        assert len(runs) == 3

    def test_persistence_across_close(self, tmp_path):
        db = str(tmp_path / "persist_runs.db")
        rec1 = RunRecorder(db)
        run_id = rec1.start_run(session_id="s1", user_message="Persist", operation_mode="offline")
        rec1.finish_run(run_id, success=True, final_response="Saved")
        rec1.close()

        rec2 = RunRecorder(db)
        run = rec2.get_run(run_id)
        rec2.close()

        assert run is not None
        assert run.session_id == "s1"
        assert run.success is True
