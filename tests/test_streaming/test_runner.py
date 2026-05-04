"""Tests for `cognithor.streaming.runner` and the `agent run` CLI."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.streaming import EventEmitter
from cognithor.streaming.runner import (
    GateOutcome,
    parse_plan_file,
    run_plan,
)
from cognithor.streaming.sinks import JsonlSink


def _write_plan(tmp_path: Path, **overrides: Any) -> Path:
    plan = {
        "run_id": "test-run-1",
        "agent_id": "tester",
        "goal": "smoke",
        "steps": [
            {"tool": "noop", "params": {"k": "v"}, "rationale": "demo"},
            {"tool": "echo", "arguments": {"msg": "hi"}},
        ],
    }
    plan.update(overrides)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# parse_plan_file
# ---------------------------------------------------------------------------


class TestParsePlanFile:
    def test_happy_path(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path)
        plan = parse_plan_file(p)
        assert plan.run_id == "test-run-1"
        assert plan.agent_id == "tester"
        assert len(plan.steps) == 2
        # Both 'params' and 'arguments' map to the canonical 'arguments' key.
        assert plan.steps[0]["arguments"] == {"k": "v"}
        assert plan.steps[1]["arguments"] == {"msg": "hi"}

    def test_missing_run_id(self, tmp_path: Path) -> None:
        p = tmp_path / "p.json"
        p.write_text(json.dumps({"steps": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="run_id"):
            parse_plan_file(p)

    def test_steps_must_be_list(self, tmp_path: Path) -> None:
        p = tmp_path / "p.json"
        p.write_text(json.dumps({"run_id": "r", "steps": "nope"}), encoding="utf-8")
        with pytest.raises(ValueError, match="steps"):
            parse_plan_file(p)

    def test_step_tool_required(self, tmp_path: Path) -> None:
        p = tmp_path / "p.json"
        p.write_text(
            json.dumps({"run_id": "r", "steps": [{"params": {}}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="tool"):
            parse_plan_file(p)


# ---------------------------------------------------------------------------
# run_plan — happy path with mock execution
# ---------------------------------------------------------------------------


class TestRunPlanMock:
    @pytest.mark.asyncio
    async def test_emits_full_lifecycle(self, tmp_path: Path) -> None:
        plan = parse_plan_file(_write_plan(tmp_path))
        buf = io.StringIO()
        emitter = EventEmitter()
        emitter.add_sink(JsonlSink(stream=buf))
        await emitter.start()
        try:
            exit_code = await run_plan(plan, emitter)
        finally:
            await emitter.stop()

        assert exit_code == 0
        events = [json.loads(line) for line in buf.getvalue().splitlines() if line]
        types = [e["event"] for e in events]
        # 1 run_started + N×(plan_step + gate_decision + tool_result) + 1 run_complete
        assert types[0] == "run_started"
        assert types[-1] == "run_complete"
        assert types.count("plan_step") == 2
        assert types.count("gate_decision") == 2
        assert types.count("tool_result") == 2
        # Mock execution → all tool_results are ok
        for evt in events:
            if evt["event"] == "tool_result":
                assert evt["ok"] is True


# ---------------------------------------------------------------------------
# run_plan — Gatekeeper block path
# ---------------------------------------------------------------------------


class TestRunPlanGateBlock:
    @pytest.mark.asyncio
    async def test_red_block_emits_run_error(self, tmp_path: Path) -> None:
        plan = parse_plan_file(_write_plan(tmp_path))
        buf = io.StringIO()
        emitter = EventEmitter()
        emitter.add_sink(JsonlSink(stream=buf))
        await emitter.start()
        try:
            exit_code = await run_plan(
                plan,
                emitter,
                evaluate_gate=lambda _t, _p: GateOutcome(
                    status="red",
                    rule_id="exec.shell.block",
                    rule_source="cognithor.core.gatekeeper",
                    matched_pattern="rm -rf /",
                    reason="destructive shell pattern",
                ),
            )
        finally:
            await emitter.stop()

        assert exit_code == 1
        events = [json.loads(line) for line in buf.getvalue().splitlines() if line]
        types = [e["event"] for e in events]
        # Must terminate with run_error, NOT run_complete.
        assert types[-1] == "run_error"
        assert "run_complete" not in types
        # The gate_decision before run_error must carry the explanation.
        gate_evt = next(e for e in events if e["event"] == "gate_decision")
        assert gate_evt["status"] == "red"
        assert gate_evt["explanation"]["rule_id"] == "exec.shell.block"
        # FailureMode is policy_block per TRUST-3 contract.
        run_error = events[-1]
        assert run_error["failure_mode"] == "policy_block"


# ---------------------------------------------------------------------------
# run_plan — real execute_tool callable
# ---------------------------------------------------------------------------


class TestRunPlanRealExecute:
    @pytest.mark.asyncio
    async def test_execute_tool_callable_drives_tool_result(
        self,
        tmp_path: Path,
    ) -> None:
        plan = parse_plan_file(_write_plan(tmp_path))
        buf = io.StringIO()
        emitter = EventEmitter()
        emitter.add_sink(JsonlSink(stream=buf))

        async def execute(tool: str, params: dict[str, Any]) -> dict[str, Any]:
            return {
                "ok": True,
                "preview": f"ran {tool} with {params}",
                "chunks": 3,
            }

        await emitter.start()
        try:
            exit_code = await run_plan(plan, emitter, execute_tool=execute)
        finally:
            await emitter.stop()

        assert exit_code == 0
        events = [json.loads(line) for line in buf.getvalue().splitlines() if line]
        tool_evts = [e for e in events if e["event"] == "tool_result"]
        assert len(tool_evts) == 2
        for evt in tool_evts:
            assert evt["ok"] is True
            assert evt["chunks"] == 3
            assert "ran" in evt["preview"]

    @pytest.mark.asyncio
    async def test_execute_tool_raise_emits_run_error(
        self,
        tmp_path: Path,
    ) -> None:
        plan = parse_plan_file(_write_plan(tmp_path))
        buf = io.StringIO()
        emitter = EventEmitter()
        emitter.add_sink(JsonlSink(stream=buf))

        async def execute(_tool: str, _params: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("network down")

        await emitter.start()
        try:
            exit_code = await run_plan(plan, emitter, execute_tool=execute)
        finally:
            await emitter.stop()

        assert exit_code == 1
        events = [json.loads(line) for line in buf.getvalue().splitlines() if line]
        run_error = events[-1]
        assert run_error["event"] == "run_error"
        assert run_error["failure_mode"] == "internal_error"
        assert "RuntimeError" in run_error["error"]
