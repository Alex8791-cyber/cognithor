"""Tests for `cognithor.streaming.events` — schema round-trip + envelope invariants.

Covers H1 (per-event schema_version), H2 (all 7 lifecycle event
types defined day 1), and H5 (events round-trip through the
machine-readable JSON Schema at
``src/cognithor/streaming/schemas/v1/events.json``).
"""

from __future__ import annotations

import json
import re

import jsonschema
import pytest

from cognithor.streaming.events import (
    CRITICAL_EVENT_TYPES,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    DecisionExplanation,
    GateDecision,
    PlanStep,
    RunCancelled,
    RunComplete,
    RunError,
    RunStarted,
    SinkDropped,
    StreamEvent,
    ToolResult,
)

# --- Schema fixtures ---------------------------------------------------------


@pytest.fixture(scope="module")
def schema() -> dict[str, object]:
    """Load the v1 JSON Schema once per test module."""

    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate(schema: dict[str, object], payload: dict[str, object]) -> None:
    jsonschema.validate(instance=payload, schema=schema)


# --- Envelope invariants -----------------------------------------------------


class TestEnvelope:
    def test_schema_path_exists(self) -> None:
        assert SCHEMA_PATH.exists(), f"events schema missing at {SCHEMA_PATH}"

    def test_schema_version_label(self) -> None:
        assert SCHEMA_VERSION == "v1"

    def test_run_id_required(self) -> None:
        with pytest.raises(ValueError, match="run_id"):
            RunStarted(run_id="", plan_path="plan.json", step_count=3)

    def test_subclass_must_set_event_type(self) -> None:
        # Bare StreamEvent has empty EVENT_TYPE → must raise.
        with pytest.raises(TypeError, match="EVENT_TYPE"):
            StreamEvent(run_id="abc")

    def test_ts_is_iso8601_utc(self) -> None:
        evt = RunStarted(run_id="abc", plan_path="plan.json", step_count=0)
        # Pattern: 2026-05-04T18:30:42.123Z
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
            evt.ts,
        ), evt.ts

    def test_per_event_schema_version_not_envelope(self, schema: dict) -> None:
        """H1 — schema_version is per-event, NOT envelope-level.

        We assert this by checking that no envelope-level
        ``schema_version`` exists in the top-level schema, while
        every event variant has its own ``schema_version`` const.
        """

        assert "schema_version" not in schema.get("properties", {}), (
            "envelope must NOT carry a top-level schema_version (H1)"
        )
        # Each event def must require schema_version
        for name in (
            "run_started",
            "plan_step",
            "gate_decision",
            "tool_result",
            "run_complete",
            "run_error",
            "run_cancelled",
            "sink_dropped",
        ):
            event_schema = schema["$defs"][name]
            sv = event_schema["properties"]["schema_version"]
            assert "const" in sv, f"{name}.schema_version must be const for H1"


# --- H2 — all seven lifecycle event types defined ----------------------------


class TestLifecycleEventsAllDefined:
    """H2 — run_started, run_error, run_cancelled MUST exist day 1.

    If these are missing, any client that switches on a closed set
    of event types breaks the moment they're added later. Adding
    them in v1 prevents that.
    """

    def test_run_started_exists(self) -> None:
        assert RunStarted.EVENT_TYPE == "run_started"

    def test_run_error_exists(self) -> None:
        assert RunError.EVENT_TYPE == "run_error"

    def test_run_cancelled_exists(self) -> None:
        assert RunCancelled.EVENT_TYPE == "run_cancelled"

    def test_critical_set_includes_terminals_and_drop_notice(self) -> None:
        # H4 — critical events bypass the bounded-buffer drop path.
        assert (
            frozenset({"run_complete", "run_error", "run_cancelled", "sink_dropped"})
            == CRITICAL_EVENT_TYPES
        )


# --- Round-trip every event type through the schema -------------------------


class TestSchemaRoundTrip:
    def test_run_started_minimal(self, schema: dict) -> None:
        evt = RunStarted(run_id="r1", plan_path="plan.json", step_count=3)
        _validate(schema, evt.to_dict())

    def test_run_started_with_agent_id(self, schema: dict) -> None:
        evt = RunStarted(
            run_id="r1",
            plan_path="plan.json",
            step_count=3,
            agent_id="planner-1",
        )
        _validate(schema, evt.to_dict())

    def test_plan_step_full(self, schema: dict) -> None:
        evt = PlanStep(
            run_id="r1",
            step=0,
            action={
                "tool": "web_search",
                "arguments": {"query": "TRUST-7"},
                "rationale": "user asked",
            },
        )
        _validate(schema, evt.to_dict())

    def test_gate_decision_block_with_explanation(self, schema: dict) -> None:
        evt = GateDecision(
            run_id="r1",
            step=0,
            status="red",
            explanation=DecisionExplanation(
                rule_id="exec.shell.path-traversal",
                rule_source="cognithor.core.gatekeeper._classify_risk",
                matched_pattern="rm -rf /",
            ),
        )
        _validate(schema, evt.to_dict())

    def test_gate_decision_green_no_explanation(self, schema: dict) -> None:
        evt = GateDecision(run_id="r1", step=1, status="green")
        _validate(schema, evt.to_dict())

    def test_tool_result_ok(self, schema: dict) -> None:
        evt = ToolResult(
            run_id="r1",
            step=1,
            ok=True,
            duration_ms=42,
            chunks=4,
            preview="result preview",
        )
        _validate(schema, evt.to_dict())

    def test_tool_result_error(self, schema: dict) -> None:
        evt = ToolResult(
            run_id="r1",
            step=1,
            ok=False,
            duration_ms=10,
            error="ConnectionRefusedError",
        )
        _validate(schema, evt.to_dict())

    def test_run_complete_with_receipt(self, schema: dict) -> None:
        evt = RunComplete(
            run_id="r1",
            duration_ms=1234,
            receipt={"trace_id": "r1", "trust": {"cost": {"micro_usd": 17}}},
        )
        _validate(schema, evt.to_dict())

    def test_run_error_with_failure_mode(self, schema: dict) -> None:
        evt = RunError(
            run_id="r1",
            failure_mode="policy_block",
            error="GREEN risk floor",
            step=2,
        )
        _validate(schema, evt.to_dict())

    def test_run_cancelled(self, schema: dict) -> None:
        evt = RunCancelled(run_id="r1", reason="user-abort", step=2)
        _validate(schema, evt.to_dict())

    def test_sink_dropped(self, schema: dict) -> None:
        evt = SinkDropped(run_id="r1", sink="websocket", dropped_count=42)
        _validate(schema, evt.to_dict())


# --- Producer-side invariants ------------------------------------------------


class TestProducerInvariants:
    def test_decision_explanation_truncates_matched_pattern(self) -> None:
        long_pattern = "x" * 500
        d = DecisionExplanation(
            rule_id="r",
            rule_source="s",
            matched_pattern=long_pattern,
        )
        out = d.to_dict()
        assert len(out["matched_pattern"]) == 200

    def test_tool_result_truncates_preview_to_500(self, schema: dict) -> None:
        evt = ToolResult(run_id="r", step=0, ok=True, preview="x" * 1000)
        payload = evt.to_dict()
        assert len(payload["preview"]) == 500
        _validate(schema, payload)

    def test_emitted_event_is_dict_not_str(self) -> None:
        evt = PlanStep(run_id="r", step=0, action={"tool": "noop"})
        out = evt.to_dict()
        assert isinstance(out, dict)
        # Must be JSON-serialisable.
        json.dumps(out)
