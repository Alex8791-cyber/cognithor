"""AgentRunner — execute a serialised :class:`ActionPlan` and stream events.

The thin connector between Sprint-27's `cognithor agent run`
subcommand and the existing Cognithor execution surface
(:class:`Gatekeeper`, MCP dispatch). Loads a JSON plan file,
walks the steps, and for each:

1. emits a ``plan_step`` event,
2. asks the :class:`Gatekeeper` for a decision (when wired) and
   emits the corresponding ``gate_decision`` event,
3. executes the tool (real dispatch when ``execute_tool`` is
   provided; otherwise a deterministic mock that returns ``ok=true``
   for the JSONL-stream contract demo) and emits ``tool_result``,
4. on terminal exit, emits ``run_complete`` / ``run_error`` /
   ``run_cancelled`` with an optional TRUST-1 receipt bundle.

This module is intentionally NOT the place to plumb the full PGE
loop — Gateway integration happens in Sprint-27 PR-K's end-to-end
smoke. PR-B's contract is "given a plan file, produce a faithful
event stream"; the streaming primitives are what the extension
binds against.

Plan-file shape (JSON, all fields top-level):

.. code-block:: json

    {
      "run_id": "trace-2026-05-04-1234",
      "agent_id": "planner-default",
      "goal": "fetch the README",
      "steps": [
        {"tool": "read_file", "params": {"path": "README.md"}}
      ]
    }

``run_id`` is required and is reused as the ``trace_id`` for the
TRUST-1 receipt fetch (when wired).
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cognithor.streaming.events import (
    DecisionExplanation as StreamDecisionExplanation,
)
from cognithor.streaming.events import (
    GateDecision,
    PlanStep,
    RunCancelled,
    RunComplete,
    RunError,
    RunStarted,
    ToolResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.streaming.emitter import EventEmitter

# Type alias for the optional real-dispatch callable. The extension
# in PR-K (or external orchestrators) supply it; PR-B's tests rely
# on the mock fallback so they don't need an MCP environment.
ExecuteTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

# Type alias for the optional Gatekeeper-evaluate callable. Kept
# loosely typed so callers can pass any function with the right
# call shape — the runner doesn't import Gatekeeper directly to
# keep PR-B isolated.
EvaluateGate = Callable[
    [str, dict[str, Any]],  # tool, params
    "GateOutcome",
]


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Provider-agnostic Gatekeeper decision shape consumed by the runner."""

    status: str  # one of green / yellow / orange_approved / orange_blocked / red
    rule_id: str | None = None
    rule_source: str | None = None
    matched_pattern: str | None = None
    reason: str | None = None

    @property
    def is_block(self) -> bool:
        return self.status in ("red", "orange_blocked")


@dataclass(frozen=True, slots=True)
class _ParsedPlan:
    run_id: str
    agent_id: str | None
    plan_path: str
    steps: list[dict[str, Any]]


def parse_plan_file(path: Path) -> _ParsedPlan:
    """Load + minimally validate a plan-file. Raises ``ValueError`` on bad shape.

    The runner accepts a deliberately small slice of the
    :class:`cognithor.models.ActionPlan` Pydantic model — enough
    to drive the event stream without forcing every external
    orchestrator to pull in the full Pydantic schema.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "plan-file must decode to a JSON object"
        raise ValueError(msg)

    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        msg = "plan-file: 'run_id' must be a non-empty string"
        raise ValueError(msg)

    agent_id_raw = raw.get("agent_id")
    agent_id: str | None = agent_id_raw if isinstance(agent_id_raw, str) and agent_id_raw else None

    steps = raw.get("steps")
    if not isinstance(steps, list):
        msg = "plan-file: 'steps' must be a list"
        raise ValueError(msg)

    normalised: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            msg = f"plan-file: step[{i}] must be a JSON object"
            raise ValueError(msg)
        tool = step.get("tool")
        if not isinstance(tool, str) or not tool:
            msg = f"plan-file: step[{i}].tool must be a non-empty string"
            raise ValueError(msg)
        params = step.get("params") or step.get("arguments") or {}
        if not isinstance(params, dict):
            msg = f"plan-file: step[{i}].params must be an object"
            raise ValueError(msg)
        rationale = step.get("rationale")
        normalised.append(
            {
                "tool": tool,
                "arguments": dict(params),
                "rationale": rationale if isinstance(rationale, str) else None,
            },
        )

    return _ParsedPlan(
        run_id=run_id,
        agent_id=agent_id,
        plan_path=str(path),
        steps=normalised,
    )


async def run_plan(
    plan: _ParsedPlan,
    emitter: EventEmitter,
    *,
    evaluate_gate: EvaluateGate | None = None,
    execute_tool: ExecuteTool | None = None,
    receipt_builder: Callable[[str], dict[str, Any]] | None = None,
) -> int:
    """Walk the plan, emit events, return process exit-code.

    ``0`` on success, ``1`` on run_error, ``130`` on user-cancel.
    The runner deliberately catches ``KeyboardInterrupt`` so the
    extension reading a subprocess stream sees ``run_cancelled``
    rather than a torn-off pipe.
    """

    started_at = time.monotonic()

    emitter.emit(
        RunStarted(
            run_id=plan.run_id,
            plan_path=plan.plan_path,
            step_count=len(plan.steps),
            agent_id=plan.agent_id,
        ),
    )

    last_step_index: int | None = None
    try:
        for step_index, step in enumerate(plan.steps):
            last_step_index = step_index
            emitter.emit(
                PlanStep(
                    run_id=plan.run_id,
                    step=step_index,
                    action={k: v for k, v in step.items() if v is not None},
                ),
            )

            outcome = (
                evaluate_gate(step["tool"], step["arguments"])
                if evaluate_gate is not None
                else GateOutcome(status="green")
            )
            explanation: StreamDecisionExplanation | None = None
            if outcome.rule_id is not None and outcome.rule_source is not None:
                explanation = StreamDecisionExplanation(
                    rule_id=outcome.rule_id,
                    rule_source=outcome.rule_source,
                    matched_pattern=outcome.matched_pattern,
                )
            emitter.emit(
                GateDecision(
                    run_id=plan.run_id,
                    step=step_index,
                    status=outcome.status,  # type: ignore[arg-type]
                    explanation=explanation,
                ),
            )
            if outcome.is_block:
                # Blocked tools terminate the run with run_error so
                # the extension can render the failure clearly.
                emitter.emit(
                    RunError(
                        run_id=plan.run_id,
                        failure_mode="policy_block",
                        error=outcome.reason or "blocked by Gatekeeper",
                        step=step_index,
                        receipt=(
                            receipt_builder(plan.run_id) if receipt_builder is not None else None
                        ),
                    ),
                )
                return 1

            tool_started = time.monotonic()
            try:
                if execute_tool is not None:
                    result = await execute_tool(step["tool"], step["arguments"])
                    ok = bool(result.get("ok", True))
                    error = result.get("error")
                    chunks_raw = result.get("chunks")
                    chunks = chunks_raw if isinstance(chunks_raw, int) and chunks_raw >= 0 else None
                    preview_raw = result.get("preview")
                    preview = preview_raw if isinstance(preview_raw, str) else None
                else:
                    # Mock execution — preserves the JSONL stream
                    # contract for tests + extension demos without
                    # requiring an MCP environment.
                    ok = True
                    error = None
                    chunks = None
                    preview = None
            except Exception as exc:
                emitter.emit(
                    RunError(
                        run_id=plan.run_id,
                        failure_mode="internal_error",
                        error=f"{type(exc).__name__}: {exc}",
                        step=step_index,
                        receipt=(
                            receipt_builder(plan.run_id) if receipt_builder is not None else None
                        ),
                    ),
                )
                return 1

            emitter.emit(
                ToolResult(
                    run_id=plan.run_id,
                    step=step_index,
                    ok=ok,
                    duration_ms=int((time.monotonic() - tool_started) * 1000),
                    chunks=chunks,
                    preview=preview,
                    error=str(error) if error else None,
                ),
            )

        emitter.emit(
            RunComplete(
                run_id=plan.run_id,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                receipt=(receipt_builder(plan.run_id) if receipt_builder is not None else {}),
            ),
        )
        return 0

    except KeyboardInterrupt:
        emitter.emit(
            RunCancelled(
                run_id=plan.run_id,
                reason="user-abort",
                step=last_step_index,
                receipt=(receipt_builder(plan.run_id) if receipt_builder is not None else None),
            ),
        )
        return 130
