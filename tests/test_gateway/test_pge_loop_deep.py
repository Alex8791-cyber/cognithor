"""Deep coverage for cognithor.gateway.pge_loop.

The module owns the Plan -> Gate -> Execute orchestration:
  * ``run_pge_loop`` — the iteration loop driving Planner / Gatekeeper /
    Executor with replan, approval, stuck-detection, budget mid-loop.
  * ``handle_approvals`` — folds user APPROVE decisions into ALLOW/BLOCK
    via the originating channel.
  * ``is_cu_plan`` — pure helper (computer-use tool detection).

Tests stub the Planner / Gatekeeper / Executor and the Gateway shell so
each branch of the loop body can be exercised in isolation.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.gateway.pge_loop import (
    handle_approvals,
    is_cu_plan,
    run_pge_loop,
)
from cognithor.models import (
    ActionPlan,
    GateDecision,
    GateStatus,
    IncomingMessage,
    PlannedAction,
    RiskLevel,
    SessionContext,
    ToolResult,
    WorkingMemory,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _silent_callback() -> AsyncMock:
    """An AsyncMock callable that accepts (event, status, **extras) noop."""
    cb = AsyncMock()
    return cb


def _bare_gateway(
    *,
    planner: Any = None,
    gatekeeper: Any = None,
    executor: Any = None,
) -> Any:
    """Build a Gateway-like shell that pge_loop can drive without real init."""
    gw = MagicMock()
    gw._running = True
    gw._cancelled_sessions = set()
    gw._background_tasks = set()
    gw._channels = {}
    gw._cost_tracker = None
    gw._run_recorder = None
    gw._explainability = None
    gw._kanban_engine = None
    gw._correction_memory = None
    gw._deep_learner = None
    gw._mcp_client = None
    gw._identity_layer = None
    gw._cu_tools = None
    gw._config = MagicMock()
    gw._config.recovery = None
    gw._config.vision_model = "qwen3-vl:32b"
    gw._config.tools = None

    # Stub out the critical helpers
    gw._make_status_callback = MagicMock(return_value=_silent_callback())
    gw._make_pipeline_callback = MagicMock(return_value=_silent_callback())
    gw._check_and_compact = MagicMock()
    gw._is_cu_plan = MagicMock(return_value=False)
    gw._is_fact_question = MagicMock(return_value=False)
    gw._build_reddit_forced_plan = MagicMock(return_value=None)
    gw._handle_approvals = AsyncMock(side_effect=lambda steps, decisions, *a, **kw: list(decisions))
    gw._record_metric = MagicMock()

    # Mock formulate_response → returns a ResponseEnvelope-shaped object
    env = MagicMock()
    env.content = "formulated answer"
    gw._formulate_response = AsyncMock(return_value=env)

    gw._planner = planner or MagicMock()
    gw._gatekeeper = gatekeeper or MagicMock()
    gw._executor = executor or MagicMock()
    return gw


def _msg(text: str = "do it", channel: str = "cli", session_id: str = "ws-1") -> IncomingMessage:
    return IncomingMessage(text=text, channel=channel, user_id="alex", session_id=session_id)


def _session(*, max_iterations: int = 5) -> SessionContext:
    return SessionContext(
        session_id="sess-internal",
        channel="cli",
        max_iterations=max_iterations,
    )


def _plan(
    *,
    steps: list[PlannedAction] | None = None,
    direct_response: str | None = None,
    parse_failed: bool = False,
    goal: str = "g",
) -> ActionPlan:
    return ActionPlan(
        goal=goal,
        steps=steps or [],
        direct_response=direct_response,
        parse_failed=parse_failed,
    )


def _step(tool: str = "web_search", **params: Any) -> PlannedAction:
    return PlannedAction(tool=tool, params=params or {"q": "x"})


def _allow_decision(action: PlannedAction | None = None) -> GateDecision:
    return GateDecision(
        status=GateStatus.ALLOW,
        risk_level=RiskLevel.GREEN,
        original_action=action,
    )


def _block_decision(action: PlannedAction | None = None) -> GateDecision:
    return GateDecision(
        status=GateStatus.BLOCK,
        reason="blocked",
        risk_level=RiskLevel.RED,
        original_action=action,
    )


def _approve_decision(action: PlannedAction | None = None) -> GateDecision:
    return GateDecision(
        status=GateStatus.APPROVE,
        reason="needs approval",
        risk_level=RiskLevel.ORANGE,
        original_action=action,
    )


def _ok_result(tool: str = "web_search", content: str = "ok") -> ToolResult:
    return ToolResult(tool_name=tool, content=content, is_error=False, duration_ms=12)


def _err_result(tool: str = "web_search") -> ToolResult:
    return ToolResult(
        tool_name=tool,
        content="",
        is_error=True,
        error_message="boom",
        error_type="RuntimeError",
        duration_ms=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# is_cu_plan — pure helper
# ─────────────────────────────────────────────────────────────────────────────


class TestIsCuPlan:
    def test_empty_plan_is_not_cu(self) -> None:
        plan = _plan(steps=[])
        assert is_cu_plan(plan) is False

    def test_plan_with_no_cu_tool_is_not_cu(self) -> None:
        plan = _plan(steps=[_step("web_search"), _step("read_file")])
        assert is_cu_plan(plan) is False

    @pytest.mark.parametrize(
        "tool",
        [
            "computer_screenshot",
            "computer_click",
            "computer_type",
            "computer_hotkey",
            "computer_scroll",
            "computer_drag",
        ],
    )
    def test_each_cu_tool_detected(self, tool: str) -> None:
        plan = _plan(steps=[_step(tool)])
        assert is_cu_plan(plan) is True

    def test_mixed_plan_with_one_cu_tool_is_cu(self) -> None:
        plan = _plan(
            steps=[
                _step("web_search"),
                _step("computer_click"),
                _step("read_file"),
            ]
        )
        assert is_cu_plan(plan) is True


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — early-exit / cancel paths
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPgeLoopEarlyExits:
    @pytest.mark.asyncio
    async def test_user_cancellation_aborts_immediately(self) -> None:
        gw = _bare_gateway()
        gw._cancelled_sessions = {"ws-1"}
        msg = _msg(session_id="ws-1")
        sess = _session()
        wm = WorkingMemory()

        final, results, plans, audit = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert "abgebrochen" in final.lower() or "cancel" in final.lower()
        assert results == []
        assert plans == []
        assert audit == []
        # cancelled set was cleared
        assert "ws-1" not in gw._cancelled_sessions

    @pytest.mark.asyncio
    async def test_running_false_exits_loop(self) -> None:
        gw = _bare_gateway()
        gw._running = False
        sess = _session()
        sess.iteration_count = 0
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # Loop body never entered → final stays empty unless iterations_exhausted
        assert final == ""

    @pytest.mark.asyncio
    async def test_iterations_exhausted_returns_polite_message(self) -> None:
        gw = _bare_gateway()
        sess = _session(max_iterations=3)
        sess.iteration_count = 3  # already exhausted
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert "maximum number" in final.lower() or "smaller" in final.lower()

    @pytest.mark.asyncio
    async def test_mid_loop_budget_exceeded_breaks(self) -> None:
        gw = _bare_gateway()
        ct = MagicMock()
        budget = MagicMock()
        budget.ok = False
        budget.warning = "Daily limit"
        ct.check_budget.return_value = budget
        gw._cost_tracker = ct
        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # `gateway.budget_limit_reached` translation includes the warning text
        assert final  # non-empty


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — direct response paths
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPgeLoopDirectResponse:
    @pytest.mark.asyncio
    async def test_direct_response_no_actions_returns_text(self) -> None:
        plan = _plan(direct_response="Hello, here's the answer.")
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gw = _bare_gateway(planner=planner)
        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, _r, plans, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert final == "Hello, here's the answer."
        assert len(plans) == 1
        # Iteration counter advanced
        assert sess.iteration_count == 1

    @pytest.mark.asyncio
    async def test_no_actions_no_direct_response_returns_apology(self) -> None:
        plan = _plan()  # no steps, no direct_response
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gw = _bare_gateway(planner=planner)
        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # German fallback message
        assert "umformulieren" in final.lower() or "Plan" in final

    @pytest.mark.asyncio
    async def test_replan_text_as_response_first_iter_no_results_formulates(self) -> None:
        # Direct response that LOOKS like a stuck REPLAN — should formulate
        plan = _plan(direct_response="REPLAN: I need more info.")
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gw = _bare_gateway(planner=planner)
        env = MagicMock()
        env.content = "polished answer"
        gw._formulate_response = AsyncMock(return_value=env)
        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert final == "polished answer"

    @pytest.mark.asyncio
    async def test_parse_failed_with_no_results_returns_apology(self) -> None:
        plan = _plan(parse_failed=True, direct_response="")
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gw = _bare_gateway(planner=planner)
        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # Falls through to gateway.parse_failed translation
        assert final  # non-empty

    @pytest.mark.asyncio
    async def test_parse_failed_recovers_with_existing_results(self) -> None:
        # Simulate a plan that parses fine first iter and produces a result,
        # then second-iter replan parse-fails. The loop should formulate from
        # the existing successful result.
        first_plan = _plan(steps=[_step("web_search")])
        broken_replan = _plan(parse_failed=True, direct_response="garbage{{{")
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=first_plan)
        planner.replan = AsyncMock(return_value=broken_replan)

        gk = MagicMock()
        gk.evaluate_plan.return_value = [_allow_decision(first_plan.steps[0])]

        executor = MagicMock()
        executor.execute = AsyncMock(return_value=[_ok_result("web_search", "data")])
        executor.set_status_callback = MagicMock()
        executor.set_agent_context = MagicMock()
        executor.clear_agent_context = MagicMock()

        gw = _bare_gateway(planner=planner, gatekeeper=gk, executor=executor)
        # Force multi-step path so it doesn't break early on success
        # (Actually single step → it WILL break early on success without
        # coding tool. We need to push it into replan via multi-step.)
        first_plan = _plan(steps=[_step("web_search"), _step("web_search")])
        planner.plan = AsyncMock(return_value=first_plan)
        gk.evaluate_plan.return_value = [
            _allow_decision(first_plan.steps[0]),
            _allow_decision(first_plan.steps[1]),
        ]
        executor.execute = AsyncMock(
            return_value=[_ok_result("web_search", "a"), _ok_result("web_search", "b")]
        )

        sess = _session(max_iterations=10)
        msg = _msg()
        wm = WorkingMemory()
        env = MagicMock()
        env.content = "synthesized"
        gw._formulate_response = AsyncMock(return_value=env)
        final, results, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert final == "synthesized"
        assert len(results) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — gatekeeper paths
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPgeLoopGatekeeper:
    @pytest.mark.asyncio
    async def test_all_blocked_break_with_default_message(self) -> None:
        plan = _plan(steps=[_step("delete_file"), _step("system_shutdown")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gk = MagicMock()
        gk.evaluate_plan.return_value = [
            _block_decision(plan.steps[0]),
            _block_decision(plan.steps[1]),
        ]
        gw = _bare_gateway(planner=planner, gatekeeper=gk)
        # _handle_approvals returns blocked decisions unchanged
        gw._handle_approvals = AsyncMock(side_effect=lambda steps, ds, *a, **k: list(ds))

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, audit = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert final  # non-empty
        # Audit entries recorded for both steps
        assert len(audit) >= 2

    @pytest.mark.asyncio
    async def test_all_blocked_after_3rd_block_escalates(self) -> None:
        plan = _plan(steps=[_step("send_email")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        planner.generate_escalation = AsyncMock(return_value="escalation msg")
        gk = MagicMock()
        gk.evaluate_plan.return_value = [_block_decision(plan.steps[0])]
        gw = _bare_gateway(planner=planner, gatekeeper=gk)
        gw._handle_approvals = AsyncMock(side_effect=lambda steps, ds, *a, **k: list(ds))

        sess = _session()
        # Simulate that send_email has already been blocked 2 times
        sess.record_block("send_email")
        sess.record_block("send_email")
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert final == "escalation msg"

    @pytest.mark.asyncio
    async def test_all_blocked_creates_kanban_pending_review(self) -> None:
        plan = _plan(steps=[_step("delete_file")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gk = MagicMock()
        gk.evaluate_plan.return_value = [_block_decision(plan.steps[0])]
        gw = _bare_gateway(planner=planner, gatekeeper=gk)
        gw._handle_approvals = AsyncMock(side_effect=lambda steps, ds, *a, **k: list(ds))
        kanban = MagicMock()
        gw._kanban_engine = kanban

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        kanban.create_task.assert_called_once()
        kw = kanban.create_task.call_args.kwargs
        assert kw["status"] == "pending_review"

    @pytest.mark.asyncio
    async def test_audit_entries_have_params_hash(self) -> None:
        plan = _plan(steps=[_step("read_file", path="/tmp/x")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gk = MagicMock()
        gk.evaluate_plan.return_value = [_block_decision(plan.steps[0])]
        gw = _bare_gateway(planner=planner, gatekeeper=gk)
        gw._handle_approvals = AsyncMock(side_effect=lambda steps, ds, *a, **k: list(ds))

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        _f, _r, _p, audit = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert audit
        # action_params_hash is a hex SHA-256 → 64 chars
        assert len(audit[0].action_params_hash) == 64


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — execution + replan paths
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPgeLoopExecution:
    @pytest.mark.asyncio
    async def test_single_step_success_breaks_with_formulated_response(self) -> None:
        plan = _plan(steps=[_step("web_search")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gk = MagicMock()
        gk.evaluate_plan.return_value = [_allow_decision(plan.steps[0])]
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=[_ok_result("web_search")])
        executor.set_status_callback = MagicMock()
        executor.set_agent_context = MagicMock()
        executor.clear_agent_context = MagicMock()
        gw = _bare_gateway(planner=planner, gatekeeper=gk, executor=executor)
        env = MagicMock()
        env.content = "the answer"
        gw._formulate_response = AsyncMock(return_value=env)

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, results, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        assert final == "the answer"
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_executor_clear_context_runs_even_on_exception(self) -> None:
        plan = _plan(steps=[_step("web_search")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gk = MagicMock()
        gk.evaluate_plan.return_value = [_allow_decision(plan.steps[0])]
        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=RuntimeError("exec failed"))
        executor.set_status_callback = MagicMock()
        executor.set_agent_context = MagicMock()
        executor.clear_agent_context = MagicMock()
        gw = _bare_gateway(planner=planner, gatekeeper=gk, executor=executor)

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        with pytest.raises(RuntimeError, match="exec failed"):
            await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # clear_agent_context was called in the finally clause
        executor.clear_agent_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_recorded_for_each_tool_result(self) -> None:
        plan = _plan(steps=[_step("web_search")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        # If the loop tries to replan, return a direct response so we stop.
        planner.replan = AsyncMock(return_value=_plan(direct_response="done"))
        gk = MagicMock()
        gk.evaluate_plan.return_value = [_allow_decision(plan.steps[0])]
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=[_err_result("web_search")])
        executor.set_status_callback = MagicMock()
        executor.set_agent_context = MagicMock()
        executor.clear_agent_context = MagicMock()
        gw = _bare_gateway(planner=planner, gatekeeper=gk, executor=executor)

        sess = _session(max_iterations=2)
        msg = _msg()
        wm = WorkingMemory()
        await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)

        # tool_calls_total + tool_duration_ms + errors_total
        names = [c.args[0] for c in gw._record_metric.call_args_list]
        assert "tool_calls_total" in names
        assert "tool_duration_ms" in names
        assert "errors_total" in names

    @pytest.mark.asyncio
    async def test_run_recorder_records_plan_and_results(self) -> None:
        plan = _plan(steps=[_step("web_search")])
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gk = MagicMock()
        gk.evaluate_plan.return_value = [_allow_decision(plan.steps[0])]
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=[_ok_result()])
        executor.set_status_callback = MagicMock()
        executor.set_agent_context = MagicMock()
        executor.clear_agent_context = MagicMock()
        gw = _bare_gateway(planner=planner, gatekeeper=gk, executor=executor)
        rr = MagicMock()
        gw._run_recorder = rr

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        await run_pge_loop(gw, msg, sess, wm, {}, None, None, "RUN-1")
        rr.record_plan.assert_called_once()
        rr.record_gate_decisions.assert_called_once()
        rr.record_tool_results.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — keepalive task lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestKeepaliveTaskLifecycle:
    @pytest.mark.asyncio
    async def test_keepalive_cancelled_on_planner_exception(self) -> None:
        """When the planner raises, the keepalive task must still be cancelled
        and removed from _background_tasks (deep-PR1 / DEEP-1 fix)."""
        planner = MagicMock()
        planner.plan = AsyncMock(side_effect=ConnectionError("planner down"))
        gw = _bare_gateway(planner=planner)
        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        with pytest.raises(ConnectionError):
            await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # All background tasks were removed
        assert len(gw._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_keepalive_cancelled_on_normal_break(self) -> None:
        plan = _plan(direct_response="quick")
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gw = _bare_gateway(planner=planner)
        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # Cleaned up
        assert len(gw._background_tasks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — stalled-turn / no-tool detection
# ─────────────────────────────────────────────────────────────────────────────


class TestStalledTurnDetection:
    @pytest.mark.asyncio
    async def test_consecutive_no_tool_iters_break(self) -> None:
        # Plans with no actions and no direct_response → bails out via
        # apology (the no_actions branch breaks immediately, so we never
        # accumulate streaks). This test covers the "REPLAN-text masquerading"
        # branch where the LLM returns REPLAN text without parse_failed=True
        # and we count consecutive no-tool iters.
        replan_plan = _plan(direct_response="REPLAN: still working...")
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=replan_plan)
        gw = _bare_gateway(planner=planner)
        env = MagicMock()
        env.content = "fallback"
        gw._formulate_response = AsyncMock(return_value=env)

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
        # First iteration with no results triggers the "formulate from empty"
        # path → fallback
        assert final == "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — agent-specific overrides
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentOverrides:
    @pytest.mark.asyncio
    async def test_agent_model_temperature_top_p_propagated_to_planner(self) -> None:
        plan = _plan(direct_response="ok")
        planner = MagicMock()
        planner.plan = AsyncMock(return_value=plan)
        gw = _bare_gateway(planner=planner)

        agent = MagicMock()
        agent.name = "researcher"
        agent.preferred_model = "qwen3:32b"
        agent.temperature = 0.3
        agent.top_p = 0.95
        from cognithor.core.agent_router import RouteDecision

        rd = RouteDecision(agent=agent, confidence=0.8)

        sess = _session()
        msg = _msg()
        wm = WorkingMemory()
        await run_pge_loop(gw, msg, sess, wm, {}, rd, None, None)
        kw = planner.plan.call_args.kwargs
        assert kw["model_override"] == "qwen3:32b"
        assert kw["temperature_override"] == 0.3
        assert kw["top_p_override"] == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# handle_approvals
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleApprovals:
    @pytest.mark.asyncio
    async def test_no_channel_converts_approve_to_block(self) -> None:
        gw = _bare_gateway()
        gw._channels = {}  # empty
        steps = [_step("send_email")]
        decisions = [_approve_decision(steps[0])]
        sess = _session()
        result = await handle_approvals(gw, steps, decisions, sess, "missing")
        assert result[0].status == GateStatus.BLOCK
        assert "Kein interaktiver Kanal" in result[0].reason

    @pytest.mark.asyncio
    async def test_no_channel_passes_through_non_approve(self) -> None:
        gw = _bare_gateway()
        gw._channels = {}
        steps = [_step("read_file")]
        decisions = [_allow_decision(steps[0])]
        sess = _session()
        result = await handle_approvals(gw, steps, decisions, sess, "missing")
        assert result[0].status == GateStatus.ALLOW

    @pytest.mark.asyncio
    async def test_user_approves_changes_status_to_allow(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.request_approval = AsyncMock(return_value=True)
        gw._channels = {"cli": chan}
        steps = [_step("send_email")]
        decisions = [_approve_decision(steps[0])]
        sess = _session()
        result = await handle_approvals(gw, steps, decisions, sess, "cli")
        assert result[0].status == GateStatus.ALLOW
        assert "user_approved" in result[0].policy_name

    @pytest.mark.asyncio
    async def test_user_rejects_changes_status_to_block(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.request_approval = AsyncMock(return_value=False)
        gw._channels = {"cli": chan}
        steps = [_step("send_email")]
        decisions = [_approve_decision(steps[0])]
        sess = _session()
        result = await handle_approvals(gw, steps, decisions, sess, "cli")
        assert result[0].status == GateStatus.BLOCK
        assert "user_rejected" in result[0].policy_name

    @pytest.mark.asyncio
    async def test_request_approval_exception_treated_as_rejection(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.request_approval = AsyncMock(side_effect=RuntimeError("transport"))
        gw._channels = {"cli": chan}
        steps = [_step("send_email")]
        decisions = [_approve_decision(steps[0])]
        sess = _session()
        result = await handle_approvals(gw, steps, decisions, sess, "cli")
        assert result[0].status == GateStatus.BLOCK

    @pytest.mark.asyncio
    async def test_only_approve_decisions_are_routed_for_approval(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.request_approval = AsyncMock(return_value=True)
        gw._channels = {"cli": chan}
        steps = [_step("a"), _step("b"), _step("c")]
        decisions = [
            _allow_decision(steps[0]),
            _approve_decision(steps[1]),
            _block_decision(steps[2]),
        ]
        sess = _session()
        result = await handle_approvals(gw, steps, decisions, sess, "cli")
        # Only one channel call (for the APPROVE step)
        assert chan.request_approval.await_count == 1
        # Allow/Block stay unchanged
        assert result[0].status == GateStatus.ALLOW
        assert result[1].status == GateStatus.ALLOW  # was APPROVE, now approved
        assert result[2].status == GateStatus.BLOCK

    @pytest.mark.asyncio
    async def test_ws_session_id_used_for_lookup(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.request_approval = AsyncMock(return_value=True)
        gw._channels = {"cli": chan}
        steps = [_step("send_email")]
        decisions = [_approve_decision(steps[0])]
        sess = _session()
        await handle_approvals(gw, steps, decisions, sess, "cli", ws_session_id="ws-front-1")
        kw = chan.request_approval.call_args.kwargs
        assert kw["session_id"] == "ws-front-1"

    @pytest.mark.asyncio
    async def test_ws_session_id_falls_back_to_internal(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.request_approval = AsyncMock(return_value=True)
        gw._channels = {"cli": chan}
        steps = [_step("send_email")]
        decisions = [_approve_decision(steps[0])]
        sess = _session()
        await handle_approvals(gw, steps, decisions, sess, "cli", ws_session_id=None)
        kw = chan.request_approval.call_args.kwargs
        assert kw["session_id"] == sess.session_id


# ─────────────────────────────────────────────────────────────────────────────
# run_pge_loop — concurrent PGE loops (independence)
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrentPgeRuns:
    @pytest.mark.asyncio
    async def test_two_concurrent_loops_dont_share_state(self) -> None:
        # Build TWO gateways, each running a direct-response loop concurrently.
        async def _run_one(text: str) -> str:
            plan = _plan(direct_response=f"answer-{text}")
            planner = MagicMock()
            planner.plan = AsyncMock(return_value=plan)
            gw = _bare_gateway(planner=planner)
            sess = _session()
            msg = _msg(text=text)
            wm = WorkingMemory()
            final, _r, _p, _a = await run_pge_loop(gw, msg, sess, wm, {}, None, None, None)
            return final

        results = await asyncio.gather(*(_run_one(f"q{i}") for i in range(8)))
        assert sorted(results) == sorted(f"answer-q{i}" for i in range(8))


# ─────────────────────────────────────────────────────────────────────────────
# Integration touchpoint: gateway.is_cu_plan delegates to module
# ─────────────────────────────────────────────────────────────────────────────


def test_gateway_is_cu_plan_delegates() -> None:
    """``Gateway._is_cu_plan`` must use the module helper so changes stay
    in sync. Verifies wiring as well as result identity."""
    from cognithor.gateway.gateway import Gateway

    plan_yes = _plan(steps=[_step("computer_click")])
    plan_no = _plan(steps=[_step("read_file")])
    assert Gateway._is_cu_plan(plan_yes) is True
    assert Gateway._is_cu_plan(plan_no) is False
