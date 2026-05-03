"""Cognithor · Gateway PGE-Loop — extracted from `gateway.py`.

The Plan → Gate → Execute orchestration that drives every agent turn:
  * :func:`run_pge_loop` — Phase 3: planner produces an `ActionPlan`,
    gatekeeper risk-classifies each step, executor runs ALLOW/INFORM/MASK
    actions, replan loop with stuck-detection.
  * :func:`handle_approvals` — Holds ORANGE actions for user approval
    via the originating channel and folds the decisions back into the
    risk-classified plan.
  * :func:`is_cu_plan` — pure helper checking whether a plan uses any
    Computer-Use tool.

`MAX_STALLED_MODEL_TURNS` and `advance_stalled_count` stay on
`cognithor.gateway.gateway` (single source of truth — `tests/test_core/
test_stalled_counter.py` imports them directly). This module reads them
through that import.

Part of the staged `gateway.py` split — see
`project_v0960_refactor_backlog.md` and the architect blueprint.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json as _json
import time
from typing import TYPE_CHECKING, Any

from cognithor.gateway.gateway import (
    _TOOL_STATUS_KEYS,
    MAX_STALLED_MODEL_TURNS,
    _sanitize_broken_llm_output,
    advance_stalled_count,
)
from cognithor.i18n import t
from cognithor.models import (
    ActionPlan,
    AuditEntry,
    GateDecision,
    GateStatus,
    Message,
    MessageRole,
    ToolResult,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.core.agent_router import RouteDecision
    from cognithor.gateway.gateway import Gateway
    from cognithor.models import IncomingMessage, SessionContext, WorkingMemory

log = get_logger(__name__)


def is_cu_plan(plan: ActionPlan) -> bool:
    """Check if a plan uses Computer Use tools."""
    _CU_TOOLS = frozenset(
        {
            "computer_screenshot",
            "computer_click",
            "computer_type",
            "computer_hotkey",
            "computer_scroll",
            "computer_drag",
        }
    )
    return plan.has_actions and any(step.tool in _CU_TOOLS for step in plan.steps)


async def run_pge_loop(
    gw: Gateway,
    msg: IncomingMessage,
    session: SessionContext,
    wm: WorkingMemory,
    tool_schemas: dict[str, Any],
    route_decision: RouteDecision | None,
    agent_workspace: Any,
    run_id: str | None,
    stream_callback: Any | None = None,
    active_skill: Any | None = None,
) -> tuple[str, list[ToolResult], list[ActionPlan], list[AuditEntry]]:
    """Phase 3: Plan → Gate → Execute Loop.

    Args:
        stream_callback: Optionaler async Callback fuer Streaming-Events.

    Returns:
        (final_response, all_results, all_plans, all_audit)
    """
    all_results: list[ToolResult] = []
    all_plans: list[ActionPlan] = []
    all_audit: list[AuditEntry] = []
    final_response = ""
    _expl_trail_id: str | None = None  # Explainability not wired into PGE loop yet
    _consecutive_no_tool_iters = 0  # Detect stuck replan loops
    _max_no_tool_iters = 2  # After 2 iters without tool execution, stop

    # Status callback for progress feedback
    # Nutze msg.session_id (Client/WS-ID), nicht session.session_id (intern)
    _status_cb = gw._make_status_callback(msg.channel, msg.session_id)
    # Pipeline callback for live PGE visualization (WebUI only)
    _pipeline_cb = gw._make_pipeline_callback(msg.channel, msg.session_id)

    # Identity Layer reference (set during Phase init)
    _identity = getattr(gw, "_identity_layer", None)

    # Agent-specific LLM overrides (preferred_model, temperature, top_p)
    _agent_model: str | None = None
    _agent_temperature: float | None = None
    _agent_top_p: float | None = None
    if route_decision and route_decision.agent:
        _agent = route_decision.agent
        if _agent.preferred_model:
            _agent_model = _agent.preferred_model
        if _agent.temperature is not None:
            _agent_temperature = _agent.temperature
        if getattr(_agent, "top_p", None) is not None:
            _agent_top_p = _agent.top_p
        if _agent_model or _agent_temperature is not None or _agent_top_p is not None:
            log.info(
                "agent_llm_overrides",
                agent=_agent.name,
                model=_agent_model,
                temperature=_agent_temperature,
                top_p=_agent_top_p,
            )

    # ── Live Correction Detection ─────────────────────────────
    _CORRECTION_TRIGGERS = frozenset(
        {
            "nein",
            "stopp",
            "stop",
            "halt",
            "falsch",
            "nicht so",
            "stattdessen",
            "anders",
            "korrigier",
            "abbrech",
            "cancel",
            "wrong",
            "lass das",
            "vergiss das",
            "mach anders",
        }
    )
    _lower_msg = msg.text.lower().strip()
    _is_correction = any(t in _lower_msg for t in _CORRECTION_TRIGGERS)

    if _is_correction and session.iteration_count > 0:
        log.info("live_correction_detected", text=msg.text[:80])
        if hasattr(gw, "_correction_memory") and gw._correction_memory:
            gw._correction_memory.store(
                user_message=getattr(session, "last_user_message", "") or "",
                correction_text=msg.text,
            )
        # Feed correction into Evolution Engine as learning gap
        if hasattr(gw, "_deep_learner") and gw._deep_learner:
            try:
                last_msg = getattr(session, "last_user_message", "") or msg.text
                gap = f"User-Korrektur: {last_msg[:100]} → {msg.text[:100]}"
                config = getattr(gw, "_config", None)
                if config and hasattr(config, "evolution"):
                    goals = list(getattr(config.evolution, "learning_goals", []) or [])
                    if gap not in goals and len(goals) < 20:
                        goals.append(gap)
                        config.evolution.learning_goals = goals
                        log.info("evolution_gap_from_correction", correction=msg.text[:60])
            except Exception:
                log.debug("evolution_correction_injection_failed", exc_info=True)
        wm.add_message(
            Message(
                role=MessageRole.SYSTEM,
                content=t("gateway.correction_hint", text=msg.text),
                channel=msg.channel,
            )
        )

    while not session.iterations_exhausted and gw._running:
        # Cancel-Check: User hat /stop oder cancel gesendet
        if msg.session_id in gw._cancelled_sessions:
            gw._cancelled_sessions.discard(msg.session_id)
            log.info("pge_cancelled_by_user", session=session.session_id[:8])
            final_response = t("gateway.processing_cancelled")
            break

        # Mid-loop cost budget check — abort if daily/monthly limit exceeded
        if hasattr(gw, "_cost_tracker") and gw._cost_tracker:
            try:
                _budget = gw._cost_tracker.check_budget()
                if not _budget.ok:
                    log.warning("pge_budget_exceeded_mid_loop", session=session.session_id[:8])
                    final_response = t("gateway.budget_limit_reached", warning=_budget.warning)
                    break
            except Exception:
                log.debug("pge_budget_check_failed", exc_info=True)

        session.iteration_count += 1
        await _pipeline_cb("iteration", "start", iteration=session.iteration_count)

        # Check token budget and compact if necessary
        gw._check_and_compact(wm, session)

        log.info(
            "agent_loop_iteration",
            iteration=session.iteration_count,
            session=session.session_id[:8],
            chat_messages=len(wm.chat_history),
            token_estimate=wm.token_count,
        )

        # Status: Thinking (with periodic keepalive for long-running plans)
        await _status_cb("thinking", t("gateway.status_thinking"))
        await _pipeline_cb("plan", "start", iteration=session.iteration_count)

        # Keepalive: send periodic status updates while planner works
        _keepalive_event = asyncio.Event()

        async def _thinking_keepalive(stop: asyncio.Event) -> None:
            """Send periodic status updates so the UI shows activity."""
            _elapsed = 0
            _messages = [
                t("gateway.status_thinking"),
                t("gateway.status_planning"),
                t("gateway.status_analyzing"),
                t("gateway.status_creating_plan"),
                t("gateway.status_working"),
            ]
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=5)
                    break  # stop was set
                except TimeoutError:
                    pass
                _elapsed += 5
                _msg = _messages[min(_elapsed // 10, len(_messages) - 1)]
                with contextlib.suppress(Exception):
                    await _status_cb("thinking", f"{_msg} ({_elapsed}s)")

        _keepalive_task = asyncio.create_task(_thinking_keepalive(_keepalive_event))
        gw._background_tasks.add(_keepalive_task)

        # Identity: enrich context before planning (first iteration only)
        if session.iteration_count == 1 and _identity is not None:
            try:
                _id_ctx = _identity.enrich_context(msg.text)
                _cognitive_text = _id_ctx.get("cognitive_context", "")
                if _cognitive_text:
                    wm.add_message(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=f"[Cognitive Identity]\n{_cognitive_text}",
                            channel=msg.channel,
                        )
                    )
            except Exception:
                log.debug("identity_enrich_failed", exc_info=True)

        # Hard-routing: skills with force_tool bypass the Planner's tool selection
        _force_plan = None
        if (
            active_skill is not None
            and session.iteration_count == 1
            and hasattr(active_skill, "skill")
            and active_skill.skill is not None
            and active_skill.skill.slug == "reddit_lead_hunter"
        ):
            _force_plan = gw._build_reddit_forced_plan(msg.text)
            if _force_plan is not None:
                log.info("reddit_hard_route_applied", goal=_force_plan.goal)

        # Planner
        if _force_plan is not None:
            plan = _force_plan
        elif session.iteration_count == 1:
            plan = await gw._planner.plan(
                user_message=msg.text,
                working_memory=wm,
                tool_schemas=tool_schemas,
                model_override=_agent_model,
                temperature_override=_agent_temperature,
                top_p_override=_agent_top_p,
            )
        else:
            plan = await gw._planner.replan(
                original_goal=msg.text,
                results=all_results,
                working_memory=wm,
                tool_schemas=tool_schemas,
                model_override=_agent_model,
                temperature_override=_agent_temperature,
                top_p_override=_agent_top_p,
            )

        # Stop keepalive once planner responds
        _keepalive_event.set()
        _keepalive_task.cancel()
        with contextlib.suppress(BaseException):
            await _keepalive_task
        gw._background_tasks.discard(_keepalive_task)

        all_plans.append(plan)
        await _pipeline_cb(
            "plan",
            "done",
            iteration=session.iteration_count,
            has_actions=plan.has_actions,
            steps=len(plan.steps) if plan.has_actions else 0,
        )

        # ── Pre-Flight Notification (non-blocking, agentic-first) ──
        _recovery_cfg = getattr(gw._config, "recovery", None)
        if (
            _recovery_cfg
            and getattr(_recovery_cfg, "pre_flight_enabled", False)
            and plan.has_actions
            and len(plan.steps) >= getattr(_recovery_cfg, "pre_flight_min_steps", 2)
        ):
            _timeout = getattr(_recovery_cfg, "pre_flight_timeout_seconds", 3)
            _timeout = min(_timeout, 30)  # Hard upper bound
            _steps_summary = [
                {"tool": s.tool, "rationale": (s.rationale or "")[:80]} for s in plan.steps[:5]
            ]
            await _status_cb(
                "pre_flight",
                _json.dumps(
                    {
                        "goal": plan.goal or msg.text[:100],
                        "steps": _steps_summary,
                        "timeout": _timeout,
                        "session_id": msg.session_id,
                    }
                ),
            )
            _pf_start = time.monotonic()
            _pf_cancelled = False
            while (time.monotonic() - _pf_start) < _timeout:
                if msg.session_id in gw._cancelled_sessions:
                    gw._cancelled_sessions.discard(msg.session_id)
                    _pf_cancelled = True
                    break
                await asyncio.sleep(0.5)
            if _pf_cancelled:
                log.info("pre_flight_cancelled", session=session.session_id[:8])
                final_response = "Plan abgebrochen. Was soll ich stattdessen tun?"
                break
            log.debug("pre_flight_auto_execute", session=session.session_id[:8])

        # Emit plan detail for UI Plan Review panel
        if plan.has_actions:
            _plan_steps = []
            for step in plan.steps:
                _plan_steps.append(
                    {
                        "tool": step.tool,
                        "params": {k: str(v)[:100] for k, v in step.params.items()},
                        "rationale": step.rationale,
                        "risk_estimate": step.risk_estimate.value
                        if hasattr(step.risk_estimate, "value")
                        else str(step.risk_estimate),
                        "depends_on": step.depends_on,
                    }
                )
            channel = gw._channels.get(msg.channel)
            if channel and hasattr(channel, "send_plan_detail"):
                try:
                    await channel.send_plan_detail(
                        msg.session_id,
                        {
                            "iteration": session.iteration_count,
                            "goal": plan.goal,
                            "reasoning": plan.reasoning,
                            "confidence": plan.confidence,
                            "steps": _plan_steps,
                        },
                    )
                except Exception:
                    log.debug("plan_detail_send_failed", exc_info=True)

        if run_id and gw._run_recorder:
            try:
                gw._run_recorder.record_plan(run_id, plan)
            except Exception:
                log.debug("run_recorder_plan_failed", exc_info=True)

        # Computer Use: delegate to CUAgentExecutor for multi-turn interaction
        if gw._is_cu_plan(plan):
            from cognithor.core.cu_agent import CUAgentConfig, CUAgentExecutor

            _vision_model = getattr(gw._config, "vision_model", "qwen3-vl:32b")
            _allowed_tools = getattr(
                getattr(gw._config, "tools", None),
                "computer_use_allowed_tools",
                None,
            )
            cu_agent = CUAgentExecutor(
                planner=gw._planner,
                mcp_client=gw._mcp_client,
                gatekeeper=gw._gatekeeper,
                working_memory=wm,
                tool_schemas=tool_schemas,
                config=CUAgentConfig(
                    max_iterations=30,
                    max_duration_seconds=480,
                    vision_model=_vision_model,
                ),
                allowed_tools=_allowed_tools,
                session_context=session,
                cu_tools=getattr(gw, "_cu_tools", None),
            )
            cu_result = await cu_agent.execute(
                goal=msg.text,
                initial_plan=plan,
                status_callback=_status_cb,
                cancel_check=lambda: msg.session_id in gw._cancelled_sessions,
            )
            all_results.extend(cu_result.tool_results)
            if cu_result.action_history:
                wm.add_message(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            "[Computer Use Ergebnis]\n"
                            + "\n".join(cu_result.action_history[-10:])
                            + f"\n\nAbschluss: {cu_result.abort_reason}"
                            + (
                                f"\nZusammenfassung: {cu_result.task_summary}"
                                if cu_result.task_summary
                                else ""
                            )
                            + (
                                f"\nErstellte Dateien: {', '.join(cu_result.output_files)}"
                                if cu_result.output_files
                                else ""
                            )
                            + (
                                f"\nExtrahierter Text:\n{cu_result.extracted_content[:2000]}"
                                if cu_result.extracted_content
                                else ""
                            )
                        ),
                        channel=msg.channel,
                    )
                )
            await _status_cb("finishing", "Formuliere Antwort...")
            _envelope = await gw._formulate_response(
                msg.text,
                all_results,
                wm,
                stream_callback,
            )
            final_response = _envelope.content
            break

        # JSON parse failed even after retry — recover gracefully
        if getattr(plan, "parse_failed", False):
            log.warning(
                "pge_plan_parse_failed",
                iteration=session.iteration_count,
                confidence=plan.confidence,
                preview=(plan.direct_response or "")[:200],
            )
            # Recovery: if successful tool results already exist,
            # formulate a clean response from them (instead of giving up)
            if all_results and any(r.success for r in all_results):
                await _status_cb("finishing", "Composing response...")
                _envelope = await gw._formulate_response(
                    msg.text,
                    all_results,
                    wm,
                    stream_callback,
                )
                final_response = _envelope.content
            else:
                # No context -- sanitized fallback or error message
                _raw = plan.direct_response or ""
                _sanitized = _sanitize_broken_llm_output(_raw)
                if _sanitized and len(_sanitized) > 20:
                    # LLM hat brauchbaren Text produziert, nur JSON-Artefakte entfernt
                    final_response = _sanitized
                else:
                    final_response = t("gateway.parse_failed")
            break

        # Direkte Antwort — but detect REPLAN text masquerading as response
        if not plan.has_actions and plan.direct_response:
            _resp = plan.direct_response.strip()
            # If the LLM returned REPLAN reasoning instead of a real answer
            # or a JSON plan, it's stuck — don't echo it to the user.
            _is_replan_text = (
                _resp.startswith("REPLAN")
                or _resp.startswith("KORRIGIERTER PLAN")
                or _resp.startswith("BETROFFENE SCHRITTE")
                or _resp.startswith("AKTUALISIERTE RISIKOBEWERTUNG")
                or "REPLAN-GRUND" in _resp[:200]
                or "CORRECTED PLAN" in _resp[:200]
            )
            if _is_replan_text:
                _consecutive_no_tool_iters += 1
                log.warning(
                    "pge_replan_text_as_response",
                    iteration=session.iteration_count,
                    no_tool_streak=_consecutive_no_tool_iters,
                    preview=_resp[:100],
                )
                # On first iteration with no tool results, the LLM is
                # hallucinating REPLAN text for a conversational message.
                # Don't retry — immediately formulate a direct response.
                if session.iteration_count == 1 and not all_results:
                    await _status_cb("finishing", "Composing response...")
                    _envelope = await gw._formulate_response(
                        msg.text,
                        [],
                        wm,
                        stream_callback,
                    )
                    final_response = _envelope.content
                    break
                # Allow max 2 replan-text retries, then break
                if (
                    _consecutive_no_tool_iters < _max_no_tool_iters
                    and session.iteration_count < session.max_iterations
                ):
                    continue
                # Stuck — never send raw REPLAN text to the user
                if all_results and any(r.success for r in all_results):
                    await _status_cb("finishing", "Composing response...")
                    _envelope = await gw._formulate_response(
                        msg.text,
                        all_results,
                        wm,
                        stream_callback,
                    )
                    final_response = _envelope.content
                else:
                    final_response = (
                        "I'm stuck in a planning loop and can't make progress. "
                        "Please try rephrasing your request more concretely — e.g. "
                        "'Write a Pac-Man main.py' instead of 'Create a game'."
                    )
                break

            # If we already have successful tool results but the replan
            # returned text instead of JSON, formulate a proper response
            if all_results and any(r.success for r in all_results):
                await _status_cb("finishing", "Composing response...")
                _envelope = await gw._formulate_response(
                    msg.text,
                    all_results,
                    wm,
                    stream_callback,
                )
                final_response = _envelope.content
                break

            final_response = plan.direct_response
            break

        if not plan.has_actions:
            # If there are prior successful results, summarize them
            if all_results and any(r.success for r in all_results):
                await _status_cb("finishing", "Composing response...")
                _envelope = await gw._formulate_response(
                    msg.text,
                    all_results,
                    wm,
                    stream_callback,
                )
                final_response = _envelope.content
                break
            final_response = (
                "Ich konnte keinen Plan dafuer erstellen. Kannst du deine Frage umformulieren?"
            )
            break

        # Gatekeeper
        await _pipeline_cb("gate", "start", iteration=session.iteration_count)
        decisions = gw._gatekeeper.evaluate_plan(plan.steps, session)

        for step, decision in zip(plan.steps, decisions, strict=False):
            params_hash = hashlib.sha256(
                _json.dumps(step.params, sort_keys=True, default=str).encode()
            ).hexdigest()
            all_audit.append(
                AuditEntry(
                    session_id=session.session_id,
                    action_tool=step.tool,
                    action_params_hash=params_hash,
                    decision_status=decision.status,
                    decision_reason=decision.reason,
                )
            )

        # Approvals — use msg.session_id (client-facing WS session)
        # instead of session.session_id (internal gateway key) so the
        # channel can find the active WebSocket connection.
        approved_decisions = await gw._handle_approvals(
            plan.steps,
            decisions,
            session,
            msg.channel,
            ws_session_id=msg.session_id,
        )

        _n_blocked = sum(1 for d in approved_decisions if d.status == GateStatus.BLOCK)
        _n_allowed = sum(1 for d in approved_decisions if d.status != GateStatus.BLOCK)
        await _pipeline_cb(
            "gate",
            "done",
            iteration=session.iteration_count,
            blocked=_n_blocked,
            allowed=_n_allowed,
        )

        all_blocked = all(d.status == GateStatus.BLOCK for d in approved_decisions)
        if all_blocked:
            # Create pending_review Kanban task for blocked actions
            _kanban = getattr(gw, "_kanban_engine", None)
            if _kanban:
                for step, decision in zip(plan.steps, approved_decisions, strict=False):
                    if decision.status == GateStatus.BLOCK:
                        try:
                            _kanban.create_task(
                                title=f"Review: {step.tool} blocked by Gatekeeper",
                                description=(
                                    f"**Tool:** {step.tool}\n"
                                    f"**Reason:** {decision.reason}\n"
                                    f"**Risk:** {decision.risk_level.value}\n"
                                    f"**Params:** {str(step.params)[:200]}\n\n"
                                    "Approve this task to allow execution, "
                                    "or reject to cancel."
                                ),
                                priority="high",
                                status="pending_review",
                                source="system",
                                source_ref=f"gatekeeper:{step.tool}",
                                created_by="gatekeeper",
                            )
                            log.info(
                                "kanban_pending_review_created",
                                tool=step.tool,
                                risk=decision.risk_level.value,
                            )
                        except Exception:
                            log.debug("kanban_pending_review_failed", exc_info=True)

            for step, decision in zip(plan.steps, approved_decisions, strict=False):
                block_count = session.record_block(step.tool)
                if block_count >= 3:
                    escalation = await gw._planner.generate_escalation(
                        tool=step.tool,
                        reason=decision.reason,
                        working_memory=wm,
                    )
                    final_response = escalation
                    break
            else:
                try:
                    from cognithor.utils.error_messages import all_actions_blocked_message

                    final_response = all_actions_blocked_message(plan.steps, approved_decisions)
                except Exception:
                    final_response = "All planned actions were blocked by the Gatekeeper."
            break

        # Status: Tool-specific progress message
        for step in plan.steps:
            _status_key = _TOOL_STATUS_KEYS.get(step.tool)
            tool_status = (
                t(_status_key) if _status_key else t("status.tool_running", tool=step.tool)
            )
            await _status_cb("executing", tool_status)
            break  # Only send the first tool's status

        # Stream: tool_start events for each planned step
        if stream_callback is not None:
            for step in plan.steps:
                try:
                    _sk = _TOOL_STATUS_KEYS.get(step.tool)
                    _st = t(_sk) if _sk else t("status.tool_running", tool=step.tool)
                    await stream_callback(
                        "tool_start",
                        {
                            "tool": step.tool,
                            "status": _st,
                        },
                    )
                except Exception:
                    log.debug("stream_tool_start_failed", exc_info=True)

        # Set status callback on executor for retry visibility
        gw._executor.set_status_callback(_status_cb)
        await _pipeline_cb(
            "execute",
            "start",
            iteration=session.iteration_count,
            tools=[s.tool for s in plan.steps],
        )

        # Executor
        if route_decision and route_decision.agent.name != "jarvis":
            gw._executor.set_agent_context(
                workspace_dir=str(agent_workspace) if agent_workspace else None,
                sandbox_overrides=route_decision.agent.get_sandbox_config(),
                agent_name=route_decision.agent.name,
                session_id=session.session_id,
            )
        else:
            gw._executor.set_agent_context(session_id=session.session_id)

        # Faktenfrage: cross_check fuer search_and_read auto-injizieren
        # (muss NACH set_agent_context, da dieses clear_agent_context aufruft)
        if gw._is_fact_question(msg.text):
            gw._executor.set_fact_question_context(True)

        try:
            results = await gw._executor.execute(plan.steps, approved_decisions)
        finally:
            gw._executor.clear_agent_context()

        # Stream: tool_result events for each completed tool
        if stream_callback is not None:
            for result in results:
                try:
                    await stream_callback(
                        "tool_result",
                        {
                            "tool": result.tool_name,
                            "success": result.success,
                            "result": (result.content[:200] if result.success else "")
                            if hasattr(result, "content")
                            else "",
                        },
                    )
                except Exception:
                    log.debug("stream_tool_result_failed", exc_info=True)

        if run_id and gw._run_recorder:
            try:
                gw._run_recorder.record_gate_decisions(run_id, approved_decisions)
                gw._run_recorder.record_tool_results(run_id, results)
            except Exception:
                log.debug("run_recorder_results_failed", exc_info=True)

        all_results.extend(results)
        await _pipeline_cb(
            "execute",
            "done",
            iteration=session.iteration_count,
            success=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if r.is_error),
            total_ms=int(sum(r.duration_ms or 0 for r in results)),
        )

        # Identity: process execution results
        if _identity is not None:
            try:
                _tool_summary = "; ".join(
                    f"{r.tool_name}: {'OK' if r.success else 'FAIL'}" for r in results
                )
                _identity.process_interaction("assistant", f"[Tools] {_tool_summary}")
            except Exception:
                log.debug("identity_process_failed", exc_info=True)

        for result in results:
            all_audit.append(
                AuditEntry(
                    session_id=session.session_id,
                    action_tool=result.tool_name,
                    action_params_hash="",
                    decision_status=GateStatus.ALLOW,
                    decision_reason=f"executed success={result.success}",
                    execution_result="ok" if result.success else result.error_message or "error",
                )
            )
            # Prometheus: Tool-Aufruf-Metriken
            gw._record_metric("tool_calls_total", 1, tool_name=result.tool_name)
            if hasattr(result, "duration_ms") and result.duration_ms:
                gw._record_metric(
                    "tool_duration_ms",
                    result.duration_ms,
                    tool_name=result.tool_name,
                )
            if result.is_error:
                gw._record_metric(
                    "errors_total",
                    1,
                    channel=msg.channel,
                    error_type="tool_error",
                )

        # Explainability: record gatekeeper decision + execution outcome
        if getattr(gw, "_explainability", None) and _expl_trail_id:
            for _step, decision, result in zip(
                plan.steps,
                approved_decisions,
                results,
                strict=False,
            ):
                try:
                    gw._explainability.record_decision(
                        _expl_trail_id,
                        tool_name=result.tool_name,
                        gate_status=decision.status.value,
                        risk_level=decision.risk_level.value,
                        reason=decision.reason,
                        outcome="ok" if result.success else (result.error_message or "error"),
                        duration_ms=getattr(result, "duration_ms", 0) or 0,
                        success=result.success,
                    )
                except Exception:
                    log.debug("explainability_record_failed", exc_info=True)

        for result in results:
            wm.add_tool_result(result)

        has_errors = any(r.is_error for r in results)
        has_success = any(r.success for r in results)

        # Track consecutive iterations without any tool execution
        if results:
            _consecutive_no_tool_iters = 0
        else:
            _consecutive_no_tool_iters += 1
            if _consecutive_no_tool_iters >= _max_no_tool_iters:
                log.warning("pge_stuck_no_tools", iterations=session.iteration_count)
                if all_results and any(r.success for r in all_results):
                    await _status_cb("finishing", "Composing response...")
                    _envelope = await gw._formulate_response(
                        msg.text,
                        all_results,
                        wm,
                        stream_callback,
                    )
                    final_response = _envelope.content
                else:
                    final_response = (
                        "I'm stuck in a planning loop without making progress. "
                        "Please try a more specific request — e.g. "
                        "'Write a Pac-Man main.py' instead of 'Create a game'."
                    )
                break

        # ── Formal stalled-turn counter (session-scoped) ────────
        _tool_call_count = len(results)
        _successful_call_count = sum(1 for r in results if r.success)
        session.stalled_turn_count = advance_stalled_count(
            session.stalled_turn_count,
            _tool_call_count,
            _successful_call_count,
        )
        if session.stalled_turn_count >= MAX_STALLED_MODEL_TURNS:
            log.warning(
                "pge_stalled_turn_limit",
                stalled_turns=session.stalled_turn_count,
                iterations=session.iteration_count,
            )
            if all_results and any(r.success for r in all_results):
                await _status_cb("finishing", "Composing response...")
                _envelope = await gw._formulate_response(
                    msg.text,
                    all_results,
                    wm,
                    stream_callback,
                )
                final_response = _envelope.content
            else:
                final_response = (
                    "The model has been unable to make progress for "
                    f"{session.stalled_turn_count} consecutive turns. "
                    "Please simplify your request or try a different approach."
                )
            break

        # Check if the plan had MULTIPLE steps (multi-step task)
        _current_plan = all_plans[-1] if all_plans else None
        _is_multi_step = (
            _current_plan is not None
            and hasattr(_current_plan, "steps")
            and len(_current_plan.steps) > 1
        )

        # Coding tools: do not break immediately -- replan decides
        # whether further steps are needed (test, analyze, fix code)
        _coding_tools = {
            "run_python",
            "exec_command",
            "write_file",
            "edit_file",
            "analyze_code",
        }
        used_coding_tool = any(r.tool_name in _coding_tools for r in results)

        # ── Break conditions ─────────────────────────────────────────
        # Single-step non-coding tasks: respond immediately after success
        if has_success and not has_errors and not used_coding_tool and not _is_multi_step:
            await _status_cb("finishing", "Composing response...")
            _envelope = await gw._formulate_response(
                msg.text,
                all_results,
                wm,
                stream_callback,
            )
            final_response = _envelope.content
            break

        # Multi-step / coding tasks: let replan decide if more steps needed.
        # Caps scale with user's max_iterations setting to prevent infinite loops
        # while respecting the configured iteration budget.
        _successful_iters = sum(1 for r in all_results if r.success)
        # Scale coding cap: 80% of max_iterations (min 4, reserve room for formulate)
        _max_coding_iters = max(4, int(session.max_iterations * 0.8))
        _max_coding_iters = min(_max_coding_iters, session.max_iterations - 1)
        # Scale success threshold: ~30% of max_iterations (min 3)
        _success_threshold = max(3, int(session.max_iterations * 0.3))
        if (
            has_success
            and (used_coding_tool or _is_multi_step)
            and (
                session.iteration_count >= _max_coding_iters
                or _successful_iters >= _success_threshold
            )
        ):
            await _status_cb("finishing", "Composing response...")
            _envelope = await gw._formulate_response(
                msg.text,
                all_results,
                wm,
                stream_callback,
            )
            final_response = _envelope.content
            break
            # Otherwise: continue to replan for more steps (normal)

        # Failure-Threshold: give planner room for alternative strategies
        # Only give up after 70% of max_iterations with no success at all
        _failure_threshold = max(5, int(session.max_iterations * 0.7))
        if not has_success and session.iteration_count >= _failure_threshold:
            await _status_cb("finishing", "Composing response...")
            _envelope = await gw._formulate_response(
                msg.text,
                all_results,
                wm,
                stream_callback,
            )
            final_response = _envelope.content
            break

    if session.iterations_exhausted and not final_response:
        final_response = (
            "I've reached the maximum number of processing steps "
            "without fully completing the task. "
            "Please try a more specific request or break the task "
            "into smaller steps — happy to help!"
        )

    await _pipeline_cb(
        "complete",
        "done",
        iterations=session.iteration_count,
        tools_used=len(all_results),
    )

    # Identity: save state after PGE loop
    if _identity is not None:
        try:
            _identity.process_interaction("assistant", final_response[:500])
            _identity.save()
        except Exception:
            log.debug("identity_save_failed", exc_info=True)

    return final_response, all_results, all_plans, all_audit


async def handle_approvals(
    gw: Gateway,
    steps: list[Any],
    decisions: list[GateDecision],
    session: SessionContext,
    channel_name: str,
    *,
    ws_session_id: str | None = None,
) -> list[GateDecision]:
    """Holt User-Bestaetigungen fuer ORANGE-Aktionen ein.

    Args:
        ws_session_id: Client-facing session ID (e.g. from the WS URL).
                       Used for connection lookup instead of the internal
                       ``session.session_id``.

    Returns:
        Aktualisierte Liste von Entscheidungen (APPROVE → ALLOW oder BLOCK).
    """
    channel = gw._channels.get(channel_name)
    if channel is None:
        # No channel available — convert unresolved APPROVE to BLOCK
        result = list(decisions)
        for i, decision in enumerate(result):
            if decision.status == GateStatus.APPROVE:
                _no_channel_reason = (
                    f"Kein interaktiver Kanal verfuegbar fuer Bestaetigung: {decision.reason}"
                )
                result[i] = GateDecision(
                    status=GateStatus.BLOCK,
                    reason=_no_channel_reason,
                    risk_level=decision.risk_level,
                    original_action=decision.original_action,
                    policy_name=f"{decision.policy_name}:no_channel",
                )
                log.warning(
                    "approval_no_channel", tool=getattr(decision.original_action, "tool", "?")
                )
        return result

    # Use client-facing session ID for WS connection lookup
    _approval_sid = ws_session_id or session.session_id
    result = list(decisions)  # Kopie

    for i, (step, decision) in enumerate(zip(steps, decisions, strict=False)):
        if decision.status != GateStatus.APPROVE:
            continue

        # User fragen
        try:
            approved = await channel.request_approval(
                session_id=_approval_sid,
                action=step,
                reason=decision.reason,
            )
        except Exception:
            log.warning("approval_request_failed", tool=step.tool, exc_info=True)
            approved = False

        if approved:
            result[i] = GateDecision(
                status=GateStatus.ALLOW,
                reason=f"User-Bestätigung für: {decision.reason}",
                risk_level=decision.risk_level,
                original_action=step,
                policy_name=f"{decision.policy_name}:user_approved",
            )
            log.info("user_approved_action", tool=step.tool)
        else:
            result[i] = GateDecision(
                status=GateStatus.BLOCK,
                reason=f"User-Ablehnung für: {decision.reason}",
                risk_level=decision.risk_level,
                original_action=step,
                policy_name=f"{decision.policy_name}:user_rejected",
            )
            log.info("user_rejected_action", tool=step.tool)

    return result
