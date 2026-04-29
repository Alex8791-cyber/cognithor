"""Cognithor · Gateway post-processing — extracted from `gateway.py`.

Runs after the PGE loop completes:
  * :func:`run_post_processing` — orchestrates the post-PGE pipeline
    (reflexion, skill tracking, telemetry, persistence).
  * :func:`maybe_record_pattern` — opportunistic pattern-mining hook.
  * :func:`persist_session` — writes session + working memory to SQLite.
  * :func:`persist_key_tool_results` — file-based artefact persistence.

Each function takes the `Gateway` instance as its first argument (`gw`)
and reads gateway-internal state through it. No new instance attributes.

Part of the staged `gateway.py` split — see
`project_v0960_refactor_backlog.md` and the architect blueprint.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.models import AgentResult, Message, MessageRole
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.gateway import Gateway
    from cognithor.models import (
        SessionContext,
        ToolResult,
        WorkingMemory,
    )

log = get_logger(__name__)


async def run_post_processing(
    gw: Gateway,
    session: SessionContext,
    wm: WorkingMemory,
    agent_result: AgentResult,
    active_skill: Any,
    run_id: str | None,
) -> None:
    """Phase 4: Reflection, Skill-Tracking, Telemetry, Profiler, Run-Recording."""
    if gw._reflector and gw._reflector.should_reflect(agent_result):
        try:
            reflection = await gw._reflector.reflect(session, wm, agent_result)
            agent_result.reflection = reflection
            log.info(
                "reflection_done",
                session=session.session_id[:8],
                score=reflection.success_score,
            )
            # Apply reflection to memory tiers (episodic, semantic, procedural)
            if gw._memory_manager:
                try:
                    counts = await gw._reflector.apply(reflection, gw._memory_manager)
                    log.info(
                        "reflection_applied",
                        session=session.session_id[:8],
                        episodic=counts.get("episodic", 0),
                        semantic=counts.get("semantic", 0),
                        procedural=counts.get("procedural", 0),
                    )
                except Exception as apply_exc:
                    log.error("reflection_apply_error", error=str(apply_exc))
            if run_id and gw._run_recorder:
                try:
                    gw._run_recorder.record_reflection(run_id, reflection)
                except Exception:
                    log.debug("run_recorder_reflection_failed", exc_info=True)
            # ── Feed weak results into Evolution Engine ──────────
            # If reflection shows poor quality or user corrected us,
            # create a learning goal so the Evolution Engine researches it.
            if (
                reflection.success_score < 0.5
                and hasattr(gw, "_deep_learner")
                and gw._deep_learner
                and hasattr(gw, "_evolution_loop")
                and gw._evolution_loop
            ):
                try:
                    user_msg = (session.messages[-1].content[:200] if session.messages else "")[
                        :200
                    ]
                    gap_description = (
                        f"Schwache Antwort (Score {reflection.success_score:.1f}) auf: {user_msg}"
                    )
                    # Add as learning goal for next evolution cycle
                    config = getattr(gw, "_config", None)
                    if config and hasattr(config, "evolution"):
                        goals = list(getattr(config.evolution, "learning_goals", []) or [])
                        # Avoid duplicates
                        if gap_description not in goals and len(goals) < 20:
                            goals.append(gap_description)
                            config.evolution.learning_goals = goals
                            log.info(
                                "evolution_gap_from_chat",
                                score=reflection.success_score,
                                query=user_msg[:60],
                            )
                except Exception:
                    log.debug("evolution_gap_injection_failed", exc_info=True)
        except Exception as exc:
            log.error("reflection_error", error=str(exc))

    # Meta-Reasoning: record strategy outcome
    if getattr(gw, "_strategy_memory", None):
        try:
            from cognithor.learning.strategy_memory import (
                StrategyRecord,
                classify_task_type,
            )

            tools_used = [r.tool_name for r in agent_result.tool_results if r.tool_name]
            if tools_used:
                task_type = classify_task_type(tools_used)
                strategy = " -> ".join(dict.fromkeys(tools_used))
                success = any(r.success for r in agent_result.tool_results)
                total_ms = sum(getattr(r, "duration_ms", 0) or 0 for r in agent_result.tool_results)
                gw._strategy_memory.record(
                    StrategyRecord(
                        task_type=task_type,
                        strategy=strategy[:200],
                        success=success,
                        duration_ms=total_ms,
                        tool_count=len(tools_used),
                    )
                )
                log.debug(
                    "strategy_recorded",
                    task_type=task_type,
                    strategy=strategy[:60],
                    success=success,
                )
        except Exception:
            log.debug("strategy_record_failed", exc_info=True)

    if active_skill and gw._skill_registry:
        try:
            success = agent_result.success
            score = (
                agent_result.reflection.success_score
                if agent_result.reflection
                else (0.8 if success else 0.3)
            )
            gw._skill_registry.record_usage(
                active_skill.skill.slug,
                success=success,
                score=score,
            )
            # Store failure pattern in procedure (for learning effect)
            if not success and gw._memory_manager and active_skill.procedure_name:
                try:
                    error_summary = agent_result.error[:200] if agent_result.error else "unknown"
                    gw._memory_manager.procedural.add_failure_pattern(
                        active_skill.procedure_name,
                        error_summary,
                    )
                except Exception:
                    log.debug("procedure_failure_pattern_save_failed", exc_info=True)

            # Gap Detection: Melde niedrige Erfolgsrate
            if not success and hasattr(gw, "_skill_generator") and gw._skill_generator:
                try:
                    skill_obj = active_skill.skill
                    if skill_obj.total_uses >= 3 and skill_obj.total_uses > 0:
                        success_rate = skill_obj.success_count / skill_obj.total_uses
                        if success_rate < 0.4:
                            gw._skill_generator.gap_detector.report_low_success_rate(
                                skill_obj.slug,
                                success_rate,
                            )
                except Exception:
                    log.debug("skill_gap_detection_failed", exc_info=True)
        except Exception:
            log.debug("skill_usage_tracking_skipped", exc_info=True)

    if hasattr(gw, "_task_telemetry") and gw._task_telemetry:
        try:
            all_results = agent_result.tool_results
            tools_used = [r.tool_name for r in all_results]
            error_type = ""
            error_msg = ""
            for r in all_results:
                if r.is_error:
                    error_type = r.error_type or ""
                    error_msg = r.content[:200]
                    break
            gw._task_telemetry.record_task(
                session_id=session.session_id,
                success=agent_result.success,
                duration_ms=float(agent_result.total_duration_ms),
                tool_calls=tools_used,
                error_type=error_type,
                error_message=error_msg,
            )
        except Exception:
            log.debug("task_telemetry_record_failed", exc_info=True)

    if hasattr(gw, "_task_profiler") and gw._task_profiler:
        try:
            score = (
                agent_result.reflection.success_score
                if agent_result.reflection
                else (0.8 if agent_result.success else 0.3)
            )
            gw._task_profiler.finish_task(
                session_id=session.session_id,
                success_score=score,
            )
        except Exception:
            log.debug("task_profiler_finish_failed", exc_info=True)

    if run_id and hasattr(gw, "_run_recorder") and gw._run_recorder:
        try:
            gw._run_recorder.finish_run(
                run_id,
                success=agent_result.success,
                final_response=agent_result.response[:500],
            )
        except Exception:
            log.debug("run_recorder_finish_failed", exc_info=True)

    # Prompt-Evolution: Record session reward for A/B testing
    if getattr(gw, "_prompt_evolution", None) and gw._planner:
        try:
            version_id = getattr(gw._planner, "_current_prompt_version_id", None)
            if version_id:
                reward_score = (
                    agent_result.reflection.success_score
                    if agent_result.reflection
                    else (0.8 if agent_result.success else 0.3)
                )
                gw._prompt_evolution.record_session(
                    session_id=session.session_id,
                    prompt_version_id=version_id,
                    reward=reward_score,
                )
        except Exception:
            log.debug("prompt_evolution_record_failed", exc_info=True)

    # GEPA: Collect execution trace
    if getattr(gw, "_trace_store", None):
        try:
            import time as _time
            import uuid as _uuid

            from cognithor.learning.execution_trace import ExecutionTrace, TraceStep

            # Extract user goal from working memory
            _goal = ""
            for _m in wm.messages:
                if getattr(_m, "role", None) and _m.role.value == "user":
                    _goal = getattr(_m, "content", "")[:1000]
                    break

            _reward = (
                agent_result.reflection.success_score
                if agent_result.reflection
                else (0.8 if agent_result.success else 0.3)
            )
            trace = ExecutionTrace(
                trace_id=_uuid.uuid4().hex[:16],
                session_id=session.session_id,
                goal=_goal,
                total_duration_ms=int(agent_result.total_duration_ms),
                success_score=_reward,
                model_used=agent_result.model_used or "",
                created_at=_time.time(),
            )
            # Build steps from tool_results
            for _tr in agent_result.tool_results or []:
                step = TraceStep(
                    step_id=_uuid.uuid4().hex[:16],
                    parent_id=None,
                    tool_name=getattr(_tr, "tool_name", "") or "",
                    input_summary=str(getattr(_tr, "input", ""))[:500],
                    output_summary=str(getattr(_tr, "content", ""))[:500],
                    status="error" if getattr(_tr, "is_error", False) else "success",
                    error_detail=str(getattr(_tr, "error_type", ""))
                    if getattr(_tr, "is_error", False)
                    else "",
                    duration_ms=int(getattr(_tr, "duration_ms", 0)),
                    timestamp=_time.time(),
                )
                trace.steps.append(step)
            gw._trace_store.save_trace(trace)
            log.debug("gepa_trace_saved", trace_id=trace.trace_id, steps=len(trace.steps))
        except Exception:
            log.debug("gepa_trace_save_failed", exc_info=True)

    # Reflexion: check for known solutions before recording new errors
    if getattr(gw, "_reflexion_memory", None) and hasattr(agent_result, "tool_results"):
        try:
            for tr in agent_result.tool_results or []:
                if getattr(tr, "is_error", False) or getattr(tr, "error", None):
                    tool = getattr(tr, "tool_name", "") or str(getattr(tr, "name", ""))
                    error_msg = str(getattr(tr, "error", "") or getattr(tr, "error_type", ""))
                    known = gw._reflexion_memory.get_solution(tool, "unknown", error_msg)
                    if known:
                        log.info(
                            "reflexion_known_error",
                            tool=tool,
                            solution=known.prevention_rule,
                        )
                    else:
                        _msg_text = ""
                        for _m in wm.messages:
                            if getattr(_m, "role", None) and _m.role.value == "user":
                                _msg_text = getattr(_m, "content", "")
                                break
                        gw._reflexion_memory.record_error(
                            tool_name=tool,
                            error_category="unknown",
                            error_message=error_msg,
                            root_cause="auto-detected",
                            prevention_rule="",
                            task_context=_msg_text[:200] if _msg_text else "",
                            channel=getattr(session, "channel", ""),
                        )
        except Exception:
            log.debug("reflexion_post_processing_failed", exc_info=True)

    # GEPA: Run evolution cycle if due
    if getattr(gw, "_evolution_orchestrator", None):
        try:
            import time as _time

            orch = gw._evolution_orchestrator
            gepa_cfg = getattr(gw._config, "gepa", None)
            interval = (gepa_cfg.evolution_interval_hours * 3600) if gepa_cfg else 21600
            if _time.time() - getattr(orch, "_last_cycle_time", 0) > interval:
                evo_result = orch.run_evolution_cycle()
                log.info(
                    "gepa_evolution_cycle_completed",
                    cycle_id=evo_result.cycle_id,
                    traces=evo_result.traces_analyzed,
                    proposals=evo_result.proposals_generated,
                    applied=evo_result.proposal_applied,
                    rollbacks=evo_result.auto_rollbacks,
                )
        except Exception:
            log.debug("gepa_evolution_cycle_failed", exc_info=True)

    # Session-Analyse: Failure-Clustering und Feedback-Loop
    if getattr(gw, "_session_analyzer", None):
        try:
            improvements = await gw._session_analyzer.analyze_session(
                session_id=session.session_id,
                agent_result=agent_result,
                reflection=agent_result.reflection,
            )
            for imp in improvements:
                log.info(
                    "session_improvement_proposed",
                    action=imp.action_type,
                    target=imp.target,
                    priority=imp.priority,
                )
                try:
                    applied = gw._session_analyzer.apply_improvement(imp)
                    if applied:
                        log.info(
                            "session_improvement_applied",
                            action=imp.action_type,
                            target=imp.target,
                        )
                except Exception:
                    log.debug("session_improvement_apply_failed", exc_info=True)
        except Exception:
            log.debug("session_analysis_failed", exc_info=True)

    # Pattern Documentation: record successful tool sequences
    if gw._memory_manager:
        try:
            gw._maybe_record_pattern(session, wm, agent_result)
        except Exception:
            log.debug("pattern_documentation_post_failed", exc_info=True)

    # Self-Learning: Process actionable skill gaps (auto-generate new tools)
    if hasattr(gw, "_skill_generator") and gw._skill_generator:
        try:
            generated = await gw._skill_generator.process_all_gaps(
                skill_registry=gw._skill_registry if hasattr(gw, "_skill_registry") else None,
            )
            newly_registered = False
            for skill in generated:
                log.info(
                    "skill_auto_generated",
                    name=skill.name,
                    status=skill.status.value,
                    version=skill.version,
                )
                if skill.status.value == "registered":
                    newly_registered = True
            # CORE.md aktualisieren wenn neue Skills registriert wurden
            if newly_registered:
                try:
                    gw._sync_core_inventory()
                except Exception:
                    log.debug("core_inventory_sync_after_skill_gen_failed", exc_info=True)
        except Exception:
            log.debug("skill_gap_processing_failed", exc_info=True)


# ── Pattern Documentation ────────────────────────────────────
# `_PATTERN_MAX_PER_HOUR` lives as a ClassVar on `Gateway` (defined in
# `gateway.py`). `maybe_record_pattern` reads it via `gw._PATTERN_MAX_PER_HOUR`.
# Class-attr lookup works on bare `Gateway()` and `Gateway.__new__(Gateway)`
# instances alike — important because tests instantiate without `__init__`.


def maybe_record_pattern(
    gw: Gateway,
    session: SessionContext,
    wm: WorkingMemory,
    agent_result: AgentResult,
) -> None:
    """Extract and store execution patterns for procedural memory.

    After successful execution, extracts the tool sequence and user intent,
    checks for similar existing patterns, and stores new ones.
    Rate limited to max 5 recordings per hour.
    """
    try:
        # Only record successful executions with tool results
        if not agent_result.success or not agent_result.tool_results:
            return

        # Check for errors in tool results
        if any(getattr(tr, "is_error", False) for tr in agent_result.tool_results):
            return

        # Rate limit check
        now = time.monotonic()
        # Prune old timestamps (older than 1 hour)
        gw._pattern_record_timestamps[:] = [
            ts for ts in gw._pattern_record_timestamps if now - ts < 3600
        ]
        if len(gw._pattern_record_timestamps) >= gw._PATTERN_MAX_PER_HOUR:
            return

        # Extract tool sequence
        tool_sequence = [
            getattr(tr, "tool_name", "") or ""
            for tr in agent_result.tool_results
            if getattr(tr, "tool_name", "")
        ]
        if not tool_sequence:
            return

        # Extract user intent keywords from working memory
        user_text = ""
        for m in getattr(wm, "chat_history", []):
            if getattr(m, "role", None) and m.role.value == "user":
                user_text = getattr(m, "content", "")
                break
        if not user_text:
            return

        # Build keywords (simple: take significant words)
        keywords = [
            w
            for w in user_text.lower().split()
            if len(w) > 3
            and w
            not in {
                "bitte",
                "kannst",
                "koenntest",
                "wuerdest",
                "mach",
                "zeig",
                "dass",
                "diese",
                "dieser",
                "dieses",
                "eine",
                "einen",
                "einem",
                "einer",
                "the",
                "and",
                "for",
                "that",
                "this",
                "with",
                "please",
                "could",
                "would",
                "show",
                "make",
            }
        ][:5]

        if not keywords:
            return

        channel = getattr(session, "channel", "")
        tools_str = ", ".join(tool_sequence)
        keywords_str = ", ".join(keywords)

        # Check if similar pattern exists (fuzzy match via procedural memory)
        if gw._memory_manager:
            procedural = getattr(gw._memory_manager, "procedural", None)
            if procedural is not None:
                # Check for existing procedures with similar tool sequences
                existing = getattr(procedural, "search_procedures", None)
                if existing:
                    try:
                        matches = existing(keywords_str)
                        if matches and tools_str in str(matches):
                            log.debug(
                                "pattern_already_documented",
                                tools=tools_str,
                            )
                            return
                    except Exception:
                        log.debug("procedural_search_failed", exc_info=True)

                # Store new pattern as procedure with human-readable name
                pattern_body = (
                    f"When user asks about {keywords_str}, "
                    f"use tools [{tools_str}]. "
                    f"Context: {user_text[:200]}"
                )
                try:
                    from cognithor.models import ProcedureMetadata

                    # Generate readable name from keywords (max 5 words, slugified)
                    _name_words = [
                        re.sub(r"[^\w]", "", k.lower()) for k in keywords[:3] if len(k) > 2
                    ]
                    if _name_words:
                        name = "-".join(_name_words)
                    else:
                        # Fallback: use first meaningful words from user text
                        _text_words = [
                            w.lower() for w in user_text.split()[:4] if len(w) > 2 and w.isalpha()
                        ]
                        name = "-".join(_text_words) if _text_words else f"auto-{int(now)}"
                    # Ensure uniqueness by appending short hash if file exists
                    _base_name = name
                    _proc_dir = getattr(procedural, "_dir", None)
                    if _proc_dir is not None and isinstance(_proc_dir, str | Path):
                        _proc_dir = Path(_proc_dir)
                        if (_proc_dir / f"{name}.md").exists():
                            _short = hashlib.sha256(
                                f"{tools_str}:{keywords_str}".encode()
                            ).hexdigest()[:6]
                            name = f"{_base_name}-{_short}"
                            # If even that exists, skip (true duplicate)
                            if (_proc_dir / f"{name}.md").exists():
                                log.debug("pattern_duplicate_skipped", name=name)
                                return

                    procedural.save_procedure(
                        name=name,
                        body=pattern_body,
                        metadata=ProcedureMetadata(
                            name=name,
                            trigger_keywords=keywords,
                            tools_required=tool_sequence,
                        ),
                    )
                    gw._pattern_record_timestamps.append(now)
                    log.info(
                        "pattern_documented",
                        name=name,
                        tools=tools_str,
                        keywords=keywords_str,
                        channel=channel,
                    )
                except Exception:
                    log.debug("pattern_save_failed", exc_info=True)
    except Exception:
        log.debug("pattern_documentation_failed", exc_info=True)


async def persist_session(
    gw: Gateway,
    session: SessionContext,
    wm: WorkingMemory,
) -> None:
    """Phase 5: Session persistieren."""
    # Incognito: nur Session-Metadaten speichern, keine Chat-History
    if session.incognito:
        if gw._session_store:
            try:
                gw._session_store.save_session(session)
            except Exception as exc:
                log.warning("session_persist_error", error=str(exc))
        return
    if gw._session_store:
        try:
            gw._session_store.save_session(session)
            gw._session_store.save_chat_history(
                session.session_id,
                wm.chat_history,
            )
        except Exception as exc:
            log.warning("session_persist_error", error=str(exc))
        # Auto-Titel aus erster User-Message generieren
        if hasattr(gw._session_store, "auto_title"):
            try:
                gw._session_store.auto_title(session.session_id)
            except Exception:
                log.debug("auto_title_failed", exc_info=True)


# =========================================================================
# Agent-zu-Agent Delegation
# =========================================================================


def persist_key_tool_results(
    gw: Gateway,
    wm: WorkingMemory,
    results: list[ToolResult],
) -> None:
    """Persistiert wichtige Tool-Ergebnisse als TOOL-Messages in der Chat-History.

    Damit behaelt der Planner bei Folge-Requests den vollen Kontext,
    z.B. extrahierter Text aus Bildern, Analyse-Ergebnisse, Suchergebnisse.
    """
    for result in results:
        if not result.success:
            continue
        if result.tool_name not in gw._CONTEXT_TOOLS:
            continue
        if not result.content.strip():
            continue

        content = result.content[: gw._CONTEXT_RESULT_LIMIT]
        if len(result.content) > gw._CONTEXT_RESULT_LIMIT:
            content += "\n[... gekürzt]"

        wm.add_message(
            Message(
                role=MessageRole.TOOL,
                content=content,
                name=result.tool_name,
            )
        )
        log.debug(
            "tool_result_persisted",
            tool=result.tool_name,
            chars=len(content),
        )


# `_ATTACHMENT_TOOLS` and `_ATTACHMENT_EXTENSIONS` are class-level constants
# on `Gateway` (defined in `gateway.py`). Helpers here read them via
# `gw._ATTACHMENT_TOOLS` etc. — single source of truth stays on the class.

# ── Prometheus Metric Recording ──────────────────────────────
