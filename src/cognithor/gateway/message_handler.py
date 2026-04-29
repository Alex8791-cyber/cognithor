"""Cognithor · Gateway message-handler — extracted from `gateway.py`.

The user-facing entry point of every agent turn:
  * :func:`handle_message` — orchestrates the full message lifecycle
    (Phase 1 routing → Phase 2 execution context → Phase 3 PGE loop →
    Phase 4 post-processing → Phase 5 persistence).
  * :func:`resolve_agent_route` — Phase 1: agent routing, session,
    working memory, skill matching, workspace.
  * :func:`prepare_execution_context` — Phase 2: profiler, budget,
    run-recorder, policy snapshot.
  * :func:`make_status_callback`, :func:`make_pipeline_callback` —
    fire-and-forget callback factories for live UI feedback.
  * :func:`formulate_response` — observer-aware response synthesis.

Every function takes the `Gateway` instance as `gw`; instance state stays
on the Gateway. Part of the staged `gateway.py` split — see
`project_v0960_refactor_backlog.md` and the architect blueprint.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from cognithor.core.agent_router import RouteDecision
from cognithor.core.autonomous_orchestrator import AutonomousOrchestrator
from cognithor.gateway.gateway import (
    _build_video_attachment,
    _classify_attachments,
    _extract_uuid_from_path,
)
from cognithor.i18n import t
from cognithor.models import (
    ActionPlan,
    AgentResult,
    AuditEntry,
    IncomingMessage,
    Message,
    MessageRole,
    OutgoingMessage,
    ToolResult,
    WorkingMemory,
)
from cognithor.security.compliance_engine import ComplianceViolation
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.core.planner import ResponseEnvelope
    from cognithor.gateway.gateway import Gateway
    from cognithor.models import SessionContext

log = get_logger(__name__)


async def handle_message(
    gw: Gateway,
    msg: IncomingMessage,
    stream_callback: Any | None = None,
) -> OutgoingMessage:
    """Verarbeitet eine eingehende Nachricht. [B§3.4]

    Orchestriert den PGE-Zyklus (Plan → Gate → Execute → Replan).

    Args:
        msg: Eingehende Nachricht.
        stream_callback: Optionaler async Callback fuer Streaming-Events.
            Signatur: async (event_type: str, data: dict) -> None.
            Wird vom WebUI-Channel gesetzt, um Tokens und Status-Events
            in Echtzeit an den Client zu senden.

    Returns:
        OutgoingMessage mit der Jarvis-Antwort.
    """
    _handle_start = time.monotonic()

    # Notify idle detector IMMEDIATELY so background tasks (ATL, evolution)
    # yield the GPU to user requests. Previously this was at the END of
    # handle_message, meaning the evolution loop kept blocking the LLM
    # for the entire duration of the user's wait.
    if hasattr(gw, "_idle_detector") and gw._idle_detector:
        gw._idle_detector.notify_activity()
    if gw._active_learner is not None:
        gw._active_learner.notify_activity()

    # --- Sub-Agent depth guard ---
    # The _agent_runner passes depth in msg.metadata. Enforce max depth
    # to prevent infinite recursive sub-agent delegation.
    _depth = msg.metadata.get("depth", 0) if msg.metadata else 0
    _max_depth = getattr(gw._config.security, "max_sub_agent_depth", 3)
    if _depth > _max_depth:
        log.warning(
            "sub_agent_depth_exceeded",
            depth=_depth,
            max_depth=_max_depth,
            channel=msg.channel,
            user_id=msg.user_id,
        )
        return OutgoingMessage(
            channel=msg.channel,
            text=t(
                "gateway.sub_agent_recursion",
                depth=_depth,
                max_depth=_max_depth,
            ),
            is_final=True,
        )

    # Detect system-internal messages (cron jobs, sub-agents, etc.)
    _uid = (msg.user_id or "").lower()
    _is_system = (
        _uid.startswith("cron")
        or _uid.startswith("heartbeat")
        or _uid.startswith("agent:")
        or (msg.channel or "") in ("cron", "sub_agent", "system", "evolution", "heartbeat")
        or (msg.metadata or {}).get("cron_job")
    )

    # Handle consent responses (works for all channels including WebUI)
    # Skip for system messages — they don't need consent
    if (
        (
            not _is_system
            and msg.text
            and msg.text.strip().lower() in ("akzeptieren", "accept", "ja", "yes")
        )
        and hasattr(gw, "_consent_manager")
        and gw._consent_manager
    ):
        _uid = msg.user_id or msg.session_id or "unknown"
        _ch = msg.channel or "unknown"
        if gw._consent_manager.requires_consent(_uid, _ch):
            gw._consent_manager.grant_consent(
                _uid, _ch, "data_processing", context=msg.session_id or ""
            )
            return OutgoingMessage(
                channel=msg.channel,
                text=t("gateway.consent_granted"),
                session_id=msg.session_id or msg.id or "consent",
                is_final=True,
            )

    # GDPR compliance gate — check consent before processing
    if hasattr(gw, "_compliance_engine") and gw._compliance_engine:
        try:
            from cognithor.security.gdpr import DataPurpose, ProcessingBasis

            _channel = msg.channel or "unknown"
            _user = msg.user_id or msg.session_id or "unknown"
            # System-internal messages use legitimate interest, not consent
            # Detect via channel name, user_id prefix, or metadata
            _basis = ProcessingBasis.LEGITIMATE_INTEREST if _is_system else ProcessingBasis.CONSENT
            gw._compliance_engine.check(
                user_id=_user,
                channel=_channel,
                legal_basis=_basis,
                purpose=DataPurpose.CONVERSATION,
            )
        except ComplianceViolation as e:
            log.info("compliance_blocked", reason=str(e)[:100])
            # Return a user-friendly consent prompt
            consent_text = str(e)
            if "consent" in consent_text.lower():
                consent_text = t("privacy.consent_prompt")
            return OutgoingMessage(
                channel=msg.channel,
                text=consent_text,
                session_id=msg.session_id or msg.id or "compliance",
                is_final=True,
            )

    # Prometheus: count incoming requests
    gw._record_metric("requests_total", 1, channel=msg.channel)

    # User-Feedback erkennen und speichern (vor PGE-Zyklus)
    if getattr(gw, "_session_analyzer", None):
        try:
            signal = gw._session_analyzer._extract_feedback_signal(msg.text)
            if signal is not None:
                fb_type, detail = signal
                sid = msg.session_id if hasattr(msg, "session_id") else ""
                gw._session_analyzer.record_user_feedback(
                    session_id=sid,
                    message_id=getattr(msg, "message_id", ""),
                    feedback_type=fb_type,
                    detail=detail,
                )
                log.info("user_feedback_recorded", type=fb_type)
        except Exception:
            log.debug("user_feedback_detection_failed", exc_info=True)

    # Phase 1: Agent-Routing, Session, WM, Skills, Workspace
    (
        route_decision,
        session,
        wm,
        active_skill,
        agent_workspace,
        agent_name,
        _expl_trail_id,
    ) = await gw._resolve_agent_route(msg)

    # Phase 2: Profiler, Budget, Run-Recording, Policy-Snapshot
    run_id, budget_response = await gw._prepare_execution_context(
        msg,
        session,
        wm,
        route_decision,
    )
    if budget_response is not None:
        return budget_response

    # Phase 2.3+2.5: Execute in parallel (#43 optimization)
    # Context pipeline, coding classification, and presearch are independent
    # and can run in parallel.

    # Tool-Schemas (gefiltert nach Agent-Rechten) — synchron, schnell
    tool_schemas = gw._mcp_client.get_tool_schemas() if gw._mcp_client else {}
    if route_decision and route_decision.agent.has_tool_restrictions:
        tool_schemas = route_decision.agent.filter_tools(tool_schemas)

    # Subsystem checks
    if gw._planner is None or gw._gatekeeper is None or gw._executor is None:
        raise RuntimeError("Gateway.initialize() must be called before handle_message()")

    async def _run_context_pipeline():
        if session.incognito:
            log.info("incognito_skip_context", session=session.session_id[:8])
            return
        if gw._context_pipeline is not None:
            try:
                ctx_result = await gw._context_pipeline.enrich(msg.text, wm)
                if not ctx_result.skipped:
                    log.info(
                        "context_enriched",
                        memories=len(ctx_result.memory_results),
                        vault=len(ctx_result.vault_snippets),
                        episodes=len(ctx_result.episode_snippets),
                        ms=f"{ctx_result.duration_ms:.1f}",
                    )
            except Exception:
                log.warning("context_pipeline_failed", exc_info=True)

    async def _run_coding_classification():
        _is_coding = False
        _coding_model = ""
        _coding_complexity = "simple"
        try:
            _is_coding, _coding_complexity = await gw._classify_coding_task(msg.text)
            if _is_coding and gw._model_router:
                if _coding_complexity == "complex":
                    _coding_model = gw._model_router._config.models.coder.name
                else:
                    _coding_model = gw._model_router._config.models.coder_fast.name
                # NOTE: Do NOT call set_coding_override() here — asyncio.create_task()
                # runs in a copied context, so ContextVar changes are invisible to the
                # parent. The override is applied in the parent after await (line below).
                log.info("coding_task_detected", complexity=_coding_complexity, model=_coding_model)
        except Exception:
            log.debug("coding_classification_skipped", exc_info=True)
        return _is_coding, _coding_model, _coding_complexity

    async def _run_presearch():
        return await gw._maybe_presearch(msg, wm)

    import asyncio as _aio

    _ctx_task = _aio.create_task(_run_context_pipeline())
    _coding_task = _aio.create_task(_run_coding_classification())
    _presearch_task = _aio.create_task(_run_presearch())

    await _ctx_task  # Muss vor PGE fertig sein (modifiziert wm)
    is_coding, coding_model, coding_complexity = await _coding_task
    presearch_results = await _presearch_task

    # Apply coding override in parent context (ContextVar must be set here,
    # not inside the create_task — asyncio tasks get a copied context)
    if coding_model and gw._model_router:
        gw._model_router.set_coding_override(coding_model)

    # Coding-Tasks: mehr Iterationen fuer iteratives Fixen, Debuggen, Optimieren
    # Cognithor soll autonom arbeiten bis die Aufgabe erledigt ist
    if is_coding and session.max_iterations < 50:
        session.max_iterations = 50

    # ── Token Budget (complexity-based) ──
    _token_budget = None
    try:
        from cognithor.core.token_budget import TokenBudgetManager

        _complexity = TokenBudgetManager.detect_complexity(msg.text)
        _token_budget = TokenBudgetManager(complexity=_complexity, channel=msg.channel)
        log.debug(
            "token_budget_allocated",
            complexity=_complexity,
            channel=msg.channel,
            total=_token_budget.total,
        )
    except Exception:
        log.debug("token_budget_skipped", exc_info=True)

    # ── Sentiment Detection (Modul 3) ──
    try:
        from cognithor.core.sentiment import (
            Sentiment,
            detect_sentiment,
            get_sentiment_system_message,
        )

        sentiment_result = detect_sentiment(msg.text)
        if sentiment_result.sentiment != Sentiment.NEUTRAL:
            hint = get_sentiment_system_message(sentiment_result.sentiment)
            if hint:
                wm.add_message(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=hint,
                        channel=msg.channel,
                    )
                )
                log.info(
                    "sentiment_detected",
                    sentiment=sentiment_result.sentiment,
                    confidence=sentiment_result.confidence,
                    trigger=sentiment_result.trigger_phrase[:50],
                )
    except Exception:
        log.debug("sentiment_detection_skipped", exc_info=True)

    # ── User Preferences (Modul 4) ──
    if hasattr(gw, "_user_pref_store") and gw._user_pref_store is not None:
        try:
            pref = gw._user_pref_store.record_interaction(msg.user_id, len(msg.text))
            verbosity_hint = pref.verbosity_hint
            if verbosity_hint:
                wm.add_message(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=verbosity_hint,
                        channel=msg.channel,
                    )
                )
        except Exception:
            log.debug("user_preferences_skipped", exc_info=True)

    # ── Channel Flags (Modul 5) ──
    _channel_flags = None
    try:
        from cognithor.core.channel_flags import get_channel_flags

        _channel_flags = get_channel_flags(msg.channel)
        if _channel_flags.compact_output or _channel_flags.token_efficient:
            wm.add_message(
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        f"Channel: {msg.channel}. "
                        + (
                            t("gateway.channel_compact") + " "
                            if _channel_flags.compact_output
                            else ""
                        )
                        + (
                            t(
                                "gateway.channel_max_length",
                                max=_channel_flags.max_response_length,
                            )
                            + " "
                            if _channel_flags.max_response_length
                            else ""
                        )
                        + (
                            t("gateway.channel_no_markdown") + " "
                            if not _channel_flags.allow_markdown
                            else ""
                        )
                        + (
                            t("gateway.channel_no_code_blocks") + " "
                            if not _channel_flags.allow_code_blocks
                            else ""
                        )
                    ).strip(),
                    channel=msg.channel,
                )
            )
            log.debug(
                "channel_flags_applied",
                channel=msg.channel,
                compact=_channel_flags.compact_output,
                max_len=_channel_flags.max_response_length,
            )
    except Exception:
        log.debug("channel_flags_skipped", exc_info=True)

    # ── Autonomous Orchestration (complex/recurring tasks) ──
    auto_task = None
    if hasattr(gw, "_autonomous_orchestrator") and gw._autonomous_orchestrator.should_orchestrate(
        msg.text
    ):
        auto_task = gw._autonomous_orchestrator.create_task(msg.text, session.session_id)
        orchestration_context = gw._autonomous_orchestrator.get_orchestration_prompt(auto_task)
        wm.add_message(
            Message(
                role=MessageRole.SYSTEM,
                content=orchestration_context,
                channel=msg.channel,
            )
        )
        log.info("autonomous_orchestration_active", task_id=auto_task.task_id)

    all_results: list[ToolResult] = []
    all_plans: list[ActionPlan] = []
    all_audit: list[AuditEntry] = []

    # Hilfsfunktion: ToolEnforcer-State sicher aufraemen
    def _cleanup_skill_state() -> None:
        """Setzt active_skill und ToolEnforcer Call-Counter zurueck."""
        if hasattr(gw._gatekeeper, "set_active_skill"):
            gw._gatekeeper.set_active_skill(None)
        if (
            active_skill is not None
            and hasattr(gw._gatekeeper, "_tool_enforcer")
            and gw._gatekeeper._tool_enforcer is not None
            and hasattr(active_skill, "skill")
            and active_skill.skill is not None
        ):
            gw._gatekeeper._tool_enforcer.reset_call_count(active_skill.skill.slug)

    # Pipeline callback fuer Presearch + PGE-Loop
    # msg.session_id = WS-URL session_id (vom Client),
    # session.session_id = interner Gateway-Key.
    # Channels nutzen msg.session_id fuer Connection-Lookup.
    _pipeline_cb = gw._make_pipeline_callback(msg.channel, msg.session_id)

    try:
        if presearch_results:
            # Direktantwort aus Suchergebnissen generieren (PGE-Bypass)
            # Pipeline-Events auch im Presearch-Pfad senden
            await _pipeline_cb("iteration", "start", iteration=1)
            await _pipeline_cb("plan", "start", iteration=1)
            await _pipeline_cb(
                "plan",
                "done",
                iteration=1,
                has_actions=False,
                steps=0,
                presearch=True,
            )
            await _pipeline_cb("execute", "start", iteration=1, tools=["presearch"])
            final_response = await gw._answer_from_presearch(msg.text, presearch_results)
            if not final_response:
                await _pipeline_cb("execute", "done", iteration=1, success=0, failed=1, total_ms=0)
                # Fallback: normaler PGE-Loop wenn Antwort-Generierung fehlschlug
                if active_skill is not None and hasattr(gw._gatekeeper, "set_active_skill"):
                    gw._gatekeeper.set_active_skill(
                        active_skill.skill if hasattr(active_skill, "skill") else None,
                    )
                final_response, all_results, all_plans, all_audit = await gw._run_pge_loop(
                    msg,
                    session,
                    wm,
                    tool_schemas,
                    route_decision,
                    agent_workspace,
                    run_id,
                    stream_callback=stream_callback,
                    active_skill=active_skill,
                )
            else:
                await _pipeline_cb("execute", "done", iteration=1, success=1, failed=0, total_ms=0)
                await _pipeline_cb("complete", "done", iterations=1, tools_used=1)
                log.info("presearch_bypass_used", response_chars=len(final_response))
        else:
            # Phase 3: PGE-Loop (regulaerer Ablauf)
            # Community-Skill ToolEnforcer: Aktiven Skill an Gatekeeper weiterreichen
            if active_skill is not None and hasattr(gw._gatekeeper, "set_active_skill"):
                gw._gatekeeper.set_active_skill(
                    active_skill.skill if hasattr(active_skill, "skill") else None,
                )
            final_response, all_results, all_plans, all_audit = await gw._run_pge_loop(
                msg,
                session,
                wm,
                tool_schemas,
                route_decision,
                agent_workspace,
                run_id,
                stream_callback=stream_callback,
                active_skill=active_skill,
            )
    finally:
        _cleanup_skill_state()

    # Coding-Override aufraeumen
    if gw._model_router:
        gw._model_router.clear_coding_override()

    # ── Autonomous Task Evaluation ──
    if auto_task is not None:
        auto_task.quality_score = gw._autonomous_orchestrator.evaluate_result(
            auto_task, final_response, all_results
        )
        auto_task.status = (
            "completed"
            if auto_task.quality_score >= AutonomousOrchestrator.QUALITY_THRESHOLD
            else "needs_improvement"
        )
        log.info(
            "autonomous_task_evaluated",
            task_id=auto_task.task_id,
            quality=auto_task.quality_score,
            status=auto_task.status,
        )

    # User- und Antwort-Nachricht in Working Memory speichern (nach PGE-Loop)
    wm.add_message(Message(role=MessageRole.USER, content=msg.text, channel=msg.channel))

    # Persist important tool results as TOOL messages in chat history,
    # so follow-up requests have full context (e.g. vision text for PDF export)
    gw._persist_key_tool_results(wm, all_results)

    wm.add_message(Message(role=MessageRole.ASSISTANT, content=final_response))

    # ── ConversationTree: Store nodes for chat branching ──────
    if hasattr(gw, "_conversation_tree") and gw._conversation_tree and not session.incognito:
        try:
            # Create conversation if not yet assigned
            if not session.conversation_id:
                session.conversation_id = gw._conversation_tree.create_conversation(
                    title=msg.text[:60]
                )
            # Store user message node
            user_node_id = gw._conversation_tree.add_node(
                session.conversation_id,
                role="user",
                text=msg.text,
                parent_id=session.active_leaf_id or None,
                agent_name=agent_name,
            )
            # Store assistant response node
            asst_node_id = gw._conversation_tree.add_node(
                session.conversation_id,
                role="assistant",
                text=final_response,
                parent_id=user_node_id,
                agent_name=agent_name,
            )
            session.active_leaf_id = asst_node_id
            log.debug(
                "tree_nodes_stored",
                conv=session.conversation_id[:12],
                user_node=user_node_id[:12],
                asst_node=asst_node_id[:12],
            )
        except Exception:
            log.debug("tree_node_storage_failed", exc_info=True)

    # Phase 4: Reflexion, Skill-Tracking, Telemetry, Profiler, Run-Recording
    # Sum token counts across all plans
    _total_input = sum(getattr(p, "input_tokens", 0) for p in all_plans)
    _total_output = sum(getattr(p, "output_tokens", 0) for p in all_plans)
    _backend = getattr(gw._config, "llm_backend_type", "ollama") if gw._config else ""

    agent_result = AgentResult(
        response=final_response,
        plans=all_plans,
        tool_results=all_results,
        audit_entries=all_audit,
        total_iterations=session.iteration_count,
        total_duration_ms=int((time.monotonic() - _handle_start) * 1000),
        model_used=coding_model
        if is_coding
        else (gw._model_router.select_model("planning", "high") if gw._model_router else ""),
        input_tokens=_total_input,
        output_tokens=_total_output,
        backend_type=_backend,
        success=not any(r.is_error for r in all_results) if all_results else True,
    )
    # Post-processing (reflection, skill tracking, telemetry) runs in background
    # so handle_message returns the response immediately without blocking on
    # the 30-60s reflection LLM call.
    import asyncio as _aio_pp

    _pp_task = _aio_pp.create_task(
        gw._run_post_processing(session, wm, agent_result, active_skill, run_id)
    )
    gw._background_tasks.add(_pp_task)
    _pp_task.add_done_callback(gw._background_tasks.discard)

    # Complete explainability trail
    if getattr(gw, "_explainability", None) and _expl_trail_id:
        try:
            gw._explainability.complete_trail(_expl_trail_id)
        except Exception:
            log.debug("explainability_complete_failed", exc_info=True)

    # Phase 5: Session persistieren
    await gw._persist_session(session, wm)

    # Prometheus: Request-Dauer und Token-Metriken
    _duration_ms = (time.monotonic() - _handle_start) * 1000
    gw._record_metric("request_duration_ms", _duration_ms, channel=msg.channel)
    _model_used = agent_result.model_used or ""
    if _model_used:
        gw._record_metric("tokens_used_total", 1, model=_model_used, role="request")

    # Extract attachments from tool results (e.g. document_export)
    attachments = gw._extract_attachments(all_results)

    # Notify active learner of user activity (resets idle timer)
    if gw._active_learner is not None:
        gw._active_learner.notify_activity()

    if hasattr(gw, "_idle_detector") and gw._idle_detector:
        gw._idle_detector.notify_activity()

    # Build metadata with token/model info for the UI
    _meta: dict[str, Any] = {}
    if agent_result.input_tokens or agent_result.output_tokens:
        _meta["input_tokens"] = agent_result.input_tokens
        _meta["output_tokens"] = agent_result.output_tokens
    if agent_result.model_used:
        _meta["model"] = agent_result.model_used
    if agent_result.backend_type:
        _meta["backend"] = agent_result.backend_type
    if agent_result.total_duration_ms:
        _meta["duration_ms"] = agent_result.total_duration_ms

    return OutgoingMessage(
        channel=msg.channel,
        text=final_response,
        session_id=session.session_id,
        is_final=True,
        attachments=attachments,
        metadata=_meta,
    )


# ── handle_message sub-methods ────────────────────────────────


async def resolve_agent_route(
    gw: Gateway,
    msg: IncomingMessage,
) -> tuple[RouteDecision | None, SessionContext, WorkingMemory, Any, Any, str, str | None]:
    """Phase 1: Agent-Routing, Session, Working Memory, Skills, Workspace."""
    route_decision = None
    agent_workspace = None
    agent_name = "jarvis"

    if gw._agent_router is not None:
        target_agent = msg.metadata.get("target_agent")
        if target_agent:
            target_profile = gw._agent_router.get_agent(target_agent)
            if target_profile:
                route_decision = RouteDecision(
                    agent=target_profile,
                    confidence=1.0,
                    reason=f"Explicit target: {target_agent}",
                )
                log.info(
                    "agent_explicit_target",
                    agent=target_agent,
                    source=msg.metadata.get("cron_job", "delegation"),
                )

        if route_decision is None:
            from cognithor.core.bindings import MessageContext as _MsgCtx

            msg_context = _MsgCtx.from_incoming(msg)
            route_decision = gw._agent_router.route(
                msg.text,
                context=msg_context,
            )

        agent_name = route_decision.agent.name

    session = gw._get_or_create_session(msg.channel, msg.user_id, agent_name)
    session.touch()
    session.reset_iteration()

    wm = gw._get_or_create_working_memory(session)
    wm.clear_for_new_request()

    # Route image/video attachments to the VLM for this turn. Cleared by
    # clear_for_new_request() so it only affects the current turn.
    if msg.attachments:
        images, video_path, rejected_videos = _classify_attachments(msg.attachments)
        wm.image_attachments = images
        if video_path is not None:
            try:
                wm.video_attachment = await asyncio.to_thread(
                    _build_video_attachment, video_path, gw._config
                )
                uuid = _extract_uuid_from_path(video_path)
                if uuid is not None and gw._video_cleanup is not None:
                    gw._video_cleanup.register_upload(uuid, session.session_id)
            except Exception:
                log.warning("video_attachment_build_failed", path=video_path, exc_info=True)
        if rejected_videos:
            log.warning(
                "video_validation_extras_rejected",
                session_id=session.session_id,
                rejected=rejected_videos,
            )
    else:
        wm.image_attachments = []

    # Start explainability trail for this request
    _expl_trail_id: str | None = None
    if getattr(gw, "_explainability", None) is not None:
        try:
            _trail = gw._explainability.start_trail(
                request_id=session.session_id,
                agent_id=agent_name,
            )
            _expl_trail_id = _trail.trail_id
        except Exception:
            log.debug("explainability_start_failed", exc_info=True)

    if gw._audit_logger:
        gw._audit_logger.log_user_input(
            msg.channel,
            msg.text[:100],
            agent_name=agent_name,
        )

    if route_decision and route_decision.agent.system_prompt:
        wm.add_message(
            Message(
                role=MessageRole.SYSTEM,
                content=route_decision.agent.system_prompt,
                channel=msg.channel,
            )
        )

    # Gap Detection: detect explicit tool/skill creation requests
    if hasattr(gw, "_skill_generator") and gw._skill_generator:
        _lower = msg.text.lower()
        _tool_request_triggers = (
            "erstelle ein tool",
            "erstelle einen skill",
            "baue ein tool",
            "create a tool",
            "build a tool",
            "neues tool",
            "neuer skill",
            "tool erstellen",
            "skill erstellen",
            "ich brauche ein tool",
            "kannst du ein tool",
            "mach ein tool",
        )
        for trigger in _tool_request_triggers:
            if trigger in _lower:
                gw._skill_generator.gap_detector.report_user_request(
                    msg.text[:200],
                    context=msg.text,
                )
                break

    active_skill = None
    if gw._skill_registry is not None:
        try:
            tool_list = gw._mcp_client.get_tool_list() if gw._mcp_client else []
            active_skill = gw._skill_registry.inject_into_working_memory(
                msg.text,
                wm,
                available_tools=tool_list,
            )
            # Gap Detection: Melde wenn kein Skill zur Anfrage passt
            if active_skill is None and hasattr(gw, "_skill_generator") and gw._skill_generator:
                gw._skill_generator.gap_detector.report_no_skill_match(msg.text)
        except Exception as exc:
            log.debug("skill_match_error", error=str(exc))

    if gw._agent_router is not None and route_decision:
        agent_workspace = gw._agent_router.resolve_agent_workspace(
            route_decision.agent.name,
            gw._config.workspace_dir,
        )
        log.debug(
            "agent_workspace_resolved",
            agent=route_decision.agent.name,
            workspace=str(agent_workspace),
            shared=route_decision.agent.shared_workspace,
        )

    return (
        route_decision,
        session,
        wm,
        active_skill,
        agent_workspace,
        agent_name,
        _expl_trail_id,
    )


async def prepare_execution_context(
    gw: Gateway,
    msg: IncomingMessage,
    session: SessionContext,
    wm: WorkingMemory,
    route_decision: RouteDecision | None,
) -> tuple[str | None, OutgoingMessage | None]:
    """Phase 2: Profiler, Budget, Run-Recording, Policy-Snapshot.

    Returns:
        (run_id, budget_response) -- budget_response is not None if budget exceeded.
    """
    if hasattr(gw, "_task_profiler") and gw._task_profiler:
        try:
            gw._task_profiler.start_task(
                session_id=session.session_id,
                task_description=msg.text[:200],
            )
        except Exception:
            log.debug("task_profiler_start_failed", exc_info=True)

    if hasattr(gw, "_cost_tracker") and gw._cost_tracker:
        try:
            budget = gw._cost_tracker.check_budget()
            if not budget.ok:
                return None, OutgoingMessage(
                    channel=msg.channel,
                    text=t("gateway.budget_limit_reached", warning=budget.warning),
                    session_id=session.session_id,
                    is_final=True,
                )
        except Exception:
            log.debug("budget_check_failed", exc_info=True)

    run_id = None
    if hasattr(gw, "_run_recorder") and gw._run_recorder:
        try:
            run_id = gw._run_recorder.start_run(
                session_id=session.session_id,
                user_message=msg.text[:500],
                operation_mode=str(getattr(gw._config, "resolved_operation_mode", "")),
            )
        except Exception:
            log.debug("run_recorder_start_failed", exc_info=True)

    if run_id and gw._run_recorder and gw._gatekeeper:
        try:
            policies = gw._gatekeeper.get_policies()
            if policies:
                gw._run_recorder.record_policy_snapshot(
                    run_id, {"rules": [r.model_dump() for r in policies]}
                )
        except Exception:
            log.debug("run_recorder_policy_snapshot_failed", exc_info=True)

    return run_id, None


def make_status_callback(
    gw: Gateway,
    channel_name: str,
    session_id: str,
) -> Any:
    """Creates a fire-and-forget status callback for the current channel.

    Returns an async callable (status_type: str, text: str) -> None.
    """

    async def _send_status(status_type: str, text: str) -> None:
        channel = gw._channels.get(channel_name)
        if channel is None:
            return
        try:
            from cognithor.channels.base import StatusType

            try:
                st = StatusType(status_type) if isinstance(status_type, str) else status_type
            except ValueError:
                st = StatusType.PROCESSING
            await asyncio.wait_for(
                channel.send_status(session_id, st, text),
                timeout=2.0,
            )
        except Exception:
            log.debug("status_send_failed", exc_info=True)  # fire-and-forget

    return _send_status


def make_pipeline_callback(
    gw: Gateway,
    channel_name: str,
    session_id: str,
) -> Any:
    """Creates a fire-and-forget pipeline event callback.

    Returns an async callable for sending structured PGE pipeline
    events to the frontend for the live pipeline visualization.
    """
    _start_mono = time.monotonic()

    async def _send_pipeline(phase: str, status: str, **extra: Any) -> None:
        channel = gw._channels.get(channel_name)
        if channel is None or not hasattr(channel, "send_pipeline_event"):
            return
        try:
            await asyncio.wait_for(
                channel.send_pipeline_event(
                    session_id,
                    {
                        "phase": phase,
                        "status": status,
                        "elapsed_ms": int((time.monotonic() - _start_mono) * 1000),
                        **extra,
                    },
                ),
                timeout=2.0,
            )
        except Exception:
            log.debug("pipeline_event_send_failed", exc_info=True)

    return _send_pipeline


async def formulate_response(
    gw: Gateway,
    msg_text: str,
    all_results: list[ToolResult],
    wm: WorkingMemory,
    stream_callback: Any | None = None,
) -> ResponseEnvelope:
    """Formulate response, optionally streaming tokens to the client.

    Non-streaming path uses ``run_pge_with_observer_directive`` so Observer
    ``pge_reloop`` directives trigger a Gateway-level re-entry.
    Streaming path delegates to ``formulate_response_stream`` directly; the
    Planner's internal observer loop still runs, but PGE-reloop directives
    from the streaming path are not currently re-entered (limitation).
    """
    if stream_callback is not None and hasattr(gw._planner, "formulate_response_stream"):
        try:
            return await gw._planner.formulate_response_stream(
                user_message=msg_text,
                results=all_results,
                working_memory=wm,
                stream_callback=stream_callback,
            )
        except Exception:
            log.debug("streaming_formulate_failed_fallback", exc_info=True)
            # Fall through to non-streaming
    # Look up via the gateway module namespace so monkeypatches on
    # `gateway_module.run_pge_with_observer_directive` (used by
    # `tests/test_integration/test_observer_flow.py`) intercept the call.
    from cognithor.gateway import gateway as _gw_mod

    return await _gw_mod.run_pge_with_observer_directive(
        planner=gw._planner,
        user_message=msg_text,
        results=all_results,
        working_memory=wm,
        session_state=wm.session_state,
        config=gw._config,
    )
