"""Gateway: Central entry point and agent loop.

The Gateway:
  - Receives messages from all channels
  - Manages sessions
  - Orchestrates the PGE cycle (Plan -> Gate -> Execute -> Replan)
  - Returns responses to channels
  - Starts and stops all subsystems

Bible reference: §9.1 (Gateway), §3.4 (Complete cycle)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json as _json
import signal
import time
from typing import TYPE_CHECKING, Any

from jarvis.config import JarvisConfig, load_config
from jarvis.core.agent_router import RouteDecision
from jarvis.gateway.phases import (
    apply_phase,
    declare_advanced_attrs,
    declare_agents_attrs,
    declare_compliance_attrs,
    declare_core_attrs,
    declare_memory_attrs,
    declare_pge_attrs,
    declare_security_attrs,
    declare_tools_attrs,
    init_advanced,
    init_agents,
    init_core,
    init_memory,
    init_pge,
    init_security,
    init_tools,
)
from jarvis.mcp.client import JarvisMCPClient
from jarvis.models import (
    ActionPlan,
    AgentResult,
    AuditEntry,
    GateDecision,
    GateStatus,
    IncomingMessage,
    Message,
    MessageRole,
    OutgoingMessage,
    SessionContext,
    ToolResult,
    WorkingMemory,
)

from jarvis.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from jarvis.channels.base import Channel

log = get_logger(__name__)


class Gateway:
    """Central entry point. Connects all Jarvis subsystems. [B§9.1]"""

    # Session TTL: sessions older than 24 hours are considered stale
    _SESSION_TTL_SECONDS: float = 24 * 60 * 60  # 24h
    # Minimum interval between stale-session cleanup sweeps
    _CLEANUP_INTERVAL_SECONDS: float = 60 * 60  # 1h

    def __init__(self, config: JarvisConfig | None = None) -> None:
        """Initialisiert das Gateway mit PGE-Trinität, MCP-Client und Memory."""
        self._config = config or load_config()
        self._channels: dict[str, Channel] = {}
        self._sessions: dict[str, SessionContext] = {}
        self._working_memories: dict[str, WorkingMemory] = {}
        self._session_last_accessed: dict[str, float] = {}
        self._last_session_cleanup: float = time.monotonic()
        self._running = False

        # Declare all subsystem attributes via phase modules
        apply_phase(self, declare_core_attrs(self._config))
        apply_phase(self, declare_security_attrs(self._config))
        apply_phase(self, declare_tools_attrs(self._config))
        apply_phase(self, declare_memory_attrs(self._config))
        apply_phase(self, declare_pge_attrs(self._config))
        apply_phase(self, declare_agents_attrs(self._config))
        apply_phase(self, declare_compliance_attrs(self._config))
        apply_phase(self, declare_advanced_attrs(self._config))

    async def initialize(self) -> None:
        """Initialisiert alle Subsysteme in der richtigen Reihenfolge.

        Dependency graph (→ = depends on):
          core        (independent)
          security    → core (_llm)
          memory      → security (_audit_logger)
          tools       → memory, core (_memory_manager, _interop)
          pge         → core, security, tools (_llm, _model_router, _mcp_client, ...)
          agents      → memory, tools, security (_memory_manager, _mcp_client, ...)

        Independent phases are run in parallel via asyncio.gather where possible.
        """
        # 1. Logging
        setup_logging(
            level=self._config.log_level,
            log_dir=self._config.logs_dir,
        )
        log.info("gateway_init_start", version=self._config.version)

        # 2. Verzeichnisse sicherstellen
        self._config.ensure_directories()
        self._config.ensure_default_files()

        # --- Phase A: Core (independent) ---
        core_result = await init_core(self._config)
        llm_ok = core_result.pop("__llm_ok", False)
        apply_phase(self, core_result)

        # --- Phase B: Security (depends on core for _llm) ---
        security_result = await init_security(self._config, llm_backend=self._llm)
        apply_phase(self, security_result)

        # --- Phase C: Memory (depends on security for _audit_logger) ---
        memory_result = await init_memory(self._config, audit_logger=self._audit_logger)
        apply_phase(self, memory_result)

        # --- Phase D: Tools (depends on memory + core) ---
        mcp_client = JarvisMCPClient(self._config)
        tools_result = await init_tools(
            self._config,
            mcp_client=mcp_client,
            memory_manager=self._memory_manager,
            interop=self._interop,
            handle_message=self.handle_message,
        )
        apply_phase(self, tools_result)

        # --- Phase E: PGE + Agents in parallel (both depend on phases A-D) ---
        pge_coro = init_pge(
            self._config,
            llm=self._llm,
            model_router=self._model_router,
            mcp_client=self._mcp_client,
            runtime_monitor=self._runtime_monitor,
            audit_logger=self._audit_logger,
            memory_manager=self._memory_manager,
            cost_tracker=self._cost_tracker,
        )
        agents_coro = init_agents(
            self._config,
            memory_manager=self._memory_manager,
            mcp_client=self._mcp_client,
            audit_logger=self._audit_logger,
            jarvis_home=self._config.jarvis_home,
            handle_message=self.handle_message,
            heartbeat_config=self._config.heartbeat,
        )
        pge_result, agents_result = await asyncio.gather(pge_coro, agents_coro)
        apply_phase(self, pge_result)
        apply_phase(self, agents_result)

        # --- Phase F: Advanced (depends on PGE + tools) ---
        advanced_result = await init_advanced(
            self._config,
            task_telemetry=self._task_telemetry,
            error_clusterer=self._error_clusterer,
            task_profiler=self._task_profiler,
            cost_tracker=self._cost_tracker,
            run_recorder=self._run_recorder,
            gatekeeper=self._gatekeeper,
        )
        apply_phase(self, advanced_result)

        # Governance-Cron-Job registrieren (taeglich um 02:00)
        if self._cron_engine and hasattr(self, "_governance_agent") and self._governance_agent:
            try:
                from jarvis.cron.jobs import governance_analysis
                self._cron_engine.add_system_job(
                    name="governance_analysis",
                    schedule="0 2 * * *",
                    callback=governance_analysis,
                    args=[self],
                )
            except Exception:
                log.debug("governance_cron_registration_skipped", exc_info=True)

        log.info(
            "gateway_init_complete",
            ollama_available=llm_ok,
            tools=self._mcp_client.get_tool_list(),
            cron_jobs=self._cron_engine.job_count if self._cron_engine else 0,
        )

        # Audit: System-Start protokollieren
        if self._audit_logger:
            self._audit_logger.log_system(
                "startup",
                description=f"Jarvis gestartet (LLM={llm_ok}, Tools={len(self._mcp_client.get_tool_list())})",
            )

    def register_channel(self, channel: Channel) -> None:
        """Registriert einen Kommunikationskanal."""
        self._channels[channel.name] = channel
        log.info("channel_registered", channel=channel.name)

    async def start(self) -> None:
        """Startet den Gateway und alle Channels + Cron."""
        self._running = True

        # Signal-Handler für Graceful Shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, OSError):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # Cron-Engine starten (wenn konfiguriert)
        if self._cron_engine and self._cron_engine.has_enabled_jobs:
            await self._cron_engine.start()
            log.info("cron_engine_started", jobs=self._cron_engine.job_count)

        # MCP-Server starten (OPTIONAL — nur wenn Bridge aktiviert)
        if self._mcp_bridge and self._mcp_bridge.enabled:
            try:
                await self._mcp_bridge.start()
            except Exception as exc:
                log.warning("mcp_bridge_start_failed", error=str(exc))

        # A2A-Server starten (OPTIONAL)
        if self._a2a_adapter and self._a2a_adapter.enabled:
            try:
                await self._a2a_adapter.start()
                # A2A HTTP-Routes in WebUI-App registrieren
                for channel in self._channels.values():
                    if hasattr(channel, "app") and channel.app is not None:
                        try:
                            from jarvis.a2a.http_handler import A2AHTTPHandler
                            a2a_http = A2AHTTPHandler(self._a2a_adapter)
                            a2a_http.register_routes(channel.app)
                        except Exception as exc:
                            log.debug("a2a_http_routes_skip", error=str(exc))
            except Exception as exc:
                log.warning("a2a_adapter_start_failed", error=str(exc))

        # Channels starten
        tasks = []
        for channel in self._channels.values():
            task = asyncio.create_task(
                channel.start(self.handle_message),
                name=f"channel-{channel.name}",
            )
            tasks.append(task)

        if tasks:
            # Warte bis alle Channels beendet sind
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            log.warning("no_channels_registered")

    async def shutdown(self) -> None:
        """Fährt den Gateway sauber herunter mit Session-Persistierung."""
        log.info("gateway_shutdown_start")
        self._running = False

        # Audit log BEFORE closing resources
        if self._audit_logger:
            self._audit_logger.log_system("shutdown", description="Jarvis heruntergefahren")

        # Cron-Engine stoppen
        if self._cron_engine:
            await self._cron_engine.stop()

        # Channels stoppen
        for channel in self._channels.values():
            try:
                await channel.stop()
            except Exception as exc:
                log.warning("channel_stop_error", channel=channel.name, error=str(exc))

        # Sessions persistieren
        if self._session_store:
            saved_count = 0
            for _key, session in self._sessions.items():
                try:
                    self._session_store.save_session(session)
                    # Chat-History speichern
                    wm = self._working_memories.get(session.session_id)
                    if wm and wm.chat_history:
                        self._session_store.save_chat_history(
                            session.session_id,
                            wm.chat_history,
                        )
                    saved_count += 1
                except Exception as exc:
                    log.warning(
                        "session_save_error",
                        session=session.session_id[:8],
                        error=str(exc),
                    )
            log.info("sessions_persisted", count=saved_count)
            self._session_store.close()

        # Memory-Manager schließen
        if hasattr(self, "_memory_manager") and self._memory_manager:
            try:
                await self._memory_manager.close()
            except Exception as exc:
                log.warning("memory_close_error", error=str(exc))

        # A2A-Adapter stoppen (optional)
        if self._a2a_adapter:
            try:
                await self._a2a_adapter.stop()
            except Exception:
                log.debug("a2a_adapter_stop_skipped", exc_info=True)

        # Browser-Agent stoppen (optional)
        if self._browser_agent:
            try:
                await self._browser_agent.stop()
            except Exception:
                log.debug("browser_agent_stop_skipped", exc_info=True)

        # MCP-Bridge stoppen (optional)
        if self._mcp_bridge:
            try:
                await self._mcp_bridge.stop()
            except Exception:
                log.debug("mcp_bridge_stop_skipped", exc_info=True)

        # CostTracker schliessen
        if hasattr(self, "_cost_tracker") and self._cost_tracker:
            try:
                self._cost_tracker.close()
            except Exception:
                log.debug("cost_tracker_close_skipped", exc_info=True)

        # RunRecorder schliessen
        if hasattr(self, "_run_recorder") and self._run_recorder:
            try:
                self._run_recorder.close()
            except Exception:
                log.debug("run_recorder_close_skipped", exc_info=True)

        # GovernanceAgent schliessen
        if hasattr(self, "_governance_agent") and self._governance_agent:
            try:
                self._governance_agent.close()
            except Exception:
                log.debug("governance_agent_close_skipped", exc_info=True)

        # MCP-Client trennen
        if self._mcp_client:
            await self._mcp_client.disconnect_all()

        # Ollama-Client schließen
        if self._llm:
            await self._llm.close()

        log.info("gateway_shutdown_complete")

    async def handle_message(self, msg: IncomingMessage) -> OutgoingMessage:
        """Verarbeitet eine eingehende Nachricht. [B§3.4]

        Orchestriert den PGE-Zyklus (Plan → Gate → Execute → Replan).

        Returns:
            OutgoingMessage mit der Jarvis-Antwort.
        """
        _handle_start = time.monotonic()

        # Phase 1: Agent-Routing, Session, WM, Skills, Workspace
        route_decision, session, wm, active_skill, agent_workspace, agent_name = \
            await self._resolve_agent_route(msg)

        # Phase 2: Profiler, Budget, Run-Recording, Policy-Snapshot
        run_id, budget_response = await self._prepare_execution_context(
            msg, session, wm, route_decision,
        )
        if budget_response is not None:
            return budget_response

        # Tool-Schemas (gefiltert nach Agent-Rechten)
        tool_schemas = self._mcp_client.get_tool_schemas() if self._mcp_client else {}
        if route_decision and route_decision.agent.has_tool_restrictions:
            tool_schemas = route_decision.agent.filter_tools(tool_schemas)

        # Subsystem checks
        if self._planner is None or self._gatekeeper is None or self._executor is None:
            raise RuntimeError("Gateway.initialize() must be called before handle_message()")

        # Phase 3: PGE-Loop
        final_response, all_results, all_plans, all_audit = await self._run_pge_loop(
            msg, session, wm, tool_schemas, route_decision, agent_workspace, run_id,
        )

        # User- und Antwort-Nachricht in Working Memory speichern (nach PGE-Loop)
        wm.add_message(Message(role=MessageRole.USER, content=msg.text, channel=msg.channel))
        wm.add_message(Message(role=MessageRole.ASSISTANT, content=final_response))

        # Phase 4: Reflexion, Skill-Tracking, Telemetry, Profiler, Run-Recording
        agent_result = AgentResult(
            response=final_response,
            plans=all_plans,
            tool_results=all_results,
            audit_entries=all_audit,
            total_iterations=session.iteration_count,
            total_duration_ms=int((time.monotonic() - _handle_start) * 1000),
            model_used=self._model_router.select_model("planning", "high")
            if self._model_router
            else "",
            success=not any(r.is_error for r in all_results) if all_results else True,
        )
        await self._run_post_processing(session, wm, agent_result, active_skill, run_id)

        # Phase 5: Session persistieren
        await self._persist_session(session, wm)

        return OutgoingMessage(
            channel=msg.channel,
            text=final_response,
            session_id=session.session_id,
            is_final=True,
        )

    # ── handle_message sub-methods ────────────────────────────────

    async def _resolve_agent_route(
        self, msg: IncomingMessage,
    ) -> tuple[RouteDecision | None, "SessionContext", "WorkingMemory", Any, Any, str]:
        """Phase 1: Agent-Routing, Session, Working Memory, Skills, Workspace."""
        route_decision = None
        agent_workspace = None
        agent_name = "jarvis"

        if self._agent_router is not None:
            target_agent = msg.metadata.get("target_agent")
            if target_agent:
                target_profile = self._agent_router.get_agent(target_agent)
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
                from jarvis.core.bindings import MessageContext as _MsgCtx
                msg_context = _MsgCtx.from_incoming(msg)
                route_decision = self._agent_router.route(
                    msg.text, context=msg_context,
                )

            agent_name = route_decision.agent.name

        session = self._get_or_create_session(msg.channel, msg.user_id, agent_name)
        session.touch()
        session.reset_iteration()

        wm = self._get_or_create_working_memory(session)
        wm.clear_for_new_request()

        if self._audit_logger:
            self._audit_logger.log_user_input(
                msg.channel, msg.text[:100],
                agent_name=agent_name,
            )

        if route_decision and route_decision.agent.system_prompt:
            wm.add_message(Message(
                role=MessageRole.SYSTEM,
                content=route_decision.agent.system_prompt,
                channel=msg.channel,
            ))

        active_skill = None
        if self._skill_registry is not None:
            try:
                tool_list = self._mcp_client.get_tool_list() if self._mcp_client else []
                active_skill = self._skill_registry.inject_into_working_memory(
                    msg.text, wm, available_tools=tool_list,
                )
            except Exception as exc:
                log.debug("skill_match_error", error=str(exc))

        if self._agent_router is not None and route_decision:
            agent_workspace = self._agent_router.resolve_agent_workspace(
                route_decision.agent.name,
                self._config.workspace_dir,
            )
            log.debug(
                "agent_workspace_resolved",
                agent=route_decision.agent.name,
                workspace=str(agent_workspace),
                shared=route_decision.agent.shared_workspace,
            )

        return route_decision, session, wm, active_skill, agent_workspace, agent_name

    async def _prepare_execution_context(
        self,
        msg: IncomingMessage,
        session: "SessionContext",
        wm: "WorkingMemory",
        route_decision: RouteDecision | None,
    ) -> tuple[str | None, OutgoingMessage | None]:
        """Phase 2: Profiler, Budget, Run-Recording, Policy-Snapshot.

        Returns:
            (run_id, budget_response) — budget_response is not None if budget exceeded.
        """
        if hasattr(self, "_task_profiler") and self._task_profiler:
            try:
                self._task_profiler.start_task(
                    session_id=session.session_id,
                    task_description=msg.text[:200],
                )
            except Exception:
                pass

        if hasattr(self, "_cost_tracker") and self._cost_tracker:
            try:
                budget = self._cost_tracker.check_budget()
                if not budget.ok:
                    return None, OutgoingMessage(
                        channel=msg.channel,
                        text=f"Budget-Limit erreicht: {budget.warning}",
                        session_id=session.session_id,
                        is_final=True,
                    )
            except Exception:
                log.debug("budget_check_failed", exc_info=True)

        run_id = None
        if hasattr(self, "_run_recorder") and self._run_recorder:
            try:
                run_id = self._run_recorder.start_run(
                    session_id=session.session_id,
                    user_message=msg.text[:500],
                    operation_mode=str(getattr(self._config, "resolved_operation_mode", "")),
                )
            except Exception:
                log.debug("run_recorder_start_failed", exc_info=True)

        if run_id and self._run_recorder and self._gatekeeper:
            try:
                policies = self._gatekeeper.get_policies()
                if policies:
                    self._run_recorder.record_policy_snapshot(
                        run_id, {"rules": [r.model_dump() for r in policies]}
                    )
            except Exception:
                log.debug("run_recorder_policy_snapshot_failed", exc_info=True)

        return run_id, None

    async def _run_pge_loop(
        self,
        msg: IncomingMessage,
        session: "SessionContext",
        wm: "WorkingMemory",
        tool_schemas: dict[str, Any],
        route_decision: RouteDecision | None,
        agent_workspace: Any,
        run_id: str | None,
    ) -> tuple[str, list[ToolResult], list[ActionPlan], list[AuditEntry]]:
        """Phase 3: Plan → Gate → Execute Loop.

        Returns:
            (final_response, all_results, all_plans, all_audit)
        """
        all_results: list[ToolResult] = []
        all_plans: list[ActionPlan] = []
        all_audit: list[AuditEntry] = []
        final_response = ""

        while not session.iterations_exhausted and self._running:
            session.iteration_count += 1

            log.info(
                "agent_loop_iteration",
                iteration=session.iteration_count,
                session=session.session_id[:8],
            )

            # Planner
            if session.iteration_count == 1:
                plan = await self._planner.plan(
                    user_message=msg.text,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                )
            else:
                plan = await self._planner.replan(
                    original_goal=msg.text,
                    results=all_results,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                )

            all_plans.append(plan)

            if run_id and self._run_recorder:
                try:
                    self._run_recorder.record_plan(run_id, plan)
                except Exception:
                    pass

            # Direkte Antwort
            if not plan.has_actions and plan.direct_response:
                final_response = plan.direct_response
                break

            if not plan.has_actions:
                final_response = (
                    "Ich konnte keinen Plan erstellen. Kannst du deine Frage umformulieren?"
                )
                break

            # Gatekeeper
            decisions = self._gatekeeper.evaluate_plan(plan.steps, session)

            for step, decision in zip(plan.steps, decisions, strict=False):
                params_hash = hashlib.sha256(
                    _json.dumps(step.params, sort_keys=True, default=str).encode()
                ).hexdigest()
                all_audit.append(AuditEntry(
                    session_id=session.session_id,
                    action_tool=step.tool,
                    action_params_hash=params_hash,
                    decision_status=decision.status,
                    decision_reason=decision.reason,
                ))

            # Approvals
            approved_decisions = await self._handle_approvals(
                plan.steps, decisions, session, msg.channel,
            )

            all_blocked = all(d.status == GateStatus.BLOCK for d in approved_decisions)
            if all_blocked:
                for step, decision in zip(plan.steps, approved_decisions, strict=False):
                    block_count = session.record_block(step.tool)
                    if block_count >= 3:
                        escalation = await self._planner.generate_escalation(
                            tool=step.tool,
                            reason=decision.reason,
                            working_memory=wm,
                        )
                        final_response = escalation
                        break
                else:
                    final_response = "Alle geplanten Aktionen wurden vom Gatekeeper blockiert."
                break

            # Executor
            if route_decision and route_decision.agent.name != "jarvis":
                self._executor.set_agent_context(
                    workspace_dir=str(agent_workspace) if agent_workspace else None,
                    sandbox_overrides=route_decision.agent.get_sandbox_config(),
                    agent_name=route_decision.agent.name,
                    session_id=session.session_id,
                )
            else:
                self._executor.set_agent_context(session_id=session.session_id)

            try:
                results = await self._executor.execute(plan.steps, approved_decisions)
            finally:
                self._executor.clear_agent_context()

            if run_id and self._run_recorder:
                try:
                    self._run_recorder.record_gate_decisions(run_id, approved_decisions)
                    self._run_recorder.record_tool_results(run_id, results)
                except Exception:
                    pass

            all_results.extend(results)

            for result in results:
                all_audit.append(AuditEntry(
                    session_id=session.session_id,
                    action_tool=result.tool_name,
                    action_params_hash="",
                    decision_status=GateStatus.ALLOW,
                    decision_reason=f"executed success={result.success}",
                    execution_result="ok" if result.success else result.error_message or "error",
                ))

            for result in results:
                wm.add_tool_result(result)

            has_errors = any(r.is_error for r in results)
            has_success = any(r.success for r in results)

            if has_success and not has_errors:
                final_response = await self._planner.formulate_response(
                    user_message=msg.text,
                    results=all_results,
                    working_memory=wm,
                )
                break

            if not has_success and session.iteration_count >= 3:
                final_response = await self._planner.formulate_response(
                    user_message=msg.text,
                    results=all_results,
                    working_memory=wm,
                )
                break

        if session.iterations_exhausted and not final_response:
            final_response = (
                "Ich habe das Iterationslimit erreicht, ohne die Aufgabe abzuschließen. "
                "Bitte versuche es mit einer spezifischeren Anfrage."
            )

        return final_response, all_results, all_plans, all_audit

    async def _run_post_processing(
        self,
        session: "SessionContext",
        wm: "WorkingMemory",
        agent_result: AgentResult,
        active_skill: Any,
        run_id: str | None,
    ) -> None:
        """Phase 4: Reflection, Skill-Tracking, Telemetry, Profiler, Run-Recording."""
        if self._reflector and self._reflector.should_reflect(agent_result):
            try:
                reflection = await self._reflector.reflect(session, wm, agent_result)
                agent_result.reflection = reflection
                log.info(
                    "reflection_done",
                    session=session.session_id[:8],
                    score=reflection.success_score,
                )
                if run_id and self._run_recorder:
                    try:
                        self._run_recorder.record_reflection(run_id, reflection)
                    except Exception:
                        pass
            except Exception as exc:
                log.error("reflection_error", error=str(exc))

        if active_skill and self._skill_registry:
            try:
                success = agent_result.success
                score = (
                    agent_result.reflection.success_score
                    if agent_result.reflection
                    else (0.8 if success else 0.3)
                )
                self._skill_registry.record_usage(
                    active_skill.skill.slug, success=success, score=score,
                )
            except Exception:
                log.debug("skill_usage_tracking_skipped", exc_info=True)

        if hasattr(self, "_task_telemetry") and self._task_telemetry:
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
                self._task_telemetry.record_task(
                    session_id=session.session_id,
                    success=agent_result.success,
                    duration_ms=float(agent_result.total_duration_ms),
                    tool_calls=tools_used,
                    error_type=error_type,
                    error_message=error_msg,
                )
            except Exception:
                log.debug("task_telemetry_record_failed", exc_info=True)

        if hasattr(self, "_task_profiler") and self._task_profiler:
            try:
                score = (
                    agent_result.reflection.success_score
                    if agent_result.reflection
                    else (0.8 if agent_result.success else 0.3)
                )
                self._task_profiler.finish_task(
                    session_id=session.session_id,
                    success_score=score,
                )
            except Exception:
                log.debug("task_profiler_finish_failed", exc_info=True)

        if run_id and hasattr(self, "_run_recorder") and self._run_recorder:
            try:
                self._run_recorder.finish_run(
                    run_id,
                    success=agent_result.success,
                    final_response=agent_result.response[:500],
                )
            except Exception:
                log.debug("run_recorder_finish_failed", exc_info=True)

    async def _persist_session(
        self, session: "SessionContext", wm: "WorkingMemory",
    ) -> None:
        """Phase 5: Session persistieren."""
        if self._session_store:
            try:
                self._session_store.save_session(session)
                self._session_store.save_chat_history(
                    session.session_id,
                    wm.chat_history,
                )
            except Exception as exc:
                log.warning("session_persist_error", error=str(exc))

    # =========================================================================
    # Agent-zu-Agent Delegation
    # =========================================================================

    async def execute_delegation(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        session: "SessionContext",
        parent_wm: "WorkingMemory",
    ) -> str:
        """Führt eine echte Agent-zu-Agent-Delegation aus.

        Der delegierte Agent bekommt:
          - Eigenen System-Prompt
          - Eigenen Workspace (isoliert)
          - Eigene Sandbox-Config
          - Eigene Tool-Filterung
          - Die Aufgabe als User-Nachricht

        Das Ergebnis fließt als Text zurück zum aufrufenden Agenten.

        Args:
            from_agent: Name des delegierenden Agenten.
            to_agent: Name des Ziel-Agenten.
            task: Die delegierte Aufgabe.
            session: Aktuelle Session.
            parent_wm: Working Memory des Eltern-Agenten.

        Returns:
            Ergebnis-Text der Delegation.
        """
        if not self._agent_router:
            return f"Agent-Router nicht verfügbar. Delegation an {to_agent} fehlgeschlagen."

        # Delegation erstellen und validieren
        delegation = self._agent_router.create_delegation(from_agent, to_agent, task)
        if delegation is None:
            return (
                f"Delegation von {from_agent} an {to_agent} nicht erlaubt. "
                f"Ich bearbeite die Aufgabe selbst."
            )

        target = delegation.target_profile
        if not target:
            return f"Agent {to_agent} nicht gefunden."

        log.info(
            "delegation_executing",
            from_=from_agent,
            to=to_agent,
            task=task[:200],
            depth=delegation.depth,
        )

        # Eigene Working Memory für delegierten Agenten
        sub_wm = WorkingMemory(session_id=session.session_id)

        # System-Prompt des Ziel-Agenten injizieren
        if target.system_prompt:
            sub_wm.add_message(Message(
                role=MessageRole.SYSTEM,
                content=target.system_prompt,
            ))

        # Aufgabe als User-Nachricht
        sub_wm.add_message(Message(
            role=MessageRole.USER,
            content=task,
        ))

        # Workspace des Ziel-Agenten auflösen
        target_workspace = self._agent_router.resolve_agent_workspace(
            to_agent, self._config.workspace_dir,
        )

        # Tool-Schemas für Ziel-Agenten filtern
        tool_schemas = self._mcp_client.get_tool_schemas() if self._mcp_client else {}
        if target.has_tool_restrictions:
            tool_schemas = target.filter_tools(tool_schemas)

        # Planner mit Ziel-Agent-Kontext aufrufen
        if self._planner is None:
            raise RuntimeError("Planner nicht initialisiert — Delegation nicht möglich")

        plan = await self._planner.plan(
            user_message=task,
            working_memory=sub_wm,
            tool_schemas=tool_schemas,
        )

        # Direkte Antwort?
        if not plan.has_actions and plan.direct_response:
            delegation.result = plan.direct_response
            delegation.success = True
            return plan.direct_response

        if not plan.has_actions:
            delegation.result = "Kein Plan erstellt."
            delegation.success = False
            return delegation.result

        # Gatekeeper prüfen
        if self._gatekeeper is None:
            raise RuntimeError("Gatekeeper nicht initialisiert — Delegation nicht möglich")
        decisions = self._gatekeeper.evaluate_plan(plan.steps, session)

        # APPROVE/BLOCK-Entscheidungen in Delegationen blockieren (kein HITL moeglich)
        blocked = [
            d for d in decisions
            if d.status in (GateStatus.APPROVE, GateStatus.BLOCK)
        ]
        if blocked:
            reasons = "; ".join(d.reason for d in blocked[:3])
            delegation.result = f"Delegation blockiert: {reasons}"
            delegation.success = False
            return delegation.result

        # Executor mit Ziel-Agent-Kontext
        assert self._executor is not None
        self._executor.set_agent_context(
            workspace_dir=str(target_workspace),
            sandbox_overrides=target.get_sandbox_config(),
            agent_name=target.name,
        )

        try:
            results = await self._executor.execute(plan.steps, decisions)
        finally:
            self._executor.clear_agent_context()

        # Ergebnis formulieren
        if any(r.success for r in results):
            response = await self._planner.formulate_response(
                user_message=task,
                results=results,
                working_memory=sub_wm,
            )
            delegation.result = response
            delegation.success = True
        else:
            delegation.result = "Delegation fehlgeschlagen: Keine erfolgreichen Aktionen."
            delegation.success = False

        log.info(
            "delegation_complete",
            from_=from_agent,
            to=to_agent,
            success=delegation.success,
            result_len=len(delegation.result or ""),
        )

        return delegation.result or ""

    # =========================================================================
    # Private Methoden
    # =========================================================================

    def _cleanup_stale_sessions(self) -> None:
        """Remove sessions that have not been accessed for more than _SESSION_TTL_SECONDS.

        This is called periodically (guarded by _CLEANUP_INTERVAL_SECONDS) to
        prevent unbounded growth of the in-memory session and working-memory dicts.
        """
        now = time.monotonic()
        stale_keys = [
            key
            for key, last_ts in self._session_last_accessed.items()
            if (now - last_ts) > self._SESSION_TTL_SECONDS
        ]
        for key in stale_keys:
            session = self._sessions.pop(key, None)
            if session:
                self._working_memories.pop(session.session_id, None)
            self._session_last_accessed.pop(key, None)
        if stale_keys:
            log.info("stale_sessions_cleaned", count=len(stale_keys))
        self._last_session_cleanup = now

    def _maybe_cleanup_sessions(self) -> None:
        """Trigger stale session cleanup if enough time has passed since the last sweep."""
        now = time.monotonic()
        if (now - self._last_session_cleanup) >= self._CLEANUP_INTERVAL_SECONDS:
            self._cleanup_stale_sessions()

    def _get_or_create_session(
        self,
        channel: str,
        user_id: str,
        agent_name: str = "jarvis",
    ) -> SessionContext:
        """Lädt oder erstellt eine Session für Channel+User+Agent.

        Per-Agent-Isolation: Jeder Agent hat seine eigene Session.
        Das verhindert dass Working Memories vermischt werden.

        Reihenfolge:
          0. Periodic stale-session cleanup
          1. Im RAM-Cache nachschauen
          2. Aus SQLite laden (Session-Persistenz)
          3. Neue Session erstellen
        """
        # 0. Periodically clean up stale sessions
        self._maybe_cleanup_sessions()

        key = f"{channel}:{user_id}:{agent_name}"

        # 1. RAM-Cache
        if key in self._sessions:
            self._session_last_accessed[key] = time.monotonic()
            return self._sessions[key]

        # 2. SQLite-Persistenz
        if self._session_store:
            stored = self._session_store.load_session(channel, user_id, agent_name)
            if stored and stored.agent_name == agent_name:
                self._sessions[key] = stored
                self._session_last_accessed[key] = time.monotonic()
                log.info(
                    "session_restored",
                    session=stored.session_id[:8],
                    channel=channel,
                    agent=agent_name,
                    messages=stored.message_count,
                )
                return stored

        # 3. Neue Session
        session = SessionContext(
            user_id=user_id,
            channel=channel,
            agent_name=agent_name,
            max_iterations=self._config.security.max_iterations,
        )
        self._sessions[key] = session
        self._session_last_accessed[key] = time.monotonic()

        # Sofort persistieren
        if self._session_store:
            self._session_store.save_session(session)

        log.info(
            "session_created",
            session=session.session_id[:8],
            channel=channel,
            agent=agent_name,
        )
        return session

    def _get_or_create_working_memory(self, session: SessionContext) -> WorkingMemory:
        """Lädt oder erstellt Working Memory für eine Session.

        Bei existierenden Sessions wird die Chat-History aus SQLite geladen.
        """
        if session.session_id not in self._working_memories:
            wm = WorkingMemory(
                session_id=session.session_id,
                max_tokens=self._config.models.planner.context_window,
            )

            # Core Memory laden (wenn vorhanden)
            core_path = self._config.core_memory_path
            if core_path.exists():
                try:
                    wm.core_memory_text = core_path.read_text(encoding="utf-8")
                except Exception as exc:
                    log.warning("core_memory_load_failed", error=str(exc))

            # Chat-History aus SessionStore wiederherstellen
            if self._session_store:
                try:
                    history = self._session_store.load_chat_history(
                        session.session_id,
                        limit=20,
                    )
                    if history:
                        wm.chat_history = history
                        log.info(
                            "chat_history_restored",
                            session=session.session_id[:8],
                            messages=len(history),
                        )
                except Exception as exc:
                    log.warning("chat_history_load_failed", error=str(exc))

            self._working_memories[session.session_id] = wm

        return self._working_memories[session.session_id]

    async def _handle_approvals(
        self,
        steps: list[Any],
        decisions: list[GateDecision],
        session: SessionContext,
        channel_name: str,
    ) -> list[GateDecision]:
        """Holt User-Bestätigungen für ORANGE-Aktionen ein.

        Returns:
            Aktualisierte Liste von Entscheidungen (APPROVE → ALLOW oder BLOCK).
        """
        channel = self._channels.get(channel_name)
        if channel is None:
            return decisions

        result = list(decisions)  # Kopie

        for i, (step, decision) in enumerate(zip(steps, decisions, strict=False)):
            if decision.status != GateStatus.APPROVE:
                continue

            # User fragen
            approved = await channel.request_approval(
                session_id=session.session_id,
                action=step,
                reason=decision.reason,
            )

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
