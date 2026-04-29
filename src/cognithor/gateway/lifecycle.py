"""Cognithor · Gateway lifecycle — extracted from `gateway.py`.

Channel + cron startup, graceful shutdown, vLLM orchestrator hooks,
backend switch via UnifiedLLMClient rebuild.

Each function takes the `Gateway` instance as `gw`. Instance state lives
on Gateway. Wrappers in `gateway.py` keep the public API
(`Gateway.start()`, `Gateway.shutdown()`, etc.) unchanged.

Final step in the staged `gateway.py` split (PR 6/6) — see
`project_v0960_refactor_backlog.md` and the architect blueprint.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import TYPE_CHECKING, Any

from cognithor.i18n import t
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.gateway import Gateway
    from cognithor.models import ActionPlan

log = get_logger(__name__)


async def start(gw: Gateway) -> None:
    """Startet den Gateway und alle Channels + Cron."""
    gw._running = True

    # Signal handler for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, OSError):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(gw.shutdown()))

    # Cron-Engine starten (wenn konfiguriert)
    if gw._cron_engine and gw._cron_engine.has_enabled_jobs:
        await gw._cron_engine.start()
        log.info("cron_engine_started", jobs=gw._cron_engine.job_count)

    # MCP-Server starten (OPTIONAL -- nur wenn Bridge aktiviert)
    if gw._mcp_bridge and gw._mcp_bridge.enabled:
        try:
            await gw._mcp_bridge.start()
        except Exception as exc:
            log.warning("mcp_bridge_start_failed", error=str(exc))

    # A2A-Server starten (OPTIONAL)
    if gw._a2a_adapter and gw._a2a_adapter.enabled:
        try:
            await gw._a2a_adapter.start()
            # A2A HTTP-Routes in WebUI-App registrieren
            for channel in gw._channels.values():
                if hasattr(channel, "app") and channel.app is not None:
                    try:
                        from cognithor.a2a.http_handler import A2AHTTPHandler

                        a2a_http = A2AHTTPHandler(gw._a2a_adapter)
                        a2a_http.register_routes(channel.app)
                    except Exception as exc:
                        log.debug("a2a_http_routes_skip", error=str(exc))
        except Exception as exc:
            log.warning("a2a_adapter_start_failed", error=str(exc))

    # Auto-update: community skill sync if plugins.auto_update is enabled
    if getattr(gw._config.plugins, "auto_update", False) or getattr(
        gw._config.marketplace, "auto_update", False
    ):
        _task = asyncio.create_task(gw._auto_update_skills(), name="auto-update-skills")
        gw._background_tasks.add(_task)
        _task.add_done_callback(gw._background_tasks.discard)

    # Start active learning (background file watcher)
    if gw._active_learner is not None:
        try:
            gw._active_learner._memory = getattr(gw, "_memory_manager", None)
            # D3: Default watch_dirs if empty — scan vault + memory for new files
            if not gw._active_learner._watch_dirs:
                vault_dir = gw._config.cognithor_home / "vault"
                wissen_dir = gw._config.cognithor_home / "vault" / "wissen"
                for d in [vault_dir, wissen_dir]:
                    if d.exists():
                        gw._active_learner._watch_dirs.append(str(d))
            await gw._active_learner.start()
            log.info("active_learner_started")
        except Exception:
            log.debug("active_learner_start_failed", exc_info=True)

    # Start curiosity gap detection (runs every 5 minutes)
    if gw._curiosity_engine is not None:

        async def _curiosity_loop() -> None:
            while True:
                await asyncio.sleep(300)  # 5 minutes
                try:
                    mm = getattr(gw, "_memory_manager", None)
                    if mm and hasattr(mm, "semantic") and mm.semantic:
                        entities: list[dict[str, Any]] = []
                        try:
                            raw = mm.semantic.list_entities(limit=100)
                            entities = [e if isinstance(e, dict) else {"id": str(e)} for e in raw]
                        except Exception:
                            log.debug("curiosity_entity_list_failed", exc_info=True)
                        if entities:
                            await gw._curiosity_engine.detect_gaps("", entities)
                            log.debug(
                                "curiosity_gaps_detected",
                                count=gw._curiosity_engine.open_gap_count,
                            )
                except Exception:
                    log.debug("curiosity_loop_error", exc_info=True)

        task = asyncio.create_task(_curiosity_loop())
        gw._background_tasks.add(task)
        task.add_done_callback(gw._background_tasks.discard)

    # Start background process monitor
    _bg_manager = getattr(gw, "_bg_manager", None)
    if _bg_manager is not None:
        try:
            from cognithor.mcp.background_tasks import ProcessMonitor

            async def _notify_status_change(job_id, old, new, job):
                channel_name = job.get("channel", "")
                session_id = job.get("session_id", "")
                cmd_short = job.get("command", "")[:60]
                text = f"Background job {job_id} {new}: {cmd_short}"
                if job.get("exit_code") is not None:
                    text += f" (exit code: {job['exit_code']})"
                if channel_name and session_id:
                    cb = gw._make_status_callback(channel_name, session_id)
                    await cb("background", text)
                log.info("background_job_status_change", job_id=job_id, old=old, new=new)

            gw._process_monitor = ProcessMonitor(
                _bg_manager,
                on_status_change=_notify_status_change,
            )
            gw._process_monitor._running = True
            _mon_task = asyncio.create_task(
                gw._process_monitor._loop(),
                name="bg-process-monitor",
            )
            gw._background_tasks.add(_mon_task)
            _mon_task.add_done_callback(gw._background_tasks.discard)
            log.info("process_monitor_started")
        except Exception:
            log.debug("process_monitor_start_failed", exc_info=True)

    # Daily audit log retention cleanup
    async def _daily_retention_cleanup():
        while True:
            await asyncio.sleep(86400)  # 24 hours
            try:
                if (
                    hasattr(gw, "_audit_logger")
                    and gw._audit_logger
                    and hasattr(gw._audit_logger, "cleanup_old_entries")
                ):
                    removed = gw._audit_logger.cleanup_old_entries()
                    log.info("audit_retention_cleanup", removed=removed)
                if hasattr(gw, "_bg_manager") and gw._bg_manager:
                    removed_logs = gw._bg_manager.cleanup_old_logs()
                    log.info("background_log_cleanup", removed=removed_logs)
                # RFC 3161 TSA: Daily timestamp on audit anchor
                if (
                    getattr(gw._config, "audit", None)
                    and getattr(gw._config.audit, "tsa_enabled", False)
                    and hasattr(gw, "_audit_trail")
                    and gw._audit_trail
                ):
                    try:
                        from datetime import UTC, datetime

                        from cognithor.security.tsa import TSAClient

                        anchor = gw._audit_trail.get_anchor()
                        if anchor["entry_count"] > 0:
                            date_str = datetime.now(UTC).strftime("%Y-%m-%d")
                            tsa_url = getattr(
                                gw._config.audit, "tsa_url", "https://freetsa.org/tsr"
                            )
                            tsa_dir = gw._config.cognithor_home / "tsa"
                            tsa_client = TSAClient(tsa_url=tsa_url, storage_dir=tsa_dir)
                            tsr_path = tsa_client.request_timestamp(anchor["hash"], date_str)
                            if tsr_path:
                                log.info(
                                    "tsa_daily_timestamp_created",
                                    date=date_str,
                                    anchor_hash=anchor["hash"][:16],
                                    entry_count=anchor["entry_count"],
                                    tsr_path=str(tsr_path),
                                )
                            else:
                                log.warning("tsa_daily_timestamp_failed", date=date_str)
                    except Exception:
                        log.debug("tsa_daily_failed", exc_info=True)
                # WORM: Upload audit files to S3/MinIO with Object Lock
                if (
                    getattr(gw._config, "audit", None)
                    and getattr(gw._config.audit, "worm_backend", "none") != "none"
                ):
                    try:
                        from cognithor.audit.worm import WORMUploader

                        worm_audit_dir = gw._config.cognithor_home / "data" / "audit"
                        uploader = WORMUploader(gw._config.audit, gw._config.cognithor_home)
                        uploaded = uploader.upload_daily(worm_audit_dir)
                        if uploaded:
                            log.info(
                                "worm_daily_upload_complete",
                                count=len(uploaded),
                                files=uploaded,
                            )
                    except Exception:
                        log.debug("worm_daily_upload_failed", exc_info=True)
            except Exception:
                log.debug("retention_cleanup_failed", exc_info=True)

    _retention_task = asyncio.create_task(
        _daily_retention_cleanup(), name="daily-retention-cleanup"
    )
    gw._background_tasks.add(_retention_task)
    _retention_task.add_done_callback(gw._background_tasks.discard)

    # Skill lifecycle: daily audit
    async def _daily_skill_lifecycle():
        while True:
            await asyncio.sleep(86400)  # 24 hours
            try:
                if hasattr(gw, "_skill_lifecycle") and gw._skill_lifecycle:
                    audit_results = gw._skill_lifecycle.audit_all()
                    healthy = sum(1 for r in audit_results if r.status == "healthy")
                    unhealthy = len(audit_results) - healthy
                    if unhealthy > 0:
                        log.info(
                            "skill_lifecycle_audit",
                            total=len(audit_results),
                            healthy=healthy,
                            unhealthy=unhealthy,
                        )
            except Exception:
                log.debug("skill_lifecycle_cron_failed", exc_info=True)

    _skill_lifecycle_task = asyncio.create_task(
        _daily_skill_lifecycle(), name="daily-skill-lifecycle"
    )
    gw._background_tasks.add(_skill_lifecycle_task)
    _skill_lifecycle_task.add_done_callback(gw._background_tasks.discard)

    # Start confidence decay (runs every 24 hours)
    if gw._confidence_manager is not None:

        async def _decay_loop() -> None:
            while True:
                await asyncio.sleep(86400)  # 24 hours
                try:
                    mm = getattr(gw, "_memory_manager", None)
                    idx = getattr(mm, "_indexer", None) if mm else None
                    if idx and hasattr(idx, "list_entities_for_decay"):
                        decay_entities = idx.list_entities_for_decay()
                        for ent in decay_entities:
                            eid = ent.get("id", "")
                            conf = ent.get("confidence", 1.0)
                            updated = ent.get("updated_at", "")
                            if updated:
                                from datetime import UTC, datetime

                                try:
                                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                                    days = (datetime.now(UTC) - dt).days
                                    new_conf = gw._confidence_manager.decay(conf, days)
                                    if abs(new_conf - conf) > 0.01:
                                        idx.update_entity_confidence(eid, new_conf)
                                except (ValueError, TypeError):
                                    pass
                        log.info("confidence_decay_applied")
                except Exception:
                    log.debug("decay_loop_error", exc_info=True)

        task = asyncio.create_task(_decay_loop())
        gw._background_tasks.add(task)
        task.add_done_callback(gw._background_tasks.discard)

    # Breach detection (GDPR Art. 33)
    if getattr(gw._config, "audit", None) and getattr(
        gw._config.audit, "breach_notification_enabled", True
    ):
        try:
            from cognithor.audit.breach_detector import BreachDetector

            _breach_state = gw._config.cognithor_home / "breach_state.json"
            _cooldown = getattr(gw._config.audit, "breach_cooldown_hours", 1)
            gw._breach_detector = BreachDetector(
                state_path=_breach_state,
                cooldown_hours=_cooldown,
            )

            async def _breach_scan_loop():
                while True:
                    await asyncio.sleep(300)  # Every 5 minutes
                    try:
                        if hasattr(gw, "_audit_logger") and gw._audit_logger:
                            breaches = gw._breach_detector.scan(gw._audit_logger)
                            if breaches:
                                log.critical(
                                    "gdpr_breach_notification",
                                    count=len(breaches),
                                    article="Art. 33 DSGVO",
                                )
                    except Exception:
                        log.debug("breach_scan_failed", exc_info=True)

            _breach_task = asyncio.create_task(_breach_scan_loop(), name="breach-detector")
            gw._background_tasks.add(_breach_task)
            _breach_task.add_done_callback(gw._background_tasks.discard)
            log.info("breach_detector_started")
        except Exception:
            log.debug("breach_detector_start_failed", exc_info=True)

    # Start Evolution Loop (idle-time learning)
    if hasattr(gw, "_evolution_loop") and gw._evolution_loop:
        try:
            await gw._evolution_loop.start()
            if gw._evolution_loop._task:
                gw._background_tasks.add(gw._evolution_loop._task)
                gw._evolution_loop._task.add_done_callback(gw._background_tasks.discard)
            log.info("evolution_loop_started")
        except Exception:
            log.debug("evolution_loop_start_failed", exc_info=True)

    # Skill Lifecycle: initial audit + periodic background task
    try:
        skill_registry = getattr(gw, "_skill_registry", None)
        sl_cfg = getattr(gw._config, "skill_lifecycle", None)
        if skill_registry and (sl_cfg is None or sl_cfg.enabled):
            from cognithor.skills.lifecycle import SkillLifecycleManager

            generated_dir = gw._config.cognithor_home / "skills" / "generated"
            gw._skill_lifecycle = SkillLifecycleManager(skill_registry, generated_dir)
            report = gw._skill_lifecycle.get_report()
            log.info("skill_lifecycle_audit", report=report[:200])

            # Auto-repair broken skills
            if sl_cfg is None or sl_cfg.auto_repair:
                for broken in gw._skill_lifecycle.get_broken_skills():
                    gw._skill_lifecycle.repair_skill(broken.slug)

            # Periodic audit background task
            interval_h = getattr(sl_cfg, "audit_interval_hours", 24) if sl_cfg else 24

            async def _periodic_skill_audit() -> None:
                """Run skill audit periodically in the background."""
                import asyncio as _aio

                while True:
                    await _aio.sleep(interval_h * 3600)
                    try:
                        mgr = gw._skill_lifecycle
                        mgr.audit_all()
                        if sl_cfg is None or sl_cfg.auto_repair:
                            for b in mgr.get_broken_skills():
                                mgr.repair_skill(b.slug)
                        if sl_cfg is None or getattr(sl_cfg, "suggest_new", True):
                            mgr.suggest_skills()
                        log.info("skill_lifecycle_periodic_audit_done")
                    except Exception:
                        log.debug("skill_lifecycle_periodic_audit_failed", exc_info=True)

            _audit_task = asyncio.create_task(_periodic_skill_audit())
            gw._background_tasks.add(_audit_task)
            _audit_task.add_done_callback(gw._background_tasks.discard)
            log.info("skill_lifecycle_periodic_started", interval_hours=interval_h)
    except Exception:
        log.debug("skill_lifecycle_init_failed", exc_info=True)

    # Episodic Compression (daily maintenance)
    if gw._memory_manager and hasattr(gw._memory_manager, "compressor"):
        try:
            from datetime import date

            async def _periodic_episode_compression() -> None:
                await asyncio.sleep(3600)  # First run after 1 hour
                while True:
                    try:
                        compressor = gw._memory_manager.compressor
                        ep_dir = gw._config.cognithor_home / "memory" / "episodes"
                        if ep_dir.exists():
                            dates = []
                            for f in ep_dir.glob("*.md"):
                                with contextlib.suppress(ValueError):
                                    dates.append(date.fromisoformat(f.stem))
                            compressible = compressor.identify_compressible(dates)
                            if compressible:
                                weeks = compressor.group_into_weeks(compressible)
                                for start, end in weeks[:3]:  # Max 3 per run
                                    entries = []
                                    for d in compressible:
                                        if start <= d <= end:
                                            p = ep_dir / f"{d.isoformat()}.md"
                                            if p.exists():
                                                entries.append(p.read_text(encoding="utf-8")[:2000])
                                    if entries:
                                        compressed = compressor.compress_heuristic(
                                            entries,
                                            start.isoformat(),
                                            end.isoformat(),
                                        )
                                        if compressed:
                                            log.info(
                                                "episodic_compression_done",
                                                start=start.isoformat(),
                                                end=end.isoformat(),
                                                entries=len(entries),
                                            )
                    except Exception:
                        log.debug("episodic_compression_failed", exc_info=True)
                    await asyncio.sleep(86400)  # Daily

            _comp_task = asyncio.create_task(_periodic_episode_compression())
            gw._background_tasks.add(_comp_task)
            _comp_task.add_done_callback(gw._background_tasks.discard)
            log.info("episodic_compression_scheduled")
        except Exception:
            log.debug("episodic_compression_init_failed", exc_info=True)

    # Channels starten
    tasks = []
    for channel in gw._channels.values():
        task = asyncio.create_task(
            channel.start(gw.handle_message),
            name=f"channel-{channel.name}",
        )
        tasks.append(task)

    if tasks:
        # Warte bis alle Channels beendet sind
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for task, result in zip(tasks, results, strict=False):
            if isinstance(result, BaseException):
                ch_name = task.get_name()
                log.error(
                    "channel_start_failed",
                    channel=ch_name,
                    error=str(result),
                    error_type=type(result).__name__,
                )
    else:
        log.warning("no_channels_registered")


async def auto_update_skills(gw: Gateway) -> None:
    """Background task: sync community registry periodically (daily)."""
    while gw._running:
        try:
            from cognithor.skills.community.sync import RegistrySync

            sync = RegistrySync(
                community_dir=gw._config.cognithor_home / "skills" / "community",
                skill_registry=gw._skill_registry if hasattr(gw, "_skill_registry") else None,
            )
            result = await sync.sync_once()
            if result.success:
                log.info(
                    "auto_update_sync_done",
                    skills=result.registry_skills,
                    recalls=len(result.new_recalls),
                )
            else:
                log.warning("auto_update_sync_failed", errors=result.errors)
        except Exception as exc:
            log.debug("auto_update_skipped", reason=str(exc))
        await asyncio.sleep(86400)  # Daily


def on_startup_vllm(gw: Gateway):
    """Called during init. If a cognithor-managed vLLM container is
    already running (from a previous session with auto_stop_on_close=False,
    or because the user ran the container manually), adopt it — no restart."""
    if not gw._config.vllm.enabled or gw._vllm_orchestrator is None:
        return None
    try:
        return gw._vllm_orchestrator.reuse_existing()
    except Exception as exc:
        log.warning("vllm_reuse_existing_failed", error=str(exc))
        return None


def on_shutdown_vllm(gw: Gateway) -> None:
    """Called on Gateway.shutdown(). Stops the container only if the user
    has opted in via config.vllm.auto_stop_on_close."""
    if not gw._config.vllm.enabled or gw._vllm_orchestrator is None:
        return
    if gw._config.vllm.auto_stop_on_close:
        try:
            gw._vllm_orchestrator.stop_container()
        except Exception as exc:
            log.warning("vllm_shutdown_failed", error=str(exc))


async def shutdown(gw: Gateway) -> None:
    """Faehrt den Gateway sauber herunter mit Session-Persistierung."""
    log.info("gateway_shutdown_start")
    gw._running = False

    # Cancel all background tasks
    for task in list(gw._background_tasks):
        task.cancel()
    if gw._background_tasks:
        with contextlib.suppress(Exception):
            await asyncio.gather(*gw._background_tasks, return_exceptions=True)
    gw._background_tasks.clear()

    # Stop background process monitor
    if hasattr(gw, "_process_monitor") and gw._process_monitor:
        await gw._process_monitor.stop()

    if hasattr(gw, "_evolution_loop") and gw._evolution_loop:
        gw._evolution_loop.stop()

    # Audit log BEFORE closing resources
    if gw._audit_logger:
        gw._audit_logger.log_system("shutdown", description=t("gateway.shutdown_description"))

    # Active learner stoppen
    if gw._active_learner is not None:
        with contextlib.suppress(Exception):
            gw._active_learner.stop()

    # Cron-Engine stoppen
    if gw._cron_engine:
        await gw._cron_engine.stop()

    # Channels stoppen
    for channel in gw._channels.values():
        try:
            await channel.stop()
        except Exception as exc:
            log.warning("channel_stop_error", channel=channel.name, error=str(exc))

    # Sessions persistieren
    if gw._session_store:
        saved_count = 0
        for _key, session in gw._sessions.items():
            try:
                gw._session_store.save_session(session)
                # Chat-History speichern
                wm = gw._working_memories.get(session.session_id)
                if wm and wm.chat_history:
                    gw._session_store.save_chat_history(
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
        gw._session_store.close()

    # Close memory manager
    if hasattr(gw, "_memory_manager") and gw._memory_manager:
        try:
            await gw._memory_manager.close()
        except Exception as exc:
            log.warning("memory_close_error", error=str(exc))

    # A2A-Adapter stoppen (optional)
    if gw._a2a_adapter:
        try:
            await gw._a2a_adapter.stop()
        except Exception:
            log.debug("a2a_adapter_stop_skipped", exc_info=True)

    # Browser-Agent stoppen (optional)
    if gw._browser_agent:
        try:
            await gw._browser_agent.stop()
        except Exception:
            log.debug("browser_agent_stop_skipped", exc_info=True)

    # MCP-Bridge stoppen (optional)
    if gw._mcp_bridge:
        try:
            await gw._mcp_bridge.stop()
        except Exception:
            log.debug("mcp_bridge_stop_skipped", exc_info=True)

    # CostTracker schliessen
    if hasattr(gw, "_cost_tracker") and gw._cost_tracker:
        try:
            gw._cost_tracker.close()
        except Exception:
            log.debug("cost_tracker_close_skipped", exc_info=True)

    # RunRecorder schliessen
    if hasattr(gw, "_run_recorder") and gw._run_recorder:
        try:
            gw._run_recorder.close()
        except Exception:
            log.debug("run_recorder_close_skipped", exc_info=True)

    # GovernanceAgent schliessen
    if hasattr(gw, "_governance_agent") and gw._governance_agent:
        try:
            gw._governance_agent.close()
        except Exception:
            log.debug("governance_agent_close_skipped", exc_info=True)

    # Flush gatekeeper audit buffer (prevent losing entries)
    if gw._gatekeeper:
        try:
            gw._gatekeeper._flush_audit_buffer()
        except Exception:
            log.debug("gatekeeper_flush_skipped", exc_info=True)

    # Close UserPreferenceStore
    if hasattr(gw, "_user_pref_store") and gw._user_pref_store:
        try:
            gw._user_pref_store.close()
        except Exception:
            log.debug("user_pref_store_close_skipped", exc_info=True)

    # vLLM video cleanup + media server teardown
    if gw._video_cleanup is not None:
        try:
            await gw._video_cleanup.stop()
        except Exception:
            log.debug("video_cleanup_stop_skipped", exc_info=True)
    if gw._media_server is not None:
        try:
            await gw._media_server.stop()
        except Exception:
            log.debug("media_server_stop_skipped", exc_info=True)

    # MCP-Client trennen
    if gw._mcp_client:
        await gw._mcp_client.disconnect_all()

    # Close Ollama client
    if gw._llm:
        await gw._llm.close()

    log.info("gateway_shutdown_complete")


def rebuild_llm_client(gw: Gateway, new_backend_type: str) -> None:
    """Re-init UnifiedLLMClient for a new backend type.

    Called from the FastAPI /api/backends/active endpoint when the user
    switches backends from the Flutter UI. No app restart needed.
    """
    from cognithor.core.unified_llm import UnifiedLLMClient

    gw._config.llm_backend_type = new_backend_type
    gw._llm = UnifiedLLMClient.create(gw._config)


async def execute_workflow(gw: Gateway, workflow_yaml: str) -> dict[str, Any]:
    """Execute a workflow via the DAG WorkflowEngine.

    Parses a YAML workflow definition and runs it through the wired
    WorkflowEngine. Returns the WorkflowRun as a dictionary.

    Args:
        workflow_yaml: YAML string defining the workflow.

    Returns:
        Dict with workflow run results.

    Raises:
        RuntimeError: If DAG WorkflowEngine is not available.
    """
    engine = getattr(gw, "_dag_workflow_engine", None)
    if engine is None:
        raise RuntimeError("DAG WorkflowEngine ist nicht verfügbar")

    from cognithor.core.workflow_schema import WorkflowDefinition

    workflow = WorkflowDefinition.from_yaml(workflow_yaml)

    errors = engine.validate(workflow)
    if errors:
        return {"success": False, "errors": errors}

    run = await engine.execute(workflow)
    return run.model_dump(mode="json")


async def execute_action_plan_as_workflow(gw: Gateway, plan: ActionPlan) -> dict[str, Any]:
    """Execute an ActionPlan through the DAG WorkflowEngine.

    Bridges PGE-style ActionPlans with the full WorkflowEngine.
    Useful for complex multi-step plans that benefit from the
    engine's checkpoint/resume, retry strategies, and status callbacks.

    Args:
        plan: PGE ActionPlan to execute.

    Returns:
        Dict with workflow run results.

    Raises:
        RuntimeError: If DAG WorkflowEngine is not available.
    """
    engine = getattr(gw, "_dag_workflow_engine", None)
    if engine is None:
        raise RuntimeError("DAG WorkflowEngine ist nicht verfügbar")

    from cognithor.core.workflow_adapter import action_plan_to_workflow

    workflow = action_plan_to_workflow(
        plan,
        max_parallel=getattr(gw._config.executor, "max_parallel_tools", 4),
    )

    errors = engine.validate(workflow)
    if errors:
        return {"success": False, "errors": errors}

    run = await engine.execute(workflow)
    return run.model_dump(mode="json")
