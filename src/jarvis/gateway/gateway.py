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
import re
import signal
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

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
    init_compliance,
    init_core,
    init_memory,
    init_pge,
    init_security,
    init_tools,
)
from jarvis.i18n import t
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
from jarvis.core.autonomous_orchestrator import AutonomousOrchestrator
from jarvis.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from jarvis.channels.base import Channel
    from jarvis.core.message_queue import DurableMessageQueue
    from jarvis.models import SubAgentConfig

log = get_logger(__name__)

# Presearch result markers — used to detect empty/failed search results
_PRESEARCH_NO_RESULTS = "Keine Ergebnisse"
_PRESEARCH_NO_ENGINE = "Keine Suchengine"

# ── Tool status map for progress feedback ────────────────────────

_TOOL_STATUS_MAP: dict[str, str] = {
    "web_search": "Searching the web...",
    "web_news_search": "Searching news...",
    "search_and_read": "Researching online...",
    "web_fetch": "Fetching webpage...",
    "read_file": "Reading file...",
    "write_file": "Writing file...",
    "edit_file": "Editing file...",
    "exec_command": "Running command...",
    "run_python": "Running Python code...",
    "search_memory": "Searching knowledge...",
    "save_to_memory": "Saving knowledge...",
    "document_export": "Creating document...",
    "media_analyze_image": "Analyzing image...",
    "media_transcribe_audio": "Transcribing audio...",
    "media_extract_text": "Extracting text...",
    "media_tts": "Generating speech...",
    "vault_search": "Searching vault...",
    "vault_write": "Writing to vault...",
    "analyze_code": "Analyzing code...",
    "list_directory": "Listing directory...",
    "browser_navigate": "Navigating browser...",
    "browser_screenshot": "Taking screenshot...",
}


def _sanitize_broken_llm_output(text: str) -> str:
    """Entfernt JSON-Artefakte aus einer kaputten LLM-Antwort.

    Wenn das LLM einen Mix aus Freitext und kaputtem JSON produziert hat,
    extrahiert diese Funktion den lesbaren Textanteil.

    Returns:
        Bereinigter Text oder leerer String wenn nichts Brauchbares uebrig bleibt.
    """
    if not text:
        return ""

    import re as _re

    # 1. Code-Bloecke entfernen (```json ... ```)
    cleaned = _re.sub(r"```(?:json)?\s*\n?.*?\n?\s*```", "", text, flags=_re.DOTALL)

    # 2. JSON-Objekte entfernen ({ ... } Bloecke die JSON-Keys enthalten)
    cleaned = _re.sub(r"\{[^{}]*\"[^{}]*\"[^{}]*\}", "", cleaned)

    # 3. Stray JSON-Fragmente entfernen (Keys ohne zugehoerige Objekte)
    cleaned = _re.sub(r"\"(?:goal|steps|tool|params|reasoning|confidence)\":\s*", "", cleaned)

    # 4. Leere Klammern, Kommas und Whitespace aufraeumen
    cleaned = _re.sub(r"[{}\[\]]", "", cleaned)
    cleaned = _re.sub(r"\s*,\s*,\s*", " ", cleaned)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


class Gateway:
    """Central entry point. Connects all Jarvis subsystems. [B§9.1]"""

    # Session TTL: sessions older than 24 hours are considered stale
    _SESSION_TTL_SECONDS: float = 24 * 60 * 60  # 24h
    # Minimum interval between stale-session cleanup sweeps
    _CLEANUP_INTERVAL_SECONDS: float = 60 * 60  # 1h

    def __init__(self, config: JarvisConfig | None = None) -> None:
        """Initialisiert das Gateway mit PGE-Trinitaet, MCP-Client und Memory."""
        self._config = config or load_config()
        self._channels: dict[str, Channel] = {}
        self._sessions: dict[str, SessionContext] = {}
        self._working_memories: dict[str, WorkingMemory] = {}
        self._session_last_accessed: dict[str, float] = {}
        self._last_session_cleanup: float = time.monotonic()
        self._session_lock = threading.Lock()
        self._running = False
        self._cancelled_sessions: set[str] = set()
        self._context_pipeline = None
        self._message_queue: DurableMessageQueue | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._pattern_record_timestamps: list[float] = []

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
            gateway=self,
        )
        apply_phase(self, tools_result)

        # --- Phase D.1: Context Pipeline (depends on memory + tools) ---
        try:
            from jarvis.core.context_pipeline import ContextPipeline

            cp_config = getattr(self._config, "context_pipeline", None)
            if cp_config is None:
                from jarvis.config import ContextPipelineConfig

                cp_config = ContextPipelineConfig()
            if cp_config.enabled:
                self._context_pipeline = ContextPipeline(cp_config)
                self._context_pipeline.set_memory_manager(self._memory_manager)
                if hasattr(self, "_vault_tools") and self._vault_tools:
                    self._context_pipeline.set_vault_tools(self._vault_tools)
                log.info("context_pipeline_initialized")
        except Exception:
            log.debug("context_pipeline_init_skipped", exc_info=True)

        # --- Phase D.2: Message Queue (optional, durable message buffering) ---
        if self._config.queue.enabled:
            try:
                from jarvis.core.message_queue import DurableMessageQueue as _Dmq

                queue_path = self._config.jarvis_home / "memory" / "message_queue.db"
                self._message_queue = _Dmq(
                    queue_path,
                    max_size=self._config.queue.max_size,
                    max_retries=self._config.queue.max_retries,
                    ttl_hours=self._config.queue.ttl_hours,
                )
                # Trigger lazy DB init
                _ = self._message_queue.conn
                log.info("message_queue_initialized", db=str(queue_path))
            except Exception:
                log.warning("message_queue_init_failed", exc_info=True)

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

        # Wire Orchestrator runner (Sub-Agent execution via handle_message)
        if getattr(self, "_orchestrator", None):
            try:

                async def _agent_runner(
                    config: SubAgentConfig,
                    agent_name: str,
                ) -> AgentResult:
                    msg = IncomingMessage(
                        channel="sub_agent",
                        user_id=f"agent:{agent_name}",
                        text=config.task,
                        metadata={
                            "agent_type": config.agent_type.value,
                            "parent_agent": agent_name,
                            "max_iterations": config.max_iterations,
                            "depth": config.depth + 1,
                        },
                    )
                    try:
                        response = await asyncio.wait_for(
                            self.handle_message(msg),
                            timeout=config.timeout_seconds,
                        )
                        return AgentResult(
                            response=response.text,
                            success=True,
                            model_used=config.model,
                        )
                    except TimeoutError:
                        return AgentResult(
                            response="",
                            success=False,
                            error=f"Sub-Agent Timeout nach {config.timeout_seconds}s",
                        )
                    except Exception as exc:
                        return AgentResult(
                            response="",
                            success=False,
                            error=str(exc),
                        )

                self._orchestrator.set_runner(_agent_runner)
                log.info("orchestrator_runner_wired")
            except Exception:
                log.debug("orchestrator_runner_wiring_skipped", exc_info=True)

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

        # Wire DAG WorkflowEngine with MCP client + Gatekeeper
        if getattr(self, "_dag_workflow_engine", None):
            try:
                self._dag_workflow_engine._mcp_client = self._mcp_client
                self._dag_workflow_engine._gatekeeper = self._gatekeeper
            except Exception:
                log.debug("dag_workflow_engine_wiring_skipped", exc_info=True)

        # Wire prompt_evolution to planner (created in advanced, after PGE)
        if getattr(self, "_prompt_evolution", None) and getattr(self, "_planner", None):
            self._planner._prompt_evolution = self._prompt_evolution

        # Wire prompt_evolution LLM client (meta-prompt generation)
        if getattr(self, "_prompt_evolution", None) and self._llm and self._model_router:
            try:

                async def _pe_llm_call(prompt: str) -> str:
                    model = self._model_router.select_model("planning", "high")
                    resp = await self._llm.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                    )
                    return resp.get("message", {}).get("content", "")

                self._prompt_evolution._llm_client = _pe_llm_call
            except Exception:
                log.debug("prompt_evolution_llm_wiring_skipped", exc_info=True)

        # Wire prompt_evolution interval from config
        if getattr(self, "_prompt_evolution", None):
            try:
                self._prompt_evolution.set_evolution_interval_hours(
                    self._config.prompt_evolution.evolution_interval_hours
                )
            except Exception:
                log.debug("prompt_evolution_interval_config_skipped", exc_info=True)

        # --- Phase G: Compliance validation ---
        compliance_attrs = {
            k: getattr(self, f"_{k}", None)
            for k in (
                "compliance_framework",
                "decision_log",
                "remediation_tracker",
                "economic_governor",
                "compliance_exporter",
                "impact_assessor",
                "explainability",
            )
        }
        await init_compliance(self._config, **compliance_attrs)

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

        # Prompt-Evolution-Cron-Job registrieren
        if self._cron_engine and getattr(self, "_prompt_evolution", None):
            try:
                from jarvis.cron.jobs import prompt_evolution_check

                interval_h = self._config.prompt_evolution.evolution_interval_hours
                cron_expr = (
                    f"0 */{interval_h} * * *" if interval_h < 24 else f"0 {interval_h % 24} * * *"
                )
                self._cron_engine.add_system_job(
                    name="prompt_evolution_check",
                    schedule=cron_expr,
                    callback=prompt_evolution_check,
                    args=[self],
                )
            except Exception:
                log.debug("prompt_evolution_cron_registration_skipped", exc_info=True)

        # --- Autonomous Orchestrator (connects PGE + SkillGenerator + Reflector) ---
        self._autonomous_orchestrator = AutonomousOrchestrator(
            gateway=self,
            skill_generator=getattr(self, "_skill_generator", None),
            reflector=getattr(self, "_reflector", None),
        )

        # --- Feedback Store (thumbs up/down rating) ---
        try:
            from jarvis.core.feedback import FeedbackStore

            self._feedback_store = FeedbackStore(db_path=self._config.jarvis_home / "feedback.db")
            log.info("feedback_store_initialized")
        except Exception:
            log.debug("feedback_store_init_failed", exc_info=True)
            self._feedback_store = None

        # --- Correction Memory (Smart Recovery) ---
        try:
            from jarvis.core.correction_memory import CorrectionMemory

            _proactive = 3
            if hasattr(self._config, "recovery"):
                _proactive = getattr(self._config.recovery, "correction_proactive_threshold", 3)
            self._correction_memory = CorrectionMemory(
                db_path=self._config.jarvis_home / "corrections.db",
                proactive_threshold=_proactive,
            )
            log.info("correction_memory_initialized")
            # Wire into context pipeline
            if hasattr(self, "_context_pipeline") and self._context_pipeline:
                self._context_pipeline.set_correction_memory(self._correction_memory)
                log.debug("correction_memory_wired_to_pipeline")
        except Exception:
            log.debug("correction_memory_init_failed", exc_info=True)
            self._correction_memory = None

        # Conversation Tree (Chat Branching)
        try:
            from jarvis.core.conversation_tree import ConversationTree

            self._conversation_tree = ConversationTree(
                db_path=self._config.jarvis_home / "conversations.db"
            )
            log.info("conversation_tree_initialized")
        except Exception:
            log.debug("conversation_tree_init_failed", exc_info=True)
            self._conversation_tree = None

        # System Detector (hardware profiling)
        try:
            from jarvis.system.detector import SystemDetector

            _detector = SystemDetector()
            _cache = self._config.jarvis_home / "system_profile.json"
            self._system_profile = _detector.run_quick_scan(cache_path=_cache)
            self._system_profile.save(_cache)
            log.info(
                "system_profile_detected",
                tier=self._system_profile.get_tier(),
                mode=self._system_profile.get_recommended_mode(),
                results=len(self._system_profile.results),
            )
        except Exception:
            log.debug("system_detector_failed", exc_info=True)
            self._system_profile = None

        # ResourceMonitor (lightweight psutil-based, always available)
        self._resource_monitor = None
        try:
            from jarvis.system.resource_monitor import ResourceMonitor

            self._resource_monitor = ResourceMonitor()
            log.info("resource_monitor_initialized")
        except Exception:
            log.debug("resource_monitor_init_skipped", exc_info=True)

        # CheckpointStore (persistent evolution checkpoints)
        self._checkpoint_store = None
        try:
            from jarvis.core.checkpointing import CheckpointStore

            self._checkpoint_store = CheckpointStore(
                self._config.jarvis_home / "checkpoints"
            )
            log.info("checkpoint_store_initialized")
        except Exception:
            log.debug("checkpoint_store_init_skipped", exc_info=True)

        # LLM call function for Evolution Engine + DeepLearner
        self._llm_call = None
        if self._llm and self._model_router:
            async def _evolution_llm_call(prompt: str) -> str:
                model = self._model_router.select_model("planning", "medium")
                resp = await self._llm.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return resp.get("message", {}).get("content", "")

            self._llm_call = _evolution_llm_call

        # Evolution Engine (idle-time autonomous learning)
        self._idle_detector = None
        self._evolution_loop = None
        if getattr(self._config, "evolution", None) and self._config.evolution.enabled:
            try:
                from jarvis.evolution.idle_detector import IdleDetector
                from jarvis.evolution.loop import EvolutionLoop

                idle_minutes = getattr(self._config.evolution, "idle_minutes", 5)
                self._idle_detector = IdleDetector(idle_threshold_seconds=idle_minutes * 60)
                op_mode = str(
                    getattr(self._config, "resolved_operation_mode", "offline")
                )
                self._evolution_loop = EvolutionLoop(
                    idle_detector=self._idle_detector,
                    curiosity_engine=getattr(self, "_curiosity_engine", None),
                    skill_generator=getattr(self, "_skill_generator", None),
                    memory_manager=getattr(self, "_memory_manager", None),
                    config=self._config.evolution,
                    resource_monitor=self._resource_monitor,
                    cost_tracker=self._cost_tracker,
                    checkpoint_store=self._checkpoint_store,
                    operation_mode=op_mode,
                    mcp_client=getattr(self, "_mcp_client", None),
                    llm_fn=getattr(self, "_llm_call", None),
                    skill_registry=getattr(self, "_skill_registry", None),
                    session_analyzer=getattr(self, "_session_analyzer", None),
                )
                log.info("evolution_engine_initialized", idle_minutes=idle_minutes)
            except Exception:
                log.debug("evolution_engine_init_failed", exc_info=True)

            # DeepLearner (deep autonomous learning)
            self._deep_learner = None
            if self._evolution_loop:
                try:
                    from jarvis.evolution.deep_learner import DeepLearner

                    self._deep_learner = DeepLearner(
                        llm_fn=getattr(self, "_llm_call", None),
                        plans_dir=self._config.jarvis_home / "evolution" / "plans",
                        mcp_client=getattr(self, "_mcp_client", None),
                        memory_manager=getattr(self, "_memory_manager", None),
                        skill_registry=getattr(self, "_skill_registry", None),
                        skill_generator=getattr(self, "_skill_generator", None),
                        cron_engine=getattr(self, "_cron_engine", None),
                        cost_tracker=self._cost_tracker,
                        resource_monitor=self._resource_monitor,
                        checkpoint_store=self._checkpoint_store,
                        config=self._config.evolution,
                        idle_detector=self._idle_detector,
                        operation_mode=op_mode,
                    )
                    self._evolution_loop._deep_learner = self._deep_learner
                    log.info("deep_learner_initialized")
                except Exception:
                    log.debug("deep_learner_init_failed", exc_info=True)

        log.info(
            "gateway_init_complete",
            llm_available=llm_ok,
            tools=self._mcp_client.get_tool_list(),
            cron_jobs=self._cron_engine.job_count if self._cron_engine else 0,
        )

        # Audit: System-Start protokollieren
        if self._audit_logger:
            self._audit_logger.log_system(
                "startup",
                description=(
                    f"Jarvis gestartet (LLM={llm_ok}, "
                    f"Tools={len(self._mcp_client.get_tool_list())})"
                ),
            )

        # CORE.md: Tool/Skill-Inventar aktualisieren
        try:
            self._sync_core_inventory()
        except Exception:
            log.debug("core_inventory_sync_failed", exc_info=True)

    def _sync_core_inventory(self) -> None:
        """Aktualisiert den INVENTAR-Abschnitt in CORE.md mit aktuellen Tools/Skills.

        Verwendet ToolRegistryDB fuer datenbankgestuetzte, lokalisierte und
        rollenbasierte Tool-Abschnitte. Faellt auf die alte statische Methode
        zurueck, wenn die DB nicht verfuegbar ist.
        """
        core_path = self._config.core_memory_file
        if not core_path or not core_path.exists():
            return

        content = core_path.read_text(encoding="utf-8")
        language = getattr(self._config, "language", "de")

        # Try DB-backed generation
        tool_count = 0
        try:
            from jarvis.mcp.tool_registry_db import (
                _SECTION_HEADERS,
                ToolRegistryDB,
                _ProcedureEntry,
                deduplicate_procedures,
            )

            db_path = self._config.jarvis_home / "tool_registry.db"
            registry_db = ToolRegistryDB(db_path)

            # Tools aus MCP-Client synchronisieren
            if self._mcp_client:
                registry_db.sync_from_mcp(self._mcp_client)

            tool_count = registry_db.tool_count()
            registry_db.close()
        except Exception:
            log.debug("tool_registry_db_failed_falling_back", exc_info=True)
            # Fallback: legacy method just to validate MCP is alive
            if self._sync_core_inventory_legacy() is None:
                return
            tool_count = 0

        # Compile skill list
        skill_lines: list[str] = []
        if hasattr(self, "_skill_registry") and self._skill_registry:
            try:
                for slug, skill in self._skill_registry._skills.items():
                    status = "active" if skill.enabled else "inactive"
                    skill_lines.append(f"- **{skill.name}** (`{slug}`) -- {status}")
            except Exception:
                log.debug("core_inventory_skills_failed", exc_info=True)
        if not skill_lines:
            skill_lines = ["- (no skills registered)"]

        # Procedure list with deduplication
        proc_lines: list[str] = []
        if self._memory_manager:
            try:
                from jarvis.mcp.tool_registry_db import (
                    _ProcedureEntry,
                    deduplicate_procedures,
                )

                procedural = self._memory_manager.procedural
                raw_procs = [
                    _ProcedureEntry(
                        name=meta.name,
                        total_uses=meta.total_uses,
                        trigger_keywords=list(meta.trigger_keywords),
                    )
                    for meta in procedural.list_procedures()
                ]
                proc_lines = deduplicate_procedures(
                    raw_procs,
                    language=language,
                )
            except Exception:
                log.debug("core_inventory_procedures_dedup_failed", exc_info=True)
                # Fallback: simple list
                try:
                    procedural = self._memory_manager.procedural
                    for meta in procedural.list_procedures():
                        uses = f"{meta.total_uses}x" if meta.total_uses else "0x"
                        kw = ", ".join(meta.trigger_keywords[:3]) if meta.trigger_keywords else ""
                        suffix = f" [{kw}]" if kw else ""
                        proc_lines.append(f"- `{meta.name}` ({uses} used){suffix}")
                except Exception:
                    log.debug("core_inventory_procedures_failed", exc_info=True)

        if not proc_lines:
            proc_lines = ["- (no procedures stored)"]

        # Lokalisierte Header
        try:
            from jarvis.mcp.tool_registry_db import _SECTION_HEADERS

            headers = _SECTION_HEADERS.get(language, _SECTION_HEADERS["en"])
        except Exception:
            headers = {
                "inventory_title": "INVENTORY (auto-updated)",
                "skills_title": "Installed Skills ({count})",
                "procedures_title": "Learned Procedures ({count})",
            }

        inv_title = headers["inventory_title"]
        skills_title = headers["skills_title"].format(count=len(skill_lines))
        procs_title = headers["procedures_title"].format(count=len(proc_lines))

        # Tool descriptions are injected directly into the Planner prompt
        # via {tools_section} — no need to duplicate them in CORE.md
        tool_ref = (
            f"*{tool_count} Tools registriert (werden direkt in den Planner-Prompt injiziert)*"
        )

        inventory = (
            f"## {inv_title}\n\n"
            + tool_ref
            + "\n\n"
            + f"### {skills_title}\n"
            + "\n".join(skill_lines)
            + "\n\n"
            + f"### {procs_title}\n"
            + "\n".join(proc_lines)
        )

        # Bestehenden INVENTAR/INVENTORY-Abschnitt ersetzen oder am Ende anhaengen
        marker_candidates = [
            "## INVENTAR (auto-aktualisiert)",
            "## INVENTAR (automatisch aktualisiert)",
            "## INVENTORY (auto-updated)",
            f"## {inv_title}",
        ]
        marker_start = None
        for marker in marker_candidates:
            if marker in content:
                marker_start = marker
                break

        if marker_start:
            pattern = re.escape(marker_start) + r".*?(?=\n## (?!INVENT|清单)|\Z)"
            content = re.sub(pattern, inventory, content, flags=re.DOTALL)
        else:
            content = content.rstrip() + "\n\n---\n\n" + inventory + "\n"

        core_path.write_text(content, encoding="utf-8")
        log.info(
            "core_inventory_synced",
            tools=tool_count,
            skills=len(skill_lines),
            procedures=len(proc_lines),
        )

    def _sync_core_inventory_legacy(self) -> str | None:
        """Alte statische Tool-Liste als Fallback (ohne DB).

        Returns:
            Formatierter Tool-Abschnitt oder None bei Fehler.
        """
        tool_schemas = self._mcp_client.get_tool_schemas() if self._mcp_client else {}
        if not tool_schemas:
            return None

        tool_lines: list[str] = []
        for name in sorted(tool_schemas):
            schema = tool_schemas[name]
            desc = schema.get("description", "")
            props = schema.get("inputSchema", {}).get("properties", {})
            required = set(schema.get("inputSchema", {}).get("required", []))
            if props:
                parts = []
                for k, v in props.items():
                    typ = v.get("type", "?")
                    req = " *" if k in required else ""
                    parts.append(f"{k}: {typ}{req}")
                param_str = ", ".join(parts)
                tool_lines.append(f"- `{name}({param_str})` -- {desc}")
            else:
                tool_lines.append(f"- `{name}()` -- {desc}")

        tool_count = len(tool_schemas)
        return (
            f"### Registered Tools ({tool_count})\n"
            + "Parameters marked with * are required.\n\n"
            + "\n".join(tool_lines)
        )

    def cancel_session(self, session_id: str) -> bool:
        """Bricht die aktive Verarbeitung einer Session ab.

        Der PGE-Loop prueft dieses Flag und bricht beim naechsten
        Iterationsschritt sauber ab.

        Returns:
            True wenn die Session gefunden und als cancelled markiert wurde.
        """
        self._cancelled_sessions.add(session_id)
        log.info("session_cancelled", session=session_id[:8])
        return True

    def register_channel(self, channel: Channel) -> None:
        """Registriert einen Kommunikationskanal."""
        self._channels[channel.name] = channel
        # Wire up cancel callback for channels that support it (e.g. WebUI)
        if hasattr(channel, "_cancel_callback"):
            channel._cancel_callback = self.cancel_session
        log.info("channel_registered", channel=channel.name)

    async def start(self) -> None:
        """Startet den Gateway und alle Channels + Cron."""
        self._running = True

        # Signal handler for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, OSError):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # Cron-Engine starten (wenn konfiguriert)
        if self._cron_engine and self._cron_engine.has_enabled_jobs:
            await self._cron_engine.start()
            log.info("cron_engine_started", jobs=self._cron_engine.job_count)

        # MCP-Server starten (OPTIONAL -- nur wenn Bridge aktiviert)
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

        # Auto-update: community skill sync if plugins.auto_update is enabled
        if getattr(self._config.plugins, "auto_update", False) or getattr(
            self._config.marketplace, "auto_update", False
        ):
            _task = asyncio.create_task(self._auto_update_skills(), name="auto-update-skills")
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)

        # Start active learning (background file watcher)
        if self._active_learner is not None:
            try:
                self._active_learner._memory = getattr(self, "_memory_manager", None)
                await self._active_learner.start()
                log.info("active_learner_started")
            except Exception:
                log.debug("active_learner_start_failed", exc_info=True)

        # Start curiosity gap detection (runs every 5 minutes)
        if self._curiosity_engine is not None:

            async def _curiosity_loop() -> None:
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    try:
                        mm = getattr(self, "_memory_manager", None)
                        if mm and hasattr(mm, "semantic") and mm.semantic:
                            entities: list[dict[str, Any]] = []
                            try:
                                raw = mm.semantic.list_entities(limit=100)
                                entities = [
                                    e if isinstance(e, dict) else {"id": str(e)} for e in raw
                                ]
                            except Exception:
                                log.debug("curiosity_entity_list_failed", exc_info=True)
                            if entities:
                                await self._curiosity_engine.detect_gaps("", entities)
                                log.debug(
                                    "curiosity_gaps_detected",
                                    count=self._curiosity_engine.open_gap_count,
                                )
                    except Exception:
                        log.debug("curiosity_loop_error", exc_info=True)

            task = asyncio.create_task(_curiosity_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Start background process monitor
        _bg_manager = getattr(self, "_bg_manager", None)
        if _bg_manager is not None:
            try:
                from jarvis.mcp.background_tasks import ProcessMonitor

                async def _notify_status_change(job_id, old, new, job):
                    channel_name = job.get("channel", "")
                    session_id = job.get("session_id", "")
                    cmd_short = job.get("command", "")[:60]
                    text = f"Background job {job_id} {new}: {cmd_short}"
                    if job.get("exit_code") is not None:
                        text += f" (exit code: {job['exit_code']})"
                    if channel_name and session_id:
                        cb = self._make_status_callback(channel_name, session_id)
                        await cb("background", text)
                    log.info("background_job_status_change", job_id=job_id, old=old, new=new)

                self._process_monitor = ProcessMonitor(
                    _bg_manager,
                    on_status_change=_notify_status_change,
                )
                self._process_monitor._running = True
                _mon_task = asyncio.create_task(
                    self._process_monitor._loop(),
                    name="bg-process-monitor",
                )
                self._background_tasks.add(_mon_task)
                _mon_task.add_done_callback(self._background_tasks.discard)
                log.info("process_monitor_started")
            except Exception:
                log.debug("process_monitor_start_failed", exc_info=True)

        # Daily audit log retention cleanup
        async def _daily_retention_cleanup():
            while True:
                await asyncio.sleep(86400)  # 24 hours
                try:
                    if hasattr(self, "_audit_logger") and self._audit_logger:
                        if hasattr(self._audit_logger, "cleanup_old_entries"):
                            removed = self._audit_logger.cleanup_old_entries()
                            log.info("audit_retention_cleanup", removed=removed)
                    if hasattr(self, "_bg_manager") and self._bg_manager:
                        removed_logs = self._bg_manager.cleanup_old_logs()
                        log.info("background_log_cleanup", removed=removed_logs)
                    # RFC 3161 TSA: Daily timestamp on audit anchor
                    if (
                        getattr(self._config, "audit", None)
                        and getattr(self._config.audit, "tsa_enabled", False)
                        and hasattr(self, "_audit_trail")
                        and self._audit_trail
                    ):
                        try:
                            from jarvis.security.tsa import TSAClient
                            from datetime import UTC, datetime

                            anchor = self._audit_trail.get_anchor()
                            if anchor["entry_count"] > 0:
                                date_str = datetime.now(UTC).strftime("%Y-%m-%d")
                                tsa_url = getattr(
                                    self._config.audit, "tsa_url", "https://freetsa.org/tsr"
                                )
                                tsa_dir = self._config.jarvis_home / "tsa"
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
                        getattr(self._config, "audit", None)
                        and getattr(self._config.audit, "worm_backend", "none") != "none"
                    ):
                        try:
                            from jarvis.audit.worm import WORMUploader

                            worm_audit_dir = self._config.jarvis_home / "data" / "audit"
                            uploader = WORMUploader(self._config.audit, self._config.jarvis_home)
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
        self._background_tasks.add(_retention_task)
        _retention_task.add_done_callback(self._background_tasks.discard)

        # Start confidence decay (runs every 24 hours)
        if self._confidence_manager is not None:

            async def _decay_loop() -> None:
                while True:
                    await asyncio.sleep(86400)  # 24 hours
                    try:
                        mm = getattr(self, "_memory_manager", None)
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
                                        new_conf = self._confidence_manager.decay(conf, days)
                                        if abs(new_conf - conf) > 0.01:
                                            idx.update_entity_confidence(eid, new_conf)
                                    except (ValueError, TypeError):
                                        pass
                            log.info("confidence_decay_applied")
                    except Exception:
                        log.debug("decay_loop_error", exc_info=True)

            task = asyncio.create_task(_decay_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Breach detection (GDPR Art. 33)
        if getattr(self._config, "audit", None) and getattr(
            self._config.audit, "breach_notification_enabled", True
        ):
            try:
                from jarvis.audit.breach_detector import BreachDetector

                _breach_state = self._config.jarvis_home / "breach_state.json"
                _cooldown = getattr(self._config.audit, "breach_cooldown_hours", 1)
                self._breach_detector = BreachDetector(
                    state_path=_breach_state,
                    cooldown_hours=_cooldown,
                )

                async def _breach_scan_loop():
                    while True:
                        await asyncio.sleep(300)  # Every 5 minutes
                        try:
                            if hasattr(self, "_audit_logger") and self._audit_logger:
                                breaches = self._breach_detector.scan(self._audit_logger)
                                if breaches:
                                    log.critical(
                                        "gdpr_breach_notification",
                                        count=len(breaches),
                                        article="Art. 33 DSGVO",
                                    )
                        except Exception:
                            log.debug("breach_scan_failed", exc_info=True)

                _breach_task = asyncio.create_task(_breach_scan_loop(), name="breach-detector")
                self._background_tasks.add(_breach_task)
                _breach_task.add_done_callback(self._background_tasks.discard)
                log.info("breach_detector_started")
            except Exception:
                log.debug("breach_detector_start_failed", exc_info=True)

        # Start Evolution Loop (idle-time learning)
        if hasattr(self, "_evolution_loop") and self._evolution_loop:
            try:
                await self._evolution_loop.start()
                if self._evolution_loop._task:
                    self._background_tasks.add(self._evolution_loop._task)
                    self._evolution_loop._task.add_done_callback(self._background_tasks.discard)
                log.info("evolution_loop_started")
            except Exception:
                log.debug("evolution_loop_start_failed", exc_info=True)

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

    async def _auto_update_skills(self) -> None:
        """Background task: sync community registry if auto_update is enabled."""
        try:
            from jarvis.skills.community.sync import RegistrySync

            sync = RegistrySync(
                community_dir=self._config.jarvis_home / "skills" / "community",
                skill_registry=self._skill_registry if hasattr(self, "_skill_registry") else None,
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

    async def shutdown(self) -> None:
        """Faehrt den Gateway sauber herunter mit Session-Persistierung."""
        log.info("gateway_shutdown_start")
        self._running = False

        # Cancel all background tasks
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            with contextlib.suppress(Exception):
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Stop background process monitor
        if hasattr(self, "_process_monitor") and self._process_monitor:
            await self._process_monitor.stop()

        if hasattr(self, "_evolution_loop") and self._evolution_loop:
            self._evolution_loop.stop()

        # Audit log BEFORE closing resources
        if self._audit_logger:
            self._audit_logger.log_system("shutdown", description="Jarvis heruntergefahren")

        # Active learner stoppen
        if self._active_learner is not None:
            with contextlib.suppress(Exception):
                self._active_learner.stop()

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

        # Close memory manager
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

        # Flush gatekeeper audit buffer (prevent losing entries)
        if self._gatekeeper:
            try:
                self._gatekeeper._flush_audit_buffer()
            except Exception:
                log.debug("gatekeeper_flush_skipped", exc_info=True)

        # Close UserPreferenceStore
        if hasattr(self, "_user_pref_store") and self._user_pref_store:
            try:
                self._user_pref_store.close()
            except Exception:
                log.debug("user_pref_store_close_skipped", exc_info=True)

        # MCP-Client trennen
        if self._mcp_client:
            await self._mcp_client.disconnect_all()

        # Close Ollama client
        if self._llm:
            await self._llm.close()

        log.info("gateway_shutdown_complete")

    async def execute_workflow(self, workflow_yaml: str) -> dict[str, Any]:
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
        engine = getattr(self, "_dag_workflow_engine", None)
        if engine is None:
            raise RuntimeError("DAG WorkflowEngine ist nicht verfügbar")

        from jarvis.core.workflow_schema import WorkflowDefinition

        workflow = WorkflowDefinition.from_yaml(workflow_yaml)

        errors = engine.validate(workflow)
        if errors:
            return {"success": False, "errors": errors}

        run = await engine.execute(workflow)
        return run.model_dump(mode="json")

    async def execute_action_plan_as_workflow(self, plan: ActionPlan) -> dict[str, Any]:
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
        engine = getattr(self, "_dag_workflow_engine", None)
        if engine is None:
            raise RuntimeError("DAG WorkflowEngine ist nicht verfügbar")

        from jarvis.core.workflow_adapter import action_plan_to_workflow

        workflow = action_plan_to_workflow(
            plan,
            max_parallel=getattr(self._config.executor, "max_parallel_tools", 4),
        )

        errors = engine.validate(workflow)
        if errors:
            return {"success": False, "errors": errors}

        run = await engine.execute(workflow)
        return run.model_dump(mode="json")

    def reload_components(
        self,
        *,
        prompts: bool = False,
        policies: bool = False,
        config: bool = False,
        core_memory: bool = False,
        skills: bool = False,
    ) -> dict:
        """Reload-Koordinator fuer Live-Updates vom UI."""
        reloaded = []
        if prompts and self._planner:
            self._planner.reload_prompts()
            reloaded.append("prompts")
        if policies and self._gatekeeper:
            self._gatekeeper.reload_policies()
            reloaded.append("policies")
        if core_memory:
            core_path = self._config.core_memory_path
            if core_path.exists():
                try:
                    text = core_path.read_text(encoding="utf-8")
                    for wm in self._working_memories.values():
                        wm.core_memory_text = text
                    reloaded.append("core_memory")
                except Exception:
                    log.debug("reload_core_memory_failed", exc_info=True)
        if skills and self._skill_registry:
            try:
                skill_dirs = [
                    self._config.jarvis_home / "data" / "procedures",
                    self._config.jarvis_home / self._config.plugins.skills_dir,
                ]
                self._skill_registry.load_from_directories(skill_dirs)
                reloaded.append("skills")
            except Exception:
                log.warning("skills_reload_failed", exc_info=True)
        if config:
            # Reload config.yaml from disk
            try:
                new_config = load_config(self._config.config_file)
                self._config = new_config
            except Exception:
                log.debug("config_file_reload_failed", exc_info=True)
                new_config = self._config

            # Live-update i18n locale from config
            try:
                import os

                from jarvis.i18n import set_locale

                _lang = (
                    os.environ.get("JARVIS_LANGUAGE")
                    or os.environ.get("COGNITHOR_LANGUAGE")
                    or new_config.language
                )
                set_locale(_lang)
            except Exception:
                log.debug("i18n_locale_reload_failed", exc_info=True)

            # Live-update Executor runtime parameters
            if self._executor and hasattr(self._executor, "reload_config"):
                try:
                    self._executor.reload_config(new_config)
                except Exception:
                    log.debug("executor_config_reload_failed", exc_info=True)

            # Live-update ModelRouter with new config + schedule model list refresh
            if self._model_router and hasattr(self._model_router, "_config"):
                try:
                    self._model_router._config = new_config
                    # Schedule async re-initialization to refresh _available_models
                    import asyncio

                    try:
                        loop = asyncio.get_running_loop()
                        _task = loop.create_task(self._model_router.initialize())
                        self._background_tasks.add(_task)
                        _task.add_done_callback(self._background_tasks.discard)
                    except RuntimeError:
                        pass  # no loop — model list refresh skipped
                    log.info("model_router_config_reloaded")
                except Exception:
                    log.debug("model_router_config_reload_failed", exc_info=True)

            # Recreate UnifiedLLMClient if backend type changed
            if self._llm is not None:
                old_backend = getattr(self._llm, "backend_type", "ollama")
                new_backend = new_config.llm_backend_type
                if old_backend != new_backend:
                    try:
                        from jarvis.core.unified_llm import UnifiedLLMClient

                        old_llm = self._llm
                        self._llm = UnifiedLLMClient.create(new_config)
                        # Update references in Planner/Executor
                        if self._planner and hasattr(self._planner, "_ollama"):
                            self._planner._ollama = self._llm
                        if self._executor and hasattr(self._executor, "_ollama"):
                            self._executor._ollama = self._llm
                        # Close old client
                        import asyncio

                        try:
                            loop = asyncio.get_running_loop()
                            _task = loop.create_task(old_llm.close())
                            self._background_tasks.add(_task)
                            _task.add_done_callback(self._background_tasks.discard)
                        except RuntimeError:
                            pass
                        log.info(
                            "llm_backend_switched",
                            old=old_backend,
                            new=new_backend,
                        )
                    except Exception:
                        log.warning("llm_backend_switch_failed", exc_info=True)

            # Live-update Planner with new config
            if self._planner and hasattr(self._planner, "_config"):
                try:
                    self._planner._config = new_config
                except Exception:
                    log.debug("planner_config_reload_failed", exc_info=True)

            # Live-update WebTools runtime parameters
            web_tools = None
            if self._mcp_client:
                handler = self._mcp_client.get_handler("web_search")
                if handler is not None:
                    web_tools = getattr(handler, "__self__", None)
            if web_tools and hasattr(web_tools, "reload_config"):
                try:
                    web_tools.reload_config(new_config)
                except Exception:
                    log.debug("web_tools_config_reload_failed", exc_info=True)

            # Live-update Gatekeeper tool toggles (disabled_tools list)
            if self._gatekeeper and hasattr(self._gatekeeper, "reload_disabled_tools"):
                try:
                    self._gatekeeper.reload_disabled_tools()
                    reloaded.append("tool_toggles")
                except Exception:
                    log.debug("gatekeeper_tool_toggles_reload_failed", exc_info=True)

            reloaded.append("config")
        log.info("gateway_components_reloaded", components=reloaded)
        return {"reloaded": reloaded}

    async def switch_branch(
        self, conversation_id: str, leaf_id: str, session: SessionContext
    ) -> WorkingMemory:
        """Switch to a different branch by replaying its message history."""
        if not self._conversation_tree:
            raise RuntimeError("ConversationTree not initialized")

        messages = self._conversation_tree.get_messages_for_replay(conversation_id, leaf_id)

        wm = WorkingMemory(
            session_id=session.session_id,
            max_tokens=getattr(self._config.planner, "context_window", 32768),
        )

        core_path = getattr(self._config, "core_memory_path", None)
        if core_path and hasattr(core_path, "exists") and core_path.exists():
            wm.core_memory_text = core_path.read_text(encoding="utf-8")

        for msg_data in messages:
            role = MessageRole.USER if msg_data["role"] == "user" else MessageRole.ASSISTANT
            wm.add_message(
                Message(
                    role=role,
                    content=msg_data["text"],
                    channel="webui",
                )
            )

        self._conversation_tree.set_active_leaf(conversation_id, leaf_id)
        with self._session_lock:
            self._working_memories[session.session_id] = wm

        log.info(
            "branch_switched",
            conversation=conversation_id[:12],
            leaf=leaf_id[:12],
            messages=len(messages),
        )
        return wm

    async def handle_message(
        self,
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

        # --- Sub-Agent depth guard ---
        # The _agent_runner passes depth in msg.metadata. Enforce max depth
        # to prevent infinite recursive sub-agent delegation.
        _depth = msg.metadata.get("depth", 0) if msg.metadata else 0
        _max_depth = getattr(self._config.security, "max_sub_agent_depth", 3)
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
                text=(
                    f"Sub-Agent Rekursion abgebrochen: "
                    f"Tiefe {_depth} überschreitet "
                    f"Maximum {_max_depth}."
                ),
                is_final=True,
            )

        # Prometheus: count incoming requests
        self._record_metric("requests_total", 1, channel=msg.channel)

        # User-Feedback erkennen und speichern (vor PGE-Zyklus)
        if getattr(self, "_session_analyzer", None):
            try:
                signal = self._session_analyzer._extract_feedback_signal(msg.text)
                if signal is not None:
                    fb_type, detail = signal
                    sid = msg.session_id if hasattr(msg, "session_id") else ""
                    self._session_analyzer.record_user_feedback(
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
        ) = await self._resolve_agent_route(msg)

        # Phase 2: Profiler, Budget, Run-Recording, Policy-Snapshot
        run_id, budget_response = await self._prepare_execution_context(
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
        tool_schemas = self._mcp_client.get_tool_schemas() if self._mcp_client else {}
        if route_decision and route_decision.agent.has_tool_restrictions:
            tool_schemas = route_decision.agent.filter_tools(tool_schemas)

        # Subsystem checks
        if self._planner is None or self._gatekeeper is None or self._executor is None:
            raise RuntimeError("Gateway.initialize() must be called before handle_message()")

        async def _run_context_pipeline():
            if session.incognito:
                log.info("incognito_skip_context", session=session.session_id[:8])
                return
            if self._context_pipeline is not None:
                try:
                    ctx_result = await self._context_pipeline.enrich(msg.text, wm)
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
                _is_coding, _coding_complexity = await self._classify_coding_task(msg.text)
                if _is_coding and self._model_router:
                    if _coding_complexity == "complex":
                        _coding_model = self._model_router._config.models.coder.name
                    else:
                        _coding_model = self._model_router._config.models.coder_fast.name
                    # NOTE: Do NOT call set_coding_override() here — asyncio.create_task()
                    # runs in a copied context, so ContextVar changes are invisible to the
                    # parent. The override is applied in the parent after await (line below).
                    log.info(
                        "coding_task_detected", complexity=_coding_complexity, model=_coding_model
                    )
            except Exception:
                log.debug("coding_classification_skipped", exc_info=True)
            return _is_coding, _coding_model, _coding_complexity

        async def _run_presearch():
            return await self._maybe_presearch(msg, wm)

        import asyncio as _aio

        _ctx_task = _aio.create_task(_run_context_pipeline())
        _coding_task = _aio.create_task(_run_coding_classification())
        _presearch_task = _aio.create_task(_run_presearch())

        await _ctx_task  # Muss vor PGE fertig sein (modifiziert wm)
        is_coding, coding_model, coding_complexity = await _coding_task
        presearch_results = await _presearch_task

        # Apply coding override in parent context (ContextVar must be set here,
        # not inside the create_task — asyncio tasks get a copied context)
        if coding_model and self._model_router:
            self._model_router.set_coding_override(coding_model)

        # Coding-Tasks: mehr Iterationen fuer iteratives Fixen, Debuggen, Optimieren
        # Cognithor soll autonom arbeiten bis die Aufgabe erledigt ist
        if is_coding and session.max_iterations < 50:
            session.max_iterations = 50

        # ── Token Budget (complexity-based) ──
        _token_budget = None
        try:
            from jarvis.core.token_budget import TokenBudgetManager

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
            from jarvis.core.sentiment import (
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
        if hasattr(self, "_user_pref_store") and self._user_pref_store is not None:
            try:
                pref = self._user_pref_store.record_interaction(msg.user_id, len(msg.text))
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
            from jarvis.core.channel_flags import get_channel_flags

            _channel_flags = get_channel_flags(msg.channel)
            if _channel_flags.compact_output or _channel_flags.token_efficient:
                wm.add_message(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            f"Channel: {msg.channel}. "
                            + (
                                "Halte Antworten kurz und kompakt. "
                                if _channel_flags.compact_output
                                else ""
                            )
                            + (
                                f"Max {_channel_flags.max_response_length} Zeichen. "
                                if _channel_flags.max_response_length
                                else ""
                            )
                            + (
                                "Kein Markdown verwenden. "
                                if not _channel_flags.allow_markdown
                                else ""
                            )
                            + (
                                "Keine Code-Bloecke. "
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
        if hasattr(
            self, "_autonomous_orchestrator"
        ) and self._autonomous_orchestrator.should_orchestrate(msg.text):
            auto_task = self._autonomous_orchestrator.create_task(msg.text, session.session_id)
            orchestration_context = self._autonomous_orchestrator.get_orchestration_prompt(
                auto_task
            )
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
            if hasattr(self._gatekeeper, "set_active_skill"):
                self._gatekeeper.set_active_skill(None)
            if (
                active_skill is not None
                and hasattr(self._gatekeeper, "_tool_enforcer")
                and self._gatekeeper._tool_enforcer is not None
                and hasattr(active_skill, "skill")
                and active_skill.skill is not None
            ):
                self._gatekeeper._tool_enforcer.reset_call_count(active_skill.skill.slug)

        # Pipeline callback fuer Presearch + PGE-Loop
        # msg.session_id = WS-URL session_id (vom Client),
        # session.session_id = interner Gateway-Key.
        # Channels nutzen msg.session_id fuer Connection-Lookup.
        _pipeline_cb = self._make_pipeline_callback(msg.channel, msg.session_id)

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
                final_response = await self._answer_from_presearch(msg.text, presearch_results)
                if not final_response:
                    await _pipeline_cb(
                        "execute", "done", iteration=1, success=0, failed=1, total_ms=0
                    )
                    # Fallback: normaler PGE-Loop wenn Antwort-Generierung fehlschlug
                    if active_skill is not None and hasattr(self._gatekeeper, "set_active_skill"):
                        self._gatekeeper.set_active_skill(
                            active_skill.skill if hasattr(active_skill, "skill") else None,
                        )
                    final_response, all_results, all_plans, all_audit = await self._run_pge_loop(
                        msg,
                        session,
                        wm,
                        tool_schemas,
                        route_decision,
                        agent_workspace,
                        run_id,
                        stream_callback=stream_callback,
                    )
                else:
                    await _pipeline_cb(
                        "execute", "done", iteration=1, success=1, failed=0, total_ms=0
                    )
                    await _pipeline_cb("complete", "done", iterations=1, tools_used=1)
                    log.info("presearch_bypass_used", response_chars=len(final_response))
            else:
                # Phase 3: PGE-Loop (regulaerer Ablauf)
                # Community-Skill ToolEnforcer: Aktiven Skill an Gatekeeper weiterreichen
                if active_skill is not None and hasattr(self._gatekeeper, "set_active_skill"):
                    self._gatekeeper.set_active_skill(
                        active_skill.skill if hasattr(active_skill, "skill") else None,
                    )
                final_response, all_results, all_plans, all_audit = await self._run_pge_loop(
                    msg,
                    session,
                    wm,
                    tool_schemas,
                    route_decision,
                    agent_workspace,
                    run_id,
                    stream_callback=stream_callback,
                )
        finally:
            _cleanup_skill_state()

        # Coding-Override aufraeumen
        if self._model_router:
            self._model_router.clear_coding_override()

        # ── Autonomous Task Evaluation ──
        if auto_task is not None:
            auto_task.quality_score = self._autonomous_orchestrator.evaluate_result(
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
        self._persist_key_tool_results(wm, all_results)

        wm.add_message(Message(role=MessageRole.ASSISTANT, content=final_response))

        # ── ConversationTree: Store nodes for chat branching ──────
        if (
            hasattr(self, "_conversation_tree")
            and self._conversation_tree
            and not session.incognito
        ):
            try:
                # Create conversation if not yet assigned
                if not session.conversation_id:
                    session.conversation_id = self._conversation_tree.create_conversation(
                        title=msg.text[:60]
                    )
                # Store user message node
                user_node_id = self._conversation_tree.add_node(
                    session.conversation_id,
                    role="user",
                    text=msg.text,
                    parent_id=session.active_leaf_id or None,
                    agent_name=agent_name,
                )
                # Store assistant response node
                asst_node_id = self._conversation_tree.add_node(
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
        agent_result = AgentResult(
            response=final_response,
            plans=all_plans,
            tool_results=all_results,
            audit_entries=all_audit,
            total_iterations=session.iteration_count,
            total_duration_ms=int((time.monotonic() - _handle_start) * 1000),
            model_used=coding_model
            if is_coding
            else (
                self._model_router.select_model("planning", "high") if self._model_router else ""
            ),
            success=not any(r.is_error for r in all_results) if all_results else True,
        )
        await self._run_post_processing(session, wm, agent_result, active_skill, run_id)

        # Phase 5: Session persistieren
        await self._persist_session(session, wm)

        # Prometheus: Request-Dauer und Token-Metriken
        _duration_ms = (time.monotonic() - _handle_start) * 1000
        self._record_metric("request_duration_ms", _duration_ms, channel=msg.channel)
        _model_used = agent_result.model_used or ""
        if _model_used:
            self._record_metric("tokens_used_total", 1, model=_model_used, role="request")

        # Extract attachments from tool results (e.g. document_export)
        attachments = self._extract_attachments(all_results)

        # Notify active learner of user activity (resets idle timer)
        if self._active_learner is not None:
            self._active_learner.notify_activity()

        if hasattr(self, "_idle_detector") and self._idle_detector:
            self._idle_detector.notify_activity()

        return OutgoingMessage(
            channel=msg.channel,
            text=final_response,
            session_id=session.session_id,
            is_final=True,
            attachments=attachments,
        )

    # ── handle_message sub-methods ────────────────────────────────

    async def _resolve_agent_route(
        self,
        msg: IncomingMessage,
    ) -> tuple[RouteDecision | None, SessionContext, WorkingMemory, Any, Any, str]:
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
                    msg.text,
                    context=msg_context,
                )

            agent_name = route_decision.agent.name

        session = self._get_or_create_session(msg.channel, msg.user_id, agent_name)
        session.touch()
        session.reset_iteration()

        wm = self._get_or_create_working_memory(session)
        wm.clear_for_new_request()

        if self._audit_logger:
            self._audit_logger.log_user_input(
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
        if hasattr(self, "_skill_generator") and self._skill_generator:
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
                    self._skill_generator.gap_detector.report_user_request(
                        msg.text[:200],
                        context=msg.text,
                    )
                    break

        active_skill = None
        if self._skill_registry is not None:
            try:
                tool_list = self._mcp_client.get_tool_list() if self._mcp_client else []
                active_skill = self._skill_registry.inject_into_working_memory(
                    msg.text,
                    wm,
                    available_tools=tool_list,
                )
                # Gap Detection: Melde wenn kein Skill zur Anfrage passt
                if (
                    active_skill is None
                    and hasattr(self, "_skill_generator")
                    and self._skill_generator
                ):
                    self._skill_generator.gap_detector.report_no_skill_match(msg.text)
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
        session: SessionContext,
        wm: WorkingMemory,
        route_decision: RouteDecision | None,
    ) -> tuple[str | None, OutgoingMessage | None]:
        """Phase 2: Profiler, Budget, Run-Recording, Policy-Snapshot.

        Returns:
            (run_id, budget_response) -- budget_response is not None if budget exceeded.
        """
        if hasattr(self, "_task_profiler") and self._task_profiler:
            try:
                self._task_profiler.start_task(
                    session_id=session.session_id,
                    task_description=msg.text[:200],
                )
            except Exception:
                log.debug("task_profiler_start_failed", exc_info=True)

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

    def _make_status_callback(
        self,
        channel_name: str,
        session_id: str,
    ) -> Any:
        """Creates a fire-and-forget status callback for the current channel.

        Returns an async callable (status_type: str, text: str) -> None.
        """

        async def _send_status(status_type: str, text: str) -> None:
            channel = self._channels.get(channel_name)
            if channel is None:
                return
            try:
                from jarvis.channels.base import StatusType

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

    def _make_pipeline_callback(
        self,
        channel_name: str,
        session_id: str,
    ) -> Any:
        """Creates a fire-and-forget pipeline event callback.

        Returns an async callable for sending structured PGE pipeline
        events to the frontend for the live pipeline visualization.
        """
        _start_mono = time.monotonic()

        async def _send_pipeline(phase: str, status: str, **extra: Any) -> None:
            channel = self._channels.get(channel_name)
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

    async def _formulate_response(
        self,
        msg_text: str,
        all_results: list[ToolResult],
        wm: WorkingMemory,
        stream_callback: Any | None = None,
    ) -> str:
        """Formulate response, optionally streaming tokens to the client.

        If stream_callback is set and the planner supports streaming,
        tokens are sent as stream_token events in real time.
        Falls back to non-streaming formulate_response() otherwise.
        """
        if stream_callback is not None and hasattr(self._planner, "formulate_response_stream"):
            try:
                return await self._planner.formulate_response_stream(
                    user_message=msg_text,
                    results=all_results,
                    working_memory=wm,
                    stream_callback=stream_callback,
                )
            except Exception:
                log.debug("streaming_formulate_failed_fallback", exc_info=True)
                # Fall through to non-streaming
        return await self._planner.formulate_response(
            user_message=msg_text,
            results=all_results,
            working_memory=wm,
        )

    async def _run_pge_loop(
        self,
        msg: IncomingMessage,
        session: SessionContext,
        wm: WorkingMemory,
        tool_schemas: dict[str, Any],
        route_decision: RouteDecision | None,
        agent_workspace: Any,
        run_id: str | None,
        stream_callback: Any | None = None,
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
        _consecutive_no_tool_iters = 0  # Detect stuck replan loops
        _max_no_tool_iters = 2  # After 2 iters without tool execution, stop

        # Status callback for progress feedback
        # Nutze msg.session_id (Client/WS-ID), nicht session.session_id (intern)
        _status_cb = self._make_status_callback(msg.channel, msg.session_id)
        # Pipeline callback for live PGE visualization (WebUI only)
        _pipeline_cb = self._make_pipeline_callback(msg.channel, msg.session_id)

        # Identity Layer reference (set during Phase init)
        _identity = getattr(self, "_identity_layer", None)

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
            if hasattr(self, "_correction_memory") and self._correction_memory:
                self._correction_memory.store(
                    user_message=getattr(session, "last_user_message", "") or "",
                    correction_text=msg.text,
                )
            # Feed correction into Evolution Engine as learning gap
            if hasattr(self, "_deep_learner") and self._deep_learner:
                try:
                    last_msg = getattr(session, "last_user_message", "") or msg.text
                    gap = f"User-Korrektur: {last_msg[:100]} → {msg.text[:100]}"
                    config = getattr(self, "_config", None)
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
                    content=(
                        f"[KORREKTUR] Der User hat korrigiert: "
                        f'"{msg.text}". Passe deinen Plan an. '
                        f"Fuehre NICHT die vorherige Aktion erneut aus."
                    ),
                    channel=msg.channel,
                )
            )

        while not session.iterations_exhausted and self._running:
            # Cancel-Check: User hat /stop oder cancel gesendet
            if msg.session_id in self._cancelled_sessions:
                self._cancelled_sessions.discard(msg.session_id)
                log.info("pge_cancelled_by_user", session=session.session_id[:8])
                final_response = "Verarbeitung abgebrochen. Was kann ich stattdessen fuer dich tun?"
                break

            session.iteration_count += 1
            await _pipeline_cb("iteration", "start", iteration=session.iteration_count)

            # Check token budget and compact if necessary
            self._check_and_compact(wm, session)

            log.info(
                "agent_loop_iteration",
                iteration=session.iteration_count,
                session=session.session_id[:8],
                chat_messages=len(wm.chat_history),
                token_estimate=wm.token_count,
            )

            # Status: Thinking (with periodic keepalive for long-running plans)
            await _status_cb("thinking", "Denke nach...")
            await _pipeline_cb("plan", "start", iteration=session.iteration_count)

            # Keepalive: send periodic status updates while planner works
            _keepalive_running = True

            async def _thinking_keepalive() -> None:
                """Send periodic status updates so the UI shows activity."""
                _elapsed = 0
                _messages = [
                    "Denke nach...",
                    "Plane Schritte...",
                    "Analysiere Aufgabe...",
                    "Erstelle Plan...",
                    "Arbeite daran...",
                ]
                while _keepalive_running:
                    await asyncio.sleep(5)
                    if not _keepalive_running:
                        break
                    _elapsed += 5
                    _msg = _messages[min(_elapsed // 10, len(_messages) - 1)]
                    with contextlib.suppress(Exception):
                        await _status_cb("thinking", f"{_msg} ({_elapsed}s)")

            _keepalive_task = asyncio.create_task(_thinking_keepalive())
            self._background_tasks.add(_keepalive_task)

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

            # Planner
            if session.iteration_count == 1:
                plan = await self._planner.plan(
                    user_message=msg.text,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                    model_override=_agent_model,
                    temperature_override=_agent_temperature,
                    top_p_override=_agent_top_p,
                )
            else:
                plan = await self._planner.replan(
                    original_goal=msg.text,
                    results=all_results,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                    model_override=_agent_model,
                    temperature_override=_agent_temperature,
                    top_p_override=_agent_top_p,
                )

            # Stop keepalive once planner responds
            _keepalive_running = False
            _keepalive_task.cancel()
            with contextlib.suppress(BaseException):
                await _keepalive_task
            self._background_tasks.discard(_keepalive_task)

            all_plans.append(plan)
            await _pipeline_cb(
                "plan",
                "done",
                iteration=session.iteration_count,
                has_actions=plan.has_actions,
                steps=len(plan.steps) if plan.has_actions else 0,
            )

            # ── Pre-Flight Notification (non-blocking, agentic-first) ──
            _recovery_cfg = getattr(self._config, "recovery", None)
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
                    if msg.session_id in self._cancelled_sessions:
                        self._cancelled_sessions.discard(msg.session_id)
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
                channel = self._channels.get(msg.channel)
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

            if run_id and self._run_recorder:
                try:
                    self._run_recorder.record_plan(run_id, plan)
                except Exception:
                    log.debug("run_recorder_plan_failed", exc_info=True)

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
                    final_response = await self._formulate_response(
                        msg.text,
                        all_results,
                        wm,
                        stream_callback,
                    )
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
                        final_response = await self._formulate_response(
                            msg.text,
                            [],
                            wm,
                            stream_callback,
                        )
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
                        final_response = await self._formulate_response(
                            msg.text,
                            all_results,
                            wm,
                            stream_callback,
                        )
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
                    final_response = await self._formulate_response(
                        msg.text,
                        all_results,
                        wm,
                        stream_callback,
                    )
                    break

                final_response = plan.direct_response
                break

            if not plan.has_actions:
                # If there are prior successful results, summarize them
                if all_results and any(r.success for r in all_results):
                    await _status_cb("finishing", "Composing response...")
                    final_response = await self._formulate_response(
                        msg.text,
                        all_results,
                        wm,
                        stream_callback,
                    )
                    break
                final_response = (
                    "I couldn't create a plan for this. Could you rephrase your question?"
                )
                break

            # Gatekeeper
            await _pipeline_cb("gate", "start", iteration=session.iteration_count)
            decisions = self._gatekeeper.evaluate_plan(plan.steps, session)

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

            # Approvals
            approved_decisions = await self._handle_approvals(
                plan.steps,
                decisions,
                session,
                msg.channel,
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
                    try:
                        from jarvis.utils.error_messages import all_actions_blocked_message

                        final_response = all_actions_blocked_message(plan.steps, approved_decisions)
                    except Exception:
                        final_response = "All planned actions were blocked by the Gatekeeper."
                break

            # Status: Tool-specific progress message
            for step in plan.steps:
                tool_status = _TOOL_STATUS_MAP.get(step.tool, f"Running {step.tool}...")
                await _status_cb("executing", tool_status)
                break  # Only send the first tool's status

            # Stream: tool_start events for each planned step
            if stream_callback is not None:
                for step in plan.steps:
                    try:
                        await stream_callback(
                            "tool_start",
                            {
                                "tool": step.tool,
                                "status": _TOOL_STATUS_MAP.get(
                                    step.tool, f"Running {step.tool}..."
                                ),
                            },
                        )
                    except Exception:
                        log.debug("stream_tool_start_failed", exc_info=True)

            # Set status callback on executor for retry visibility
            self._executor.set_status_callback(_status_cb)
            await _pipeline_cb(
                "execute",
                "start",
                iteration=session.iteration_count,
                tools=[s.tool for s in plan.steps],
            )

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

            # Faktenfrage: cross_check fuer search_and_read auto-injizieren
            # (muss NACH set_agent_context, da dieses clear_agent_context aufruft)
            if self._is_fact_question(msg.text):
                self._executor.set_fact_question_context(True)

            try:
                results = await self._executor.execute(plan.steps, approved_decisions)
            finally:
                self._executor.clear_agent_context()

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

            if run_id and self._run_recorder:
                try:
                    self._run_recorder.record_gate_decisions(run_id, approved_decisions)
                    self._run_recorder.record_tool_results(run_id, results)
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
                        execution_result="ok"
                        if result.success
                        else result.error_message or "error",
                    )
                )
                # Prometheus: Tool-Aufruf-Metriken
                self._record_metric("tool_calls_total", 1, tool_name=result.tool_name)
                if hasattr(result, "duration_ms") and result.duration_ms:
                    self._record_metric(
                        "tool_duration_ms",
                        result.duration_ms,
                        tool_name=result.tool_name,
                    )
                if result.is_error:
                    self._record_metric(
                        "errors_total",
                        1,
                        channel=msg.channel,
                        error_type="tool_error",
                    )

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
                        final_response = await self._formulate_response(
                            msg.text,
                            all_results,
                            wm,
                            stream_callback,
                        )
                    else:
                        final_response = (
                            "I'm stuck in a planning loop without making progress. "
                            "Please try a more specific request — e.g. "
                            "'Write a Pac-Man main.py' instead of 'Create a game'."
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
                final_response = await self._formulate_response(
                    msg.text,
                    all_results,
                    wm,
                    stream_callback,
                )
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
                final_response = await self._formulate_response(
                    msg.text,
                    all_results,
                    wm,
                    stream_callback,
                )
                break
                # Otherwise: continue to replan for more steps (normal)

            # Failure-Threshold: give planner room for alternative strategies
            # Only give up after 70% of max_iterations with no success at all
            _failure_threshold = max(5, int(session.max_iterations * 0.7))
            if not has_success and session.iteration_count >= _failure_threshold:
                await _status_cb("finishing", "Composing response...")
                final_response = await self._formulate_response(
                    msg.text,
                    all_results,
                    wm,
                    stream_callback,
                )
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

    async def _run_post_processing(
        self,
        session: SessionContext,
        wm: WorkingMemory,
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
                # Apply reflection to memory tiers (episodic, semantic, procedural)
                if self._memory_manager:
                    try:
                        counts = await self._reflector.apply(reflection, self._memory_manager)
                        log.info(
                            "reflection_applied",
                            session=session.session_id[:8],
                            episodic=counts.get("episodic", 0),
                            semantic=counts.get("semantic", 0),
                            procedural=counts.get("procedural", 0),
                        )
                    except Exception as apply_exc:
                        log.error("reflection_apply_error", error=str(apply_exc))
                if run_id and self._run_recorder:
                    try:
                        self._run_recorder.record_reflection(run_id, reflection)
                    except Exception:
                        log.debug("run_recorder_reflection_failed", exc_info=True)
                # ── Feed weak results into Evolution Engine ──────────
                # If reflection shows poor quality or user corrected us,
                # create a learning goal so the Evolution Engine researches it.
                if (
                    reflection.success_score < 0.5
                    and hasattr(self, "_deep_learner")
                    and self._deep_learner
                    and hasattr(self, "_evolution_loop")
                    and self._evolution_loop
                ):
                    try:
                        user_msg = msg.text[:200] if hasattr(msg, "text") else ""
                        gap_description = (
                            f"Schwache Antwort (Score {reflection.success_score:.1f}) "
                            f"auf: {user_msg}"
                        )
                        # Add as learning goal for next evolution cycle
                        config = getattr(self, "_config", None)
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

        if active_skill and self._skill_registry:
            try:
                success = agent_result.success
                score = (
                    agent_result.reflection.success_score
                    if agent_result.reflection
                    else (0.8 if success else 0.3)
                )
                self._skill_registry.record_usage(
                    active_skill.skill.slug,
                    success=success,
                    score=score,
                )
                # Store failure pattern in procedure (for learning effect)
                if not success and self._memory_manager and active_skill.procedure_name:
                    try:
                        error_summary = (
                            agent_result.error[:200] if agent_result.error else "unknown"
                        )
                        self._memory_manager.procedural.add_failure_pattern(
                            active_skill.procedure_name,
                            error_summary,
                        )
                    except Exception:
                        log.debug("procedure_failure_pattern_save_failed", exc_info=True)

                # Gap Detection: Melde niedrige Erfolgsrate
                if not success and hasattr(self, "_skill_generator") and self._skill_generator:
                    try:
                        skill_obj = active_skill.skill
                        if skill_obj.total_uses >= 3 and skill_obj.total_uses > 0:
                            success_rate = skill_obj.success_count / skill_obj.total_uses
                            if success_rate < 0.4:
                                self._skill_generator.gap_detector.report_low_success_rate(
                                    skill_obj.slug,
                                    success_rate,
                                )
                    except Exception:
                        log.debug("skill_gap_detection_failed", exc_info=True)
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

        # Prompt-Evolution: Record session reward for A/B testing
        if getattr(self, "_prompt_evolution", None) and self._planner:
            try:
                version_id = getattr(self._planner, "_current_prompt_version_id", None)
                if version_id:
                    reward_score = (
                        agent_result.reflection.success_score
                        if agent_result.reflection
                        else (0.8 if agent_result.success else 0.3)
                    )
                    self._prompt_evolution.record_session(
                        session_id=session.session_id,
                        prompt_version_id=version_id,
                        reward=reward_score,
                    )
            except Exception:
                log.debug("prompt_evolution_record_failed", exc_info=True)

        # GEPA: Collect execution trace
        if getattr(self, "_trace_store", None):
            try:
                import time as _time
                import uuid as _uuid

                from jarvis.learning.execution_trace import ExecutionTrace, TraceStep

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
                self._trace_store.save_trace(trace)
                log.debug("gepa_trace_saved", trace_id=trace.trace_id, steps=len(trace.steps))
            except Exception:
                log.debug("gepa_trace_save_failed", exc_info=True)

        # Reflexion: check for known solutions before recording new errors
        if getattr(self, "_reflexion_memory", None) and hasattr(agent_result, "tool_results"):
            try:
                for tr in agent_result.tool_results or []:
                    if getattr(tr, "is_error", False) or getattr(tr, "error", None):
                        tool = getattr(tr, "tool_name", "") or str(getattr(tr, "name", ""))
                        error_msg = str(getattr(tr, "error", "") or getattr(tr, "error_type", ""))
                        known = self._reflexion_memory.get_solution(tool, "unknown", error_msg)
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
                            self._reflexion_memory.record_error(
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
        if getattr(self, "_evolution_orchestrator", None):
            try:
                import time as _time

                orch = self._evolution_orchestrator
                gepa_cfg = getattr(self._config, "gepa", None)
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
        if getattr(self, "_session_analyzer", None):
            try:
                improvements = await self._session_analyzer.analyze_session(
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
            except Exception:
                log.debug("session_analysis_failed", exc_info=True)

        # Pattern Documentation: record successful tool sequences
        if self._memory_manager:
            try:
                self._maybe_record_pattern(session, wm, agent_result)
            except Exception:
                log.debug("pattern_documentation_post_failed", exc_info=True)

        # Self-Learning: Process actionable skill gaps (auto-generate new tools)
        if hasattr(self, "_skill_generator") and self._skill_generator:
            try:
                generated = await self._skill_generator.process_all_gaps(
                    skill_registry=self._skill_registry
                    if hasattr(self, "_skill_registry")
                    else None,
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
                        self._sync_core_inventory()
                    except Exception:
                        log.debug("core_inventory_sync_after_skill_gen_failed", exc_info=True)
            except Exception:
                log.debug("skill_gap_processing_failed", exc_info=True)

    # ── Pattern Documentation ────────────────────────────────────

    # Rate limiter for pattern recordings: max 5 per hour
    _PATTERN_MAX_PER_HOUR: ClassVar[int] = 5

    def _maybe_record_pattern(
        self,
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
            self._pattern_record_timestamps[:] = [
                ts for ts in self._pattern_record_timestamps if now - ts < 3600
            ]
            if len(self._pattern_record_timestamps) >= self._PATTERN_MAX_PER_HOUR:
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
            if self._memory_manager:
                procedural = getattr(self._memory_manager, "procedural", None)
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
                        from jarvis.models import ProcedureMetadata

                        # Generate readable name from keywords (max 5 words, slugified)
                        _name_words = [
                            re.sub(r"[^\w]", "", k.lower()) for k in keywords[:3] if len(k) > 2
                        ]
                        if _name_words:
                            name = "-".join(_name_words)
                        else:
                            # Fallback: use first meaningful words from user text
                            _text_words = [
                                w.lower()
                                for w in user_text.split()[:4]
                                if len(w) > 2 and w.isalpha()
                            ]
                            name = "-".join(_text_words) if _text_words else f"auto-{int(now)}"
                        # Ensure uniqueness by appending short hash if file exists
                        _base_name = name
                        if (procedural._dir / f"{name}.md").exists():
                            _short = hashlib.sha256(
                                f"{tools_str}:{keywords_str}".encode()
                            ).hexdigest()[:6]
                            name = f"{_base_name}-{_short}"
                            # If even that exists, skip (true duplicate)
                            if (procedural._dir / f"{name}.md").exists():
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
                        self._pattern_record_timestamps.append(now)
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

    async def _persist_session(
        self,
        session: SessionContext,
        wm: WorkingMemory,
    ) -> None:
        """Phase 5: Session persistieren."""
        # Incognito: nur Session-Metadaten speichern, keine Chat-History
        if session.incognito:
            if self._session_store:
                try:
                    self._session_store.save_session(session)
                except Exception as exc:
                    log.warning("session_persist_error", error=str(exc))
            return
        if self._session_store:
            try:
                self._session_store.save_session(session)
                self._session_store.save_chat_history(
                    session.session_id,
                    wm.chat_history,
                )
            except Exception as exc:
                log.warning("session_persist_error", error=str(exc))
            # Auto-Titel aus erster User-Message generieren
            if hasattr(self._session_store, "auto_title"):
                try:
                    self._session_store.auto_title(session.session_id)
                except Exception:
                    log.debug("auto_title_failed", exc_info=True)

    # =========================================================================
    # Agent-zu-Agent Delegation
    # =========================================================================

    async def execute_delegation(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        session: SessionContext,
        parent_wm: WorkingMemory,
    ) -> str:
        """Fuehrt eine echte Agent-zu-Agent-Delegation aus.

        Der delegierte Agent bekommt:
          - Eigenen System-Prompt
          - Eigenen Workspace (isoliert)
          - Eigene Sandbox-Config
          - Eigene Tool-Filterung
          - Die Aufgabe als User-Nachricht

        Das Ergebnis fliesst als Text zurueck zum aufrufenden Agenten.

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
            return f"Agent router unavailable. Delegation to {to_agent} failed."

        # Delegation erstellen und validieren
        delegation = self._agent_router.create_delegation(from_agent, to_agent, task)
        if delegation is None:
            return (
                f"Delegation from {from_agent} to {to_agent} not allowed. "
                f"I'll handle the task myself."
            )

        target = delegation.target_profile
        if not target:
            return f"Agent {to_agent} not found."

        log.info(
            "delegation_executing",
            from_=from_agent,
            to=to_agent,
            task=task[:200],
            depth=delegation.depth,
        )

        # Separate working memory for delegated agent
        sub_wm = WorkingMemory(session_id=session.session_id)

        # System-Prompt des Ziel-Agenten injizieren
        if target.system_prompt:
            sub_wm.add_message(
                Message(
                    role=MessageRole.SYSTEM,
                    content=target.system_prompt,
                )
            )

        # Aufgabe als User-Nachricht
        sub_wm.add_message(
            Message(
                role=MessageRole.USER,
                content=task,
            )
        )

        # Resolve target agent's workspace
        target_workspace = self._agent_router.resolve_agent_workspace(
            to_agent,
            self._config.workspace_dir,
        )

        # Filter tool schemas for target agent
        tool_schemas = self._mcp_client.get_tool_schemas() if self._mcp_client else {}
        if target.has_tool_restrictions:
            tool_schemas = target.filter_tools(tool_schemas)

        # Planner mit Ziel-Agent-Kontext aufrufen
        if self._planner is None:
            raise RuntimeError("Planner nicht initialisiert -- Delegation nicht möglich")

        # Agent-specific LLM overrides for delegation target
        _del_model = target.preferred_model or None
        _del_temp = target.temperature
        _del_top_p = getattr(target, "top_p", None)

        plan = await self._planner.plan(
            user_message=task,
            working_memory=sub_wm,
            tool_schemas=tool_schemas,
            model_override=_del_model,
            temperature_override=_del_temp,
            top_p_override=_del_top_p,
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

        # Check gatekeeper
        if self._gatekeeper is None:
            raise RuntimeError("Gatekeeper nicht initialisiert -- Delegation nicht möglich")
        decisions = self._gatekeeper.evaluate_plan(plan.steps, session)

        # APPROVE/BLOCK-Entscheidungen in Delegationen blockieren (kein HITL moeglich)
        blocked = [d for d in decisions if d.status in (GateStatus.APPROVE, GateStatus.BLOCK)]
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
            session_id=session.session_id,
        )

        try:
            results = await self._executor.execute(plan.steps, decisions)
        finally:
            self._executor.clear_agent_context()

        # Formulate result
        if any(r.success for r in results):
            response = await self._planner.formulate_response(
                user_message=task,
                results=results,
                working_memory=sub_wm,
            )
            delegation.result = response
            delegation.success = True
        else:
            delegation.result = "Delegation failed: no successful actions."
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

    # Tools whose results should persist in chat history for follow-up requests.
    # Without this persistence, context (e.g. extracted text from images)
    # is lost on clear_for_new_request().
    _CONTEXT_TOOLS: frozenset[str] = frozenset(
        {
            "media_analyze_image",
            "media_extract_text",
            "media_transcribe_audio",
            "analyze_code",
            "run_python",
            "web_search",
            "web_fetch",
            "search_and_read",
        }
    )
    # Maximum character count for persisted tool results in chat history
    _CONTEXT_RESULT_LIMIT: int = 4000

    def _persist_key_tool_results(
        self,
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
            if result.tool_name not in self._CONTEXT_TOOLS:
                continue
            if not result.content.strip():
                continue

            content = result.content[: self._CONTEXT_RESULT_LIMIT]
            if len(result.content) > self._CONTEXT_RESULT_LIMIT:
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

    # Tools whose results are file paths that should be attached to the response
    _ATTACHMENT_TOOLS: frozenset[str] = frozenset(
        {
            "document_export",
        }
    )
    # File extensions considered valid attachments
    _ATTACHMENT_EXTENSIONS: frozenset[str] = frozenset(
        {
            ".pdf",
            ".docx",
            ".doc",
            ".xlsx",
            ".csv",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
        }
    )

    # ── Prometheus Metric Recording ──────────────────────────────

    def _record_metric(self, name: str, value: float, **labels: str) -> None:
        """Zeichnet eine Metrik auf (wenn MonitoringHub oder TelemetryHub verfuegbar).

        Schreibt in beide Subsysteme wenn vorhanden:
          - MonitoringHub.metrics (MetricCollector) -- fuer Dashboard + Prometheus
          - TelemetryHub.metrics (MetricsProvider)  -- fuer OTLP + Prometheus
        """
        # MetricCollector (gateway/monitoring.py)
        hub = getattr(self, "_monitoring_hub", None)
        if hub is not None:
            collector = getattr(hub, "metrics", None)
            if collector is not None:
                try:
                    collector.increment(name, value, **labels)
                except Exception:
                    log.debug("metric_collector_failed", metric=name, exc_info=True)

        # MetricsProvider (telemetry/metrics.py) via TelemetryHub
        telemetry = getattr(self, "_telemetry_hub", None)
        if telemetry is not None:
            provider = getattr(telemetry, "metrics", None)
            if provider is not None:
                try:
                    # Determine metric type based on name
                    if name.endswith("_ms"):
                        provider.histogram(name, value, **labels)
                    else:
                        provider.counter(name, value, **labels)
                except Exception:
                    log.debug("metric_provider_failed", metric=name, exc_info=True)

    def _extract_attachments(self, results: list[ToolResult]) -> list[str]:
        """Extrahiert Dateipfade aus Tool-Ergebnissen fuer den Anhang-Versand.

        Prueft ob das Tool-Ergebnis einen gueltigen Dateipfad enthaelt und ob
        die Datei existiert.
        """
        from pathlib import Path

        attachments: list[str] = []
        for result in results:
            if not result.success:
                continue
            if result.tool_name not in self._ATTACHMENT_TOOLS:
                continue
            # content contains the file path
            candidate = result.content.strip()
            if not candidate:
                continue
            try:
                path = Path(candidate)
                if (
                    path.exists()
                    and path.is_file()
                    and path.suffix.lower() in self._ATTACHMENT_EXTENSIONS
                ):
                    attachments.append(str(path))
                    log.info("attachment_detected", tool=result.tool_name, path=str(path))
            except (ValueError, OSError):
                continue
        return attachments

    # ── Automatic pre-search for factual questions ────────────────────

    # Regex patterns for factual questions (when/where/who/what + verb)
    _FACT_QUESTION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(wann|wo|wer|was|wie viele|welche[rsmn]?)\b"
            r".{3,}"
            r"(hat|haben|ist|sind|wurde|wurden"
            r"|war|waren|gibt|gab|passiert|geschehen"
            r"|entführ|verhaft|angegriff|getötet"
            r"|gestorben|gewählt|gestürzt"
            r"|finde[nt]?|stattfinde[nt]?|statt"
            r"|spiele[nt]?|laufe[nt]?|läuft"
            r"|komm[ent]?|beginne[nt]?|beginn"
            r"|anfange[nt]?|fängt|endet"
            r"|aufgetreten|gestartet|eröffnet"
            r"|erschien|veröffentlich)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(when|where|who|what|how many|which)\b.{3,}(did|has|have|was|were|is|are|happened)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(stimmt es|ist es wahr|hat .+ wirklich)\b",
            re.IGNORECASE,
        ),
    ]

    # Begriffe die KEINE Faktenfrage signalisieren (Smalltalk, Meinungen)
    _SKIP_PRESEARCH_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(meinst du|findest du|was denkst du|erkläre mir|was ist ein|definier)",
            re.IGNORECASE,
        ),
        # Trailing \b entfernt: "erstell" muss auch "erstelle/erstellst/erstellen" matchen
        re.compile(
            r"\b(schreib|erstell|generier|mach|öffne|lösch|speicher|such im memory)", re.IGNORECASE
        ),
    ]

    def _is_fact_question(self, text: str) -> bool:
        """Prueft ob eine Nachricht eine Faktenfrage ist, die Web-Recherche braucht."""
        # Zu kurz → wahrscheinlich kein Fakten-Query
        if len(text) < 15:
            return False

        # Check skip patterns (commands, opinions, explanations)
        for skip_pat in self._SKIP_PRESEARCH_PATTERNS:
            if skip_pat.search(text):
                return False

        # Check factual question patterns
        return any(fact_pat.search(text) for fact_pat in self._FACT_QUESTION_PATTERNS)

    async def _classify_coding_task(self, user_message: str) -> tuple[bool, str]:
        """Klassifiziert ob eine Nachricht eine Coding-Aufgabe ist und deren Komplexitaet.

        Nutzt einen schnellen LLM-Call mit dem Executor-Modell.

        Returns:
            (is_coding, complexity) -- complexity ist "simple" oder "complex"
        """
        if not self._model_router or not self._llm:
            return False, "simple"

        classify_prompt = (
            "Klassifiziere die folgende Nachricht:\n"
            "1. Ist es eine Coding/Programmier-Aufgabe? (ja/nein)\n"
            "2. Wenn ja: Ist es einfach (einzelne Funktion, kleines Fix, Snippet)\n"
            "   oder komplex (Multi-File, Architektur, Refactoring, neues Feature)?\n\n"
            'Antworte NUR mit einem JSON: {"coding": true/false, "complexity": "simple"/"complex"}'
        )

        model = self._model_router.select_model("simple_tool_call", "low")

        try:
            response = await self._llm.chat(
                model=model,
                messages=[
                    {"role": "system", "content": classify_prompt},
                    {"role": "user", "content": user_message[:500]},
                ],
                temperature=0.1,
                format_json=True,
            )

            text = response.get("message", {}).get("content", "")
            # JSON aus Antwort extrahieren
            import json as _json_mod

            # <think>...</think> Bloecke entfernen (qwen3)
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
            data = _json_mod.loads(text)
            is_coding = bool(data.get("coding", False))
            complexity = data.get("complexity", "simple")
            if complexity not in ("simple", "complex"):
                complexity = "simple"
            return is_coding, complexity

        except Exception as exc:
            log.debug("coding_classify_failed", error=str(exc)[:200])
            return False, "simple"

    @staticmethod
    def _resolve_relative_dates(text: str) -> str:
        """Ersetzt relative Zeitangaben durch konkrete Datumsangaben.

        'morgen' → '01.03.2026', 'heute' → '28.02.2026', etc.
        """
        from datetime import datetime, timedelta

        now = datetime.now()
        today = now.date()

        # Mapping: (regex_pattern, Datum-Offset oder Callback)
        replacements: list[tuple[str, str]] = [
            (r"\bheute\b", today.strftime("%d.%m.%Y")),
            (r"\bmorgen\b", (today + timedelta(days=1)).strftime("%d.%m.%Y")),
            (r"\bübermorgen\b", (today + timedelta(days=2)).strftime("%d.%m.%Y")),
            (r"\bgestern\b", (today - timedelta(days=1)).strftime("%d.%m.%Y")),
            (r"\bvorgestern\b", (today - timedelta(days=2)).strftime("%d.%m.%Y")),
        ]

        # Weekday-based resolution: "naechsten Montag", "am Freitag", etc.
        _wochentage = {
            "montag": 0,
            "dienstag": 1,
            "mittwoch": 2,
            "donnerstag": 3,
            "freitag": 4,
            "samstag": 5,
            "sonntag": 6,
        }
        for tag_name, weekday_num in _wochentage.items():
            days_ahead = (weekday_num - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # "am Montag" = nächster Montag
            target = today + timedelta(days=days_ahead)
            replacements.append(
                (
                    rf"\b(?:nächsten?\s+|am\s+|kommenden?\s+)?{tag_name}\b",
                    target.strftime("%d.%m.%Y"),
                ),
            )

        # "naechste Woche" / "diese Woche" / "dieses Wochenende"
        replacements.append(
            (
                r"\bnächste(?:r|s|n)?\s+woche\b",
                f"Woche ab {(today + timedelta(days=7 - today.weekday())).strftime('%d.%m.%Y')}",
            ),
        )
        replacements.append(
            (
                r"\bdiese(?:s|r|n)?\s+wochenende\b",
                f"{(today + timedelta(days=5 - today.weekday())).strftime('%d.%m.%Y')}",
            ),
        )

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    async def _maybe_presearch(self, msg: IncomingMessage, wm: WorkingMemory) -> str | None:
        """Fuehrt automatisch eine Web-Suche durch wenn die Nachricht eine Faktenfrage ist.

        Returns:
            Suchergebnis-Text wenn Ergebnisse gefunden wurden, sonst None.
        """
        if not self._is_fact_question(msg.text):
            return None

        # WebTools-Instanz finden
        web_tools = None
        if self._mcp_client:
            web_tools = getattr(self._mcp_client, "_web_tools", None)
            if web_tools is None:
                # Fallback: WebTools aus registrierten Handlern extrahieren
                handler = self._mcp_client.get_handler("web_search")
                if handler is not None:
                    # Handler ist eine gebundene Methode von WebTools
                    web_tools = getattr(handler, "__self__", None)

        if web_tools is None:
            log.debug("presearch_skip_no_webtools")
            return None

        # Formulate search query as keywords (not as a question)
        query = msg.text.strip()
        # Strip command suffixes ("Recherchiere das online", etc.)
        # Longer phrases first so "bitte such" matches before "such"
        for splitter in (
            "recherchiere das",
            "recherchiere",
            "recherchier",
            "bitte such",
            "such das",
            "such online",
            "finde heraus",
            "schau nach",
            "google",
        ):
            idx = query.lower().find(splitter)
            if idx > 10:  # Nur abschneiden wenn genug Fragetext davor steht
                query = query[:idx].strip()
                break
        query = query.rstrip("?!.").strip()
        # Remove question words for better search results
        for prefix in (
            "wann hat",
            "wann haben",
            "wann wurde",
            "wann war",
            "wo hat",
            "wo haben",
            "wo wurde",
            "wo war",
            "wer hat",
            "wer ist",
            "was ist mit",
            "was hat",
            "stimmt es dass",
            "ist es wahr dass",
        ):
            if query.lower().startswith(prefix):
                query = query[len(prefix) :].strip()
                break

        # Resolve relative time references to concrete dates
        query = self._resolve_relative_dates(query)

        try:
            log.info("presearch_start", query=query[:80])
            result_text = await web_tools.web_search(
                query=query,
                num_results=5,
                language="de",
                timelimit="m",
            )

            if (
                result_text
                and _PRESEARCH_NO_RESULTS not in result_text
                and _PRESEARCH_NO_ENGINE not in result_text
            ):
                log.info("presearch_found", chars=len(result_text))
                return result_text[:8000]
            else:
                log.info("presearch_no_results", query=query[:80])
                return None

        except Exception as exc:
            log.warning("presearch_failed", error=str(exc)[:200])
            return None

    async def _answer_from_presearch(self, user_message: str, search_results: str) -> str:
        """Generiert eine Antwort AUSSCHLIEẞLICH basierend auf Suchergebnissen.

        Umgeht den Planner komplett — das LLM bekommt NUR die Suchergebnisse
        und die Frage des Users, ohne Moeglichkeit auf Trainingswissen zurueckzugreifen.

        Nutzt den unified LLM-Client (funktioniert mit jedem Backend).
        """
        if not self._llm:
            return ""
        system = (
            "You are a fact assistant. You answer questions EXCLUSIVELY "
            "based on the provided search results.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Use ONLY information from the search results below.\n"
            "2. The search results are CURRENT FACTS from the internet.\n"
            "3. If the search results describe an event, it DID happen.\n"
            "4. NEVER say 'there is no evidence' or 'that did not happen'.\n"
            "5. NEVER say 'according to my knowledge' or 'as far as I know'.\n"
            "6. Cite dates, names, and facts DIRECTLY from the results.\n"
            "7. Answer in the same language as the user's question, concisely and informatively.\n"
            "8. You have NO knowledge of your own. You know ONLY the search results.\n"
            "9. Answer DIRECTLY without reasoning process. Brief and factual."
        )

        # /no_think disables qwen3's internal reasoning mode for fast response
        user_prompt = (
            f"SEARCH RESULTS:\n\n{search_results}\n\n"
            f"---\n\n"
            f"QUESTION: {user_message}\n\n"
            f"Answer the question ONLY based on the search results above. /no_think"
        )

        # Select model via ModelRouter (backend-agnostic)
        if self._model_router:
            model = self._model_router.select_model("planning", "high")
        else:
            model = self._config.models.planner.name

        try:
            response = await self._llm.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                top_p=0.9,
            )

            answer = response.get("message", {}).get("content", "")
            # qwen3 may include <think>...</think> blocks — remove them
            answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
            if answer.strip():
                log.info("presearch_answer_generated", chars=len(answer))
                return answer.strip()

        except Exception as exc:
            log.error("presearch_answer_failed", error=str(exc)[:200])

        # Fallback: use regular PGE loop
        return ""

    def _cleanup_stale_sessions(self) -> None:
        """Remove sessions that have not been accessed for more than _SESSION_TTL_SECONDS.

        This is called periodically (guarded by _CLEANUP_INTERVAL_SECONDS) to
        prevent unbounded growth of the in-memory session and working-memory dicts.
        """
        now = time.monotonic()
        with self._session_lock:
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
            # GDPR retention: also clean up persisted sessions & channel mappings
            if self._session_store:
                try:
                    self._session_store.cleanup_old_sessions(max_age_days=30)
                    self._session_store.cleanup_channel_mappings(max_age_days=30)
                except Exception as exc:
                    log.warning("gdpr_retention_cleanup_failed", error=str(exc))

    def _get_or_create_session(
        self,
        channel: str,
        user_id: str,
        agent_name: str = "jarvis",
    ) -> SessionContext:
        """Laedt oder erstellt eine Session fuer Channel+User+Agent.

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

        with self._session_lock:
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

        # Persist (outside lock, does not block other sessions)
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
        """Laedt oder erstellt Working Memory fuer eine Session.

        Bei existierenden Sessions wird die Chat-History aus SQLite geladen.
        """
        with self._session_lock:
            if session.session_id in self._working_memories:
                return self._working_memories[session.session_id]

        # Create outside lock (I/O operations do not block other sessions)
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
                history_limit = getattr(
                    getattr(self._config, "session", None),
                    "chat_history_limit",
                    100,
                )
                history = self._session_store.load_chat_history(
                    session.session_id,
                    limit=history_limit,
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

        with self._session_lock:
            # Double-check: another thread may have been faster
            if session.session_id not in self._working_memories:
                self._working_memories[session.session_id] = wm
            return self._working_memories[session.session_id]

    def _check_and_compact(self, wm: WorkingMemory, session: SessionContext) -> None:
        """Prueft Token-Budget und kompaktiert Chat-History wenn noetig.

        Nutzt den WorkingMemoryManager fuer sprachbewusste Token-Schaetzung
        und FIFO-Entfernung alter Nachrichten.
        """
        from jarvis.memory.working import WorkingMemoryManager

        mem_cfg = self._config.memory
        mgr = WorkingMemoryManager(config=mem_cfg, max_tokens=wm.max_tokens)
        mgr._memory = wm  # Manager auf aktuelle WM zeigen

        if mgr.needs_compaction:
            result = mgr.compact()
            if result.messages_removed > 0:
                log.info(
                    "auto_compaction",
                    session=session.session_id[:8],
                    messages_removed=result.messages_removed,
                    tokens_freed=result.tokens_freed,
                    usage_after=f"{mgr.usage_ratio:.0%}",
                )

    async def _handle_approvals(
        self,
        steps: list[Any],
        decisions: list[GateDecision],
        session: SessionContext,
        channel_name: str,
    ) -> list[GateDecision]:
        """Holt User-Bestaetigungen fuer ORANGE-Aktionen ein.

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
