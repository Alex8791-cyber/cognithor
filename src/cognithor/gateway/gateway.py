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
import re
import signal
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from cognithor.config import CognithorConfig, load_config
from cognithor.core.agent_router import (
    RouteDecision,  # noqa: TC001 — needed at runtime by Pydantic + wrappers
)
from cognithor.core.autonomous_orchestrator import AutonomousOrchestrator
from cognithor.core.observer import (  # noqa: TC001
    PGEReloopDirective,  # noqa: F401 — runtime import for Task 16 isinstance checks
    ResponseEnvelope,
)

# Re-export so monkeypatch-on-gateway-module tests + back-compat callers
# keep working. `message_handler.formulate_response` reads through the
# `gateway` module namespace, not directly from `observer_directive`, so
# `monkeypatch.setattr(gateway_module, "run_pge_with_observer_directive", ...)`
# intercepts the call as before. See `tests/test_integration/test_observer_flow.py::
# TestGatewayEndToEnd::test_gateway_uses_observer_wrapper`.
from cognithor.gateway.observer_directive import (  # noqa: F401
    run_pge_with_observer_directive,
)
from cognithor.gateway.phases import (
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
from cognithor.i18n import t
from cognithor.mcp.client import JarvisMCPClient
from cognithor.models import (
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
from cognithor.security.compliance_engine import ComplianceEngine
from cognithor.security.consent import ConsentManager
from cognithor.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from cognithor.channels.base import Channel
    from cognithor.core.message_queue import DurableMessageQueue
    from cognithor.models import SubAgentConfig

log = get_logger(__name__)

# ── Stalled-turn detection ──────────────────────────────────────
MAX_STALLED_MODEL_TURNS: int = 20


def advance_stalled_count(current: int, tool_calls: int, successful_calls: int) -> int:
    """Return updated stalled-turn counter.

    Resets to 0 when the model both called *and* succeeded at tools;
    otherwise increments by 1.
    """
    if tool_calls > 0 and successful_calls > 0:
        return 0
    return current + 1


# Presearch result markers — used to detect empty/failed search results
_PRESEARCH_NO_RESULTS = "Keine Ergebnisse"
_PRESEARCH_NO_ENGINE = "Keine Suchengine"

# ── Tool status map for progress feedback ────────────────────────

_TOOL_STATUS_KEYS: dict[str, str] = {
    "web_search": "gateway.status_web_search",
    "web_news_search": "gateway.status_web_news_search",
    "search_and_read": "gateway.status_search_and_read",
    "web_fetch": "gateway.status_web_fetch",
    "read_file": "gateway.status_read_file",
    "write_file": "gateway.status_write_file",
    "edit_file": "gateway.status_edit_file",
    "exec_command": "gateway.status_exec_command",
    "run_python": "gateway.status_run_python",
    "search_memory": "gateway.status_search_memory",
    "save_to_memory": "gateway.status_save_to_memory",
    "document_export": "gateway.status_document_export",
    "media_analyze_image": "gateway.status_media_analyze_image",
    "media_transcribe_audio": "gateway.status_media_transcribe_audio",
    "media_extract_text": "gateway.status_media_extract_text",
    "media_tts": "gateway.status_media_tts",
    "vault_search": "gateway.status_vault_search",
    "vault_write": "gateway.status_vault_write",
    "analyze_code": "gateway.status_analyze_code",
    "list_directory": "gateway.status_list_directory",
    "browser_navigate": "gateway.status_browser_navigate",
    "browser_screenshot": "gateway.status_browser_screenshot",
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


# ── Video attachment classification ──────────────────────────────────────────

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
_VIDEO_EXTS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi"})


def _classify_attachments(
    attachments: list[str],
) -> tuple[list[str], str | None, list[str]]:
    """Split an attachment list into images / one video / rejected extra videos.

    Returns:
        (image_attachments, first_video_or_None, rejected_extra_videos)

    Single-video-per-turn policy (spec Decision 7): the first video in the
    list wins; any additional videos go into ``rejected_extra_videos`` so the
    caller can surface a user-visible validation error.
    """
    images: list[str] = []
    video: str | None = None
    rejected: list[str] = []
    for path in attachments:
        ext = ""
        if "." in path:
            ext = "." + path.rsplit(".", 1)[-1].lower()
        if ext in _IMAGE_EXTS:
            images.append(path)
        elif ext in _VIDEO_EXTS:
            if video is None:
                video = path
            else:
                rejected.append(path)
    return images, video, rejected


def _build_video_attachment(source: str, config: CognithorConfig) -> dict[str, Any]:
    """Turn a local path OR URL into a {'url': ..., 'sampling': ...} dict
    suitable for WorkingMemory.video_attachment.

    URLs are passed through verbatim. Local paths must already be uploaded
    via /api/media/upload in production — the Flutter client does this.
    But if a raw local path arrives (e.g. from a future direct-path code
    path or a test), we still produce a sane sampling without a URL.
    """
    from cognithor.core.video_sampling import resolve_sampling

    sampling = resolve_sampling(
        source,
        ffprobe_path=config.vllm.video_ffprobe_path,
        timeout_seconds=config.vllm.video_ffprobe_timeout_seconds,
        http_timeout_seconds=config.vllm.video_ffprobe_http_timeout_seconds,
        override=config.vllm.video_sampling_mode,
    )
    return {"url": source, "sampling": sampling.as_mm_kwargs()}


def _extract_uuid_from_path(source: str) -> str | None:
    """Pull a UUID out of ``http://.../media/<uuid>.<ext>`` or ``<uuid>.<ext>``.

    Returns None for non-matching forms (e.g. external URLs like
    https://example.com/clip.mp4). The UUID is used to register the upload
    with VideoCleanupWorker; external URLs don't need cleanup.
    """
    basename = source.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    # UUIDs from uuid4().hex are 32 hex chars
    if len(stem) == 32 and all(c in "0123456789abcdef" for c in stem):
        return stem
    return None


class Gateway:
    """Central entry point. Connects all Cognithor subsystems. [B§9.1]"""

    # Session TTL: sessions older than 24 hours are considered stale
    _SESSION_TTL_SECONDS: float = 24 * 60 * 60  # 24h
    # Minimum interval between stale-session cleanup sweeps
    _CLEANUP_INTERVAL_SECONDS: float = 60 * 60  # 1h

    def __init__(self, config: CognithorConfig | None = None) -> None:
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

        # vLLM orchestrator — created lazily if enabled; lifecycle hooks use this
        self._media_server = None
        self._video_cleanup = None
        if self._config.vllm.enabled:
            from cognithor.core.vllm_orchestrator import VLLMOrchestrator

            self._vllm_orchestrator = VLLMOrchestrator(
                docker_image=self._config.vllm.docker_image,
                port=self._config.vllm.port,
                hf_token=self._config.huggingface_api_key,
                config=self._config.vllm,
            )
            from cognithor.channels.media_server import MediaUploadServer
            from cognithor.gateway.video_cleanup import VideoCleanupWorker

            self._media_server = MediaUploadServer(self._config)
            self._video_cleanup = VideoCleanupWorker(
                media_dir=self._media_server._media_dir,
                ttl_hours=self._config.vllm.video_upload_ttl_hours,
            )
        else:
            self._vllm_orchestrator = None

        # Declare all subsystem attributes via phase modules
        apply_phase(self, declare_core_attrs(self._config))
        apply_phase(self, declare_security_attrs(self._config))
        apply_phase(self, declare_tools_attrs(self._config))
        apply_phase(self, declare_memory_attrs(self._config))
        apply_phase(self, declare_pge_attrs(self._config))
        apply_phase(self, declare_agents_attrs(self._config))
        apply_phase(self, declare_compliance_attrs(self._config))
        apply_phase(self, declare_advanced_attrs(self._config))

    @property
    def consent_manager(self) -> ConsentManager | None:
        """Expose ConsentManager for channel-level consent flows."""
        return getattr(self, "_consent_manager", None)

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
            interop=getattr(self, "_interop", None),
            handle_message=self.handle_message,
            gateway=self,
        )
        apply_phase(self, tools_result)

        # --- Phase D.1: Context Pipeline (depends on memory + tools) ---
        try:
            from cognithor.core.context_pipeline import ContextPipeline

            cp_config = getattr(self._config, "context_pipeline", None)
            if cp_config is None:
                from cognithor.config import ContextPipelineConfig

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
                from cognithor.core.message_queue import DurableMessageQueue as _Dmq

                queue_path = self._config.cognithor_home / "memory" / "message_queue.db"
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
            cognithor_home=self._config.cognithor_home,
            handle_message=self.handle_message,
            heartbeat_config=self._config.heartbeat,
        )
        pge_result, agents_result = await asyncio.gather(pge_coro, agents_coro)
        apply_phase(self, pge_result)
        apply_phase(self, agents_result)

        # Wire skill registry into context pipeline for proactive skill suggestions
        if self._context_pipeline and self._skill_registry:
            self._context_pipeline.set_skill_registry(self._skill_registry)
            log.info("skill_registry_wired_to_context_pipeline")

        # Wire skill registry into skill generator for hot-reload after generation
        if self._skill_registry and hasattr(self, "_skill_generator") and self._skill_generator:
            self._skill_generator.skill_registry = self._skill_registry

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
                            error=t("gateway.sub_agent_timeout", seconds=config.timeout_seconds),
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

        # Fix A1+A2: Re-wire memory for systems created with config (not gateway)
        # ExplorationExecutor and KnowledgeIngestService were instantiated with
        # getattr(config, "_memory_manager", None) which is always None because
        # config is CognithorConfig, not the gateway. Fix by assigning the real
        # MemoryManager now that it exists on self.
        if getattr(self, "_exploration_executor", None) and self._memory_manager:
            self._exploration_executor._memory = self._memory_manager
        if getattr(self, "_knowledge_ingest", None) and self._memory_manager:
            self._knowledge_ingest._memory = self._memory_manager

        # ── Lead Service (source-agnostic) ──────────────────────────────────
        # The generic LeadService is already initialized by Phase F (advanced.py).
        # Packs register their sources via PackContext.leads.register_source().
        # Register the generic social MCP tools if the service is available.
        _leads_svc_mcp = getattr(self, "_leads_service", None)
        if _leads_svc_mcp and self._mcp_client:
            try:
                from cognithor.mcp.social_tools import register_social_tools

                register_social_tools(self._mcp_client, _leads_svc_mcp)
                log.info("social_tools_registered")
            except Exception:
                log.debug("social_tools_registration_failed", exc_info=True)

        # Agent Pack Loader — loads installed packs from ~/.cognithor/packs/
        try:
            import os as _os_packs
            from pathlib import Path as _PathPacks

            from cognithor.packs.interface import PackContext
            from cognithor.packs.loader import PackLoader

            _packs_dir_env = _os_packs.environ.get("COGNITHOR_PACKS_DIR")
            if _packs_dir_env:
                _packs_path = _PathPacks(_packs_dir_env)
            else:
                _home_env = _os_packs.environ.get("COGNITHOR_HOME")
                _packs_path = (
                    _PathPacks(_home_env) / "packs"
                    if _home_env
                    else _PathPacks.home() / ".cognithor" / "packs"
                )

            from cognithor import __version__ as _cog_version

            self._pack_loader = PackLoader(packs_dir=_packs_path, cognithor_version=_cog_version)
            _leads_svc = getattr(self, "_leads_service", None)
            _pack_context = PackContext(
                gateway=self,
                config=self._config,
                mcp_client=self._mcp_client,
                leads=_leads_svc,
            )
            self._pack_loader.load_all(_pack_context)
            log.info(
                "packs_loaded",
                count=len(self._pack_loader.loaded()),
            )
        except Exception:
            log.debug("pack_loader_init_failed", exc_info=True)

        # Identity Tools: register MCP tools for cognitive identity interface
        if getattr(self, "_identity_layer", None) and self._mcp_client:
            try:
                from cognithor.mcp.identity_tools import register_identity_tools

                register_identity_tools(self._mcp_client, self._identity_layer, config=self._config)
                log.info("identity_mcp_tools_registered")
            except Exception:
                log.debug("identity_mcp_tools_registration_failed", exc_info=True)

        if getattr(self, "_session_analyzer", None) and self._memory_manager:
            self._session_analyzer._memory_manager = self._memory_manager

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

        # D2: Wire confidence_manager to reflector (created in advanced, after PGE)
        if getattr(self, "_confidence_manager", None) and getattr(self, "_reflector", None):
            self._reflector._confidence_manager = self._confidence_manager

        # Wire strategy_memory to planner (meta-reasoning hints)
        if getattr(self, "_strategy_memory", None) and getattr(self, "_planner", None):
            self._planner._strategy_memory = self._strategy_memory

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

        # GDPR: ConsentManager + ComplianceEngine
        try:
            self._consent_manager = ConsentManager(
                db_path=str(Path(self._config.cognithor_home) / "index" / "consent.db")
            )
            policy_ver = getattr(
                getattr(self._config, "compliance", None),
                "privacy_notice_version",
                "1.0",
            )
            self._compliance_engine = ComplianceEngine(
                consent_manager=self._consent_manager,
                enabled=getattr(
                    getattr(self._config, "compliance", None),
                    "compliance_engine_enabled",
                    True,
                ),
                policy_version=policy_ver,
            )
            if getattr(getattr(self._config, "compliance", None), "privacy_mode", False):
                self._compliance_engine.set_privacy_mode(True)
            # Wire GDPRComplianceManager with retention config from config.yaml
            from cognithor.security.gdpr import GDPRComplianceManager, build_retention_policies

            _retention_cfg = getattr(self._config, "retention", None)
            _policies = build_retention_policies(_retention_cfg)
            self._gdpr_manager = GDPRComplianceManager(retention_policies=_policies)
            self._gdpr_compliance_manager = self._gdpr_manager  # alias for cron
            log.info("gdpr_compliance_engine_initialized")
        except Exception:
            log.error("gdpr_compliance_engine_init_failed", exc_info=True)
            # Fail-closed: engine without consent store blocks all consent-based processing
            self._consent_manager = None
            self._compliance_engine = ComplianceEngine(
                consent_manager=None,
                enabled=True,
                consent_required=True,
            )
            self._gdpr_manager = None
            self._gdpr_compliance_manager = None

        # Register erasure handlers for all data tiers
        if hasattr(self, "_gdpr_manager") and self._gdpr_manager:
            erasure = self._gdpr_manager.erasure

            # Memory tier: delete user's episodic/semantic/procedural memories
            if hasattr(self, "_memory_manager") and self._memory_manager:
                mm = self._memory_manager

                def _erase_memory(uid):
                    count = 0
                    try:
                        if hasattr(mm, "episodic") and hasattr(mm.episodic, "prune_old"):
                            mm.episodic.prune_old(retention_days=0)
                            count += 1
                    except Exception:
                        log.debug("erasure_memory_failed", exc_info=True)
                    return count

                erasure.register_handler(_erase_memory)

            # Session tier
            session_store = getattr(self, "_session_store", None)
            if session_store and hasattr(session_store, "delete_user_sessions"):
                erasure.register_handler(lambda uid: session_store.delete_user_sessions(uid))

            # User preferences
            pref = getattr(self, "_user_pref_store", None)
            if pref and hasattr(pref, "delete_user"):
                erasure.register_handler(lambda uid, p=pref: p.delete_user(uid))

            # Conversation tree
            ct = getattr(self, "_conversation_tree", None)
            if ct and hasattr(ct, "delete_user"):
                erasure.register_handler(lambda uid, c=ct: c.delete_user(uid))

            # Feedback
            fb = getattr(self, "_feedback_store", None)
            if fb and hasattr(fb, "delete_user"):
                erasure.register_handler(lambda uid, f=fb: f.delete_user(uid))

            # Corrections
            cm = getattr(self, "_correction_memory", None)
            if cm and hasattr(cm, "delete_user"):
                erasure.register_handler(lambda uid, c=cm: c.delete_user(uid))

            # Vault notes (delete all for single-user system)
            vault_tools = None
            for attr_name in ("_vault_tools", "_vault"):
                vt = getattr(self, attr_name, None)
                if vt and hasattr(vt, "_backend"):
                    vault_tools = vt
                    break
            if vault_tools:

                def _erase_vault(uid, vt=vault_tools):
                    try:
                        notes = vt._backend.all_notes()
                        count = 0
                        for note in notes:
                            try:
                                vt._backend.delete(note.path)
                                count += 1
                            except Exception:
                                log.debug("erasure_vault_note_failed", exc_info=True)
                        return count
                    except Exception:
                        return 0

                erasure.register_handler(_erase_vault)

        # Governance-Cron-Job registrieren (taeglich um 02:00)
        if self._cron_engine and hasattr(self, "_governance_agent") and self._governance_agent:
            try:
                from cognithor.cron.jobs import governance_analysis

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
                from cognithor.cron.jobs import prompt_evolution_check

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

        # GDPR retention enforcement (daily at 03:00)
        if self._cron_engine and hasattr(self, "_compliance_engine") and self._compliance_engine:
            try:
                self._cron_engine.add_system_job(
                    name="retention_enforcer",
                    schedule="0 3 * * *",
                    callback=self._run_retention_enforcement,
                    args=[],
                )
                log.info("gdpr_retention_cron_registered")
            except Exception:
                log.debug("gdpr_retention_cron_failed", exc_info=True)

        # Reddit Lead Scanner (if enabled)
        _social_cfg = getattr(self._config, "social", None)
        if (
            _social_cfg
            and _social_cfg.reddit_scan_enabled
            and getattr(self, "_reddit_lead_service", None)
        ):
            try:
                from cognithor.cron.jobs import CronJob

                self._cron_engine.add_runtime_job(
                    CronJob(
                        name="reddit_lead_scan",
                        schedule=f"*/{_social_cfg.reddit_scan_interval_minutes} * * * *",
                        prompt="[CRON:reddit_lead_scan] Scan Reddit for leads",
                        channel=getattr(self._config, "default_channel", "webui") or "webui",
                        enabled=True,
                    )
                )
                log.info(
                    "reddit_cron_registered",
                    interval=_social_cfg.reddit_scan_interval_minutes,
                )
            except Exception:
                log.debug("reddit_cron_registration_failed", exc_info=True)

            # Reddit Reply Performance Tracker (every 6h)
            try:
                self._cron_engine.add_system_job(
                    name="reddit_reply_tracker",
                    schedule="0 */6 * * *",
                    callback=self._track_reddit_replies,
                )
                log.info("reddit_tracker_cron_registered")
            except Exception:
                log.debug("reddit_tracker_cron_failed", exc_info=True)

            # Reddit Style Learner (weekly, Sunday 3am)
            try:
                self._cron_engine.add_system_job(
                    name="reddit_style_learner",
                    schedule="0 3 * * 0",
                    callback=self._run_reddit_learner,
                )
                log.info("reddit_learner_cron_registered")
            except Exception:
                log.debug("reddit_learner_cron_failed", exc_info=True)

        # --- Autonomous Orchestrator (connects PGE + SkillGenerator + Reflector) ---
        self._autonomous_orchestrator = AutonomousOrchestrator(
            gateway=self,
            skill_generator=getattr(self, "_skill_generator", None),
            reflector=getattr(self, "_reflector", None),
        )

        # --- Feedback Store (thumbs up/down rating) ---
        try:
            from cognithor.core.feedback import FeedbackStore

            self._feedback_store = FeedbackStore(
                db_path=self._config.cognithor_home / "feedback.db"
            )
            log.info("feedback_store_initialized")
        except Exception:
            log.debug("feedback_store_init_failed", exc_info=True)
            self._feedback_store = None

        # --- Correction Memory (Smart Recovery) ---
        try:
            from cognithor.core.correction_memory import CorrectionMemory

            _proactive = 3
            if hasattr(self._config, "recovery"):
                _proactive = getattr(self._config.recovery, "correction_proactive_threshold", 3)
            self._correction_memory = CorrectionMemory(
                db_path=self._config.cognithor_home / "corrections.db",
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
            from cognithor.core.conversation_tree import ConversationTree

            self._conversation_tree = ConversationTree(
                db_path=self._config.cognithor_home / "conversations.db"
            )
            log.info("conversation_tree_initialized")
        except Exception:
            log.debug("conversation_tree_init_failed", exc_info=True)
            self._conversation_tree = None

        # System Detector (hardware profiling)
        try:
            from cognithor.system.detector import SystemDetector

            _detector = SystemDetector()
            _cache = self._config.cognithor_home / "system_profile.json"
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
            from cognithor.system.resource_monitor import ResourceMonitor

            self._resource_monitor = ResourceMonitor()
            log.info("resource_monitor_initialized")
        except Exception:
            log.debug("resource_monitor_init_skipped", exc_info=True)

        # CheckpointStore (persistent evolution checkpoints)
        self._checkpoint_store = None
        try:
            from cognithor.core.checkpointing import CheckpointStore

            self._checkpoint_store = CheckpointStore(self._config.cognithor_home / "checkpoints")
            log.info("checkpoint_store_initialized")
        except Exception:
            log.debug("checkpoint_store_init_skipped", exc_info=True)

        # LLM call function for Evolution Engine + DeepLearner
        self._llm_call = None
        if self._llm and self._model_router:

            async def _evolution_llm_call(prompt: str) -> str:
                import asyncio as _evo_aio

                model = self._model_router.select_model("summarization", "low")
                try:
                    resp = await _evo_aio.wait_for(
                        self._llm.chat(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            options={"num_predict": 2000},
                        ),
                        timeout=120,  # ATL calls fail fast, don't block 10min
                    )
                except TimeoutError:
                    log.debug("evolution_llm_timeout", model=model)
                    return ""
                return resp.get("message", {}).get("content", "")

            self._llm_call = _evolution_llm_call

        # Evolution Engine (idle-time autonomous learning)
        self._idle_detector = None
        self._evolution_loop = None
        if getattr(self._config, "evolution", None) and self._config.evolution.enabled:
            try:
                from cognithor.evolution.idle_detector import IdleDetector
                from cognithor.evolution.loop import EvolutionLoop

                idle_minutes = getattr(self._config.evolution, "idle_minutes", 5)
                self._idle_detector = IdleDetector(idle_threshold_seconds=idle_minutes * 60)
                op_mode = str(getattr(self._config, "resolved_operation_mode", "offline"))
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
                    from cognithor.evolution.deep_learner import DeepLearner

                    self._deep_learner = DeepLearner(
                        llm_fn=getattr(self, "_llm_call", None),
                        plans_dir=self._config.cognithor_home / "evolution" / "plans",
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

                    # Wire LLM for entity extraction. Use the SAME model as the planner
                    # to avoid loading a second model in parallel (which causes VRAM
                    # thrashing / model swap and 10+ minute delays).
                    if self._llm and self._model_router:

                        async def _entity_llm_call(prompt: str) -> str:
                            _entity_model = self._config.models.planner.name
                            resp = await self._llm.chat(
                                model=_entity_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                options={"num_predict": 2000},
                            )
                            content = resp.get("message", {}).get("content", "")
                            # Strip Qwen3 think tags
                            import re

                            return re.sub(
                                r"<think>.*?</think>", "", content, flags=re.DOTALL
                            ).strip()

                        self._deep_learner._entity_llm_fn = _entity_llm_call

                    log.info("deep_learner_initialized")

                    # CycleController for autonomous exam-based learning
                    try:
                        from cognithor.evolution.cycle_controller import CycleController

                        _cycle_ctrl = CycleController(
                            plans_dir=self._config.cognithor_home / "evolution" / "plans"
                        )
                        self._deep_learner._cycle_controller = _cycle_ctrl
                        log.info("cycle_controller_initialized")
                    except Exception:
                        log.debug("cycle_controller_init_failed", exc_info=True)
                except Exception:
                    log.debug("deep_learner_init_failed", exc_info=True)

        # ATL (Autonomous Thinking Loop) wiring
        if self._evolution_loop:
            try:
                from cognithor.evolution.atl_config import ATLConfig

                atl_raw = getattr(self._config, "atl", None) or {}
                if isinstance(atl_raw, ATLConfig):
                    atl_cfg = atl_raw
                elif isinstance(atl_raw, dict) and atl_raw:
                    atl_cfg = ATLConfig(**atl_raw)
                else:
                    atl_cfg = ATLConfig()

                self._evolution_loop._atl_config = atl_cfg

                if atl_cfg.enabled:
                    from cognithor.evolution.atl_journal import ATLJournal
                    from cognithor.evolution.goal_manager import GoalManager

                    goals_path = self._config.cognithor_home / "evolution" / "goals.yaml"
                    journal_dir = self._config.cognithor_home / "evolution" / "journal"

                    gm = GoalManager(goals_path=goals_path)

                    # One-time migration of learning_goals -> Goal objects
                    if not goals_path.exists():
                        old_goals: list[str] = []
                        if hasattr(self._config, "evolution") and self._config.evolution:
                            old_goals = list(
                                getattr(self._config.evolution, "learning_goals", []) or []
                            )
                        if old_goals:
                            gm.migrate_learning_goals(old_goals)
                            log.info("atl_goals_migrated", count=len(old_goals))

                    self._evolution_loop._goal_manager = gm
                    self._evolution_loop._atl_journal = ATLJournal(journal_dir=journal_dir)
                    log.info("atl_initialized", goals=len(gm.active_goals()))

                    # Register ATL MCP tools
                    try:
                        from cognithor.mcp.atl_tools import register_atl_tools, set_atl_context

                        set_atl_context(
                            goal_manager=self._evolution_loop._goal_manager,
                            journal=self._evolution_loop._atl_journal,
                            config=atl_cfg,
                            loop=self._evolution_loop,
                        )
                        register_atl_tools(self._mcp_client)
                        log.info("atl_tools_registered")
                    except Exception:
                        log.debug("atl_tools_registration_failed", exc_info=True)
            except Exception:
                log.debug("atl_wiring_failed", exc_info=True)

        # Kanban Board
        try:
            if getattr(self._config, "kanban", None) and self._config.kanban.enabled:
                from cognithor.kanban.engine import KanbanEngine
                from cognithor.kanban.store import KanbanStore
                from cognithor.mcp.kanban_tools import register_kanban_tools

                _kanban_db = self._config.cognithor_home / "db" / "kanban.db"
                _kanban_store = KanbanStore(str(_kanban_db))
                self._kanban_engine = KanbanEngine(
                    _kanban_store,
                    max_auto_tasks=self._config.kanban.max_auto_tasks_per_session,
                    max_subtask_depth=self._config.kanban.max_subtask_depth,
                    cascade_cancel=self._config.kanban.cascade_cancel_subtasks,
                )
                register_kanban_tools(self._mcp_client, self._kanban_engine)
                log.info("kanban_engine_initialized", db=str(_kanban_db))
        except Exception:
            log.debug("kanban_init_failed", exc_info=True)
            self._kanban_engine = None

        # V6: Wire tool registry into Gatekeeper for per-tool risk annotations
        if self._gatekeeper and hasattr(self._mcp_client, "_tool_registry"):
            self._gatekeeper.set_tool_registry(self._mcp_client._tool_registry)

        # vLLM media server + cleanup worker lifecycle
        if self._media_server is not None:
            try:
                port = await self._media_server.start()
                if self._vllm_orchestrator is not None:
                    self._vllm_orchestrator.media_url = f"http://host.docker.internal:{port}"
            except Exception:
                log.warning("media_server_start_failed", exc_info=True)

        # Bug C1-r3: publish the Gateway-owned VLLMOrchestrator (with media_url
        # wired above) + MediaUploadServer onto every already-registered
        # channel's FastAPI app.state. This is the defensive path for when a
        # channel was registered BEFORE initialize(). The common path —
        # register_channel() after initialize() — is handled by
        # register_channel() itself via _publish_app_state.
        for channel in self._channels.values():
            self._publish_app_state(channel)

        if self._video_cleanup is not None:
            try:
                await self._video_cleanup.start()
            except Exception:
                log.warning("video_cleanup_start_failed", exc_info=True)

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
                description=t(
                    "gateway.startup_description",
                    llm_ok=llm_ok,
                    tool_count=len(self._mcp_client.get_tool_list()),
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
        if not self._memory_manager or not hasattr(self._memory_manager, "_core"):
            return
        core = self._memory_manager._core
        content = core.content
        if not content:
            return
        language = getattr(self._config, "language", "de")

        # Try DB-backed generation
        tool_count = 0
        try:
            from cognithor.mcp.tool_registry_db import (
                _SECTION_HEADERS,
                ToolRegistryDB,
                _ProcedureEntry,
                deduplicate_procedures,
            )

            db_path = self._config.cognithor_home / "tool_registry.db"
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
                from cognithor.mcp.tool_registry_db import (
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
            from cognithor.mcp.tool_registry_db import _SECTION_HEADERS

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

        core.save(content)
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
        # Bug C1-r3: expose Gateway-owned VLLMOrchestrator + MediaUploadServer
        # onto channel.app.state so the backends_api and media_api routers see
        # the same instances as the Gateway (media_url + media_server are
        # already wired by initialize()). Channels are typically registered
        # AFTER initialize(), so this is the primary wiring path; the duplicate
        # loop in initialize() handles the reverse ordering defensively.
        self._publish_app_state(channel)
        log.info("channel_registered", channel=channel.name)

    def _publish_app_state(self, channel: Channel) -> None:
        """Copy Gateway-owned singletons onto channel.app.state (idempotent)."""
        app = getattr(channel, "app", None)
        if app is None:
            return
        try:
            if getattr(self, "_vllm_orchestrator", None) is not None:
                app.state.vllm_orchestrator = self._vllm_orchestrator
            if getattr(self, "_media_server", None) is not None:
                app.state.media_server = self._media_server
        except Exception:
            log.debug("app_state_wiring_failed", channel=channel.name, exc_info=True)

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
                            from cognithor.a2a.http_handler import A2AHTTPHandler

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
                # D3: Default watch_dirs if empty — scan vault + memory for new files
                if not self._active_learner._watch_dirs:
                    vault_dir = self._config.cognithor_home / "vault"
                    wissen_dir = self._config.cognithor_home / "vault" / "wissen"
                    for d in [vault_dir, wissen_dir]:
                        if d.exists():
                            self._active_learner._watch_dirs.append(str(d))
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
                from cognithor.mcp.background_tasks import ProcessMonitor

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
                    if (
                        hasattr(self, "_audit_logger")
                        and self._audit_logger
                        and hasattr(self._audit_logger, "cleanup_old_entries")
                    ):
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
                            from datetime import UTC, datetime

                            from cognithor.security.tsa import TSAClient

                            anchor = self._audit_trail.get_anchor()
                            if anchor["entry_count"] > 0:
                                date_str = datetime.now(UTC).strftime("%Y-%m-%d")
                                tsa_url = getattr(
                                    self._config.audit, "tsa_url", "https://freetsa.org/tsr"
                                )
                                tsa_dir = self._config.cognithor_home / "tsa"
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
                            from cognithor.audit.worm import WORMUploader

                            worm_audit_dir = self._config.cognithor_home / "data" / "audit"
                            uploader = WORMUploader(self._config.audit, self._config.cognithor_home)
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

        # Skill lifecycle: daily audit
        async def _daily_skill_lifecycle():
            while True:
                await asyncio.sleep(86400)  # 24 hours
                try:
                    if hasattr(self, "_skill_lifecycle") and self._skill_lifecycle:
                        audit_results = self._skill_lifecycle.audit_all()
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
        self._background_tasks.add(_skill_lifecycle_task)
        _skill_lifecycle_task.add_done_callback(self._background_tasks.discard)

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
                from cognithor.audit.breach_detector import BreachDetector

                _breach_state = self._config.cognithor_home / "breach_state.json"
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

        # Skill Lifecycle: initial audit + periodic background task
        try:
            skill_registry = getattr(self, "_skill_registry", None)
            sl_cfg = getattr(self._config, "skill_lifecycle", None)
            if skill_registry and (sl_cfg is None or sl_cfg.enabled):
                from cognithor.skills.lifecycle import SkillLifecycleManager

                generated_dir = self._config.cognithor_home / "skills" / "generated"
                self._skill_lifecycle = SkillLifecycleManager(skill_registry, generated_dir)
                report = self._skill_lifecycle.get_report()
                log.info("skill_lifecycle_audit", report=report[:200])

                # Auto-repair broken skills
                if sl_cfg is None or sl_cfg.auto_repair:
                    for broken in self._skill_lifecycle.get_broken_skills():
                        self._skill_lifecycle.repair_skill(broken.slug)

                # Periodic audit background task
                interval_h = getattr(sl_cfg, "audit_interval_hours", 24) if sl_cfg else 24

                async def _periodic_skill_audit() -> None:
                    """Run skill audit periodically in the background."""
                    import asyncio as _aio

                    while True:
                        await _aio.sleep(interval_h * 3600)
                        try:
                            mgr = self._skill_lifecycle
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
                self._background_tasks.add(_audit_task)
                _audit_task.add_done_callback(self._background_tasks.discard)
                log.info("skill_lifecycle_periodic_started", interval_hours=interval_h)
        except Exception:
            log.debug("skill_lifecycle_init_failed", exc_info=True)

        # Episodic Compression (daily maintenance)
        if self._memory_manager and hasattr(self._memory_manager, "compressor"):
            try:
                from datetime import date

                async def _periodic_episode_compression() -> None:
                    await asyncio.sleep(3600)  # First run after 1 hour
                    while True:
                        try:
                            compressor = self._memory_manager.compressor
                            ep_dir = self._config.cognithor_home / "memory" / "episodes"
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
                                                    entries.append(
                                                        p.read_text(encoding="utf-8")[:2000]
                                                    )
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
                self._background_tasks.add(_comp_task)
                _comp_task.add_done_callback(self._background_tasks.discard)
                log.info("episodic_compression_scheduled")
            except Exception:
                log.debug("episodic_compression_init_failed", exc_info=True)

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
        """Background task: sync community registry periodically (daily)."""
        while self._running:
            try:
                from cognithor.skills.community.sync import RegistrySync

                sync = RegistrySync(
                    community_dir=self._config.cognithor_home / "skills" / "community",
                    skill_registry=self._skill_registry
                    if hasattr(self, "_skill_registry")
                    else None,
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

    def on_startup_vllm(self):
        """Called during init. If a cognithor-managed vLLM container is
        already running (from a previous session with auto_stop_on_close=False,
        or because the user ran the container manually), adopt it — no restart."""
        if not self._config.vllm.enabled or self._vllm_orchestrator is None:
            return None
        try:
            return self._vllm_orchestrator.reuse_existing()
        except Exception as exc:
            log.warning("vllm_reuse_existing_failed", error=str(exc))
            return None

    def on_shutdown_vllm(self) -> None:
        """Called on Gateway.shutdown(). Stops the container only if the user
        has opted in via config.vllm.auto_stop_on_close."""
        if not self._config.vllm.enabled or self._vllm_orchestrator is None:
            return
        if self._config.vllm.auto_stop_on_close:
            try:
                self._vllm_orchestrator.stop_container()
            except Exception as exc:
                log.warning("vllm_shutdown_failed", error=str(exc))

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
            self._audit_logger.log_system("shutdown", description=t("gateway.shutdown_description"))

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

        # vLLM video cleanup + media server teardown
        if self._video_cleanup is not None:
            try:
                await self._video_cleanup.stop()
            except Exception:
                log.debug("video_cleanup_stop_skipped", exc_info=True)
        if self._media_server is not None:
            try:
                await self._media_server.stop()
            except Exception:
                log.debug("media_server_stop_skipped", exc_info=True)

        # MCP-Client trennen
        if self._mcp_client:
            await self._mcp_client.disconnect_all()

        # Close Ollama client
        if self._llm:
            await self._llm.close()

        log.info("gateway_shutdown_complete")

    def rebuild_llm_client(self, new_backend_type: str) -> None:
        """Re-init UnifiedLLMClient for a new backend type.

        Called from the FastAPI /api/backends/active endpoint when the user
        switches backends from the Flutter UI. No app restart needed.
        """
        from cognithor.core.unified_llm import UnifiedLLMClient

        self._config.llm_backend_type = new_backend_type
        self._llm = UnifiedLLMClient.create(self._config)

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

        from cognithor.core.workflow_schema import WorkflowDefinition

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

        from cognithor.core.workflow_adapter import action_plan_to_workflow

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
                    self._config.cognithor_home / "data" / "procedures",
                    self._config.cognithor_home / self._config.plugins.skills_dir,
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

                from cognithor.i18n import set_locale

                _lang = os.environ.get("COGNITHOR_LANGUAGE") or new_config.language
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
                        from cognithor.core.unified_llm import UnifiedLLMClient

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

        # CAG prefix injection
        if (
            hasattr(self, "_memory_manager")
            and self._memory_manager
            and getattr(self._memory_manager, "_cag_manager", None)
        ):
            try:
                _cag_mgr = self._memory_manager._cag_manager
                if _cag_mgr.is_active:
                    _model_id = self._config.models.planner.name
                    _cag_prefix = await _cag_mgr.get_stable_prefix(wm.core_memory_text, _model_id)
                    if _cag_prefix:
                        wm.cag_prefix = _cag_prefix
            except Exception:
                log.debug("cag_prefix_preparation_failed", exc_info=True)

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
        """Verarbeitet eine eingehende Nachricht. [B§3.4]"""
        from cognithor.gateway import message_handler

        return await message_handler.handle_message(self, msg, stream_callback)

    async def _resolve_agent_route(
        self,
        msg: IncomingMessage,
    ) -> tuple[RouteDecision | None, SessionContext, WorkingMemory, Any, Any, str, str | None]:
        """Phase 1: Agent-Routing, Session, Working Memory, Skills, Workspace."""
        from cognithor.gateway import message_handler

        return await message_handler.resolve_agent_route(self, msg)

    async def _prepare_execution_context(
        self,
        msg: IncomingMessage,
        session: SessionContext,
        wm: WorkingMemory,
        route_decision: RouteDecision | None,
    ) -> tuple[str | None, OutgoingMessage | None]:
        """Phase 2: Profiler, Budget, Run-Recording, Policy-Snapshot."""
        from cognithor.gateway import message_handler

        return await message_handler.prepare_execution_context(
            self, msg, session, wm, route_decision
        )

    def _make_status_callback(
        self,
        channel_name: str,
        session_id: str,
    ) -> Any:
        """Creates a fire-and-forget status callback for the current channel."""
        from cognithor.gateway import message_handler

        return message_handler.make_status_callback(self, channel_name, session_id)

    def _make_pipeline_callback(
        self,
        channel_name: str,
        session_id: str,
    ) -> Any:
        """Creates a fire-and-forget pipeline event callback."""
        from cognithor.gateway import message_handler

        return message_handler.make_pipeline_callback(self, channel_name, session_id)

    async def _formulate_response(
        self,
        msg_text: str,
        all_results: list[ToolResult],
        wm: WorkingMemory,
        stream_callback: Any | None = None,
    ) -> ResponseEnvelope:
        """Formulate response, optionally streaming tokens to the client."""
        from cognithor.gateway import message_handler

        return await message_handler.formulate_response(
            self, msg_text, all_results, wm, stream_callback
        )

    @staticmethod
    def _is_cu_plan(plan: ActionPlan) -> bool:
        """Check if a plan uses Computer Use tools."""
        from cognithor.gateway import pge_loop

        return pge_loop.is_cu_plan(plan)

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
        active_skill: Any | None = None,
    ) -> tuple[str, list[ToolResult], list[ActionPlan], list[AuditEntry]]:
        """Phase 3: Plan -> Gate -> Execute Loop."""
        from cognithor.gateway import pge_loop

        return await pge_loop.run_pge_loop(
            self,
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

    async def _run_post_processing(
        self,
        session: SessionContext,
        wm: WorkingMemory,
        agent_result: AgentResult,
        active_skill: Any,
        run_id: str | None,
    ) -> None:
        """Phase 4: Reflection, Skill-Tracking, Telemetry, Profiler, Run-Recording."""
        from cognithor.gateway import post_processing

        return await post_processing.run_post_processing(
            self, session, wm, agent_result, active_skill, run_id
        )

    def _maybe_record_pattern(
        self,
        session: SessionContext,
        wm: WorkingMemory,
        agent_result: AgentResult,
    ) -> None:
        """Pattern-Mining: Tool-Sequenz + User-Intent in Procedural Memory speichern."""
        from cognithor.gateway import post_processing

        return post_processing.maybe_record_pattern(self, session, wm, agent_result)

    async def _persist_session(
        self,
        session: SessionContext,
        wm: WorkingMemory,
    ) -> None:
        """Phase 5: Session persistieren (Incognito-aware)."""
        from cognithor.gateway import post_processing

        return await post_processing.persist_session(self, session, wm)

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

        # Broadcast delegation status to frontend
        try:
            channel_name = session.channel or "webui"
            status_cb = self._make_status_callback(channel_name, session.session_id)
            await status_cb(
                "working",
                f"Delegation: {from_agent} -> {to_agent}: {task[:100]}",
            )
        except Exception:
            log.debug("delegation_status_broadcast_failed", exc_info=True)

        # Create forked session for delegated agent (provenance tracking)
        from cognithor.models import SessionContext as _SC

        sub_session = _SC(
            user_id=session.user_id,
            channel=session.channel,
            agent_name=to_agent,
            parent_session_id=session.session_id,
            fork_reason=f"delegated from {from_agent}: {task[:200]}",
        )
        if self._session_store:
            try:
                self._session_store.save_session(sub_session)
            except Exception:
                log.debug("delegation_session_save_skipped", exc_info=True)

        # Separate working memory for delegated agent
        sub_wm = WorkingMemory(session_id=sub_session.session_id)

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
            _envelope = await self._planner.formulate_response(
                user_message=task,
                results=results,
                working_memory=sub_wm,
            )
            response = _envelope.content
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

    # Rate limiter for pattern recordings (post-processing pipeline): at most
    # 5 procedural-memory writes per hour. Read by
    # `post_processing.maybe_record_pattern()` via `gw._PATTERN_MAX_PER_HOUR`.
    _PATTERN_MAX_PER_HOUR: ClassVar[int] = 5

    # Tools whose results are file paths that should be attached to the response.
    # Read by `message_utils.extract_attachments()` via `gw._ATTACHMENT_TOOLS`.
    _ATTACHMENT_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {
            "document_export",
        }
    )
    # File extensions considered valid attachments. Read by
    # `message_utils.extract_attachments()` via `gw._ATTACHMENT_EXTENSIONS`.
    _ATTACHMENT_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
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

    def _persist_key_tool_results(
        self,
        wm: WorkingMemory,
        results: list[ToolResult],
    ) -> None:
        """Persistiert wichtige Tool-Ergebnisse als TOOL-Messages in der Chat-History."""
        from cognithor.gateway import post_processing

        return post_processing.persist_key_tool_results(self, wm, results)

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

    # Regex patterns for factual questions (when/where/who/what + verb).
    # Read by `message_utils.is_fact_question()` via `gw._FACT_QUESTION_PATTERNS`.
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
            r"\b(when|where|who|what|how many|which)\b.{3,}"
            r"(did|has|have|was|were|is|are|happened)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(stimmt es|ist es wahr|hat .+ wirklich)\b",
            re.IGNORECASE,
        ),
    ]

    # Begriffe die KEINE Faktenfrage signalisieren (Smalltalk, Meinungen, Befehle).
    # Read by `message_utils.is_fact_question()` via `gw._SKIP_PRESEARCH_PATTERNS`.
    _SKIP_PRESEARCH_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(meinst du|findest du|was denkst du|erkläre mir|was ist ein|definier)",
            re.IGNORECASE,
        ),
        # Trailing \b entfernt: "erstell" muss auch "erstelle/erstellst/erstellen" matchen.
        re.compile(
            r"\b(schreib|erstell|generier|mach|öffne|lösch|speicher|such im memory)",
            re.IGNORECASE,
        ),
    ]

    def _extract_attachments(self, results: list[ToolResult]) -> list[str]:
        """Extrahiert Dateipfade aus Tool-Ergebnissen fuer den Anhang-Versand."""
        from cognithor.gateway import message_utils

        return message_utils.extract_attachments(self, results)

    def _is_fact_question(self, text: str) -> bool:
        """Prueft ob eine Nachricht eine Faktenfrage ist, die Web-Recherche braucht."""
        from cognithor.gateway import message_utils

        return message_utils.is_fact_question(self, text)

    async def _classify_coding_task(self, user_message: str) -> tuple[bool, str]:
        """Klassifiziert ob eine Nachricht eine Coding-Aufgabe ist und deren Komplexitaet."""
        from cognithor.gateway import message_utils

        return await message_utils.classify_coding_task(self, user_message)

    @staticmethod
    def _resolve_relative_dates(text: str) -> str:
        """Ersetzt relative Zeitangaben durch konkrete Datumsangaben."""
        from cognithor.gateway import message_utils

        return message_utils.resolve_relative_dates(text)

    def _build_reddit_forced_plan(self, user_text: str) -> ActionPlan | None:
        """Erzeugt einen Reddit-Hardrouting-Plan fuer Lead-Hunter-Anfragen."""
        from cognithor.gateway import message_utils

        return message_utils.build_reddit_forced_plan(self, user_text)

    async def _maybe_presearch(self, msg: IncomingMessage, wm: WorkingMemory) -> str | None:
        """Fuehrt eine automatische Web-Suche aus wenn die Nachricht eine Faktenfrage ist."""
        from cognithor.gateway import message_utils

        return await message_utils.maybe_presearch(self, msg, wm)

    async def _answer_from_presearch(self, user_message: str, search_results: str) -> str:
        """Erzeugt eine direkte LLM-Antwort aus den Pre-Search-Ergebnissen."""
        from cognithor.gateway import message_utils

        return await message_utils.answer_from_presearch(self, user_message, search_results)

    def _cleanup_stale_sessions(self) -> None:
        """Entfernt abgelaufene Sessions aus dem In-Memory-Cache."""
        from cognithor.gateway import session_mgmt

        return session_mgmt.cleanup_stale_sessions(self)

    def _maybe_cleanup_sessions(self) -> None:
        """Triggert das Stale-Session-Cleanup, wenn das Intervall ueberschritten ist."""
        from cognithor.gateway import session_mgmt

        return session_mgmt.maybe_cleanup_sessions(self)

    def _get_or_create_session(
        self,
        channel: str,
        user_id: str,
        agent_name: str = "jarvis",
    ) -> SessionContext:
        """Holt oder erstellt eine Session fuer (channel, user_id, agent_name)."""
        from cognithor.gateway import session_mgmt

        return session_mgmt.get_or_create_session(self, channel, user_id, agent_name)

    def _get_or_create_working_memory(self, session: SessionContext) -> WorkingMemory:
        """Holt oder erstellt das WorkingMemory fuer eine Session."""
        from cognithor.gateway import session_mgmt

        return session_mgmt.get_or_create_working_memory(self, session)

    def _check_and_compact(self, wm: WorkingMemory, session: SessionContext) -> None:
        """Prueft ob das WorkingMemory komprimiert werden muss und tut es ggf."""
        from cognithor.gateway import session_mgmt

        return session_mgmt.check_and_compact(self, wm, session)

    async def _handle_approvals(
        self,
        steps: list[Any],
        decisions: list[GateDecision],
        session: SessionContext,
        channel_name: str,
        *,
        ws_session_id: str | None = None,
    ) -> list[GateDecision]:
        """Holt User-Bestaetigungen fuer ORANGE-Aktionen ein."""
        from cognithor.gateway import pge_loop

        return await pge_loop.handle_approvals(
            self, steps, decisions, session, channel_name, ws_session_id=ws_session_id
        )

    async def _track_reddit_replies(self) -> None:
        # Legacy stub — Reddit tracking now handled by the reddit-lead-hunter-pro pack.
        log.debug("reddit_reply_tracker_stub_called")

    async def _run_reddit_learner(self) -> None:
        # Legacy stub — Reddit learning now handled by the reddit-lead-hunter-pro pack.
        log.debug("reddit_style_learner_stub_called")
