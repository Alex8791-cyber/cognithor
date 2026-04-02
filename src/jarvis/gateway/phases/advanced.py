"""Advanced phase: Monitoring, isolation, auth, connectors, workflows, etc.

Attributes handled:
  _monitoring_hub, _isolation, _workspace_guard, _auth_gateway,
  _connector_registry, _workflow_engine, _template_library,
  _ecosystem_policy, _model_registry, _i18n, _reputation_engine,
  _recall_manager, _abuse_reporter, _governance_policy, _interop,
  _governance_hub, _ecosystem_controller, _user_portal, _skill_cli,
  _setup_wizard, _perf_manager, _exploration_executor,
  _knowledge_qa, _knowledge_lineage, _knowledge_ingest
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.gateway.phases import PhaseResult

log = get_logger(__name__)


def declare_advanced_attrs(config: Any) -> PhaseResult:
    """Return default values for all advanced attributes.

    Eagerly constructs instances where possible (matching original __init__).
    """
    result: PhaseResult = {
        # Active systems (instantiated in init_advanced)
        "run_recorder": None,
        "governance_agent": None,
        "replay_engine": None,
        "improvement_gate": None,
        "prompt_evolution": None,
        "session_analyzer": None,
        "dag_workflow_engine": None,
        "curiosity_engine": None,
        "confidence_manager": None,
        "active_learner": None,
        "exploration_executor": None,
        "knowledge_qa": None,
        "knowledge_lineage": None,
        "knowledge_ingest": None,
        "reflexion_memory": None,
        "hermes_compat": None,
        "trace_store": None,
        "proposal_store": None,
        "evolution_orchestrator": None,
        "hashline_guard": None,
        "strategy_memory": None,
        "monitoring_hub": None,
        # Enterprise-deferred: declared for hasattr() but not instantiated at startup
        "isolation": None,
        "auth_gateway": None,
        "agent_heartbeat": None,
        "command_registry": None,
        "interaction_store": None,
    }

    # ── Enterprise Placeholders (deferred) ──────────────────────────────
    # These modules are prepared for Cognithor Enterprise but have no
    # runtime callers yet. They are NOT instantiated at startup to save
    # ~200ms+ of import/init time. They remain accessible via API
    # endpoints that lazy-import on first request.
    #
    # Modules deferred: MonitoringHub, MultiUserIsolation, WorkspaceGuard,
    #   AuthGateway, ConnectorRegistry, WorkflowEngine, TemplateLibrary,
    #   EcosystemPolicy, ModelExtensionRegistry, I18nManager,
    #   ReputationEngine, SkillRecallManager, AbuseReporter,
    #   GovernancePolicy, InteropProtocol, EcosystemController,
    #   GovernanceHub, UserPortal, SkillCLI, SetupWizard,
    #   PerformanceManager, SelfImprover
    #
    # To activate any of them, move the import block back here and
    # wire the key methods into handle_message() or a background task.

    # Phase 16b: DAG WorkflowEngine (this one IS used — wired in gateway.py)
    try:
        from jarvis.core.workflow_engine import WorkflowEngine as DAGWorkflowEngine

        result["dag_workflow_engine"] = DAGWorkflowEngine()
    except Exception:
        log.debug("dag_workflow_engine_init_skipped", exc_info=True)

    # GEPA — Execution Trace Store
    try:
        from jarvis.learning.execution_trace import TraceStore

        trace_db = str(config.db_path.with_name("memory_traces.db"))
        result["trace_store"] = TraceStore(Path(trace_db))
        log.info("trace_store_initialized", db=trace_db)
    except Exception:
        log.debug("trace_store_init_skipped", exc_info=True)

    # GEPA — Proposal Store
    try:
        from jarvis.learning.trace_optimizer import ProposalStore

        proposal_db = str(config.db_path.with_name("memory_proposals.db"))
        result["proposal_store"] = ProposalStore(Path(proposal_db))
        log.info("proposal_store_initialized", db=proposal_db)
    except Exception:
        log.debug("proposal_store_init_skipped", exc_info=True)

    # GEPA — Evolution Orchestrator (uses TraceStore + ProposalStore from above)
    try:
        from jarvis.learning.causal_attributor import CausalAttributor
        from jarvis.learning.evolution_orchestrator import EvolutionOrchestrator
        from jarvis.learning.trace_optimizer import TraceOptimizer

        if getattr(config, "gepa", None) and config.gepa.enabled:
            ts = result.get("trace_store")
            ps = result.get("proposal_store")
            if ts and ps:
                result["evolution_orchestrator"] = EvolutionOrchestrator(
                    trace_store=ts,
                    attributor=CausalAttributor(),
                    optimizer=TraceOptimizer(proposal_store=ps),
                    proposal_store=ps,
                    min_traces=config.gepa.min_traces_for_proposal,
                    max_active=config.gepa.max_active_optimizations,
                    rollback_threshold=config.gepa.auto_rollback_threshold,
                    auto_apply=config.gepa.auto_apply,
                )
                log.info("gepa_orchestrator_initialized")
    except Exception:
        log.debug("gepa_orchestrator_init_skipped", exc_info=True)

    # StrategyMemory (Meta-Reasoning)
    try:
        from jarvis.learning.strategy_memory import StrategyMemory

        jarvis_home = getattr(config, "jarvis_home", Path.home() / ".jarvis")
        strat_db = Path(jarvis_home) / "index" / "strategy_memory.db"
        result["strategy_memory"] = StrategyMemory(db_path=strat_db)
        log.info("strategy_memory_initialized", db=str(strat_db))
    except Exception:
        log.debug("strategy_memory_init_skipped", exc_info=True)

    # Reflexion Memory
    try:
        from jarvis.learning.reflexion import ReflexionMemory

        reflexion_dir = Path(getattr(config, "jarvis_home", Path.home() / ".jarvis")) / "memory"
        result["reflexion_memory"] = ReflexionMemory(data_dir=reflexion_dir)
        log.info("reflexion_memory_initialized", data_dir=str(reflexion_dir))
    except Exception:
        log.debug("reflexion_memory_init_skipped", exc_info=True)

    # HermesCompatLayer (agentskills.io SKILL.md import/export)
    try:
        from jarvis.skills.hermes_compat import HermesCompatLayer

        result["hermes_compat"] = HermesCompatLayer()
    except Exception:
        log.debug("hermes_compat_init_skipped", exc_info=True)

    # RunRecorder (Forensic Run Recording)
    try:
        from jarvis.forensics.run_recorder import RunRecorder

        runs_db = str(config.db_path.with_name("memory_runs.db"))
        result["run_recorder"] = RunRecorder(runs_db)
        log.info("run_recorder_initialized", db=runs_db)
    except Exception:
        log.debug("run_recorder_init_skipped", exc_info=True)

    return result


async def init_advanced(
    config: Any,
    task_telemetry: Any = None,
    error_clusterer: Any = None,
    task_profiler: Any = None,
    cost_tracker: Any = None,
    run_recorder: Any = None,
    gatekeeper: Any = None,
) -> PhaseResult:
    """Initialize advanced subsystems that depend on earlier phases.

    Args:
        config: JarvisConfig instance.
        task_telemetry: TaskTelemetryCollector (from PGE phase).
        error_clusterer: ErrorClusterer (from PGE phase).
        task_profiler: TaskProfiler (from PGE phase).
        cost_tracker: CostTracker (from tools phase).
        run_recorder: RunRecorder (from declare_advanced_attrs).
    """
    result: PhaseResult = {}

    # GovernanceAgent (needs PGE subsystems + CostTracker + RunRecorder)
    try:
        from jarvis.governance.governor import GovernanceAgent

        gov_db = str(config.db_path.with_name("memory_governance.db"))
        result["governance_agent"] = GovernanceAgent(
            task_telemetry=task_telemetry,
            error_clusterer=error_clusterer,
            task_profiler=task_profiler,
            cost_tracker=cost_tracker,
            run_recorder=run_recorder,
            db_path=gov_db,
        )
        log.info("governance_agent_initialized", db=gov_db)
    except Exception:
        log.debug("governance_agent_init_skipped", exc_info=True)

    # ImprovementGate
    try:
        from jarvis.governance.improvement_gate import ImprovementGate

        gate = ImprovementGate(config.improvement)
        if result.get("governance_agent"):
            result["governance_agent"].improvement_gate = gate
        result["improvement_gate"] = gate
        log.info("improvement_gate_initialized")
    except Exception:
        log.debug("improvement_gate_init_skipped", exc_info=True)

    # PromptEvolutionEngine
    try:
        from jarvis.learning.prompt_evolution import PromptEvolutionEngine

        if config.prompt_evolution.enabled:
            pe_db = str(config.db_path.with_name("memory_prompt_evolution.db"))
            result["prompt_evolution"] = PromptEvolutionEngine(
                db_path=pe_db,
                min_sessions_per_arm=config.prompt_evolution.min_sessions_per_arm,
                significance_threshold=config.prompt_evolution.significance_threshold,
                max_concurrent_tests=config.prompt_evolution.max_concurrent_tests,
            )
            log.info("prompt_evolution_initialized", db=pe_db)
    except Exception:
        log.debug("prompt_evolution_init_skipped", exc_info=True)

    # SessionAnalyzer (Feedback-Loop, Failure-Clustering)
    try:
        from jarvis.learning.session_analyzer import SessionAnalyzer

        sa_dir = Path(getattr(config, "jarvis_home", Path.home() / ".jarvis")) / "memory"
        result["session_analyzer"] = SessionAnalyzer(data_dir=sa_dir)
        log.info("session_analyzer_initialized", data_dir=str(sa_dir))
    except Exception:
        log.debug("session_analyzer_init_skipped", exc_info=True)

    # CuriosityEngine (Knowledge Gap Detection)
    try:
        from jarvis.learning.curiosity import CuriosityEngine

        result["curiosity_engine"] = CuriosityEngine()
        log.info("curiosity_engine_initialized")
    except Exception:
        log.debug("curiosity_engine_init_skipped", exc_info=True)

    # KnowledgeConfidenceManager (Confidence Decay & Feedback)
    try:
        from jarvis.learning.confidence import KnowledgeConfidenceManager

        result["confidence_manager"] = KnowledgeConfidenceManager()
        log.info("confidence_manager_initialized")
    except Exception:
        log.debug("confidence_manager_init_skipped", exc_info=True)

    # ActiveLearner (Background file watching & learning)
    try:
        from jarvis.learning.active_learner import ActiveLearner

        result["active_learner"] = ActiveLearner()
        log.info("active_learner_initialized")
    except Exception:
        log.debug("active_learner_init_skipped", exc_info=True)

    # ExplorationExecutor (needs CuriosityEngine)
    try:
        from jarvis.learning.explorer import ExplorationExecutor

        curiosity = result.get("curiosity_engine")
        mm = getattr(config, "_memory_manager", None)
        result["exploration_executor"] = ExplorationExecutor(
            curiosity=curiosity,
            memory=mm,
        )
        log.info("exploration_executor_initialized")
    except Exception:
        log.debug("exploration_executor_init_skipped", exc_info=True)

    # KnowledgeQAStore
    try:
        from jarvis.learning.knowledge_qa import KnowledgeQAStore

        jarvis_home = getattr(
            config,
            "jarvis_home",
            Path.home() / ".jarvis",
        )
        qa_db = Path(jarvis_home) / "memory" / "knowledge_qa.db"
        result["knowledge_qa"] = KnowledgeQAStore(db_path=qa_db)
        log.info("knowledge_qa_initialized", db=str(qa_db))
    except Exception:
        log.debug("knowledge_qa_init_skipped", exc_info=True)

    # KnowledgeLineageTracker
    try:
        from jarvis.learning.lineage import KnowledgeLineageTracker

        jarvis_home = getattr(
            config,
            "jarvis_home",
            Path.home() / ".jarvis",
        )
        lin_db = Path(jarvis_home) / "memory" / "knowledge_lineage.db"
        result["knowledge_lineage"] = KnowledgeLineageTracker(
            db_path=lin_db,
        )
        log.info("knowledge_lineage_initialized", db=str(lin_db))
    except Exception:
        log.debug("knowledge_lineage_init_skipped", exc_info=True)

    # KnowledgeIngestService (unified file/URL/YouTube ingestion)
    try:
        from jarvis.learning.knowledge_ingest import KnowledgeIngestService

        mm = getattr(config, "_memory_manager", None)
        result["knowledge_ingest"] = KnowledgeIngestService(memory=mm)
        log.info("knowledge_ingest_initialized")
    except Exception:
        log.debug("knowledge_ingest_init_skipped", exc_info=True)

    # ReplayEngine (needs Gatekeeper for policy re-evaluation)
    if gatekeeper is not None:
        try:
            from jarvis.forensics.replay_engine import ReplayEngine

            result["replay_engine"] = ReplayEngine(gatekeeper)
            log.info("replay_engine_initialized")
        except Exception:
            log.debug("replay_engine_init_skipped", exc_info=True)

    # Wire GEPA dependencies from init phase
    orch = result.get("evolution_orchestrator")
    if orch:
        if result.get("prompt_evolution"):
            orch._prompt_evolution = result["prompt_evolution"]
        if result.get("session_analyzer"):
            orch._session_analyzer = result["session_analyzer"]

    # Hashline Guard — line-level integrity for file edits
    try:
        from jarvis.hashline import HashlineGuard
        from jarvis.hashline.config import HashlineConfig as HLConfig

        hl_cfg_model = getattr(config, "hashline", None)
        hl_dict = (
            hl_cfg_model.model_dump()
            if hl_cfg_model and hasattr(hl_cfg_model, "model_dump")
            else {}
        )
        hl_cfg = HLConfig.from_dict(hl_dict)
        if hl_cfg.enabled:
            jarvis_home = getattr(config, "jarvis_home", Path.home() / ".jarvis")
            result["hashline_guard"] = HashlineGuard.create(
                config=hl_cfg, data_dir=Path(jarvis_home)
            )
            log.info("hashline_guard_initialized")
    except Exception:
        log.debug("hashline_guard_init_skipped", exc_info=True)

    return result
