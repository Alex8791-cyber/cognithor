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

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.phases import PhaseResult

from cognithor.core.safe_call import _safe_call

log = get_logger(__name__)


def _init_subsystem(name: str, result: PhaseResult, fn: Any, *args: Any, **kwargs: Any) -> None:
    """Init a subsystem via _safe_call and store in result dict."""
    value = _safe_call(name, fn, *args, **kwargs)
    if value is not None:
        result[name] = value
        log.info(f"{name}_initialized")


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
        "reddit_lead_service": None,
        "leads_service": None,
        "leads_store": None,
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

    cognithor_home = getattr(config, "cognithor_home", Path.home() / ".cognithor")

    # --- All subsystems via _safe_call (failures logged + tracked) ---

    def _init_dag() -> Any:
        from cognithor.core.workflow_engine import WorkflowEngine as DAGWorkflowEngine

        return DAGWorkflowEngine()

    _init_subsystem("dag_workflow_engine", result, _init_dag)

    def _init_trace_store() -> Any:
        from cognithor.learning.execution_trace import TraceStore

        return TraceStore(Path(str(config.db_path.with_name("memory_traces.db"))))

    _init_subsystem("trace_store", result, _init_trace_store)

    def _init_proposal_store() -> Any:
        from cognithor.learning.trace_optimizer import ProposalStore

        return ProposalStore(Path(str(config.db_path.with_name("memory_proposals.db"))))

    _init_subsystem("proposal_store", result, _init_proposal_store)

    def _init_gepa() -> Any:
        from cognithor.learning.causal_attributor import CausalAttributor
        from cognithor.learning.evolution_orchestrator import EvolutionOrchestrator
        from cognithor.learning.trace_optimizer import TraceOptimizer

        if not (getattr(config, "gepa", None) and config.gepa.enabled):
            return None
        ts = result.get("trace_store")
        ps = result.get("proposal_store")
        if not (ts and ps):
            return None
        return EvolutionOrchestrator(
            trace_store=ts,
            attributor=CausalAttributor(),
            optimizer=TraceOptimizer(proposal_store=ps),
            proposal_store=ps,
            min_traces=config.gepa.min_traces_for_proposal,
            max_active=config.gepa.max_active_optimizations,
            rollback_threshold=config.gepa.auto_rollback_threshold,
            auto_apply=config.gepa.auto_apply,
        )

    _init_subsystem("evolution_orchestrator", result, _init_gepa)

    def _init_strategy_memory() -> Any:
        from cognithor.learning.strategy_memory import StrategyMemory

        return StrategyMemory(db_path=Path(cognithor_home) / "index" / "strategy_memory.db")  # type: ignore[arg-type]

    _init_subsystem("strategy_memory", result, _init_strategy_memory)

    def _init_reflexion() -> Any:
        from cognithor.learning.reflexion import ReflexionMemory

        return ReflexionMemory(data_dir=Path(cognithor_home) / "memory")

    _init_subsystem("reflexion_memory", result, _init_reflexion)

    def _init_hermes() -> Any:
        from cognithor.skills.hermes_compat import HermesCompatLayer

        return HermesCompatLayer()

    _init_subsystem("hermes_compat", result, _init_hermes)

    def _init_run_recorder() -> Any:
        from cognithor.forensics.run_recorder import RunRecorder

        return RunRecorder(str(config.db_path.with_name("memory_runs.db")))

    _init_subsystem("run_recorder", result, _init_run_recorder)

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
        config: CognithorConfig instance.
        task_telemetry: TaskTelemetryCollector (from PGE phase).
        error_clusterer: ErrorClusterer (from PGE phase).
        task_profiler: TaskProfiler (from PGE phase).
        cost_tracker: CostTracker (from tools phase).
        run_recorder: RunRecorder (from declare_advanced_attrs).
    """
    result: PhaseResult = {}
    cognithor_home = getattr(config, "cognithor_home", Path.home() / ".cognithor")

    # --- All subsystems via _safe_call (failures logged + tracked) ---

    def _init_governance() -> Any:
        from cognithor.governance.governor import GovernanceAgent

        return GovernanceAgent(
            task_telemetry=task_telemetry,
            error_clusterer=error_clusterer,
            task_profiler=task_profiler,
            cost_tracker=cost_tracker,
            run_recorder=run_recorder,
            db_path=str(config.db_path.with_name("memory_governance.db")),
        )

    _init_subsystem("governance_agent", result, _init_governance)

    def _init_improvement_gate() -> Any:
        from cognithor.governance.improvement_gate import ImprovementGate

        gate = ImprovementGate(config.improvement)
        if result.get("governance_agent"):
            result["governance_agent"].improvement_gate = gate
        return gate

    _init_subsystem("improvement_gate", result, _init_improvement_gate)

    def _init_prompt_evolution() -> Any:
        from cognithor.learning.prompt_evolution import PromptEvolutionEngine

        if not config.prompt_evolution.enabled:
            return None
        return PromptEvolutionEngine(
            db_path=str(config.db_path.with_name("memory_prompt_evolution.db")),
            min_sessions_per_arm=config.prompt_evolution.min_sessions_per_arm,
            significance_threshold=config.prompt_evolution.significance_threshold,
            max_concurrent_tests=config.prompt_evolution.max_concurrent_tests,
        )

    _init_subsystem("prompt_evolution", result, _init_prompt_evolution)

    def _init_session_analyzer() -> Any:
        from cognithor.learning.session_analyzer import SessionAnalyzer

        return SessionAnalyzer(data_dir=Path(cognithor_home) / "memory")

    _init_subsystem("session_analyzer", result, _init_session_analyzer)

    def _init_curiosity() -> Any:
        from cognithor.learning.curiosity import CuriosityEngine

        return CuriosityEngine()

    _init_subsystem("curiosity_engine", result, _init_curiosity)

    def _init_confidence() -> Any:
        from cognithor.learning.confidence import KnowledgeConfidenceManager

        return KnowledgeConfidenceManager()

    _init_subsystem("confidence_manager", result, _init_confidence)

    def _init_active_learner() -> Any:
        from cognithor.learning.active_learner import ActiveLearner

        return ActiveLearner()

    _init_subsystem("active_learner", result, _init_active_learner)

    def _init_exploration() -> Any:
        from cognithor.learning.explorer import ExplorationExecutor

        return ExplorationExecutor(
            curiosity=result.get("curiosity_engine"),
            memory=getattr(config, "_memory_manager", None),
        )

    _init_subsystem("exploration_executor", result, _init_exploration)

    def _init_knowledge_qa() -> Any:
        from cognithor.learning.knowledge_qa import KnowledgeQAStore

        return KnowledgeQAStore(db_path=Path(cognithor_home) / "memory" / "knowledge_qa.db")

    _init_subsystem("knowledge_qa", result, _init_knowledge_qa)

    def _init_knowledge_lineage() -> Any:
        from cognithor.learning.lineage import KnowledgeLineageTracker

        return KnowledgeLineageTracker(
            db_path=Path(cognithor_home) / "memory" / "knowledge_lineage.db"
        )

    _init_subsystem("knowledge_lineage", result, _init_knowledge_lineage)

    def _init_knowledge_ingest() -> Any:
        from cognithor.learning.knowledge_ingest import KnowledgeIngestService

        return KnowledgeIngestService(
            memory=getattr(config, "_memory_manager", None),
            knowledge_builder=result.get("knowledge_builder"),
            llm_fn=result.get("llm_fn"),
        )

    _init_subsystem("knowledge_ingest", result, _init_knowledge_ingest)

    def _init_leads_service() -> Any:
        from cognithor.leads.service import LeadService
        from cognithor.leads.store import LeadStore

        _db_path = str(Path(cognithor_home) / "leads.db")
        _store = LeadStore(_db_path)
        result["leads_store"] = _store
        return LeadService(store=_store)

    _init_subsystem("leads_service", result, _init_leads_service)
    # Legacy alias — gateway code that still uses _reddit_lead_service gets None
    # which is handled gracefully by getattr(..., None) guards everywhere.
    result["reddit_lead_service"] = None

    if gatekeeper is not None:

        def _init_replay() -> Any:
            from cognithor.forensics.replay_engine import ReplayEngine

            return ReplayEngine(gatekeeper)

        _init_subsystem("replay_engine", result, _init_replay)

    # Wire GEPA dependencies
    orch = result.get("evolution_orchestrator")
    if orch:
        if result.get("prompt_evolution"):
            orch._prompt_evolution = result["prompt_evolution"]
        if result.get("session_analyzer"):
            orch._session_analyzer = result["session_analyzer"]

    def _init_hashline() -> Any:
        from cognithor.hashline import HashlineGuard
        from cognithor.hashline.config import HashlineConfig as HLConfig

        hl_cfg_model = getattr(config, "hashline", None)
        hl_dict = (
            hl_cfg_model.model_dump()
            if hl_cfg_model and hasattr(hl_cfg_model, "model_dump")
            else {}
        )
        hl_cfg = HLConfig.from_dict(hl_dict)
        if not hl_cfg.enabled:
            return None
        return HashlineGuard.create(config=hl_cfg, data_dir=Path(cognithor_home))

    _init_subsystem("hashline_guard", result, _init_hashline)

    return result
