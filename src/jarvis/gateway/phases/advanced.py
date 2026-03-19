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
        "monitoring_hub": None,
        "isolation": None,
        "workspace_guard": None,
        "auth_gateway": None,
        "connector_registry": None,
        "workflow_engine": None,
        "template_library": None,
        "ecosystem_policy": None,
        "model_registry": None,
        "i18n": None,
        "reputation_engine": None,
        "recall_manager": None,
        "abuse_reporter": None,
        "governance_policy": None,
        "interop": None,
        "governance_hub": None,
        "ecosystem_controller": None,
        "user_portal": None,
        "skill_cli": None,
        "setup_wizard": None,
        "perf_manager": None,
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
        "self_improver": None,
        "hermes_compat": None,
        "trace_store": None,
        "proposal_store": None,
        "evolution_orchestrator": None,
    }

    # Phase 5: Live-Monitoring
    try:
        from jarvis.gateway.monitoring import MonitoringHub

        result["monitoring_hub"] = MonitoringHub()
    except Exception:
        log.debug("monitoring_hub_init_skipped", exc_info=True)

    # Phase 6: Agent-Isolation
    try:
        from jarvis.core.isolation import MultiUserIsolation, WorkspaceGuard

        result["isolation"] = MultiUserIsolation()
        workspace_root = Path(config.jarvis_home) / "workspace"
        result["workspace_guard"] = WorkspaceGuard(workspace_root)
    except Exception:
        log.debug("agent_isolation_init_skipped", exc_info=True)

    # Phase 7: Auth-Gateway (SSO + Per-Agent Sessions)
    try:
        from jarvis.gateway.auth import AuthGateway

        result["auth_gateway"] = AuthGateway()
    except Exception:
        log.debug("auth_gateway_init_skipped", exc_info=True)

    # Phase 15: Enterprise-Connectors
    try:
        from jarvis.channels.connectors import ConnectorRegistry

        result["connector_registry"] = ConnectorRegistry()
    except Exception:
        log.debug("connector_registry_init_skipped", exc_info=True)

    # Phase 16: Workflow-Engine & Template-Library
    try:
        from jarvis.core.workflows import TemplateLibrary, WorkflowEngine

        result["workflow_engine"] = WorkflowEngine()
        result["template_library"] = TemplateLibrary()
    except Exception:
        log.debug("workflow_engine_init_skipped", exc_info=True)

    # Phase 16b: DAG WorkflowEngine
    try:
        from jarvis.core.workflow_engine import WorkflowEngine as DAGWorkflowEngine

        result["dag_workflow_engine"] = DAGWorkflowEngine()
    except Exception:
        log.debug("dag_workflow_engine_init_skipped", exc_info=True)

    # Phase 17: Ecosystem-Policy
    try:
        from jarvis.core.workflows import EcosystemPolicy

        result["ecosystem_policy"] = EcosystemPolicy()
    except Exception:
        log.debug("ecosystem_policy_init_skipped", exc_info=True)

    # Phase 18: Model-Extension-Registry & i18n
    try:
        from jarvis.core.extensions import I18nManager, ModelExtensionRegistry

        result["model_registry"] = ModelExtensionRegistry()
        result["i18n"] = I18nManager(default_locale="de")
    except Exception:
        log.debug("model_registry_init_skipped", exc_info=True)

    # Phase 20: Marketplace-Governance
    try:
        from jarvis.skills.governance import (
            AbuseReporter,
            GovernancePolicy,
            ReputationEngine,
            SkillRecallManager,
        )

        rep = ReputationEngine()
        result["reputation_engine"] = rep
        result["recall_manager"] = SkillRecallManager(rep)
        result["abuse_reporter"] = AbuseReporter(rep)
        result["governance_policy"] = GovernancePolicy()
    except Exception:
        log.debug("marketplace_governance_init_skipped", exc_info=True)

    # Phase 22: Cross-Agent Interop
    try:
        from jarvis.core.interop import InteropProtocol

        result["interop"] = InteropProtocol(local_agent_id="jarvis-main")
    except Exception:
        log.debug("interop_init_skipped", exc_info=True)

    # Phase 28: Ecosystem-Control + Security-Training
    try:
        from jarvis.skills.ecosystem_control import EcosystemController

        result["ecosystem_controller"] = EcosystemController()
    except Exception:
        log.debug("ecosystem_controller_init_skipped", exc_info=True)

    # Phase 31: Governance Hub (Curation, Diversity, Budget, Explainability)
    try:
        from jarvis.core.curation import GovernanceHub

        result["governance_hub"] = GovernanceHub()
    except Exception:
        log.debug("governance_hub_init_skipped", exc_info=True)

    # Phase 34: User Portal (DSGVO, Consent, Transparency)
    try:
        from jarvis.core.user_portal import UserPortal

        jarvis_home = getattr(config, "jarvis_home", Path.home() / ".jarvis")
        consent_db = jarvis_home / "consent.db"
        result["user_portal"] = UserPortal(consent_db_path=consent_db)
    except Exception:
        log.debug("user_portal_init_skipped", exc_info=True)

    # Phase 35: Skill-CLI (Developer Tools, Rewards)
    try:
        from jarvis.tools.skill_cli import SkillCLI

        result["skill_cli"] = SkillCLI()
    except Exception:
        log.debug("skill_cli_init_skipped", exc_info=True)

    # Phase 36: Setup-Wizard (Hardware-Detection, Presets)
    try:
        from jarvis.core.installer import SetupWizard

        result["setup_wizard"] = SetupWizard()
    except Exception:
        log.debug("setup_wizard_init_skipped", exc_info=True)

    # Phase 37: Performance-Manager (Vector, Balancer, Fallback)
    try:
        from jarvis.core.performance import PerformanceManager

        result["perf_manager"] = PerformanceManager()
    except Exception:
        log.debug("perf_manager_init_skipped", exc_info=True)

    # SelfImprover (Error-pattern learning)
    try:
        from jarvis.learning.self_improver import SelfImprover

        result["self_improver"] = SelfImprover()
    except Exception:
        log.debug("self_improver_init_skipped", exc_info=True)

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

    return result
