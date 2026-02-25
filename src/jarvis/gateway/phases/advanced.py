"""Advanced phase: Monitoring, isolation, auth, connectors, workflows, ecosystem, extensions, governance, portal, CLI, etc.

Attributes handled:
  _monitoring_hub, _isolation, _workspace_guard, _auth_gateway,
  _connector_registry, _workflow_engine, _template_library,
  _ecosystem_policy, _model_registry, _i18n, _reputation_engine,
  _recall_manager, _abuse_reporter, _governance_policy, _interop,
  _governance_hub, _ecosystem_controller, _user_portal, _skill_cli,
  _setup_wizard, _perf_manager
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.gateway.phases import PhaseResult
from jarvis.utils.logging import get_logger

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
        from jarvis.core.workflows import WorkflowEngine, TemplateLibrary
        result["workflow_engine"] = WorkflowEngine()
        result["template_library"] = TemplateLibrary()
    except Exception:
        log.debug("workflow_engine_init_skipped", exc_info=True)

    # Phase 17: Ecosystem-Policy
    try:
        from jarvis.core.workflows import EcosystemPolicy
        result["ecosystem_policy"] = EcosystemPolicy()
    except Exception:
        log.debug("ecosystem_policy_init_skipped", exc_info=True)

    # Phase 18: Model-Extension-Registry & i18n
    try:
        from jarvis.core.extensions import ModelExtensionRegistry, I18nManager
        result["model_registry"] = ModelExtensionRegistry()
        result["i18n"] = I18nManager(default_locale="de")
    except Exception:
        log.debug("model_registry_init_skipped", exc_info=True)

    # Phase 20: Marketplace-Governance
    try:
        from jarvis.skills.governance import (
            ReputationEngine,
            SkillRecallManager,
            AbuseReporter,
            GovernancePolicy,
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
        result["user_portal"] = UserPortal()
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

    # ReplayEngine (needs Gatekeeper for policy re-evaluation)
    if gatekeeper is not None:
        try:
            from jarvis.forensics.replay_engine import ReplayEngine
            result["replay_engine"] = ReplayEngine(gatekeeper)
            log.info("replay_engine_initialized")
        except Exception:
            log.debug("replay_engine_init_skipped", exc_info=True)

    return result
