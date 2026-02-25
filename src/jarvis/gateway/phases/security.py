"""Security phase: Runtime monitoring, audit, gatekeeper, vault, red-team, etc.

Attributes handled:
  _runtime_monitor, _audit_logger, _gatekeeper, _vault_manager,
  _isolated_sessions, _session_guard, _security_scanner, _security_pipeline,
  _security_gate, _continuous_redteam, _scan_scheduler, _webhook_notifier,
  _isolation_enforcer, _incident_tracker, _security_metrics, _security_team,
  _posture_scorer, _agent_vault_manager, _red_team, _code_auditor
"""

from __future__ import annotations

from typing import Any

from jarvis.gateway.phases import PhaseResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


def declare_security_attrs(config: Any) -> PhaseResult:
    """Return default values for all security attributes.

    Attempts eager construction where possible (matching original __init__).
    """
    result: PhaseResult = {
        "runtime_monitor": None,
        "audit_logger": None,
        "gatekeeper": None,
        "vault_manager": None,
        "isolated_sessions": None,
        "session_guard": None,
        "security_scanner": None,
        "security_pipeline": None,
        "security_gate": None,
        "continuous_redteam": None,
        "scan_scheduler": None,
        "webhook_notifier": None,
        "isolation_enforcer": None,
        "incident_tracker": None,
        "security_metrics": None,
        "security_team": None,
        "posture_scorer": None,
        "agent_vault_manager": None,
        "red_team": None,
        "code_auditor": None,
    }

    # Phase 10: Red-Team Security-Scanner
    try:
        from jarvis.security.redteam import SecurityScanner
        result["security_scanner"] = SecurityScanner()
    except Exception:
        log.debug("security_scanner_init_skipped", exc_info=True)

    # Phase 14: Encrypted Vault & Session-Isolation
    try:
        from jarvis.security.vault import (
            VaultManager,
            IsolatedSessionStore,
            SessionIsolationGuard,
        )
        vault = VaultManager()
        isolated = IsolatedSessionStore()
        result["vault_manager"] = vault
        result["isolated_sessions"] = isolated
        result["session_guard"] = SessionIsolationGuard(vault, isolated)
    except Exception:
        log.debug("vault_session_isolation_init_skipped", exc_info=True)

    # Phase 19: MLOps Security Pipeline
    try:
        from jarvis.security.mlops_pipeline import SecurityPipeline
        result["security_pipeline"] = SecurityPipeline()
    except Exception:
        log.debug("security_pipeline_init_skipped", exc_info=True)

    # Phase 21: AI Agent Security Framework
    try:
        from jarvis.security.framework import (
            IncidentTracker,
            SecurityMetrics,
            SecurityTeam,
            PostureScorer,
        )
        tracker = IncidentTracker()
        result["incident_tracker"] = tracker
        result["security_metrics"] = SecurityMetrics(tracker)
        result["security_team"] = SecurityTeam()
        result["posture_scorer"] = PostureScorer()
    except Exception:
        log.debug("security_framework_init_skipped", exc_info=True)

    # Phase 24: CI/CD Security Gate + Continuous Red-Team
    try:
        from jarvis.security.cicd_gate import (
            SecurityGate,
            ContinuousRedTeam,
            ScanScheduler,
            WebhookNotifier,
        )
        result["security_gate"] = SecurityGate()
        result["continuous_redteam"] = ContinuousRedTeam()
        result["scan_scheduler"] = ScanScheduler()
        result["webhook_notifier"] = WebhookNotifier()
    except Exception:
        log.debug("cicd_security_gate_init_skipped", exc_info=True)

    # Phase 25: Strikte Sandbox-Isolierung + Multi-Tenant
    try:
        from jarvis.security.sandbox_isolation import IsolationEnforcer
        result["isolation_enforcer"] = IsolationEnforcer()
    except Exception:
        log.debug("isolation_enforcer_init_skipped", exc_info=True)

    # Phase 29: Per-Agent Vault & Session-Isolation
    try:
        from jarvis.security.agent_vault import AgentVaultManager
        result["agent_vault_manager"] = AgentVaultManager()
    except Exception:
        log.debug("agent_vault_manager_init_skipped", exc_info=True)

    # Phase 30: Red-Team-Framework
    try:
        from jarvis.security.red_team import RedTeamFramework
        result["red_team"] = RedTeamFramework()
    except Exception:
        log.debug("red_team_framework_init_skipped", exc_info=True)

    # Phase 33: Automatisierte Code-Analyse
    try:
        from jarvis.security.code_audit import CodeAuditor
        result["code_auditor"] = CodeAuditor()
    except Exception:
        log.debug("code_auditor_init_skipped", exc_info=True)

    return result


async def init_security(config: Any, llm_backend: Any = None) -> PhaseResult:
    """Initialize runtime security subsystems (audit logger, monitor, gatekeeper).

    Args:
        config: JarvisConfig instance.
        llm_backend: Not currently used, reserved for future LLM-based security.

    Returns:
        PhaseResult with runtime_monitor, audit_logger, gatekeeper.
    """
    from jarvis.audit import AuditLogger
    from jarvis.core.gatekeeper import Gatekeeper
    from jarvis.security.monitor import RuntimeMonitor

    result: PhaseResult = {}

    # Audit Logger
    audit_log_dir = config.jarvis_home / "data" / "audit"
    audit_logger = AuditLogger(log_dir=audit_log_dir, retention_days=90)
    result["audit_logger"] = audit_logger

    # Runtime Monitor
    runtime_monitor = RuntimeMonitor(enable_defaults=True)
    result["runtime_monitor"] = runtime_monitor

    log.info(
        "security_layer_ready",
        audit_dir=str(audit_log_dir),
        monitor_rules=runtime_monitor.stats()["active_rules"],
    )

    # Gatekeeper (deterministic, no LLM needed)
    from jarvis.models import OperationMode
    op_mode = getattr(config, "resolved_operation_mode", None)
    if isinstance(op_mode, str):
        try:
            op_mode = OperationMode(op_mode)
        except ValueError:
            op_mode = None
    gatekeeper = Gatekeeper(config, audit_logger=audit_logger, operation_mode=op_mode)
    gatekeeper.initialize()
    result["gatekeeper"] = gatekeeper

    return result
