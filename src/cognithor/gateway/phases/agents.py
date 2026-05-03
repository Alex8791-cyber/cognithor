"""Agents phase: Skill registry, agent router, ingest, heartbeat, cron, commands.

Attributes handled:
  _skill_registry, _agent_router, _ingest_pipeline,
  _heartbeat_scheduler, _cron_engine, _agent_heartbeat,
  _command_registry, _interaction_store
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.phases import PhaseResult

log = get_logger(__name__)


def declare_agents_attrs(config: Any) -> PhaseResult:
    """Return default values for agent-related attributes.

    Eagerly constructs instances where possible (matching original __init__).
    """
    result: PhaseResult = {
        "skill_registry": None,
        "skill_lifecycle": None,
        "agent_router": None,
        "ingest_pipeline": None,
        "heartbeat_scheduler": None,
        "cron_engine": None,
        "agent_heartbeat": None,
        "command_registry": None,
        "interaction_store": None,
        "orchestrator": None,
    }

    # Phase 8: Per-Agent Heartbeat-Scheduler
    try:
        from cognithor.core.agent_heartbeat import AgentHeartbeatScheduler

        result["agent_heartbeat"] = AgentHeartbeatScheduler()
    except Exception:
        log.debug("agent_heartbeat_init_skipped", exc_info=True)

    # Phase 8b: Multi-Agent Orchestrator
    try:
        from cognithor.core.orchestrator import Orchestrator

        result["orchestrator"] = Orchestrator()
    except Exception:
        log.debug("orchestrator_init_skipped", exc_info=True)

    # Phase 9: Slash-Commands + Interaction-State
    try:
        from cognithor.channels.commands import CommandRegistry, InteractionStore

        result["command_registry"] = CommandRegistry()
        result["interaction_store"] = InteractionStore()
    except Exception:
        log.debug("command_registry_init_skipped", exc_info=True)

    return result


async def init_agents(
    config: Any,
    memory_manager: Any,
    mcp_client: Any,
    audit_logger: Any,
    cognithor_home: Any,
    handle_message: Any = None,
    heartbeat_config: Any = None,
    heartbeat_scheduler_instance: Any = None,
) -> PhaseResult:
    """Initialize agent subsystems: skills, router, ingest, heartbeat, cron.

    Args:
        config: CognithorConfig instance.
        memory_manager: MemoryManager instance.
        mcp_client: JarvisMCPClient instance.
        audit_logger: AuditLogger instance.
        cognithor_home: Path to jarvis home directory.
        handle_message: Gateway.handle_message callback (for cron engine).
        heartbeat_config: Heartbeat configuration (from config.heartbeat).
        heartbeat_scheduler_instance: Existing heartbeat scheduler (or None).

    Returns:
        PhaseResult with skill_registry, agent_router, ingest_pipeline,
        heartbeat_scheduler, cron_engine.
    """
    from cognithor.core.agent_router import AgentRouter
    from cognithor.cron.engine import CronEngine

    result: PhaseResult = {}

    # Skill Registry (loads skills from procedures + user skills)
    skill_registry = None
    try:
        from cognithor.skills.registry import SkillRegistry

        skill_registry = SkillRegistry()
        skill_dirs = [
            cognithor_home / "data" / "procedures",
            cognithor_home / config.plugins.skills_dir,
        ]
        # Ensure generated skills directory exists (loaded automatically
        # by SkillRegistry.load_generated_skills via the skills parent dir)
        generated_dir = cognithor_home / "skills" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        # Also check repo data/procedures directory
        repo_procedures = Path(__file__).parent.parent.parent.parent.parent / "data" / "procedures"
        if repo_procedures.exists():
            skill_dirs.insert(0, repo_procedures)
        skill_count = skill_registry.load_from_directories(skill_dirs)
        log.info("skill_registry_ready", skills=skill_count)

        # Skill-Management-Tools registrieren (create_skill, list_skills)
        try:
            from cognithor.mcp.skill_tools import register_skill_tools

            register_skill_tools(mcp_client, skill_registry, skill_dirs)
        except Exception as exc_tools:
            log.warning("skill_tools_registration_failed", error=str(exc_tools))

    except Exception as exc:
        log.warning("skill_registry_init_error", error=str(exc))
    result["skill_registry"] = skill_registry

    # Create SkillLifecycleManager for periodic auditing
    skill_lifecycle = None
    try:
        from cognithor.skills.lifecycle import SkillLifecycleManager

        generated_dir = cognithor_home / "skills" / "generated"
        skill_lifecycle = SkillLifecycleManager(
            registry=skill_registry,
            generated_dir=generated_dir,
        )
        log.info("skill_lifecycle_manager_created")
    except Exception:
        log.debug("skill_lifecycle_manager_creation_failed", exc_info=True)
    result["skill_lifecycle"] = skill_lifecycle

    # Agent Router (multi-agent routing + audit)
    agents_config = cognithor_home / "config" / "agents.yaml"
    if agents_config.exists():
        agent_router = AgentRouter.from_yaml(
            agents_config,
            audit_logger=audit_logger,
        )
    else:
        agent_router = AgentRouter(audit_logger=audit_logger)
        agent_router.initialize()
    log.info("agent_router_ready", agents=len(agent_router.list_enabled()))
    result["agent_router"] = agent_router

    # Knowledge Ingest Pipeline
    ingest_pipeline = None
    try:
        from cognithor.memory.ingest import IngestConfig, IngestPipeline

        ingest_config = IngestConfig(
            watch_dir=cognithor_home / "ingest",
            processed_dir=cognithor_home / "ingest" / "processed",
            failed_dir=cognithor_home / "ingest" / "failed",
        )
        ingest_pipeline = IngestPipeline(ingest_config, memory_manager)
        ingest_stats = ingest_pipeline.stats()
        log.info("ingest_pipeline_ready", pending=ingest_stats["pending"])
    except Exception as exc:
        log.warning("ingest_pipeline_init_error", error=str(exc))
    result["ingest_pipeline"] = ingest_pipeline

    # HeartbeatScheduler (proactive automation)
    heartbeat_scheduler = None
    try:
        from cognithor.proactive import HeartbeatScheduler

        heartbeat_scheduler = HeartbeatScheduler()
        log.info(
            "heartbeat_scheduler_ready",
            enabled_events=len(heartbeat_scheduler.enabled_configs()),
        )
    except Exception:
        log.debug("heartbeat_scheduler_init_skipped", exc_info=True)
    result["heartbeat_scheduler"] = heartbeat_scheduler

    # Cron-Engine (with HeartbeatScheduler integration)
    cron_engine = None
    try:
        cron_engine = CronEngine(
            jobs_path=config.cron_config_file,
            handler=handle_message,
            heartbeat_config=config.heartbeat,
            cognithor_home=cognithor_home,
            heartbeat_scheduler=heartbeat_scheduler,
        )
    except Exception:
        log.debug("cron_engine_init_skipped", exc_info=True)
    result["cron_engine"] = cron_engine

    return result
