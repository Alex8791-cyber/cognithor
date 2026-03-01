"""PGE phase: Planner, Gate (executor), Reflector -- the PGE trinity.

Attributes handled:
  _planner, _executor, _reflector, _skill_generator
"""

from __future__ import annotations

from typing import Any

from jarvis.gateway.phases import PhaseResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


def declare_pge_attrs(config: Any) -> PhaseResult:
    """Return default (None) values for PGE attributes."""
    return {
        "planner": None,
        "executor": None,
        "reflector": None,
        "skill_generator": None,
        "task_profiler": None,
        "task_telemetry": None,
        "error_clusterer": None,
        "causal_analyzer": None,
    }


async def init_pge(
    config: Any,
    llm: Any,
    model_router: Any,
    mcp_client: Any,
    runtime_monitor: Any,
    audit_logger: Any,
    memory_manager: Any = None,
    cost_tracker: Any = None,
) -> PhaseResult:
    """Initialize the PGE trinity (Planner, Executor, Reflector).

    Args:
        config: JarvisConfig instance.
        llm: UnifiedLLMClient instance.
        model_router: ModelRouter instance.
        mcp_client: JarvisMCPClient instance.
        runtime_monitor: RuntimeMonitor instance.
        audit_logger: AuditLogger instance.
        memory_manager: MemoryManager (optional, for new subsystems).

    Returns:
        PhaseResult with planner, executor, reflector.
    """
    from jarvis.core.executor import Executor
    from jarvis.core.planner import Planner
    from jarvis.core.reflector import Reflector

    result: PhaseResult = {}

    # Optional subsystems from memory_manager
    episodic_store = None
    weight_optimizer = None
    causal_analyzer = None
    task_profiler = None
    task_telemetry = None
    error_clusterer = None

    if memory_manager is not None:
        episodic_store = getattr(memory_manager, "episodic_store", None)
        weight_optimizer = getattr(memory_manager, "weight_optimizer", None)

    # Task Profiler (optional)
    try:
        from jarvis.core.profiler import TaskProfiler
        task_profiler = TaskProfiler()
    except Exception:
        log.debug("task_profiler_init_skipped")

    # Task Telemetry + Error Clusterer (optional)
    try:
        from jarvis.telemetry.task_telemetry import TaskTelemetryCollector
        task_telemetry = TaskTelemetryCollector()
    except Exception:
        log.debug("task_telemetry_init_skipped")

    try:
        from jarvis.telemetry.error_clustering import ErrorClusterer
        error_clusterer = ErrorClusterer()
    except Exception:
        log.debug("error_clusterer_init_skipped")

    # Causal Analyzer (optional)
    try:
        from jarvis.learning.causal import CausalAnalyzer
        causal_analyzer = CausalAnalyzer()
    except Exception:
        log.debug("causal_analyzer_init_skipped")

    # RewardCalculator (optional)
    reward_calculator = None
    try:
        from jarvis.learning.reward import RewardCalculator
        reward_calculator = RewardCalculator()
    except Exception:
        log.debug("reward_calculator_init_skipped")

    # SkillGenerator (Self-Learning Pipeline)
    # NOTE: SkillGenerator creates its own internal GapDetector.
    # We pass that SAME instance to the Executor so gaps flow into one place.
    skill_generator = None
    gap_detector = None
    try:
        from jarvis.skills.generator import SkillGenerator

        skills_dir = config.jarvis_home / "skills" / "generated"
        skills_dir.mkdir(parents=True, exist_ok=True)

        # LLM wrapper: SkillGenerator expects async (prompt: str) -> str
        async def _skill_llm_fn(prompt: str) -> str:
            model = model_router.select_model("coding", "medium")
            model_config = model_router.get_model_config(model)
            response = await llm.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_config.get("temperature", 0.3),
            )
            return response.get("message", {}).get("content", "")

        # SandboxExecutor for testing generated skills
        sandbox_executor = None
        try:
            from jarvis.core.sandbox import SandboxExecutor
            sandbox_executor = SandboxExecutor(config)
        except Exception:
            log.debug("skill_sandbox_init_skipped")

        skill_generator = SkillGenerator(
            skills_dir=skills_dir,
            sandbox_executor=sandbox_executor,
            llm_fn=_skill_llm_fn,
            require_approval=getattr(config.gatekeeper, "auto_approve_threshold", 0.7) < 0.5,
            audit_logger=audit_logger,
        )
        # Use the SkillGenerator's internal GapDetector for the Executor too
        gap_detector = skill_generator.gap_detector
        log.info("skill_generator_initialized", skills_dir=str(skills_dir))
    except Exception:
        log.debug("skill_generator_init_skipped", exc_info=True)
        # Standalone GapDetector even without SkillGenerator (for telemetry)
        try:
            from jarvis.skills.generator import GapDetector
            gap_detector = GapDetector()
        except Exception:
            pass

    # Executor (with retry/backoff + security + profiling + telemetry + gap detection)
    try:
        result["executor"] = Executor(
            config,
            mcp_client,
            gap_detector=gap_detector,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
            task_profiler=task_profiler,
            task_telemetry=task_telemetry,
            error_clusterer=error_clusterer,
        )
    except Exception:
        log.error("executor_init_failed", exc_info=True)
        raise

    # Planner (uses UnifiedLLMClient + optional causal suggestions + task profiler)
    try:
        result["planner"] = Planner(
            config, llm, model_router,
            audit_logger=audit_logger,
            causal_analyzer=causal_analyzer,
            task_profiler=task_profiler,
            cost_tracker=cost_tracker,
        )
    except Exception:
        log.error("planner_init_failed", exc_info=True)
        raise

    # Reflector (uses UnifiedLLMClient + episodic store + causal + weight optimizer + reward)
    try:
        result["reflector"] = Reflector(
            config, llm, model_router,
            audit_logger=audit_logger,
            episodic_store=episodic_store,
            causal_analyzer=causal_analyzer,
            weight_optimizer=weight_optimizer,
            reward_calculator=reward_calculator,
            cost_tracker=cost_tracker,
        )
    except Exception:
        log.error("reflector_init_failed", exc_info=True)
        raise

    # Store profiler + telemetry + skill generator for gateway access
    result["task_profiler"] = task_profiler
    result["task_telemetry"] = task_telemetry
    result["error_clusterer"] = error_clusterer
    result["causal_analyzer"] = causal_analyzer
    result["skill_generator"] = skill_generator

    return result
