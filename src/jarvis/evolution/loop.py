"""Evolution Loop — orchestrates Scout/Research/Build/Reflect cycles during idle time."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.evolution.checkpoint import EvolutionCheckpoint
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.core.checkpointing import CheckpointStore
    from jarvis.evolution.idle_detector import IdleDetector
    from jarvis.system.resource_monitor import ResourceMonitor
    from jarvis.telemetry.cost_tracker import CostTracker

log = get_logger(__name__)

__all__ = ["EvolutionLoop", "EvolutionCycleResult"]


@dataclass
class _LearningGoal:
    """Wrapper for learning goals from any source."""

    query: str = ""
    question: str = ""
    source: str = "user"  # "user" | "self_analysis" | "curiosity"
    target_skill: str = ""  # Skill slug if this is a skill improvement task

    def __post_init__(self) -> None:
        if not self.question:
            self.question = self.query

    def __str__(self) -> str:
        return self.query


@dataclass
class EvolutionCycleResult:
    """Result of one evolution cycle."""

    cycle_id: int = 0
    skipped: bool = False
    reason: str = ""
    gaps_found: int = 0
    research_topic: str = ""
    skill_created: str = ""
    duration_ms: int = 0
    source: str = ""  # "curiosity" | "user" | "self_analysis"
    steps_completed: list[str] = field(default_factory=list)


class EvolutionLoop:
    """Orchestrates autonomous learning during idle time.

    Cycle: Scout (find gaps) -> Research (deep_research) -> Build (create skill) -> Reflect
    Each step checks idle_detector — aborts immediately if user returns.
    """

    def __init__(
        self,
        idle_detector: IdleDetector,
        curiosity_engine: Any = None,
        skill_generator: Any = None,
        memory_manager: Any = None,
        config: Any = None,
        resource_monitor: ResourceMonitor | None = None,
        cost_tracker: CostTracker | None = None,
        checkpoint_store: CheckpointStore | None = None,
        operation_mode: str = "offline",
        mcp_client: Any = None,
        llm_fn: Any = None,
        skill_registry: Any = None,
        session_analyzer: Any = None,
    ) -> None:
        self._idle = idle_detector
        self._curiosity = curiosity_engine
        self._skill_gen = skill_generator
        self._memory = memory_manager
        self._config = config
        self._resource_monitor = resource_monitor
        self._cost_tracker = cost_tracker
        self._checkpoint_store = checkpoint_store
        self._operation_mode = operation_mode
        self._mcp_client = mcp_client
        self._llm_fn = llm_fn
        self._skill_registry = skill_registry
        self._session_analyzer = session_analyzer
        self._deep_learner: Any = None  # Set by gateway after construction
        self._current_checkpoint: EvolutionCheckpoint | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._cycles_today = 0
        self._last_cycle_day = ""
        self._total_cycles = 0
        self._total_skills_created = 0
        self._paused_for_resources = False
        self._results: list[EvolutionCycleResult] = []

    async def start(self) -> None:
        """Start the evolution background loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="evolution-loop")
        log.info("evolution_loop_started")

    def stop(self) -> None:
        """Stop the evolution loop."""
        self._running = False
        if self._task:
            self._task.cancel()
        log.info("evolution_loop_stopped")

    async def run_cycle(self) -> EvolutionCycleResult:
        """Run one Scout->Research->Build->Reflect cycle."""
        t0 = time.monotonic()
        self._total_cycles += 1
        result = EvolutionCycleResult(cycle_id=self._total_cycles)

        # Pre-check: still idle?
        if not self._idle.is_idle:
            result.skipped = True
            result.reason = "not_idle"
            return result

        # Pre-check: system resources available?
        if not await self._check_resources():
            result.skipped = True
            result.reason = "system_busy"
            return result

        # Pre-check: evolution budget not exhausted?
        if not self._check_evolution_budget():
            result.skipped = True
            result.reason = "budget_exhausted"
            return result

        # Step 1: Scout — find knowledge gaps
        gaps = await self._scout()
        result.steps_completed.append("scout")
        if not gaps:
            result.skipped = True
            result.reason = "no_gaps"
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            return result
        result.gaps_found = len(gaps)

        self._save_checkpoint(
            EvolutionCheckpoint(
                cycle_id=result.cycle_id,
                step_name="scout",
                step_index=0,
                gaps_found=result.gaps_found,
                steps_completed=list(result.steps_completed),
                delta={"gaps_found": result.gaps_found},
            )
        )

        if not self._idle.is_idle:
            result.skipped = True
            result.reason = "interrupted_after_scout"
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            return result

        # Resource check before research
        if not await self._check_resources():
            result.reason = "system_busy_before_research"
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            return result

        # Step 2: Research — investigate the top gap
        top_gap = gaps[0]
        result.research_topic = getattr(top_gap, "question", str(top_gap))[:100]
        result.source = getattr(top_gap, "source", "curiosity")
        research_text = await self._research(top_gap)
        result.steps_completed.append("research")

        self._save_checkpoint(
            EvolutionCheckpoint(
                cycle_id=result.cycle_id,
                step_name="research",
                step_index=1,
                gaps_found=result.gaps_found,
                research_topic=result.research_topic,
                research_text=research_text,
                steps_completed=list(result.steps_completed),
                delta={
                    "research_topic": result.research_topic,
                    "research_text": research_text[:500],
                },
            )
        )

        if not research_text:
            result.skipped = True
            result.reason = "research_empty"
            result.research_topic = ""  # Clear so it's NOT added to researched set
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            log.info("evolution_research_empty", query=getattr(top_gap, "query", "")[:60])
            return result
        if not self._idle.is_idle:
            result.reason = "interrupted_after_research"
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            return result

        # Resource check before build
        if not await self._check_resources():
            result.reason = "system_busy_before_build"
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            return result

        # Step 3: Build — create a skill from the research
        skill_name = await self._build(top_gap, research_text)
        result.steps_completed.append("build")
        if skill_name:
            result.skill_created = skill_name
            self._total_skills_created += 1

        self._save_checkpoint(
            EvolutionCheckpoint(
                cycle_id=result.cycle_id,
                step_name="build",
                step_index=2,
                gaps_found=result.gaps_found,
                research_topic=result.research_topic,
                research_text=research_text,
                skill_created=result.skill_created,
                steps_completed=list(result.steps_completed),
                delta={"skill_created": result.skill_created},
            )
        )

        # Step 4: Reflect — log what was learned
        result.steps_completed.append("reflect")
        result.duration_ms = int((time.monotonic() - t0) * 1000)

        self._save_checkpoint(
            EvolutionCheckpoint(
                cycle_id=result.cycle_id,
                step_name="reflect",
                step_index=3,
                gaps_found=result.gaps_found,
                research_topic=result.research_topic,
                skill_created=result.skill_created,
                steps_completed=list(result.steps_completed),
            )
        )

        log.info(
            "evolution_cycle_complete",
            cycle=result.cycle_id,
            gaps=result.gaps_found,
            topic=result.research_topic[:50],
            skill=result.skill_created or "none",
            duration_ms=result.duration_ms,
        )

        return result

    # -- Checkpointing ---------------------------------------------------

    def _save_checkpoint(self, cp: EvolutionCheckpoint) -> None:
        """Save evolution checkpoint to disk."""
        if not self._checkpoint_store:
            return
        from jarvis.core.checkpointing import PersistentCheckpoint

        pcp = PersistentCheckpoint(
            session_id=f"evolution-{cp.cycle_id}",
            agent_id="evolution-loop",
            state=cp.to_dict(),
        )
        self._checkpoint_store.save(pcp)
        self._current_checkpoint = cp
        log.debug("evolution_checkpoint_saved", cycle=cp.cycle_id, step=cp.step_name)

    @property
    def current_checkpoint(self) -> EvolutionCheckpoint | None:
        return self._current_checkpoint

    # -- Internal steps --------------------------------------------------

    async def _scout(self) -> list[Any]:
        """Find knowledge gaps — 3-tier priority.

        1. CuriosityEngine (detected gaps from conversations)
        2. User-defined learning goals
        3. Self-analysis (skill weaknesses, failure patterns, code quality)
        """
        import random

        # --- Tier 1: CuriosityEngine ---
        if self._curiosity:
            try:
                if self._memory and hasattr(self._memory, "semantic"):
                    entities = self._memory.semantic.list_entities(limit=50)
                    await self._curiosity.detect_gaps("", entities)
                tasks = self._curiosity.propose_exploration(max_tasks=3)
                if tasks:
                    log.info("evolution_scout_found_gaps", count=len(tasks), source="curiosity")
                    return tasks
            except Exception:
                log.debug("evolution_scout_curiosity_failed", exc_info=True)

        # --- Tier 1.5: DeepLearner active plans ---
        if self._deep_learner and self._deep_learner.has_active_plans():
            for plan in self._deep_learner.list_plans():
                if plan.status != "active":
                    continue
                sg = self._deep_learner.get_next_subgoal(plan.id)
                if sg:
                    log.info(
                        "evolution_scout_deep_plan",
                        plan=plan.goal[:40],
                        subgoal=sg.title[:40],
                    )
                    return [
                        _LearningGoal(
                            query=f"[deep:{plan.id}:{sg.id}] {sg.title}: {sg.description}",
                            source="deep_plan",
                            target_skill=sg.id,
                        )
                    ]
                # No pending SubGoals — check for stale passed SubGoals to re-test
                else:
                    try:
                        retested = await self._deep_learner.retest_stale_subgoals(plan.id)
                        if retested:
                            log.info(
                                "evolution_retested_stale", plan=plan.goal[:40], count=retested
                            )
                            # After retest, some may have gone back to researching
                            sg2 = self._deep_learner.get_next_subgoal(plan.id)
                            if sg2:
                                return [
                                    _LearningGoal(
                                        query=f"[deep:{plan.id}:{sg2.id}] {sg2.title}: {sg2.description}",
                                        source="deep_plan",
                                        target_skill=sg2.id,
                                    )
                                ]
                    except Exception:
                        log.debug("evolution_retest_failed", exc_info=True)

        # --- Tier 2: User learning goals ---
        goals: list[str] = []
        if self._config and hasattr(self._config, "learning_goals"):
            goals = self._config.learning_goals or []

        if goals:
            # Auto-promote complex goals to deep learning plans
            if (
                self._deep_learner
                and self._config
                and getattr(self._config, "deep_learning_enabled", True)
            ):
                for g in list(goals):
                    if self._deep_learner.is_complex_goal(g):
                        log.info("evolution_promoting_to_deep_plan", goal=g[:60])
                        try:
                            import asyncio

                            await self._deep_learner.create_plan(g)
                        except Exception:
                            log.debug("evolution_promote_failed", exc_info=True)

            # Remaining simple goals — only count SUCCESSFUL research as "done"
            researched = {
                r.research_topic for r in self._results[-20:] if r.research_topic and not r.skipped
            }
            available = [g for g in goals if g not in researched]
            if available:
                selected = available[:3]
                random.shuffle(selected)
                log.info("evolution_scout_using_goals", count=len(selected), goals=selected)
                return [_LearningGoal(query=g, source="user") for g in selected]

        # --- Tier 3: Self-analysis ---
        self_tasks = self._self_analyze()
        if self_tasks:
            log.info(
                "evolution_scout_self_analysis",
                count=len(self_tasks),
                tasks=[t.query[:60] for t in self_tasks[:3]],
            )
            return self_tasks

        log.info("evolution_scout_nothing_found", hint="Add learning_goals or use the system more")
        return []

    def _self_analyze(self) -> list[_LearningGoal]:
        """Analyze own codebase/skills for improvement opportunities.

        Checks:
        1. Skills with low success rate (needs_review)
        2. Failure clusters from session analysis
        3. Procedures that need review
        4. Skills with zero usage (untested)
        """
        tasks: list[_LearningGoal] = []

        # 1. Skills with low success rate
        if self._skill_registry:
            try:
                all_skills = []
                if hasattr(self._skill_registry, "list_all"):
                    all_skills = self._skill_registry.list_all()
                elif hasattr(self._skill_registry, "list_enabled"):
                    all_skills = self._skill_registry.list_enabled()

                for skill in all_skills:
                    # needs_review: high failure rate
                    sr = getattr(skill, "success_rate", 1.0)
                    uses = getattr(skill, "total_uses", 0)
                    failures = getattr(skill, "failure_count", 0)
                    name = getattr(skill, "name", getattr(skill, "slug", ""))

                    if uses >= 5 and sr < 0.5:
                        tasks.append(
                            _LearningGoal(
                                query=f"Optimize skill '{name}': {sr:.0%} success rate, "
                                f"{failures} failures — analyze failure patterns and improve",
                                source="self_analysis",
                                target_skill=getattr(skill, "slug", name),
                            )
                        )
                    elif uses == 0 and getattr(skill, "source", "") != "builtin":
                        tasks.append(
                            _LearningGoal(
                                query=f"Test skill '{name}': never used — validate with sample queries",
                                source="self_analysis",
                                target_skill=getattr(skill, "slug", name),
                            )
                        )
            except Exception:
                log.debug("evolution_self_analyze_skills_failed", exc_info=True)

        # 2. Failure clusters from session analysis
        if self._session_analyzer:
            try:
                clusters = []
                if hasattr(self._session_analyzer, "failure_clusters"):
                    clusters = self._session_analyzer.failure_clusters
                elif hasattr(self._session_analyzer, "get_failure_clusters"):
                    clusters = self._session_analyzer.get_failure_clusters()

                for cluster in clusters[:5]:
                    resolved = getattr(cluster, "is_resolved", False)
                    if resolved:
                        continue
                    category = getattr(cluster, "error_category", "unknown")
                    occurrences = getattr(cluster, "occurrences", 0)
                    pattern = getattr(cluster, "pattern_id", str(cluster))[:80]
                    if occurrences >= 2:
                        tasks.append(
                            _LearningGoal(
                                query=f"Fix recurring {category} error ({occurrences}x): {pattern}",
                                source="self_analysis",
                            )
                        )
            except Exception:
                log.debug("evolution_self_analyze_failures_failed", exc_info=True)

        # 3. Procedures that need review
        if self._memory and hasattr(self._memory, "procedures"):
            try:
                procs = self._memory.procedures
                if hasattr(procs, "list_all"):
                    for proc in procs.list_all():
                        if hasattr(proc, "needs_review") and proc.needs_review:
                            name = getattr(proc, "name", str(proc))
                            tasks.append(
                                _LearningGoal(
                                    query=f"Review procedure '{name}': low success rate, "
                                    f"find and fix failure patterns",
                                    source="self_analysis",
                                )
                            )
            except Exception:
                log.debug("evolution_self_analyze_procedures_failed", exc_info=True)

        # Deduplicate by recently researched topics
        researched = {r.research_topic for r in self._results[-20:] if r.research_topic}
        tasks = [t for t in tasks if t.query not in researched]

        # Prioritize: failures first, then untested skills
        tasks.sort(key=lambda t: (0 if "Fix" in t.query else 1 if "Optimize" in t.query else 2))
        return tasks[:3]

    async def _research(self, gap: Any) -> str:
        """Research a knowledge gap. Strategy depends on operation_mode.

        offline:  Memory search only (no LLM cost).
        hybrid:   Memory search + web search for broader context.
        online:   Memory search + web search + LLM-powered deep research.
        """
        # Deep plan goals are executed by DeepLearner directly
        if (
            hasattr(gap, "source")
            and getattr(gap, "source", "") == "deep_plan"
            and self._deep_learner
        ):
            import re as _re

            query_str = getattr(gap, "query", "")
            match = _re.match(r"\[deep:([^:]+):([^\]]+)\]", query_str)
            if match:
                plan_id, sg_id = match.group(1), match.group(2)
                log.info("evolution_deep_plan_executing", plan_id=plan_id[:8], subgoal_id=sg_id[:8])
                success = await self._deep_learner.run_subgoal(plan_id, sg_id)
                return f"DeepLearner executed subgoal: {'success' if success else 'interrupted'}"

        query = getattr(gap, "query", getattr(gap, "question", str(gap)))
        parts: list[str] = []

        # All modes: memory search
        if self._memory:
            try:
                results = self._memory.search_memory_sync(query=query, top_k=5)
                if results:
                    parts.extend(getattr(r, "text", str(r))[:200] for r in results[:3])
            except Exception:
                pass

        # hybrid + online: web search + fetch for broader context
        if self._operation_mode in ("hybrid", "online") and self._mcp_client:
            try:
                # Step 1: Search the web
                web_result = await self._mcp_client.call_tool(
                    "search_and_read",
                    {"query": query, "num_results": 3, "language": "de"},
                )
                web_text = ""
                if web_result and hasattr(web_result, "content") and not web_result.is_error:
                    web_text = web_result.content
                elif web_result and hasattr(web_result, "text"):
                    web_text = web_result.text

                if web_text:
                    parts.append(web_text[:3000])
                    log.info("evolution_web_research_found", query=query[:40], chars=len(web_text))

                    # Step 2: Persist to memory + vault
                    try:
                        await self._mcp_client.call_tool(
                            "save_to_memory",
                            {
                                "content": web_text[:5000],
                                "tier": "semantic",
                                "source_path": f"evolution/{query[:50].replace(' ', '-')}.md",
                            },
                        )
                        log.info("evolution_research_saved_to_memory", query=query[:40])
                    except Exception:
                        log.debug("evolution_memory_save_failed", exc_info=True)

                    try:
                        await self._mcp_client.call_tool(
                            "vault_save",
                            {
                                "title": f"Evolution Research: {query[:80]}",
                                "content": web_text[:10000],
                                "tags": "evolution, auto-research",
                                "folder": "wissen",
                                "sources": "",
                            },
                        )
                        log.info("evolution_research_saved_to_vault", query=query[:40])
                    except Exception:
                        log.debug("evolution_vault_save_failed", exc_info=True)
                else:
                    log.info("evolution_web_search_empty", query=query[:40])
            except Exception:
                log.debug("evolution_web_search_failed", exc_info=True)

        # online only: LLM-powered synthesis of research
        if self._operation_mode == "online" and self._llm_fn and parts:
            try:
                prompt = (
                    f"Synthesize the following research about '{query}' "
                    f"into a concise summary:\n\n" + "\n---\n".join(parts)
                )
                synthesis = await self._llm_fn(prompt)
                if synthesis:
                    return synthesis[:1000]
            except Exception:
                log.debug("evolution_llm_synthesis_failed", exc_info=True)

        return "\n".join(parts) if parts else ""

    async def _build(self, gap: Any, research: str) -> str:
        """Build a skill from research results. Returns skill name or empty.

        offline:  Generates stub skill (no LLM).
        hybrid/online: Uses LLM for skill generation when available.
        """
        if not self._skill_gen:
            return ""
        try:
            from jarvis.skills.generator import SkillGap, SkillGapType

            skill_gap = SkillGap(
                gap_type=SkillGapType.NO_SKILL_MATCH,
                description=getattr(gap, "query", str(gap))[:200],
                context=research[:500],
            )
            # hybrid/online: pass LLM function for real skill generation
            if self._operation_mode in ("hybrid", "online") and self._llm_fn:
                if hasattr(self._skill_gen, "llm_fn"):
                    self._skill_gen.llm_fn = self._llm_fn
            result = await self._skill_gen.process_gap(skill_gap)
            if result and hasattr(result, "name"):
                return result.name
        except Exception:
            log.debug("evolution_build_failed", exc_info=True)
        return ""

    # -- Resource & budget checks ----------------------------------------

    async def _check_resources(self) -> bool:
        """Check if system resources allow background work.

        Returns True if resources are available (ok to proceed).
        """
        if not self._resource_monitor:
            return True
        try:
            snap = await self._resource_monitor.sample()
            if snap.is_busy:
                self._paused_for_resources = True
                log.debug(
                    "evolution_paused_resources",
                    cpu=snap.cpu_percent,
                    ram=snap.ram_percent,
                    gpu=snap.gpu_util_percent,
                )
                return False
            self._paused_for_resources = False
            return True
        except Exception:
            return True  # On error, allow work

    def _check_evolution_budget(self) -> bool:
        """Check if per-agent evolution budget is still available.

        Returns True if budget allows more cycles.
        """
        if not self._cost_tracker or not self._config:
            return True
        agent_budgets = getattr(self._config, "agent_budgets", {})
        if not agent_budgets:
            return True
        # Check the scout agent budget (primary evolution consumer)
        for agent_name, limit in agent_budgets.items():
            if limit <= 0:
                continue
            status = self._cost_tracker.check_agent_budget(agent_name, limit)
            if not status.ok:
                log.info("evolution_budget_exhausted", agent=agent_name, warning=status.warning)
                return False
        return True

    # -- Loop control ----------------------------------------------------

    def _can_run_cycle(self) -> bool:
        """Check daily limit and cooldown."""
        today = time.strftime("%Y-%m-%d")
        if today != self._last_cycle_day:
            self._cycles_today = 0
            self._last_cycle_day = today
        max_cycles = 10
        if self._config and hasattr(self._config, "max_cycles_per_day"):
            max_cycles = self._config.max_cycles_per_day
        return self._cycles_today < max_cycles

    async def _loop(self) -> None:
        """Background loop: wait for idle -> run cycle -> cooldown."""
        cooldown = 60
        if self._config and hasattr(self._config, "cycle_cooldown_seconds"):
            cooldown = self._config.cycle_cooldown_seconds
        while self._running:
            try:
                if self._idle.is_idle and self._can_run_cycle():
                    log.info("evolution_cycle_starting", cycle=self._total_cycles + 1)
                    result = await self.run_cycle()
                    self._cycles_today += 1
                    self._results.append(result)
                    if result.skipped:
                        log.info(
                            "evolution_cycle_skipped",
                            cycle=result.cycle_id,
                            reason=result.reason,
                        )
                    if len(self._results) > 100:
                        self._results = self._results[-50:]
                    # Longer pause if skipped due to resources
                    if result.reason in (
                        "system_busy",
                        "system_busy_before_research",
                        "system_busy_before_build",
                    ):
                        await asyncio.sleep(60)  # Wait for resources to free up
                    elif result.reason == "budget_exhausted":
                        await asyncio.sleep(cooldown * 2)  # Long pause on budget
                    else:
                        await asyncio.sleep(cooldown)
                else:
                    await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("evolution_loop_error", exc_info=True)
                await asyncio.sleep(60)

    def stats(self) -> dict[str, Any]:
        """Return evolution statistics."""
        resource_info: dict[str, Any] = {"available": True}
        if self._resource_monitor and self._resource_monitor.last_snapshot:
            snap = self._resource_monitor.last_snapshot
            resource_info = {
                "available": not snap.is_busy,
                "cpu_percent": round(snap.cpu_percent, 1),
                "ram_percent": round(snap.ram_percent, 1),
                "gpu_util_percent": round(snap.gpu_util_percent, 1),
                "paused": self._paused_for_resources,
            }
        return {
            "running": self._running,
            "operation_mode": self._operation_mode,
            "is_idle": self._idle.is_idle,
            "idle_seconds": round(self._idle.idle_seconds),
            "total_cycles": self._total_cycles,
            "cycles_today": self._cycles_today,
            "total_skills_created": self._total_skills_created,
            "checkpoint": self._current_checkpoint.to_dict() if self._current_checkpoint else None,
            "resources": resource_info,
            "recent_results": [
                {
                    "cycle": r.cycle_id,
                    "skipped": r.skipped,
                    "reason": r.reason,
                    "topic": r.research_topic[:50],
                    "source": r.source,
                    "skill": r.skill_created,
                    "steps": r.steps_completed,
                    "duration_ms": r.duration_ms,
                }
                for r in self._results[-10:]
            ],
        }
