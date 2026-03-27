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
class EvolutionCycleResult:
    """Result of one evolution cycle."""

    cycle_id: int = 0
    skipped: bool = False
    reason: str = ""
    gaps_found: int = 0
    research_topic: str = ""
    skill_created: str = ""
    duration_ms: int = 0
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
    ) -> None:
        self._idle = idle_detector
        self._curiosity = curiosity_engine
        self._skill_gen = skill_generator
        self._memory = memory_manager
        self._config = config
        self._resource_monitor = resource_monitor
        self._cost_tracker = cost_tracker
        self._checkpoint_store = checkpoint_store
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

        self._save_checkpoint(EvolutionCheckpoint(
            cycle_id=result.cycle_id,
            step_name="scout",
            step_index=0,
            gaps_found=result.gaps_found,
            steps_completed=list(result.steps_completed),
            delta={"gaps_found": result.gaps_found},
        ))

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
        research_text = await self._research(top_gap)
        result.steps_completed.append("research")

        self._save_checkpoint(EvolutionCheckpoint(
            cycle_id=result.cycle_id,
            step_name="research",
            step_index=1,
            gaps_found=result.gaps_found,
            research_topic=result.research_topic,
            research_text=research_text,
            steps_completed=list(result.steps_completed),
            delta={"research_topic": result.research_topic, "research_text": research_text[:500]},
        ))

        if not research_text or not self._idle.is_idle:
            result.reason = "research_empty_or_interrupted"
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

        self._save_checkpoint(EvolutionCheckpoint(
            cycle_id=result.cycle_id,
            step_name="build",
            step_index=2,
            gaps_found=result.gaps_found,
            research_topic=result.research_topic,
            research_text=research_text,
            skill_created=result.skill_created,
            steps_completed=list(result.steps_completed),
            delta={"skill_created": result.skill_created},
        ))

        # Step 4: Reflect — log what was learned
        result.steps_completed.append("reflect")
        result.duration_ms = int((time.monotonic() - t0) * 1000)

        self._save_checkpoint(EvolutionCheckpoint(
            cycle_id=result.cycle_id,
            step_name="reflect",
            step_index=3,
            gaps_found=result.gaps_found,
            research_topic=result.research_topic,
            skill_created=result.skill_created,
            steps_completed=list(result.steps_completed),
        ))

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
        """Find knowledge gaps via CuriosityEngine."""
        if not self._curiosity:
            return []
        try:
            if self._memory and hasattr(self._memory, "semantic"):
                entities = self._memory.semantic.list_entities(limit=50)
                await self._curiosity.detect_gaps("", entities)
            tasks = self._curiosity.propose_exploration(max_tasks=3)
            return tasks
        except Exception:
            log.debug("evolution_scout_failed", exc_info=True)
            return []

    async def _research(self, gap: Any) -> str:
        """Research a knowledge gap. Returns research text."""
        query = getattr(gap, "query", getattr(gap, "question", str(gap)))
        # Use memory search as lightweight research
        if self._memory:
            try:
                results = self._memory.search_memory_sync(query=query, top_k=5)
                if results:
                    return "\n".join(getattr(r, "text", str(r))[:200] for r in results[:3])
            except Exception:
                pass
        return ""

    async def _build(self, gap: Any, research: str) -> str:
        """Build a skill from research results. Returns skill name or empty."""
        if not self._skill_gen:
            return ""
        try:
            from jarvis.skills.generator import SkillGap, SkillGapType

            skill_gap = SkillGap(
                gap_type=SkillGapType.NO_SKILL_MATCH,
                description=getattr(gap, "query", str(gap))[:200],
                context=research[:500],
            )
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
        cooldown = 300
        if self._config and hasattr(self._config, "cycle_cooldown_seconds"):
            cooldown = self._config.cycle_cooldown_seconds
        while self._running:
            try:
                if self._idle.is_idle and self._can_run_cycle():
                    result = await self.run_cycle()
                    self._cycles_today += 1
                    self._results.append(result)
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
                    "skill": r.skill_created,
                    "steps": r.steps_completed,
                    "duration_ms": r.duration_ms,
                }
                for r in self._results[-10:]
            ],
        }
