"""DeepLearner — orchestrates learning plans via StrategyPlanner and plan CRUD."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, List, Optional

from jarvis.evolution.knowledge_builder import KnowledgeBuilder
from jarvis.evolution.models import LearningPlan, SeedSource, SourceSpec, SubGoal
from jarvis.evolution.research_agent import ResearchAgent
from jarvis.evolution.strategy_planner import StrategyPlanner
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class DeepLearner:
    """High-level orchestrator for autonomous deep-learning plans.

    Delegates plan creation to StrategyPlanner and provides CRUD
    operations on persisted LearningPlan instances.
    """

    def __init__(
        self,
        llm_fn: Callable,
        plans_dir: str | None = None,
        mcp_client=None,
        memory_manager=None,
        skill_registry=None,
        skill_generator=None,
        cron_engine=None,
        cost_tracker=None,
        resource_monitor=None,
        checkpoint_store=None,
        config=None,
        idle_detector=None,
        operation_mode: str = "offline",
    ) -> None:
        if plans_dir is None:
            self._plans_dir = Path.home() / ".jarvis" / "evolution" / "plans"
        else:
            self._plans_dir = Path(plans_dir)
        self._plans_dir.mkdir(parents=True, exist_ok=True)

        self._strategy_planner = StrategyPlanner(llm_fn=llm_fn)
        self._research_agent = ResearchAgent(
            mcp_client=mcp_client,
            idle_detector=idle_detector,
        ) if mcp_client else None

        self._llm_fn = llm_fn
        self._mcp_client = mcp_client
        self._memory_manager = memory_manager
        self._skill_registry = skill_registry
        self._skill_generator = skill_generator
        self._cron_engine = cron_engine
        self._cost_tracker = cost_tracker
        self._resource_monitor = resource_monitor
        self._checkpoint_store = checkpoint_store
        self._config = config
        self._idle_detector = idle_detector
        self._operation_mode = operation_mode

    # ------------------------------------------------------------------
    # Plan CRUD
    # ------------------------------------------------------------------

    async def create_plan(
        self,
        goal: str,
        seed_sources: list[SeedSource] | None = None,
    ) -> LearningPlan:
        """Create a new learning plan via StrategyPlanner, persist to disk."""
        plan = await self._strategy_planner.create_plan(
            goal, seed_sources=seed_sources
        )
        plan.status = "active"
        plan.save(str(self._plans_dir))
        log.info("Created plan %s for goal: %s", plan.id, goal)
        return plan

    def list_plans(self) -> List[LearningPlan]:
        """Return all persisted learning plans."""
        return LearningPlan.list_plans(str(self._plans_dir))

    def get_plan(self, plan_id: str) -> LearningPlan | None:
        """Load a single plan by ID, or None if not found."""
        plan_dir = self._plans_dir / plan_id
        if not (plan_dir / "plan.json").exists():
            return None
        try:
            return LearningPlan.load(str(plan_dir))
        except Exception:
            log.warning("Failed to load plan %s", plan_id)
            return None

    def update_plan_status(self, plan_id: str, status: str) -> bool:
        """Update a plan's status and re-persist."""
        plan = self.get_plan(plan_id)
        if plan is None:
            return False
        plan.status = status
        plan.save(str(self._plans_dir))
        log.info("Plan %s status -> %s", plan_id, status)
        return True

    def delete_plan(self, plan_id: str) -> bool:
        """Remove plan directory entirely."""
        plan_dir = self._plans_dir / plan_id
        if not plan_dir.exists():
            return False
        shutil.rmtree(plan_dir)
        log.info("Deleted plan %s", plan_id)
        return True

    def get_next_subgoal(self, plan_id: str) -> SubGoal | None:
        """Return highest-priority pending SubGoal, or None if all done."""
        plan = self.get_plan(plan_id)
        if plan is None:
            return None
        pending = [sg for sg in plan.sub_goals if sg.status == "pending"]
        if not pending:
            return None
        # Sub-goals are already sorted by priority from StrategyPlanner;
        # return the first pending one (lowest priority number = highest priority).
        pending.sort(key=lambda sg: sg.priority)
        return pending[0]

    def has_active_plans(self) -> bool:
        """Return True if any plan is active with pending sub_goals."""
        for plan in self.list_plans():
            if plan.status == "active":
                if any(sg.status == "pending" for sg in plan.sub_goals):
                    return True
        return False

    def is_complex_goal(self, goal: str) -> bool:
        """Delegate complexity check to StrategyPlanner."""
        return self._strategy_planner.is_complex_goal(goal)

    # ------------------------------------------------------------------
    # Research -> Build cycle
    # ------------------------------------------------------------------

    async def run_subgoal(self, plan_id: str, subgoal_id: str) -> bool:
        """Execute Research->Build for a single SubGoal.

        Returns True if completed, False if interrupted or failed.
        """
        plan = self.get_plan(plan_id)
        if not plan:
            log.warning("deep_learner_plan_not_found", plan_id=plan_id[:8])
            return False
        subgoal = next((sg for sg in plan.sub_goals if sg.id == subgoal_id), None)
        if not subgoal:
            log.warning("deep_learner_subgoal_not_found", subgoal_id=subgoal_id[:8])
            return False
        if not self._research_agent:
            log.warning("deep_learner_no_research_agent")
            return False

        subgoal.status = "researching"
        plan.save(str(self._plans_dir))
        log.info("deep_learner_subgoal_start", plan=plan.goal[:40], subgoal=subgoal.title[:40])

        # Find sources for this subgoal
        # Use plan sources that are still pending, or discover new ones
        sources = [s for s in plan.sources if s.status == "pending"]
        if not sources:
            sources = await self._discover_sources(subgoal.title)

        builder = KnowledgeBuilder(
            mcp_client=self._mcp_client,
            llm_fn=self._llm_fn,
            goal_slug=plan.goal_slug,
        )

        for source in sources:
            # Idle check
            if self._idle_detector and not self._idle_detector.is_idle:
                log.info("deep_learner_interrupted", subgoal=subgoal.title[:40])
                plan.save(str(self._plans_dir))
                return False

            log.info("deep_learner_fetching", source=source.url[:60])
            fetch_results = await self._research_agent.fetch_source(source)
            source.status = "done" if fetch_results else "error"
            source.pages_fetched = len(fetch_results)

            # Build phase
            subgoal.status = "building"
            for fr in fetch_results:
                if self._idle_detector and not self._idle_detector.is_idle:
                    plan.save(str(self._plans_dir))
                    return False
                build_result = await builder.build(fr)
                subgoal.chunks_created += build_result.chunks_created
                subgoal.entities_created += build_result.entities_created
                if build_result.vault_path:
                    subgoal.vault_entries += 1
                subgoal.sources_fetched += 1

        # Mark as ready for quality testing (Phase 5C)
        subgoal.status = "testing"
        plan.total_chunks_indexed += subgoal.chunks_created
        plan.total_entities_created += subgoal.entities_created
        plan.total_vault_entries += subgoal.vault_entries
        plan.save(str(self._plans_dir))

        log.info(
            "deep_learner_subgoal_complete",
            subgoal=subgoal.title[:40],
            chunks=subgoal.chunks_created,
            entities=subgoal.entities_created,
            vault_entries=subgoal.vault_entries,
        )
        return True

    async def _discover_sources(self, topic: str) -> list[SourceSpec]:
        """Use web_search to find sources for a topic when none are specified."""
        if not self._mcp_client:
            return []
        try:
            result = await self._mcp_client.call_tool(
                "web_search",
                {"query": topic, "num_results": 5, "language": "de"},
            )
            if result.is_error:
                return []
            import re
            urls = re.findall(r'https?://[^\s<>"\')\]]+', result.content)
            # Deduplicate
            seen: set[str] = set()
            unique_urls: list[str] = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            return [
                SourceSpec(
                    url=url,
                    source_type="reference",
                    title=topic,
                    fetch_strategy="full_page",
                    update_frequency="once",
                )
                for url in unique_urls[:5]
            ]
        except Exception:
            log.debug("deep_learner_discover_sources_failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Future methods (not yet implemented)
    # ------------------------------------------------------------------
    # async def process_scheduled_update(self, plan_id, schedule_name) -> None: ...
    # async def run_quality_test(self, plan_id) -> dict: ...
    # async def run_horizon_scan(self, plan_id) -> dict: ...
