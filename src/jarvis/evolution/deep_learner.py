"""DeepLearner — orchestrates learning plans via StrategyPlanner and plan CRUD."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional

from jarvis.evolution.horizon_scanner import HorizonScanner
from jarvis.evolution.knowledge_builder import KnowledgeBuilder
from jarvis.evolution.models import LearningPlan, SeedSource, SourceSpec, SubGoal
from jarvis.evolution.quality_assessor import QualityAssessor
from jarvis.evolution.research_agent import ResearchAgent
from jarvis.evolution.schedule_manager import ScheduleManager
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

        self._quality_assessor = QualityAssessor(
            mcp_client=mcp_client,
            llm_fn=llm_fn,
            coverage_threshold=getattr(config, "coverage_threshold", 0.7),
            quality_threshold=getattr(config, "quality_threshold", 0.8),
        )
        self._horizon_scanner = HorizonScanner(
            llm_fn=llm_fn,
            memory_manager=memory_manager,
        )
        self._schedule_manager = ScheduleManager(cron_engine=cron_engine)

        # Knowledge validation — cross-reference claims, track confidence
        self._knowledge_validator = None
        try:
            from jarvis.evolution.knowledge_validator import KnowledgeValidator

            jarvis_home = getattr(config, "jarvis_home", None) if config else None
            if not jarvis_home and plans_dir:
                jarvis_home = Path(plans_dir).parent.parent
            if jarvis_home:
                db_path = Path(jarvis_home) / "index" / "knowledge_claims.db"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self._knowledge_validator = KnowledgeValidator(
                    db_path=db_path, llm_fn=llm_fn, mcp_client=mcp_client,
                )
                log.info("knowledge_validator_initialized", db=str(db_path))
        except Exception:
            log.debug("knowledge_validator_init_failed", exc_info=True)

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

        # Create cron schedules immediately (don't wait for all SubGoals)
        if plan.schedules:
            try:
                created = await self._schedule_manager.create_schedules(plan)
                log.info("deep_learner_schedules_created", plan=goal[:40], jobs=created)
            except Exception:
                log.debug("deep_learner_schedule_creation_failed", exc_info=True)

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
        """Return highest-priority actionable SubGoal, or None if all done.

        Picks SubGoals in this priority:
        1. 'pending' (never started)
        2. 'researching' / 'building' (interrupted, resume)
        3. 'failed' (retry)
        """
        plan = self.get_plan(plan_id)
        if plan is None:
            return None
        # Actionable statuses in priority order
        for status in ("pending", "researching", "building", "failed"):
            candidates = [sg for sg in plan.sub_goals if sg.status == status]
            if candidates:
                return candidates[0]
        return None

    def has_active_plans(self) -> bool:
        """Return True if any plan has actionable sub_goals."""
        actionable = {"pending", "researching", "building", "failed"}
        for plan in self.list_plans():
            if plan.status == "active":
                if any(sg.status in actionable for sg in plan.sub_goals):
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

        # Find sources SPECIFIC to this subgoal's topic
        # Each subgoal discovers its own sources via web search
        sources = await self._discover_sources(
            f"{subgoal.title} {subgoal.description}"[:200]
        )
        if not sources:
            # Fallback: use unfetched plan-level sources
            already_fetched = set()
            for sg in plan.sub_goals:
                if hasattr(sg, "_fetched_urls"):
                    already_fetched.update(sg._fetched_urls)
            sources = [
                s for s in plan.sources
                if s.url not in already_fetched and s.status != "error"
            ][:3]

        log.info(
            "deep_learner_sources_for_subgoal",
            subgoal=subgoal.title[:40],
            sources=[s.url[:50] for s in sources[:5]],
        )

        # Create goal-scoped index for isolated per-plan storage
        goal_index = None
        try:
            from jarvis.evolution.goal_index import GoalScopedIndex

            index_base = self._plans_dir.parent / "indexes"
            goal_index = GoalScopedIndex(goal_slug=plan.goal_slug, base_dir=index_base)
        except Exception:
            log.debug("goal_index_creation_failed", exc_info=True)

        builder = KnowledgeBuilder(
            mcp_client=self._mcp_client,
            llm_fn=self._llm_fn,
            goal_slug=plan.goal_slug,
            knowledge_validator=self._knowledge_validator,
            goal_index=goal_index,
        )

        fetched_urls: set[str] = set()
        for source in sources:
            # Idle check
            if self._idle_detector and not self._idle_detector.is_idle:
                log.info("deep_learner_interrupted", subgoal=subgoal.title[:40])
                plan.save(str(self._plans_dir))
                return False

            # Skip if already fetched by this or another subgoal
            if source.url in fetched_urls:
                continue
            fetched_urls.add(source.url)

            log.info("deep_learner_fetching", source=source.url[:60])
            fetch_results = await self._research_agent.fetch_source(source)
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

        # Quality test
        subgoal.status = "testing"
        log.info(
            "deep_learner_pre_quality",
            subgoal=subgoal.title[:40],
            chunks=subgoal.chunks_created,
            entities=subgoal.entities_created,
            vault=subgoal.vault_entries,
            sources=subgoal.sources_fetched,
        )
        quality = await self._quality_assessor.run_quality_test(subgoal, plan.goal_slug)
        subgoal.coverage_score = quality["coverage_score"]
        subgoal.quality_score = quality["quality_score"]
        if quality["passed"]:
            subgoal.status = "passed"
            log.info("deep_learner_subgoal_passed", subgoal=subgoal.title[:40],
                     quality=quality["quality_score"])
        else:
            subgoal.status = "failed"
            log.info("deep_learner_subgoal_quality_failed", subgoal=subgoal.title[:40],
                     failed=quality.get("failed_questions", []))

        # Challenge weak claims — cross-reference low-confidence facts
        if self._knowledge_validator:
            try:
                challenged = await self._knowledge_validator.challenge_weak_claims(
                    goal_slug=plan.goal_slug, max_challenges=3,
                )
                if challenged:
                    summary = self._knowledge_validator.get_claims_summary(plan.goal_slug)
                    log.info(
                        "deep_learner_claims_validated",
                        total=summary["total_claims"],
                        verified=summary["verified"],
                        disputed=summary["disputed"],
                        avg_confidence=summary["avg_confidence"],
                    )
            except Exception:
                log.debug("deep_learner_claims_challenge_failed", exc_info=True)

        # Close goal-scoped index and log stats
        if goal_index:
            try:
                log.info("goal_index_stats", **goal_index.stats())
            except Exception:
                log.debug("goal_index_stats_failed", exc_info=True)
            goal_index.close()

        # Update plan totals
        plan.total_chunks_indexed += subgoal.chunks_created
        plan.total_entities_created += subgoal.entities_created
        plan.total_vault_entries += len(subgoal.vault_entries) if isinstance(subgoal.vault_entries, list) else subgoal.vault_entries

        # Check if ALL SubGoals done → horizon scan + schedules
        all_done = all(sg.status in ("passed", "failed") for sg in plan.sub_goals)
        if all_done:
            if getattr(self._config, "auto_expand", True):
                expansions = await self._horizon_scanner.scan(plan)
                if expansions:
                    new_context = "\n".join(f"- {e['title']}: {e.get('reason', '')}" for e in expansions)
                    plan = await self._strategy_planner.replan(plan, new_context)
                    plan.expansions.extend(e["title"] for e in expansions)
                    log.info("deep_learner_horizon_expanded", count=len(expansions))
            # Setup cron schedules
            if plan.schedules:
                await self._schedule_manager.create_schedules(plan)
            # Check if plan is complete (all passed, no new pending)
            still_pending = [sg for sg in plan.sub_goals if sg.status == "pending"]
            if not still_pending:
                plan.status = "completed"
                log.info("deep_learner_plan_completed", goal=plan.goal[:40])

        # Update plan-level scores
        scored = [sg for sg in plan.sub_goals if sg.coverage_score > 0]
        if scored:
            plan.coverage_score = sum(sg.coverage_score for sg in scored) / len(scored)
            plan.quality_score = sum(sg.quality_score for sg in scored) / len(scored)

        plan.save(str(self._plans_dir))
        return True

    async def run_quality_test(self, plan_id: str, subgoal_id: str) -> dict[str, Any]:
        """Run quality test on a SubGoal. Updates SubGoal status based on result."""
        plan = self.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        subgoal = next((sg for sg in plan.sub_goals if sg.id == subgoal_id), None)
        if not subgoal:
            return {"error": "SubGoal not found"}

        result = await self._quality_assessor.run_quality_test(subgoal, plan.goal_slug)

        subgoal.coverage_score = result["coverage_score"]
        subgoal.quality_score = result["quality_score"]
        if result["passed"]:
            subgoal.status = "passed"
        else:
            subgoal.status = "researching"
        # Update plan-level scores
        done = [sg for sg in plan.sub_goals if sg.coverage_score > 0]
        if done:
            plan.coverage_score = sum(sg.coverage_score for sg in done) / len(done)
            plan.quality_score = sum(sg.quality_score for sg in done) / len(done)
        plan.save(str(self._plans_dir))
        return result

    async def run_horizon_scan(self, plan_id: str) -> list[dict[str, str]]:
        """Discover new areas beyond the literal goal. Adds new SubGoals via replan."""
        plan = self.get_plan(plan_id)
        if not plan:
            return []
        expansions = await self._horizon_scanner.scan(plan)
        if expansions:
            new_context = "HorizonScanner hat folgende Luecken gefunden:\n"
            new_context += "\n".join(f"- {e['title']}: {e.get('reason', '')}" for e in expansions)
            plan = await self._strategy_planner.replan(plan, new_context)
            plan.expansions.extend(e["title"] for e in expansions)
            plan.save(str(self._plans_dir))
            log.info("deep_learner_horizon_expanded", new_subgoals=len(expansions))
        return expansions

    async def setup_schedules(self, plan_id: str) -> int:
        """Create cron jobs for a plan's recurring sources."""
        plan = self.get_plan(plan_id)
        if not plan:
            return 0
        return await self._schedule_manager.create_schedules(plan)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _discover_sources(self, topic: str) -> list[SourceSpec]:
        """Use web_search to find topic-specific sources.

        Filters out generic homepages (< 3 path segments) and
        deduplicates against already-known plan sources.
        """
        if not self._mcp_client:
            return []
        try:
            result = await self._mcp_client.call_tool(
                "web_search",
                {"query": topic[:150], "num_results": 5, "language": "de"},
            )
            if result.is_error:
                return []
            import re
            from urllib.parse import urlparse

            urls = re.findall(r'https?://[^\s<>"\')\]]+', result.content)

            # Filter: skip bare homepages (e.g. https://www.drv.de) — want deep pages
            filtered: list[str] = []
            seen: set[str] = set()
            for url in urls:
                url = url.rstrip("/.,;:")
                if url in seen:
                    continue
                seen.add(url)
                parsed = urlparse(url)
                path = parsed.path.strip("/")
                # Accept URLs with actual content paths, skip bare domains
                if path and len(path) > 3:
                    filtered.append(url)
                elif not filtered:
                    # Accept homepage only if we have nothing else
                    filtered.append(url)

            log.info(
                "deep_learner_discovered_sources",
                topic=topic[:50],
                urls=[u[:60] for u in filtered[:5]],
            )

            return [
                SourceSpec(
                    url=url,
                    source_type="reference",
                    title=topic[:80],
                    fetch_strategy="full_page",
                    update_frequency="once",
                )
                for url in filtered[:5]
            ]
        except Exception:
            log.debug("deep_learner_discover_sources_failed", exc_info=True)
            return []
