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
        self._research_agent = (
            ResearchAgent(
                mcp_client=mcp_client,
                idle_detector=idle_detector,
            )
            if mcp_client
            else None
        )

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
                    db_path=db_path,
                    llm_fn=llm_fn,
                    mcp_client=mcp_client,
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
        plan = await self._strategy_planner.create_plan(goal, seed_sources=seed_sources)
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
        3. 'failed' (retry — only if last tested > 30 min ago)
        """
        import time as _time
        from datetime import datetime, timedelta, timezone

        plan = self.get_plan(plan_id)
        if plan is None:
            return None

        # Priority 1-2: pending, researching, building (immediate)
        for status in ("pending", "researching", "building"):
            candidates = [sg for sg in plan.sub_goals if sg.status == status]
            if candidates:
                return candidates[0]

        # Priority 3: failed — but only after a cooldown period
        # This prevents the infinite re-test loop
        cooldown = timedelta(minutes=30)
        cutoff = (datetime.now(timezone.utc) - cooldown).isoformat()
        for sg in plan.sub_goals:
            if sg.status == "failed":
                if not sg.last_tested or sg.last_tested < cutoff:
                    return sg
                else:
                    log.debug(
                        "deep_learner_skip_failed_cooldown",
                        subgoal=sg.title[:40],
                        last_tested=sg.last_tested,
                    )
        return None

    def has_active_plans(self) -> bool:
        """Return True if any plan has actionable sub_goals."""
        for plan in self.list_plans():
            if plan.status == "active":
                if self.get_next_subgoal(plan.id) is not None:
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

        # ── Iterative deep research loop ──────────────────────────────
        # Keep researching until 90% coverage. NO max rounds — the system
        # researches until the topic is actually covered, period.
        # Each round: new search query → new sources → fetch → build.
        required_coverage = 0.9
        max_pages = getattr(self._config, "max_pages_per_crawl", 50)
        fetched_urls: set[str] = set()
        research_round = 0

        while True:
            # Check if coverage is sufficient (90%)
            coverage = self._quality_assessor.check_coverage(subgoal)
            if coverage >= required_coverage:
                log.info(
                    "deep_learner_coverage_reached",
                    subgoal=subgoal.title[:40],
                    round=research_round,
                    coverage=coverage,
                    chunks=subgoal.chunks_created,
                )
                break

            # Idle check
            if self._idle_detector and not self._idle_detector.is_idle:
                log.info(
                    "deep_learner_interrupted", subgoal=subgoal.title[:40], round=research_round
                )
                plan.save(str(self._plans_dir))
                return False

            # Vary search queries per round to get diverse sources
            query_variants = [
                f"{subgoal.title} {subgoal.description}",
                f"{subgoal.title} Gesetz Paragraph Details",
                f"{subgoal.title} Beispiele Praxis Anwendung",
                f"{subgoal.title} aktuelle Rechtsprechung Urteile",
                f"{subgoal.title} Fachliteratur Kommentar Erlaeuterung",
                f"{subgoal.title} Definitionen Begriffe Grundlagen",
                f"{subgoal.title} Statistiken Zahlen Fakten",
                f"{subgoal.title} Kritik Probleme Reformbedarf",
                f"{subgoal.title} Vergleich international EU",
                f"{subgoal.title} FAQ haeufige Fragen Antworten",
            ]
            query = query_variants[research_round % len(query_variants)][:200]

            sources = await self._discover_sources(query)
            if not sources and research_round == 0:
                # Fallback: plan-level sources
                sources = [
                    s for s in plan.sources if s.url not in fetched_urls and s.status != "error"
                ][:5]

            if not sources:
                log.info("deep_learner_no_more_sources", round=research_round, coverage=coverage)
                # Wait and retry with next query variant — don't give up
                if research_round > 20:
                    log.warning(
                        "deep_learner_exhausted_search_variants", subgoal=subgoal.title[:40]
                    )
                    break
                research_round += 1
                continue

            log.info(
                "deep_learner_research_round",
                round=research_round + 1,
                query=query[:50],
                sources=len(sources),
                current_chunks=subgoal.chunks_created,
                current_coverage=coverage,
            )

            for source in sources:
                if self._idle_detector and not self._idle_detector.is_idle:
                    log.info(
                        "deep_learner_interrupted_source_loop",
                        subgoal=subgoal.title[:40],
                        round=research_round,
                    )
                    plan.save(str(self._plans_dir))
                    return False

                if source.url in fetched_urls:
                    continue
                fetched_urls.add(source.url)

                if len(fetched_urls) > max_pages:
                    log.info("deep_learner_max_pages_reached", pages=len(fetched_urls))
                    break

                log.info("deep_learner_fetching", source=source.url[:60])
                fetch_results = await self._research_agent.fetch_source(source)
                source.pages_fetched = len(fetch_results)

                # Build phase
                subgoal.status = "building"
                for fr in fetch_results:
                    if self._idle_detector and not self._idle_detector.is_idle:
                        log.info(
                            "deep_learner_interrupted_build_loop",
                            subgoal=subgoal.title[:40],
                            round=research_round,
                        )
                        plan.save(str(self._plans_dir))
                        return False
                    # Skip entity extraction if system is busy (GPU contention)
                    # Vault save + memory chunking still run (no LLM needed).
                    # Entity extraction can run later in a dedicated cycle.
                    _skip_ee = bool(
                        self._resource_monitor
                        and hasattr(self._resource_monitor, "last_snapshot")
                        and self._resource_monitor.last_snapshot
                        and self._resource_monitor.last_snapshot.is_busy
                    )
                    build_result = await builder.build(
                        fr,
                        skip_entity_extraction=_skip_ee,
                    )
                    subgoal.chunks_created += build_result.chunks_created
                    subgoal.entities_created += build_result.entities_created
                    if build_result.vault_path:
                        subgoal.vault_entries += 1
                    subgoal.sources_fetched += 1

            # Save progress after each round
            research_round += 1
            plan.save(str(self._plans_dir))
            log.info(
                "deep_learner_round_complete",
                round=research_round,
                chunks=subgoal.chunks_created,
                entities=subgoal.entities_created,
                vault=subgoal.vault_entries,
                sources=subgoal.sources_fetched,
                coverage=self._quality_assessor.check_coverage(subgoal),
            )

        # Drain entity extraction queue (deferred from busy periods)
        if builder.entity_queue_size > 0:
            _gpu_free = not (
                self._resource_monitor
                and self._resource_monitor.last_snapshot
                and self._resource_monitor.last_snapshot.is_busy
            )
            if _gpu_free:
                drained = await builder.drain_entity_queue(max_items=10)
                if drained:
                    subgoal.entities_created += drained
                    log.info(
                        "deep_learner_entity_queue_drained",
                        drained=drained,
                        remaining=builder.entity_queue_size,
                    )

        # Quality test — with timeout protection
        subgoal.status = "testing"
        import asyncio as _asyncio
        import time as _time

        log.info(
            "deep_learner_pre_quality",
            subgoal=subgoal.title[:40],
            chunks=subgoal.chunks_created,
            entities=subgoal.entities_created,
            vault=subgoal.vault_entries,
            sources=subgoal.sources_fetched,
        )
        try:
            quality = await _asyncio.wait_for(
                self._quality_assessor.run_quality_test(subgoal, plan.goal_slug),
                timeout=180,  # 3 minutes max for quality test
            )
        except (_asyncio.TimeoutError, Exception) as e:
            log.warning(
                "deep_learner_quality_test_timeout", subgoal=subgoal.title[:40], error=str(e)[:100]
            )
            quality = {
                "coverage_score": self._quality_assessor.check_coverage(subgoal),
                "quality_score": 0.0,
                "passed": False,
                "questions": [],
                "failed_questions": [],
            }
        subgoal.coverage_score = quality["coverage_score"]
        subgoal.quality_score = quality["quality_score"]
        subgoal.last_tested = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        subgoal.test_count += 1

        if quality["passed"]:
            subgoal.status = "passed"
            log.info(
                "deep_learner_subgoal_passed",
                subgoal=subgoal.title[:40],
                quality=quality["quality_score"],
                test_count=subgoal.test_count,
            )
            # Auto-generate a query skill for this knowledge area
            if subgoal.test_count == 1:  # Only on first pass, not on re-tests
                await self._generate_skill_for_subgoal(subgoal, plan)
        else:
            subgoal.status = "failed"
            # Bump coverage thresholds by reducing counts so next run does MORE research
            # This forces the system to fetch additional sources before retesting
            subgoal.vault_entries = max(0, subgoal.vault_entries - 4)
            subgoal.chunks_created = max(0, subgoal.chunks_created - 12)
            subgoal.entities_created = max(0, subgoal.entities_created - 4)
            subgoal.sources_fetched = max(0, subgoal.sources_fetched - 4)
            log.info(
                "deep_learner_subgoal_quality_failed",
                subgoal=subgoal.title[:40],
                test_count=subgoal.test_count,
                failed_count=len(quality.get("failed_questions", [])),
            )

        # CRITICAL: Save status NOW before any further processing
        # This prevents the "stuck in testing/researching" bug
        plan.save(str(self._plans_dir))

        # Challenge weak claims — cross-reference low-confidence facts
        if self._knowledge_validator:
            try:
                challenged = await self._knowledge_validator.challenge_weak_claims(
                    goal_slug=plan.goal_slug,
                    max_challenges=3,
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
        plan.total_vault_entries += (
            len(subgoal.vault_entries)
            if isinstance(subgoal.vault_entries, list)
            else subgoal.vault_entries
        )

        # Check if ALL SubGoals done → horizon scan + schedules
        all_done = all(sg.status in ("passed", "failed") for sg in plan.sub_goals)
        if all_done:
            if getattr(self._config, "auto_expand", True):
                expansions = await self._horizon_scanner.scan(plan)
                if expansions:
                    new_context = "\n".join(
                        f"- {e['title']}: {e.get('reason', '')}" for e in expansions
                    )
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
    async def retest_stale_subgoals(self, plan_id: str, max_age_days: int = 7) -> int:
        """Re-test passed SubGoals that haven't been tested in max_age_days.

        If a re-test fails, the SubGoal goes back to "researching" for
        more depth. Returns count of SubGoals re-tested.
        """
        import time as _time
        from datetime import datetime, timedelta, timezone

        plan = self.get_plan(plan_id)
        if not plan:
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        retested = 0

        for sg in plan.sub_goals:
            if sg.status != "passed":
                continue
            if not sg.last_tested or sg.last_tested < cutoff:
                log.info(
                    "deep_learner_retest",
                    subgoal=sg.title[:40],
                    last_tested=sg.last_tested,
                    test_count=sg.test_count,
                )
                result = await self._quality_assessor.run_quality_test(sg, plan.goal_slug)
                sg.last_tested = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
                sg.test_count += 1
                sg.quality_score = result["quality_score"]

                if not result["passed"]:
                    sg.status = "researching"  # Back to research!
                    log.warning(
                        "deep_learner_retest_failed",
                        subgoal=sg.title[:40],
                        quality=result["quality_score"],
                        failed_questions=[
                            q.question[:50] for q in result.get("failed_questions", [])
                        ],
                    )
                else:
                    log.info(
                        "deep_learner_retest_passed",
                        subgoal=sg.title[:40],
                        quality=result["quality_score"],
                    )
                retested += 1

        if retested:
            plan.save(str(self._plans_dir))
        return retested

    async def _generate_skill_for_subgoal(self, subgoal: SubGoal, plan: "LearningPlan") -> None:
        """Auto-generate a Markdown skill that makes this knowledge queryable.

        Creates a skill file that matches on the subgoal topic keywords
        and instructs the Planner to search vault + memory for answers.
        """
        try:
            if not self._mcp_client:
                return

            # Generate trigger keywords from the subgoal title
            keywords = [w for w in subgoal.title.split() if len(w) > 3][:5]
            slug = plan.goal_slug[:20] + "-" + subgoal.id[:8]
            skill_name = f"evolution-{slug}"

            skill_body = (
                f"---\n"
                f"name: {skill_name}\n"
                f"description: Automatisch generiertes Wissen zu '{subgoal.title}'\n"
                f"trigger_keywords: {keywords}\n"
                f"category: research\n"
                f"priority: 3\n"
                f"enabled: true\n"
                f"---\n\n"
                f"# {subgoal.title}\n\n"
                f"Du hast umfangreiches Wissen zu diesem Thema aufgebaut.\n"
                f"Durchsuche dein Vault und Memory nach relevanten Informationen:\n\n"
                f"1. Nutze `vault_search` mit Stichworten aus der Frage\n"
                f"2. Nutze `search_memory` fuer semantische Suche\n"
                f"3. Kombiniere die Ergebnisse zu einer fundierten Antwort\n"
                f"4. Zitiere Quellen wenn moeglich\n\n"
                f"Vault-Ordner: wissen/{plan.goal_slug}/\n"
                f"Domain: {plan.goal_slug}\n"
            )

            # Save skill via MCP
            result = await self._mcp_client.call_tool(
                "create_skill",
                {"name": skill_name, "content": skill_body},
            )
            if result and not result.is_error:
                subgoal.skills_generated += 1
                log.info(
                    "deep_learner_skill_generated",
                    skill=skill_name,
                    subgoal=subgoal.title[:40],
                    keywords=keywords,
                )
            else:
                log.debug("deep_learner_skill_generation_failed", result=str(result)[:100])
        except Exception:
            log.debug("deep_learner_skill_generation_error", exc_info=True)

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

            # Filter: skip bare homepages, non-relevant language domains, and noise
            _BLOCKED_DOMAINS = {
                "zhihu.com",
                "baidu.com",
                "weibo.com",
                "qq.com",  # Chinese
                "naver.com",
                "daum.net",  # Korean
                "yandex.ru",
                "vk.com",
                "mail.ru",  # Russian
                "rakuten.co.jp",
                "yahoo.co.jp",
                "ameblo.jp",  # Japanese
                "timeoutbahrain.com",
                "najiz.sa",  # Off-topic
                "web.whatsapp.com",
                "linkedin.com/in/",  # Not content
                "claude.ai",
                "chat.openai.com",  # AI chat UIs
            }
            filtered: list[str] = []
            seen: set[str] = set()
            for url in urls:
                url = url.rstrip("/.,;:")
                if url in seen:
                    continue
                seen.add(url)
                parsed = urlparse(url)
                host = parsed.netloc.lower()
                path = parsed.path.strip("/")
                # Skip blocked domains (wrong language, off-topic, not content)
                if any(blocked in host for blocked in _BLOCKED_DOMAINS):
                    continue
                # Skip non-Latin TLDs that indicate wrong-language content
                if any(host.endswith(tld) for tld in (".cn", ".jp", ".kr", ".ru", ".sa")):
                    continue
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
