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

__all__ = ["EvolutionCycleResult", "EvolutionLoop", "_match_goal_for_action"]


def _match_goal_for_action(action: Any, goals: list) -> Any | None:
    """Find the goal most relevant to an ATL action.

    Uses explicit goal_id from params if available, otherwise
    word-overlap between action text and goal titles.
    Returns None if no confident match (< 2 word overlap).
    """
    # 1. Explicit goal_id in params
    goal_id = getattr(action, "params", {}).get("goal_id", "")
    if goal_id:
        for g in goals:
            if g.id == goal_id:
                return g

    # 2. Word-overlap heuristic
    action_text = (
        f"{getattr(action, 'rationale', '')} {getattr(action, 'params', {}).get('query', '')}"
    ).lower()
    action_words = set(action_text.split())

    best, best_score = None, 0
    for g in goals:
        goal_words = set(g.title.lower().split())
        overlap = len(goal_words & action_words)
        if overlap > best_score:
            best, best_score = g, overlap

    return best if best_score >= 2 else None


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
    source: str = ""  # "curiosity" | "user" | "self_analysis" | "atl"
    steps_completed: list[str] = field(default_factory=list)
    thought: str = ""


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
        self._atl_config: Any = None  # Set by gateway: ATLConfig
        self._goal_manager: Any = None  # Set by gateway: GoalManager
        self._atl_journal: Any = None  # Set by gateway: ATLJournal
        self._atl_cycle_count = 0
        self._atl_knowledge_builders: dict[str, Any] = {}  # goal_id -> KnowledgeBuilder
        self._atl_persisted_queries: set[str] = set()
        self._last_thinking_time = time.monotonic()

    async def _synthesize_for_goal(
        self,
        research_text: str,
        goal_title: str,
        query: str,
    ) -> str | None:
        """Synthesize research findings into a structured note.

        Cognithor processes information like an expert: extracting
        what matters, connecting it to the goal, and discarding noise.
        Returns None if nothing relevant was found or on error.
        """
        if not self._llm_fn:
            return None

        prompt = (
            "Du bist ein Wissensassistent der Recherche-Ergebnisse einordnet.\n\n"
            f"Ziel: {goal_title}\n"
            f"Suchanfrage: {query}\n\n"
            f"Recherche-Ergebnis:\n{research_text[:3000]}\n\n"
            "Aufgabe:\n"
            "1. Extrahiere die 3-5 wichtigsten Fakten die fuer das Ziel relevant sind\n"
            "2. Formuliere eine strukturierte Notiz (deutsch, sachlich)\n"
            "3. Wenn nichts Relevantes gefunden wurde, antworte NUR mit KEINE_RELEVANZ\n\n"
            "Format:\n"
            "## {Thema}\n"
            "- Kernaussage 1 (Quelle: ...)\n"
            "- Kernaussage 2\n"
            "...\n"
        )

        try:
            response = await self._llm_fn(prompt)
            if not response or "KEINE_RELEVANZ" in response:
                return None
            return response.strip()
        except Exception:
            log.debug("atl_synthesis_failed", exc_info=True)
            return None

    def _create_builder_for_goal(self, goal: Any) -> Any | None:
        """Lazily create a KnowledgeBuilder for an ATL goal."""
        if not self._deep_learner or not self._mcp_client:
            return None
        try:
            from jarvis.evolution.goal_index import GoalScopedIndex
            from jarvis.evolution.knowledge_builder import KnowledgeBuilder

            # Use same index base as deep_learner
            index_base = self._deep_learner._plans_dir.parent / "indexes"
            goal_slug = goal.id
            goal_index = GoalScopedIndex(goal_slug=goal_slug, base_dir=index_base)

            return KnowledgeBuilder(
                mcp_client=self._mcp_client,
                llm_fn=self._llm_fn,
                goal_slug=goal_slug,
                goal_index=goal_index,
            )
        except Exception:
            log.debug("atl_builder_creation_failed", exc_info=True)
            return None

    async def _persist_research_result(
        self,
        tool_result: Any,
        action: Any,
        goals: list,
    ) -> None:
        """Persist a search_and_read result: match goal, dedup, synthesize, build.

        This is the core of Cognithor's intelligent learning during idle time.
        Rather than dumping raw web scrapes, it synthesizes findings like an
        expert taking structured notes.
        """
        result_text = str(tool_result)
        if len(result_text) < 200:
            return

        # Goal matching
        goal = _match_goal_for_action(action, goals)
        if not goal:
            log.debug("atl_persist_no_goal_match", rationale=getattr(action, "rationale", "")[:60])
            return

        # Dedup: skip if this query was already persisted this session
        query_key = getattr(action, "params", {}).get("query", "")[:100].lower().strip()
        if query_key in self._atl_persisted_queries:
            log.debug("atl_persist_dedup_skip", query=query_key[:60])
            return

        # Synthesis: extract relevant findings
        synthesis = await self._synthesize_for_goal(
            research_text=result_text,
            goal_title=goal.title,
            query=query_key,
        )
        if not synthesis:
            log.debug("atl_persist_no_relevance", goal=goal.title[:40], query=query_key[:40])
            return

        # Get or create KnowledgeBuilder for this goal
        builder = self._atl_knowledge_builders.get(goal.id)
        if not builder:
            builder = self._create_builder_for_goal(goal)
            if not builder:
                return
            self._atl_knowledge_builders[goal.id] = builder

        # Build knowledge from synthesized text
        from jarvis.evolution.research_agent import FetchResult

        fetch = FetchResult(
            url=query_key or "atl-research",
            text=synthesis,
            title=f"ATL: {getattr(action, 'rationale', '')[:80]}",
            source_type="atl_research",
        )
        try:
            build_result = await builder.build(fetch, skip_entity_extraction=True)
            if build_result.errors:
                log.debug("atl_persist_build_errors", errors=build_result.errors[:2])
            else:
                self._atl_persisted_queries.add(query_key)
                log.info(
                    "atl_research_persisted",
                    goal=goal.title[:40],
                    chunks=build_result.chunks_created,
                    query=query_key[:40],
                )
        except Exception:
            log.debug("atl_persist_build_failed", exc_info=True)

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

    # -- ATL Thinking Cycle ----------------------------------------------

    async def thinking_cycle(self) -> EvolutionCycleResult:
        """Run one ATL thinking cycle: evaluate goals, propose actions, journal."""
        import time as _time

        self._atl_cycle_count += 1
        result = EvolutionCycleResult(cycle_id=self._atl_cycle_count)

        # Pre-checks
        if self._atl_config and not self._atl_config.enabled:
            result.skipped = True
            result.reason = "atl_disabled"
            return result

        if not self._idle.is_idle:
            result.skipped = True
            result.reason = "not_idle"
            return result

        if self._in_quiet_hours():
            result.skipped = True
            result.reason = "quiet_hours"
            return result

        if not self._llm_fn:
            result.skipped = True
            result.reason = "no_llm"
            return result

        if not self._goal_manager:
            result.skipped = True
            result.reason = "no_goals"
            return result

        # Proactive: auto-create goals from curiosity gaps
        self._create_goals_from_curiosity()

        # Build context
        goals = self._goal_manager.active_goals()
        goals_fmt = (
            "\n".join(f"- {g.id}: {g.title} ({g.progress:.0%}) [P{g.priority}]" for g in goals)
            or "Keine aktiven Ziele."
        )

        recent = ""
        if self._atl_journal:
            entries = self._atl_journal.recent(days=2)
            recent = "\n".join(entries[:1])[:1000] if entries else ""

        # Build goal knowledge from DeepLearner progress + Memory search
        goal_knowledge_parts: list[str] = []
        if self._deep_learner:
            for plan in self._deep_learner.list_plans():
                coverage = getattr(plan, "coverage", 0.0)
                chunks = getattr(plan, "total_chunks", 0)
                if coverage > 0 or chunks > 0:
                    goal_knowledge_parts.append(
                        f"- {plan.goal[:60]}: Coverage {coverage:.0%}, "
                        f"{chunks} Chunks, Status: {plan.status}"
                    )
        if self._memory and hasattr(self._memory, "search_memory_sync"):
            for g in goals[:3]:
                try:
                    results = self._memory.search_memory_sync(
                        query=g.title,
                        top_k=2,
                    )
                    if results:
                        for r in results[:1]:
                            text = getattr(r, "text", str(r))[:150]
                            goal_knowledge_parts.append(f"- Wissen zu '{g.title[:30]}': {text}")
                except Exception:
                    pass
        goal_knowledge = "\n".join(goal_knowledge_parts)[:2000]

        from jarvis.evolution.atl_prompt import build_atl_prompt, parse_atl_response

        max_actions = 3
        if self._atl_config:
            max_actions = self._atl_config.max_actions_per_cycle

        prompt = build_atl_prompt(
            identity="Cognithor Agent OS",
            goals_formatted=goals_fmt,
            recent_events=recent or "Keine aktuellen Ereignisse.",
            goal_knowledge=goal_knowledge or "Noch kein Wissen aufgebaut.",
            now=_time.strftime("%Y-%m-%d %H:%M"),
            max_actions=max_actions,
        )

        # Call LLM
        try:
            raw = await self._llm_fn(prompt)
        except Exception:
            log.debug("atl_llm_call_failed", exc_info=True)
            raw = ""

        thought = parse_atl_response(raw or "")
        result.thought = thought.summary
        result.research_topic = thought.summary[:100]

        # Metric-based progress: compute from DeepLearner data, override LLM guesses
        self._update_goal_progress_from_metrics(goals)

        # Apply LLM goal evaluations (only if metrics didn't already set progress)
        for ev in thought.goal_evaluations:
            gid = ev.get("goal_id", "")
            delta = ev.get("progress_delta", 0.0)
            note = ev.get("note", "")
            if gid and delta != 0:
                try:
                    self._goal_manager.update_progress(gid, delta, note)
                except (KeyError, Exception):
                    log.debug("atl_goal_update_failed", goal_id=gid)

        # Dispatch proposed actions through ActionQueue + MCP
        executed_actions: list[str] = []
        if thought.proposed_actions and self._mcp_client:
            from jarvis.evolution.action_queue import ActionQueue, ATLAction

            blocked = set()
            if self._atl_config:
                blocked = set(self._atl_config.blocked_action_types)
            queue = ActionQueue(
                max_actions=self._atl_config.max_actions_per_cycle if self._atl_config else 3,
                blocked_types=blocked,
            )
            for a in thought.proposed_actions:
                queue.enqueue(
                    ATLAction(
                        type=a.get("type", "unknown"),
                        params=a.get("params", {}),
                        priority=a.get("priority", 3),
                        rationale=a.get("rationale", ""),
                    )
                )

            _action_map = {
                "research": "search_and_read",
                "memory_update": "save_to_memory",
                "notification": "send_notification",
            }
            while not queue.empty():
                action = queue.dequeue()
                if not action:
                    break
                tool_name = _action_map.get(action.type, action.type)
                # Normalize params for known tools
                params = dict(action.params)
                if tool_name == "search_and_read" and "query" not in params:
                    # LLM may put the search topic in various keys
                    query = (
                        params.pop("topic", "")
                        or params.pop("search", "")
                        or params.pop("q", "")
                        or action.rationale[:100]
                    )
                    params = {"query": query, "num_results": 3}
                elif tool_name == "save_to_memory" and "content" not in params:
                    params.setdefault("content", action.rationale[:500])
                    params.setdefault("tier", "semantic")
                try:
                    tool_result = await self._mcp_client.call_tool(tool_name, params)
                    executed_actions.append(f"[OK] {action.type}: {action.rationale[:60]}")
                    log.info("atl_action_executed", type=action.type, tool=tool_name)

                    # Auto-persist: research results -> synthesis -> KnowledgeBuilder
                    if tool_name == "search_and_read" and tool_result:
                        try:
                            await self._persist_research_result(tool_result, action, goals)
                        except Exception:
                            log.debug("atl_auto_persist_failed", exc_info=True)
                except Exception as exc:
                    executed_actions.append(f"[FAIL] {action.type}: {exc!s:.60}")
                    log.debug("atl_action_failed", type=action.type, error=str(exc)[:80])

        # Verify action outcomes: did research produce real knowledge?
        if executed_actions and self._memory and hasattr(self._memory, "search_memory_sync"):
            verified: list[str] = []
            for desc in executed_actions:
                if "[OK]" in desc and "research" in desc:
                    try:
                        query = desc.split(":", 1)[-1].strip()[:50]
                        hits = self._memory.search_memory_sync(query=query, top_k=1)
                        tag = "[VERIFIED]" if hits else "[PENDING]"
                        verified.append(f"{desc} {tag}")
                        continue
                    except Exception:
                        pass
                verified.append(desc)
            executed_actions = verified

        # Journal
        if self._atl_journal:
            try:
                actions_desc = executed_actions or [
                    f"{a.get('type', '?')}: {a.get('rationale', '')}"
                    for a in thought.proposed_actions
                ]
                await self._atl_journal.log_cycle(
                    cycle=self._atl_cycle_count,
                    summary=thought.summary,
                    goal_updates=thought.goal_evaluations,
                    actions=actions_desc,
                )
            except Exception:
                log.debug("atl_journal_failed", exc_info=True)

        # Only reset timer on meaningful cycle (not failed LLM)
        if thought.summary:
            self._last_thinking_time = _time.monotonic()

        # Set result fields
        result.source = "atl"
        result.steps_completed = ["think", "evaluate", "dispatch", "journal"]

        log.info(
            "atl_thinking_cycle_complete",
            cycle=self._atl_cycle_count,
            summary=thought.summary[:50],
            goal_evals=len(thought.goal_evaluations),
            actions_proposed=len(thought.proposed_actions),
            actions_executed=len(executed_actions),
        )

        return result

    def _create_goals_from_curiosity(self) -> int:
        """Auto-create ATL goals from CuriosityEngine knowledge gaps.

        Only creates goals for gaps with importance >= 0.6 that don't
        already have a matching goal (by entity name in title).
        Source is set to "curiosity" with priority 4 (below user goals at 3).
        """
        if not self._curiosity or not self._goal_manager:
            return 0

        try:
            gaps = self._curiosity.propose_exploration(max_tasks=5)
        except Exception:
            return 0

        if not gaps:
            return 0

        existing_titles = {g.title.lower() for g in self._goal_manager.active_goals()}
        created = 0

        for gap in gaps:
            entity = getattr(gap, "entity_name", str(gap))
            importance = getattr(gap, "importance", 0.5)
            if importance < 0.6:
                continue
            if any(entity.lower() in t for t in existing_titles):
                continue

            from jarvis.evolution.goal_manager import Goal

            goal = Goal(
                title=f"Lerne {entity}",
                description=getattr(gap, "description", f"Knowledge gap: {entity}"),
                priority=4,
                source="curiosity",
            )
            try:
                self._goal_manager.add_goal(goal)
                existing_titles.add(goal.title.lower())
                created += 1
                log.info("atl_goal_auto_created", entity=entity, importance=importance)
            except Exception:
                pass

        return created

    def _in_quiet_hours(self) -> bool:
        """Check if current time is within ATL quiet hours.

        Returns False (quiet hours disabled) when start or end is empty.
        """
        if not self._atl_config:
            return False
        start = self._atl_config.quiet_hours_start
        end = self._atl_config.quiet_hours_end
        if not start or not end:
            return False  # Disabled until user configures via Flutter UI
        from datetime import datetime

        now = datetime.now().strftime("%H:%M")
        if start <= end:
            return start <= now <= end
        # Wrap around midnight (e.g., 23:00 to 07:00)
        return now >= start or now <= end

    def _update_goal_progress_from_metrics(self, goals: list) -> None:
        """Compute goal progress from real data instead of LLM guesses.

        Reads directly from GoalScopedIndex DBs (chunks, entities) and
        SubGoal activity status. Plan-level coverage_score is often None
        because SubGoals fail quality tests — so we compute from raw data.

        Progress = 40% subgoal activity + 30% chunks + 20% entities + 10% sources
        """
        if not self._deep_learner or not self._goal_manager:
            return

        plans = self._deep_learner.list_plans()
        if not plans:
            return

        from pathlib import Path

        index_base = Path.home() / ".jarvis" / "evolution" / "indexes"

        # Stop words to exclude from matching (too generic)
        _stop = {
            "werde",
            "experte",
            "fuer",
            "die",
            "der",
            "das",
            "und",
            "den",
            "dem",
            "ein",
            "eine",
            "ist",
            "von",
            "zu",
            "mit",
            "auf",
            "in",
            "im",
            "am",
            "als",
            "bei",
            "nach",
            "ueber",
        }

        for goal in goals:
            best_plan = None
            best_score = 0.0
            goal_words = set(goal.title.lower().split()) - _stop
            for plan in plans:
                plan_words = set(plan.goal.lower().split()) - _stop
                overlap = len(goal_words & plan_words)
                if overlap > best_score:
                    best_score = overlap
                    best_plan = plan
            if not best_plan or best_score < 1:
                continue

            # Subgoal activity (any non-pending status counts)
            total_sg = len(best_plan.sub_goals) or 1
            active_sg = sum(1 for sg in best_plan.sub_goals if sg.status not in ("pending",))
            subgoal_ratio = active_sg / total_sg

            # Read real metrics from Goal Index DB
            chunks = 0
            entities = 0
            idx_dir = index_base / best_plan.goal_slug
            if idx_dir.exists():
                try:
                    from jarvis.evolution.goal_index import GoalScopedIndex

                    gi = GoalScopedIndex(best_plan.goal_slug, str(index_base))
                    stats = gi.stats()
                    chunks = stats.get("chunks", 0)
                    entities = stats.get("entities", 0)
                    gi.close()
                except Exception:
                    pass

            sources = best_plan.total_vault_entries or 0

            progress = (
                0.40 * subgoal_ratio
                + 0.30 * min(1.0, chunks / 200.0)
                + 0.20 * min(1.0, entities / 100.0)
                + 0.10 * min(1.0, sources / 10.0)
            )
            progress = round(min(1.0, max(0.0, progress)), 2)

            current = goal.progress
            if abs(progress - current) >= 0.01:
                delta = progress - current
                try:
                    self._goal_manager.update_progress(
                        goal.id,
                        delta,
                        f"Metrisch: SG={active_sg}/{total_sg} "
                        f"Chunks={chunks} Entities={entities} Sources={sources}",
                    )
                    log.info(
                        "atl_goal_progress_metric",
                        goal=goal.title[:40],
                        old=f"{current:.0%}",
                        new=f"{progress:.0%}",
                        subgoals=f"{active_sg}/{total_sg}",
                        chunks=chunks,
                        entities=entities,
                    )
                except Exception:
                    log.debug("atl_metric_progress_failed", goal_id=goal.id, exc_info=True)

    def _should_think(self) -> bool:
        """Check if it's time for a thinking cycle vs learning cycle."""
        import time as _time

        if not self._atl_config or not self._atl_config.enabled:
            return False
        interval = self._atl_config.interval_minutes * 60
        return (_time.monotonic() - self._last_thinking_time) >= interval

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
                    # ATL thinking cycle (interleaved with learning)
                    if self._atl_config and self._atl_config.enabled and self._should_think():
                        log.info("atl_thinking_cycle_starting", cycle=self._atl_cycle_count + 1)
                        result = await self.thinking_cycle()
                    else:
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
            "atl_thinking_cycles": self._atl_cycle_count,
            "atl_enabled": bool(self._atl_config and self._atl_config.enabled),
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
