"""HorizonScanner — discovers new learning opportunities via LLM exploration
and knowledge-graph gap analysis.

Part of Phase 5C: autonomous plan expansion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognithor.evolution.models import LearningPlan

logger = logging.getLogger(__name__)

LLMFunction = Callable[[str], Coroutine[Any, Any, str]]


class HorizonScanner:
    """Scans for new sub-goal candidates using two mechanisms:

    1. **LLM exploration** — asks the LLM to suggest expansions based on
       completed sub-goals and known entities.
    2. **Graph gap discovery** — finds knowledge-graph entities that have
       very few memory chunks (< 2), indicating shallow coverage.
    """

    def __init__(self, llm_fn: LLMFunction, memory_manager: Any) -> None:
        self._llm_fn = llm_fn
        self._memory = memory_manager
        self._targeted_gaps: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_targeted_expansion(self, gap_topic: str) -> None:
        """Queue a specific topic for the next scan cycle.

        Called by CycleController when exam reveals knowledge gaps.
        """
        if gap_topic not in self._targeted_gaps:
            self._targeted_gaps.append(gap_topic)
            logger.info("horizon_targeted_gap_added", topic=gap_topic)

    async def scan(self, plan: LearningPlan) -> list[dict[str, Any]]:
        """Run both discovery mechanisms and return deduplicated results."""
        llm_results = await self.explore_via_llm(plan)
        graph_results = await self.discover_graph_gaps(plan.goal_slug)

        # Add targeted gaps from CycleController exam results
        targeted = []
        if self._targeted_gaps:
            for gap in self._targeted_gaps:
                targeted.append(
                    {
                        "title": gap,
                        "reason": "Knowledge gap detected in quality exam",
                        "source": "cycle_controller",
                    }
                )
            self._targeted_gaps.clear()

        combined = llm_results + graph_results + targeted

        # Deduplicate against existing sub-goal titles (case-insensitive)
        existing_titles = {sg.title.lower() for sg in plan.sub_goals}
        deduplicated = [r for r in combined if r["title"].lower() not in existing_titles]
        return deduplicated

    async def explore_via_llm(self, plan: LearningPlan) -> list[dict[str, Any]]:
        """Ask the LLM for expansion suggestions based on the current plan."""
        completed = [sg.title for sg in plan.sub_goals if sg.status in ("passed", "done")]
        all_titles = [sg.title for sg in plan.sub_goals]

        # Gather known entities from memory
        entities: list[str] = []
        try:
            entity_objs = self._memory.semantic.list_entities()
            entities = [e.name for e in entity_objs]
        except Exception:
            logger.debug("Could not list entities from memory", exc_info=True)

        prompt = (
            f"You are expanding a learning plan about: {plan.goal}\n"
            f"Completed sub-goals: {completed}\n"
            f"All sub-goals: {all_titles}\n"
            f"Known entities: {entities}\n\n"
            "Suggest 2-5 new sub-goals that would deepen or broaden this plan. "
            'Return JSON: {"expansions": [{"title": "...", "reason": "..."}]}'
        )

        try:
            raw = await self._llm_fn(prompt)
            data = json.loads(raw)
            expansions = data.get("expansions", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("LLM returned non-JSON for horizon scan; skipping.")
            return []

        return [
            {"title": e["title"], "reason": e["reason"], "source": "llm"}
            for e in expansions
            if isinstance(e, dict) and "title" in e and "reason" in e
        ]

    async def discover_graph_gaps(self, goal_slug: str) -> list[dict[str, Any]]:
        """Find entities that have fewer than 2 memory chunks — shallow coverage."""
        results: list[dict[str, Any]] = []

        try:
            entity_objs = self._memory.semantic.list_entities()
        except Exception:
            logger.debug("Could not list entities for graph gaps", exc_info=True)
            return results

        for entity in entity_objs:
            name = entity.name
            try:
                hits = self._memory.search_memory_sync(name)
            except Exception:
                hits = []

            if len(hits) < 2:
                results.append(
                    {
                        "title": f"Vertiefe: {name}",
                        "reason": (
                            f"Entity '{name}' has only {len(hits)} memory"
                            " chunk(s) — coverage is shallow."
                        ),
                        "source": "graph",
                    }
                )

        return results
