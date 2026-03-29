"""StrategyPlanner — decomposes a high-level learning goal into a structured
LearningPlan via LLM-based reasoning."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Coroutine, List, Optional

from jarvis.evolution.models import (
    LearningPlan,
    ScheduleSpec,
    SeedSource,
    SourceSpec,
    SubGoal,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_STRATEGY_PROMPT = """\
Du bist ein Lernstrategie-Planer. Zerlege das folgende Lernziel in einen \
strukturierten Plan.

## Lernziel
{goal}

{seed_section}

## Regeln
- Erstelle 5-15 SubGoals (Teilziele), jeweils mit title, description, priority (1-10).
- Nenne autoritative Quellen (sources) mit url, source_type, title, fetch_strategy \
(single_page | sitemap_crawl | rss | api), update_frequency (once | daily | weekly).
- Erstelle schedules fuer Quellen die sich aendern (cron_expression, source_url, action, name, description).
- Denke ueber das Ziel hinaus: Welche verwandten Themen sind relevant?

## Ausgabeformat
Antworte ausschliesslich mit validem JSON:
```json
{{
  "sub_goals": [{{"title": "...", "description": "...", "priority": 10}}],
  "sources": [{{"url": "...", "source_type": "...", "title": "...", "fetch_strategy": "...", "update_frequency": "..."}}],
  "schedules": [{{"name": "...", "cron_expression": "...", "source_url": "...", "action": "...", "description": "..."}}]
}}
```
"""

_REPLAN_PROMPT = """\
Du bist ein Lernstrategie-Planer. Ein bestehender Plan wird erweitert.

## Urspruengliches Ziel
{goal}

## Bestehende Teilziele
{existing_subgoals}

## Neuer Kontext
{new_context}

## Aufgabe
Erstelle ZUSAETZLICHE sub_goals, sources und schedules die den Plan erweitern. \
Wiederhole KEINE bestehenden Teilziele. Antworte ausschliesslich mit validem JSON:
```json
{{
  "sub_goals": [{{"title": "...", "description": "...", "priority": 10}}],
  "sources": [{{"url": "...", "source_type": "...", "title": "...", "fetch_strategy": "...", "update_frequency": "..."}}],
  "schedules": [{{"name": "...", "cron_expression": "...", "source_url": "...", "action": "...", "description": "..."}}]
}}
```
"""

_STRICT_JSON_SUFFIX = (
    "\n\nWICHTIG: Antworte NUR mit einem einzigen JSON-Objekt. Kein Text davor oder danach."
)

# ---------------------------------------------------------------------------
# Complexity keywords
# ---------------------------------------------------------------------------

_COMPLEX_KEYWORDS: list[str] = [
    "experte",
    "expert",
    "master",
    "meister",
    "spezialist",
    "deep dive",
    "umfassend",
    "komplett",
    "alles ueber",
    "alles über",
    "vollstaendig",
    "vollständig",
    "grundlagen bis fortgeschritten",
    "von grund auf",
    "zertifizierung",
    "certification",
]

_MIN_WORD_COUNT_FOR_COMPLEX = 10

# Type alias for the LLM callable
LLMFn = Callable[[str], Coroutine[Any, Any, str]]


class StrategyPlanner:
    """Decomposes a learning goal into a structured LearningPlan via LLM."""

    def __init__(self, llm_fn: LLMFn, max_retries: int = 3) -> None:
        self._llm_fn = llm_fn
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_plan(
        self,
        goal: str,
        seed_sources: Optional[List[SeedSource]] = None,
    ) -> LearningPlan:
        """Build a LearningPlan by asking the LLM to decompose *goal*."""
        plan = LearningPlan(goal=goal)
        if seed_sources:
            plan.seed_sources = list(seed_sources)

        seed_section = ""
        if seed_sources:
            lines = ["## Seed-Quellen (vom Nutzer bereitgestellt)"]
            for s in seed_sources:
                lines.append(f"- [{s.title or s.value}]({s.value}) ({s.content_type})")
            seed_section = "\n".join(lines)

        prompt = _STRATEGY_PROMPT.format(goal=goal, seed_section=seed_section)
        data = await self._call_llm_json(prompt)

        if data is None:
            plan.status = "error"
            return plan

        _populate_plan(plan, data)
        plan.status = "active"
        return plan

    async def replan(self, plan: LearningPlan, new_context: str) -> LearningPlan:
        """Extend *plan* with additional sub_goals/sources/schedules."""
        existing = "\n".join(
            f"- [{sg.status}] {sg.title}: {sg.description}" for sg in plan.sub_goals
        )
        prompt = _REPLAN_PROMPT.format(
            goal=plan.goal,
            existing_subgoals=existing,
            new_context=new_context,
        )
        data = await self._call_llm_json(prompt)
        if data is None:
            log.warning("replan: LLM returned no valid JSON — plan unchanged")
            return plan

        # Merge without duplicates
        existing_titles = {sg.title for sg in plan.sub_goals}
        for sg_data in data.get("sub_goals", []):
            title = sg_data.get("title", "")
            if title and title not in existing_titles:
                plan.sub_goals.append(
                    SubGoal(
                        title=title,
                        description=sg_data.get("description", ""),
                        priority=sg_data.get("priority", 5),
                        parent_goal_id=plan.id,
                    )
                )
                existing_titles.add(title)

        existing_urls = {s.url for s in plan.sources}
        for src in data.get("sources", []):
            url = src.get("url", "")
            if url and url not in existing_urls:
                plan.sources.append(
                    SourceSpec(
                        url=url,
                        source_type=src.get("source_type", "web"),
                        title=src.get("title"),
                        fetch_strategy=src.get("fetch_strategy"),
                        update_frequency=src.get("update_frequency"),
                    )
                )
                existing_urls.add(url)

        existing_names = {s.name for s in plan.schedules}
        for sched in data.get("schedules", []):
            name = sched.get("name", "")
            if name and name not in existing_names:
                plan.schedules.append(
                    ScheduleSpec(
                        name=name,
                        cron_expression=sched.get("cron_expression", ""),
                        source_url=sched.get("source_url"),
                        action=sched.get("action", "fetch"),
                        description=sched.get("description"),
                    )
                )
                existing_names.add(name)

        # Re-sort sub_goals by priority desc
        plan.sub_goals.sort(key=lambda sg: sg.priority, reverse=True)
        plan.expansions += 1
        return plan

    def is_complex_goal(self, goal: str) -> bool:
        """Heuristic: does *goal* look like a large learning endeavour?"""
        lower = goal.lower()
        for kw in _COMPLEX_KEYWORDS:
            if kw in lower:
                return True
        if len(goal.split()) > _MIN_WORD_COUNT_FOR_COMPLEX:
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_llm_json(self, prompt: str) -> dict | None:
        """Call LLM and parse JSON from response, retrying on failure."""
        for attempt in range(self._max_retries):
            p = prompt if attempt == 0 else prompt + _STRICT_JSON_SUFFIX
            try:
                raw = await self._llm_fn(p)
                extracted = _extract_json(raw)
                if extracted is not None:
                    return json.loads(extracted)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                log.debug("LLM JSON parse attempt %d failed: %s", attempt + 1, exc)
            log.info("Retrying LLM call (%d/%d)", attempt + 1, self._max_retries)
        log.error("LLM failed to return valid JSON after %d attempts", self._max_retries)
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str | None:
    """Pull a JSON object from LLM output — handles ```json fences and raw braces."""
    if not text:
        return None
    # Try ```json ... ``` block first
    m = re.search(r"```json\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try raw { ... } extraction (outermost braces)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _populate_plan(plan: LearningPlan, data: dict) -> None:
    """Fill *plan* fields from parsed LLM dict."""
    for sg_data in data.get("sub_goals", []):
        plan.sub_goals.append(
            SubGoal(
                title=sg_data.get("title", "Untitled"),
                description=sg_data.get("description", ""),
                priority=sg_data.get("priority", 5),
                parent_goal_id=plan.id,
            )
        )
    for src in data.get("sources", []):
        plan.sources.append(
            SourceSpec(
                url=src.get("url", ""),
                source_type=src.get("source_type", "web"),
                title=src.get("title"),
                fetch_strategy=src.get("fetch_strategy"),
                update_frequency=src.get("update_frequency"),
            )
        )
    for sched in data.get("schedules", []):
        plan.schedules.append(
            ScheduleSpec(
                name=sched.get("name", ""),
                cron_expression=sched.get("cron_expression", ""),
                source_url=sched.get("source_url"),
                action=sched.get("action", "fetch"),
                description=sched.get("description"),
            )
        )
    # Sort sub_goals by priority descending
    plan.sub_goals.sort(key=lambda sg: sg.priority, reverse=True)
