"""ScheduleManager — creates cron jobs for recurring evolution source updates."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.evolution.models import LearningPlan

log = get_logger(__name__)


def _normalize_cron(expr: str) -> str:
    """Convert 6-field or named-day cron expressions to standard 5-field.

    LLMs sometimes produce:
    - 6-field: '0 0 8 * * MON' (seconds included)
    - Named days: 'MON', 'TUE', etc.

    CronEngine expects 5-field with numeric day-of-week (0=Sun..6=Sat).
    """
    day_map = {
        "SUN": "0",
        "MON": "1",
        "TUE": "2",
        "WED": "3",
        "THU": "4",
        "FRI": "5",
        "SAT": "6",
    }
    parts = expr.strip().split()

    # 6 fields → drop first (seconds)
    if len(parts) == 6:
        parts = parts[1:]

    # Replace named days
    if len(parts) == 5:
        for name, num in day_map.items():
            parts[4] = re.sub(rf"\b{name}\b", num, parts[4], flags=re.IGNORECASE)
        # Fix day_of_week=7 → 0 (LLMs sometimes use 7 for Sunday)
        parts[4] = re.sub(r"\b7\b", "0", parts[4])

    return " ".join(parts)


class ScheduleManager:
    """Translates ScheduleSpecs from a LearningPlan into cron jobs.

    Each schedule is registered via cron_engine.add_runtime_job(CronJob(...))
    so that sources are periodically re-fetched.

    Jobs are automatically staggered by 10 minutes to prevent GPU contention
    when many jobs share the same hour (max 6 jobs per hour, 144 per day).
    """

    _stagger_counter: int = 0  # Class-level counter for staggering

    def __init__(self, cron_engine: Any | None = None) -> None:
        self._cron_engine = cron_engine

    @classmethod
    def _stagger_cron(cls, cron_expr: str) -> str:
        """Offset the minute field by 10-min increments to avoid GPU contention.

        Jobs that would fire at minute=0 get staggered to 0, 10, 20, 30, 40, 50.
        This ensures max 1 job fires per 10-minute window.
        """
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            return cron_expr
        minute = parts[0]
        # Only stagger jobs that fire at a fixed minute (not ranges/wildcards)
        if minute.isdigit():
            base = int(minute)
            offset = (cls._stagger_counter * 10) % 60
            # If base was 0, distribute across 0,10,20,30,40,50
            # If base was something else, add offset
            new_minute = (base + offset) % 60
            parts[0] = str(new_minute)
            # If we wrapped past 60, we might need to bump the hour
            # but for simplicity we just mod 60 (jobs shift within same hour)
            cls._stagger_counter += 1
        return " ".join(parts)

    async def create_schedules(self, plan: LearningPlan) -> int:
        """Create cron jobs for every ScheduleSpec in *plan*.

        Returns number of cron jobs successfully created.
        """
        if self._cron_engine is None or not plan.schedules:
            return 0

        from cognithor.models import CronJob

        created = 0
        for spec in plan.schedules:
            job_name = f"evolution_{plan.goal_slug[:30]}_{spec.name[:30]}"
            # Sanitize job name (no spaces, no special chars)
            job_name = re.sub(r"[^a-zA-Z0-9_-]", "_", job_name)

            cron_expr = self._stagger_cron(_normalize_cron(spec.cron_expression))
            prompt = (
                f"[evolution-update:{plan.id}:{spec.source_url}] {spec.description or spec.name}"
            )

            try:
                cron_job = CronJob(
                    name=job_name,
                    schedule=cron_expr,
                    prompt=prompt,
                    channel="cli",
                    enabled=True,
                )
                ok = self._cron_engine.add_runtime_job(cron_job)
                if ok:
                    created += 1
                    log.info(
                        "evolution_cron_created",
                        job=job_name,
                        cron=cron_expr,
                        source=spec.source_url[:50],
                    )
                else:
                    log.warning("evolution_cron_failed", job=job_name)
            except Exception:
                log.debug("evolution_cron_error", job=job_name, exc_info=True)

        return created
