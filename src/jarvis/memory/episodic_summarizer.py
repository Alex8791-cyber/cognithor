"""LLM-gestuetzte Zusammenfassung von Episoden."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, TYPE_CHECKING

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.memory.episodic_store import EpisodicStore

log = get_logger(__name__)


class EpisodicSummarizer:
    """Erstellt Zusammenfassungen ueber Zeitraeume."""

    def __init__(self, store: EpisodicStore, llm: Any = None) -> None:
        self._store = store
        self._llm = llm

    async def summarize_day(self, target_date: date) -> str:
        """Erstellt eine Tageszusammenfassung."""
        start = target_date.isoformat()
        end = (target_date + timedelta(days=1)).isoformat()

        # Get episodes for this day
        try:
            episodes = self._store.search_episodes(
                query="*",
                limit=100,
                date_range=(start, end),
            )
        except Exception:
            episodes = []

        if not episodes:
            return f"Keine Episoden fuer {target_date.isoformat()}"

        # Build summary text
        lines = [f"Tageszusammenfassung fuer {target_date.isoformat()}:"]
        for ep in episodes:
            score_str = f"Score: {ep.success_score:.1f}"
            lines.append(f"- [{score_str}] {ep.topic}: {ep.outcome or ep.content[:100]}")

        summary = "\n".join(lines)

        # If LLM available, enhance the summary
        if self._llm:
            try:
                prompt_text = (
                    f"Fasse folgende Episoden des Tages zusammen. "
                    f"Fokus auf Erfolge, Misserfolge, Muster:\n\n{summary}"
                )
                response = await self._llm.chat(
                    model="qwen3:8b",
                    messages=[
                        {"role": "system", "content": "Erstelle eine kurze Tageszusammenfassung auf Deutsch."},
                        {"role": "user", "content": prompt_text},
                    ],
                )
                summary = response.get("message", {}).get("content", summary)
            except Exception as exc:
                log.warning("summarize_day_llm_error", error=str(exc))

        # Store the summary
        self._store.store_summary(
            period="day",
            start_date=start,
            end_date=end,
            summary=summary,
            key_learnings=[],
        )

        return summary

    async def summarize_week(self, week_start: date) -> str:
        """Erstellt eine Wochenzusammenfassung aus Tageszusammenfassungen."""
        week_end = week_start + timedelta(days=7)

        # Get daily summaries for this week
        summaries = self._store.get_summaries(period="day")
        week_summaries = [
            s for s in summaries
            if s["start_date"] >= week_start.isoformat()
            and s["start_date"] < week_end.isoformat()
        ]

        if not week_summaries:
            return f"Keine Zusammenfassungen fuer Woche ab {week_start.isoformat()}"

        lines = [f"Wochenzusammenfassung ({week_start.isoformat()} - {week_end.isoformat()}):"]
        for s in week_summaries:
            lines.append(f"\n### {s['start_date']}")
            lines.append(s["summary"][:500])

        summary = "\n".join(lines)

        self._store.store_summary(
            period="week",
            start_date=week_start.isoformat(),
            end_date=week_end.isoformat(),
            summary=summary,
        )

        return summary
