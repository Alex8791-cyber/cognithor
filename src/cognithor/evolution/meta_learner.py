"""MetaLearner -- self-improvement engine that analyses evolution cycle history.

Detects which research strategies yield the best results, recommends
adjustments to depth/breadth/frequency, and tracks overall learning
efficiency over time.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "MetaAnalysis",
    "MetaLearner",
    "StrategyAdjustment",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class StrategyAdjustment:
    """A recommended change to evolution strategy parameters."""

    parameter: str  # e.g. "research_depth", "source_count", "frequency"
    current_value: float
    recommended_value: float
    reason: str
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategyAdjustment:
        return cls(
            id=d.get("id", _new_id()),
            parameter=d["parameter"],
            current_value=d["current_value"],
            recommended_value=d["recommended_value"],
            reason=d.get("reason", ""),
        )


@dataclass
class MetaAnalysis:
    """Result of analysing a set of evolution cycles."""

    total_cycles: int
    avg_quality_score: float
    avg_coverage_score: float
    best_strategy: str
    worst_strategy: str
    efficiency_score: float  # 0-1
    trend: str  # "improving" | "stable" | "declining"
    adjustments: list[StrategyAdjustment] = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "total_cycles": self.total_cycles,
            "avg_quality_score": self.avg_quality_score,
            "avg_coverage_score": self.avg_coverage_score,
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
            "efficiency_score": self.efficiency_score,
            "trend": self.trend,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetaAnalysis:
        return cls(
            id=d.get("id", _new_id()),
            total_cycles=d["total_cycles"],
            avg_quality_score=d["avg_quality_score"],
            avg_coverage_score=d["avg_coverage_score"],
            best_strategy=d.get("best_strategy", ""),
            worst_strategy=d.get("worst_strategy", ""),
            efficiency_score=d.get("efficiency_score", 0.0),
            trend=d.get("trend", "stable"),
            adjustments=[StrategyAdjustment.from_dict(a) for a in d.get("adjustments", [])],
            created_at=d.get("created_at", datetime.now(UTC).isoformat()),
        )


# ---------------------------------------------------------------------------
# MetaLearner
# ---------------------------------------------------------------------------

# Strategy labels mapped from cycle data
_STRATEGY_LABELS = {
    "deep_research": "Tiefe Recherche mit vielen Quellen",
    "broad_scan": "Breite Suche mit wenigen Quellen pro Thema",
    "targeted": "Gezielte Recherche auf Kernthemen",
    "iterative": "Iterative Vertiefung mit Wiederholungen",
}


def _classify_strategy(cycle: dict[str, Any]) -> str:
    """Heuristic classification of the strategy used in a cycle."""
    sources = cycle.get("sources_fetched", 0)
    chunks = cycle.get("chunks_created", 0)
    rounds = cycle.get("research_rounds", 1)

    if sources > 10 and chunks > 50:
        return "deep_research"
    if sources > 5 and rounds <= 2:
        return "broad_scan"
    if rounds > 3:
        return "iterative"
    return "targeted"


class MetaLearner:
    """Analyses patterns in past evolution cycles to improve future strategies.

    Tracks which strategies (deep research, broad scan, targeted, iterative)
    produce the best quality and coverage scores, then recommends parameter
    adjustments accordingly.
    """

    def __init__(self) -> None:
        self._history: list[MetaAnalysis] = []
        # PASS-3: cap history so a long-running evolution loop
        # (thousands of cycles/day) does not accumulate forever.
        self._history_max = 200

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    async def analyze_cycle_history(
        self,
        cycles: list[dict[str, Any]],
    ) -> MetaAnalysis:
        """Analyse a list of cycle result dicts and produce a MetaAnalysis.

        Each *cycle* dict should contain at least:
        - ``quality_score`` (float, 0-1)
        - ``coverage_score`` (float, 0-1)
        - ``sources_fetched`` (int)
        - ``chunks_created`` (int)
        - ``research_rounds`` (int, optional)
        - ``duration_seconds`` (float, optional)
        """
        if not cycles:
            return MetaAnalysis(
                total_cycles=0,
                avg_quality_score=0.0,
                avg_coverage_score=0.0,
                best_strategy="",
                worst_strategy="",
                efficiency_score=0.0,
                trend="stable",
            )

        quality_scores = [c.get("quality_score", 0.0) for c in cycles]
        coverage_scores = [c.get("coverage_score", 0.0) for c in cycles]

        avg_quality = statistics.mean(quality_scores)
        avg_coverage = statistics.mean(coverage_scores)

        # Classify strategies and track per-strategy scores
        strategy_scores: dict[str, list[float]] = {}
        for cycle in cycles:
            strategy = _classify_strategy(cycle)
            combined = (
                cycle.get("quality_score", 0.0) * 0.6 + cycle.get("coverage_score", 0.0) * 0.4
            )
            strategy_scores.setdefault(strategy, []).append(combined)

        # Find best and worst strategies
        strategy_avgs = {s: statistics.mean(scores) for s, scores in strategy_scores.items()}
        best_strategy = max(strategy_avgs, key=strategy_avgs.get) if strategy_avgs else ""  # type: ignore[arg-type]
        worst_strategy = min(strategy_avgs, key=strategy_avgs.get) if strategy_avgs else ""  # type: ignore[arg-type]

        # Efficiency: ratio of quality achieved per source fetched
        total_sources = sum(c.get("sources_fetched", 1) for c in cycles)
        efficiency = avg_quality / max(total_sources / len(cycles), 1.0)
        efficiency = min(efficiency, 1.0)

        # Trend detection: compare first half vs second half
        trend = self._detect_trend(quality_scores)

        # Generate adjustments
        adjustments = self._generate_adjustments(
            avg_quality=avg_quality,
            avg_coverage=avg_coverage,
            efficiency=efficiency,
            best_strategy=best_strategy,
            worst_strategy=worst_strategy,
            cycles=cycles,
        )

        analysis = MetaAnalysis(
            total_cycles=len(cycles),
            avg_quality_score=round(avg_quality, 3),
            avg_coverage_score=round(avg_coverage, 3),
            best_strategy=best_strategy,
            worst_strategy=worst_strategy,
            efficiency_score=round(efficiency, 3),
            trend=trend,
            adjustments=adjustments,
        )
        self._history.append(analysis)
        if len(self._history) > self._history_max:
            # Keep the tail; older analyses are not consulted.
            self._history = self._history[-self._history_max :]

        log.info(
            "meta_analysis_complete",
            cycles=len(cycles),
            quality=analysis.avg_quality_score,
            coverage=analysis.avg_coverage_score,
            efficiency=analysis.efficiency_score,
            trend=trend,
            best=best_strategy,
        )
        return analysis

    # ------------------------------------------------------------------
    # Strategy recommendations
    # ------------------------------------------------------------------

    async def recommend_strategy_adjustments(self) -> list[StrategyAdjustment]:
        """Return adjustments from the most recent analysis, or empty list."""
        if not self._history:
            return []
        return self._history[-1].adjustments

    # ------------------------------------------------------------------
    # Efficiency tracking
    # ------------------------------------------------------------------

    async def get_learning_efficiency(self) -> float:
        """Return the current learning efficiency score (0-1)."""
        if not self._history:
            return 0.0
        return self._history[-1].efficiency_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_trend(scores: list[float]) -> str:
        """Compare first half vs second half of scores."""
        if len(scores) < 4:
            return "stable"
        mid = len(scores) // 2
        first_half = statistics.mean(scores[:mid])
        second_half = statistics.mean(scores[mid:])
        delta = second_half - first_half
        if delta > 0.05:
            return "improving"
        if delta < -0.05:
            return "declining"
        return "stable"

    @staticmethod
    def _generate_adjustments(
        *,
        avg_quality: float,
        avg_coverage: float,
        efficiency: float,
        best_strategy: str,
        worst_strategy: str,
        cycles: list[dict[str, Any]],
    ) -> list[StrategyAdjustment]:
        adjustments: list[StrategyAdjustment] = []

        avg_sources = (
            statistics.mean([c.get("sources_fetched", 0) for c in cycles]) if cycles else 0
        )

        # Low quality -> increase research depth
        if avg_quality < 0.5:
            adjustments.append(
                StrategyAdjustment(
                    parameter="research_depth",
                    current_value=avg_sources,
                    recommended_value=avg_sources * 1.5,
                    reason=(
                        f"Durchschnittliche Qualitaet ({avg_quality:.1%}) ist niedrig. "
                        "Mehr Quellen pro Thema konsultieren."
                    ),
                )
            )

        # Low coverage -> broaden search
        if avg_coverage < 0.5:
            adjustments.append(
                StrategyAdjustment(
                    parameter="source_count",
                    current_value=avg_sources,
                    recommended_value=avg_sources + 5,
                    reason=(
                        f"Durchschnittliche Abdeckung ({avg_coverage:.1%}) ist niedrig. "
                        "Breitere Suche empfohlen."
                    ),
                )
            )

        # Low efficiency -> reduce frequency but increase depth
        if efficiency < 0.3 and avg_sources > 5:
            adjustments.append(
                StrategyAdjustment(
                    parameter="frequency",
                    current_value=1.0,
                    recommended_value=0.5,
                    reason=(
                        f"Niedrige Effizienz ({efficiency:.1%}). "
                        "Weniger haeufig, aber gruendlicher recherchieren."
                    ),
                )
            )

        # Strategy recommendation
        if best_strategy and worst_strategy and best_strategy != worst_strategy:
            adjustments.append(
                StrategyAdjustment(
                    parameter="strategy_preference",
                    current_value=0.0,
                    recommended_value=1.0,
                    reason=(
                        f"Beste Strategie: {best_strategy}. "
                        f"Schlechteste: {worst_strategy}. "
                        f"Mehr Zyklen mit '{best_strategy}'-Ansatz empfohlen."
                    ),
                )
            )

        return adjustments

    @property
    def history(self) -> list[MetaAnalysis]:
        return list(self._history)
