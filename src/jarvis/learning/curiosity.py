"""Curiosity-driven knowledge exploration engine.

Identifies knowledge gaps in Jarvis's entity graph and proposes
exploration tasks to fill them.  Gaps arise from:

  - Low-confidence entities (< 0.5)
  - Stale entities (not updated in 90+ days)
  - Missing expected relations
  - Mentioned topics with no backing entities

Integration:
  - Reads entities from MemoryManager / SemanticMemory
  - Generates ExplorationTasks consumed by the active learner
  - API surface exposed via ``_register_learning_routes``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeGap:
    """A detected gap in Jarvis's knowledge."""

    id: str
    question: str
    topic: str
    importance: float  # 0-100
    curiosity: float  # 0-100
    suggested_sources: list[str] = field(default_factory=list)
    status: str = "open"  # open, exploring, answered, dismissed
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ExplorationTask:
    """A task for autonomous knowledge exploration."""

    gap_id: str
    query: str
    sources: list[str]  # memory, web, files
    priority: int = 5  # 1-10
    max_depth: int = 2


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CuriosityEngine:
    """Identifies knowledge gaps, ranks exploration priority, proposes research tasks."""

    #: Confidence threshold below which an entity is flagged.
    LOW_CONFIDENCE_THRESHOLD = 0.5

    #: Days without update before an entity is considered stale.
    STALE_DAYS = 90

    def __init__(self, memory: MemoryManager | None = None) -> None:
        self._memory = memory
        self._gaps: list[KnowledgeGap] = []
        self._exploration_queue: list[ExplorationTask] = []

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    async def detect_gaps(
        self,
        context: str,
        entities: list[dict[str, Any]],
    ) -> list[KnowledgeGap]:
        """Analyze *context* and *entities* to find knowledge gaps.

        Gaps are detected when:
        - An entity has low confidence (< 0.5)
        - An entity hasn't been updated in > 90 days
        """
        gaps: list[KnowledgeGap] = []

        now = datetime.now(UTC)

        for entity in entities:
            eid = entity.get("id", "")
            name = entity.get("name", "?")

            # 1. Low confidence entities
            conf = entity.get("confidence", 1.0)
            if conf < self.LOW_CONFIDENCE_THRESHOLD:
                gaps.append(
                    KnowledgeGap(
                        id=f"low_conf_{eid}",
                        question=f"What do we really know about {name}?",
                        topic=name,
                        importance=70,
                        curiosity=(1 - conf) * 100,
                    )
                )

            # 2. Stale entities (not updated in 90+ days)
            updated = entity.get("updated_at")
            if updated and isinstance(updated, str):
                try:
                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    days_old = (now - dt).days
                    if days_old > self.STALE_DAYS:
                        gaps.append(
                            KnowledgeGap(
                                id=f"stale_{eid}",
                                question=f"Is information about {name} still current?",
                                topic=name,
                                importance=40,
                                curiosity=min(days_old / 3, 100),
                            )
                        )
                except (ValueError, TypeError):
                    pass

        self._gaps = gaps
        return gaps

    # ------------------------------------------------------------------
    # Exploration tasks
    # ------------------------------------------------------------------

    def propose_exploration(self, max_tasks: int = 5) -> list[ExplorationTask]:
        """Generate exploration tasks from detected gaps, sorted by priority."""
        tasks: list[ExplorationTask] = []
        sorted_gaps = sorted(
            [g for g in self._gaps if g.status == "open"],
            key=lambda g: g.importance * 0.6 + g.curiosity * 0.4,
            reverse=True,
        )

        for gap in sorted_gaps[:max_tasks]:
            tasks.append(
                ExplorationTask(
                    gap_id=gap.id,
                    query=gap.question,
                    sources=["memory", "web"],
                    priority=max(1, min(10, int(gap.importance / 10))),
                )
            )

        self._exploration_queue = tasks
        return tasks

    # ------------------------------------------------------------------
    # Properties & mutators
    # ------------------------------------------------------------------

    @property
    def gaps(self) -> list[KnowledgeGap]:
        """Return a copy of the current gap list."""
        return list(self._gaps)

    @property
    def open_gap_count(self) -> int:
        return sum(1 for g in self._gaps if g.status == "open")

    def dismiss_gap(self, gap_id: str) -> bool:
        """Mark a gap as dismissed.  Returns ``True`` if found."""
        for g in self._gaps:
            if g.id == gap_id:
                g.status = "dismissed"
                return True
        return False

    def mark_answered(self, gap_id: str) -> bool:
        """Mark a gap as answered.  Returns ``True`` if found."""
        for g in self._gaps:
            if g.id == gap_id:
                g.status = "answered"
                return True
        return False

    def mark_exploring(self, gap_id: str) -> bool:
        """Mark a gap as currently being explored."""
        for g in self._gaps:
            if g.id == gap_id:
                g.status = "exploring"
                return True
        return False
