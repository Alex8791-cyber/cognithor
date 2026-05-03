"""AutonomousLearner -- curiosity-driven exploration and self-improvement engine.

Phase 5 of the Evolution Engine.  Extends the existing DeepLearner with
proactive knowledge-gap identification, learning-task prioritisation,
knowledge synthesis, and self-improvement proposals.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "AutonomousLearner",
    "ImprovementProposal",
    "Insight",
    "KnowledgeGap",
    "LearningOutcome",
    "LearningTask",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class GapSource(str, Enum):
    USER_INTERACTION = "user_interaction"
    FAILED_QUERY = "failed_query"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_DOMAIN = "missing_domain"


class TaskPriority(int, Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class KnowledgeGap:
    topic: str
    description: str
    source: GapSource = GapSource.USER_INTERACTION
    confidence: float = 0.0
    id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "description": self.description,
            "source": self.source.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeGap:
        return cls(
            id=d.get("id", _new_id()),
            topic=d["topic"],
            description=d.get("description", ""),
            source=GapSource(d.get("source", "user_interaction")),
            confidence=d.get("confidence", 0.0),
            created_at=d.get("created_at", datetime.now(UTC).isoformat()),
        )


@dataclass
class LearningTask:
    gap_id: str
    query: str
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_effort: float = 1.0  # hours
    id: str = field(default_factory=_new_id)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gap_id": self.gap_id,
            "query": self.query,
            "priority": self.priority.value,
            "estimated_effort": self.estimated_effort,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningTask:
        return cls(
            id=d.get("id", _new_id()),
            gap_id=d["gap_id"],
            query=d["query"],
            priority=TaskPriority(d.get("priority", 3)),
            estimated_effort=d.get("estimated_effort", 1.0),
            status=d.get("status", "pending"),
        )


@dataclass
class LearningOutcome:
    task_id: str
    summary: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    chunks_created: int = 0
    id: str = field(default_factory=_new_id)
    completed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "summary": self.summary,
            "sources": self.sources,
            "confidence": self.confidence,
            "chunks_created": self.chunks_created,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningOutcome:
        return cls(
            id=d.get("id", _new_id()),
            task_id=d["task_id"],
            summary=d["summary"],
            sources=d.get("sources", []),
            confidence=d.get("confidence", 0.0),
            chunks_created=d.get("chunks_created", 0),
            completed_at=d.get("completed_at", datetime.now(UTC).isoformat()),
        )


@dataclass
class Insight:
    title: str
    description: str
    supporting_outcomes: list[str] = field(default_factory=list)
    actionable: bool = False
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "supporting_outcomes": self.supporting_outcomes,
            "actionable": self.actionable,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Insight:
        return cls(
            id=d.get("id", _new_id()),
            title=d["title"],
            description=d.get("description", ""),
            supporting_outcomes=d.get("supporting_outcomes", []),
            actionable=d.get("actionable", False),
        )


class ProposalType(str, Enum):
    NEW_SKILL = "new_skill"
    CONFIG_CHANGE = "config_change"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    KNOWLEDGE_EXPANSION = "knowledge_expansion"


@dataclass
class ImprovementProposal:
    title: str
    proposal_type: ProposalType
    description: str
    insight_ids: list[str] = field(default_factory=list)
    estimated_impact: float = 0.5  # 0-1
    id: str = field(default_factory=_new_id)
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "proposal_type": self.proposal_type.value,
            "description": self.description,
            "insight_ids": self.insight_ids,
            "estimated_impact": self.estimated_impact,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImprovementProposal:
        return cls(
            id=d.get("id", _new_id()),
            title=d["title"],
            proposal_type=ProposalType(d.get("proposal_type", "knowledge_expansion")),
            description=d.get("description", ""),
            insight_ids=d.get("insight_ids", []),
            estimated_impact=d.get("estimated_impact", 0.5),
            status=d.get("status", "proposed"),
        )


# ---------------------------------------------------------------------------
# AutonomousLearner
# ---------------------------------------------------------------------------


# Keywords that often indicate the user does not know something
_UNCERTAINTY_KEYWORDS = {
    "was ist",
    "wie funktioniert",
    "erklaere",
    "ich weiss nicht",
    "keine ahnung",
    "was bedeutet",
    "kannst du",
    "hilf mir",
    "what is",
    "how does",
    "explain",
    "help me",
    "i don't know",
}

# Minimum interaction count before we start identifying gaps
_MIN_INTERACTIONS = 3


class AutonomousLearner:
    """Curiosity-driven exploration engine for Phase 5 of Evolution.

    Works alongside :class:`DeepLearner` to proactively identify knowledge
    gaps from user interactions, prioritise learning tasks, synthesise
    research outcomes, and propose self-improvement actions.
    """

    def __init__(
        self,
        llm_fn: Callable | None = None,  # type: ignore[type-arg]
        deep_learner: Any = None,
        memory_manager: Any = None,
    ) -> None:
        self._llm_fn = llm_fn
        self._deep_learner = deep_learner
        self._memory_manager = memory_manager
        self._outcomes: list[LearningOutcome] = []
        self._insights: list[Insight] = []
        self._proposals: list[ImprovementProposal] = []

    # ------------------------------------------------------------------
    # 1. Knowledge gap identification
    # ------------------------------------------------------------------

    async def identify_knowledge_gaps(
        self,
        recent_interactions: list[dict[str, Any]],
    ) -> list[KnowledgeGap]:
        """Analyse recent user interactions and find knowledge gaps.

        Each interaction dict should have at least ``"message"`` and
        optionally ``"response"``, ``"success"`` (bool), ``"topic"`` keys.
        """
        if len(recent_interactions) < _MIN_INTERACTIONS:
            return []

        gaps: list[KnowledgeGap] = []
        topic_counter: Counter[str] = Counter()

        for interaction in recent_interactions:
            msg = interaction.get("message", "").lower()
            topic = interaction.get("topic", "")
            success = interaction.get("success", True)

            # Count topic frequency
            if topic:
                topic_counter[topic] += 1

            # Failed queries indicate missing knowledge
            if not success:
                gaps.append(
                    KnowledgeGap(
                        topic=topic or msg[:80],
                        description=f"Fehlgeschlagene Anfrage: {msg[:200]}",
                        source=GapSource.FAILED_QUERY,
                        confidence=0.8,
                    )
                )
                continue

            # Uncertainty keywords suggest areas to explore
            for keyword in _UNCERTAINTY_KEYWORDS:
                if keyword in msg:
                    gaps.append(
                        KnowledgeGap(
                            topic=topic or msg[:80],
                            description=f"Unsicherheit erkannt: {msg[:200]}",
                            source=GapSource.USER_INTERACTION,
                            confidence=0.5,
                        )
                    )
                    break

        # Frequently asked topics with no dedicated knowledge
        for topic_name, count in topic_counter.most_common(5):
            if count >= 2:
                # Check if we already have a gap for this topic
                existing = {g.topic for g in gaps}
                if topic_name not in existing:
                    gaps.append(
                        KnowledgeGap(
                            topic=topic_name,
                            description=f"Haeufig angefragtes Thema ({count}x)",
                            source=GapSource.MISSING_DOMAIN,
                            confidence=min(0.3 + count * 0.15, 0.9),
                        )
                    )

        # Deduplicate by topic hash
        seen: set[str] = set()
        unique: list[KnowledgeGap] = []
        for gap in gaps:
            h = hashlib.md5(gap.topic.encode()).hexdigest()[:12]
            if h not in seen:
                seen.add(h)
                unique.append(gap)

        log.info("knowledge_gaps_identified", count=len(unique))
        return unique

    # ------------------------------------------------------------------
    # 2. Learning task prioritisation
    # ------------------------------------------------------------------

    async def prioritize_learning(
        self,
        gaps: list[KnowledgeGap],
    ) -> list[LearningTask]:
        """Convert knowledge gaps into prioritised learning tasks."""
        tasks: list[LearningTask] = []

        for gap in gaps:
            # Determine priority from gap source and confidence
            if gap.source == GapSource.FAILED_QUERY:
                priority = TaskPriority.HIGH
                effort = 0.5
            elif gap.source == GapSource.MISSING_DOMAIN:
                priority = TaskPriority.MEDIUM
                effort = 2.0
            elif gap.confidence >= 0.7:
                priority = TaskPriority.HIGH
                effort = 1.0
            else:
                priority = TaskPriority.LOW
                effort = 0.5

            tasks.append(
                LearningTask(
                    gap_id=gap.id,
                    query=f"{gap.topic}: {gap.description[:120]}",
                    priority=priority,
                    estimated_effort=effort,
                )
            )

        # Sort by priority (lower number = higher priority), then confidence
        gap_map = {g.id: g for g in gaps}
        tasks.sort(
            key=lambda t: (
                t.priority.value,
                -(gap_map.get(t.gap_id, KnowledgeGap(topic="", description="")).confidence),
            )
        )

        log.info("learning_tasks_prioritized", count=len(tasks))
        return tasks

    # ------------------------------------------------------------------
    # 3. Execute learning task
    # ------------------------------------------------------------------

    async def execute_learning_task(
        self,
        task: LearningTask,
    ) -> LearningOutcome:
        """Execute a single learning task.

        If a :class:`DeepLearner` is available, delegates to its plan
        creation.  Otherwise produces a minimal outcome stub that can be
        enriched later.
        """
        task.status = "running"
        log.info("learning_task_started", task_id=task.id[:8], query=task.query[:60])

        sources: list[str] = []
        summary = ""
        chunks = 0
        confidence = 0.0

        if self._deep_learner is not None:
            try:
                plan = await self._deep_learner.create_plan(task.query)
                summary = f"Lernplan erstellt: {plan.goal} ({len(plan.sub_goals)} Teilziele)"
                sources = [s.url for s in plan.sources[:5]]
                chunks = plan.total_chunks_indexed
                confidence = 0.6
            except Exception:
                log.debug("learning_task_plan_failed", exc_info=True)
                summary = f"Lernplan-Erstellung fehlgeschlagen fuer: {task.query[:100]}"
                confidence = 0.1
        else:
            summary = f"Lernaufgabe erkannt (kein DeepLearner): {task.query[:200]}"
            confidence = 0.2

        task.status = "completed"

        outcome = LearningOutcome(
            task_id=task.id,
            summary=summary,
            sources=sources,
            confidence=confidence,
            chunks_created=chunks,
        )
        self._outcomes.append(outcome)

        log.info(
            "learning_task_completed",
            task_id=task.id[:8],
            confidence=confidence,
            chunks=chunks,
        )
        return outcome

    # ------------------------------------------------------------------
    # 4. Knowledge synthesis
    # ------------------------------------------------------------------

    async def synthesize_knowledge(
        self,
        outcomes: list[LearningOutcome],
    ) -> list[Insight]:
        """Combine multiple learning outcomes into actionable insights."""
        if not outcomes:
            return []

        insights: list[Insight] = []

        # Group outcomes by common words in their summaries
        topic_groups: dict[str, list[LearningOutcome]] = {}
        for outcome in outcomes:
            words = set(outcome.summary.lower().split())
            # Use the longest word > 4 chars as a rough topic key
            key_words = sorted(
                (w for w in words if len(w) > 4 and w.isalpha()),
                key=len,
                reverse=True,
            )
            key = key_words[0] if key_words else "general"
            topic_groups.setdefault(key, []).append(outcome)

        for topic_key, group in topic_groups.items():
            if len(group) == 0:
                continue

            avg_confidence = sum(o.confidence for o in group) / len(group)
            total_chunks = sum(o.chunks_created for o in group)
            all_sources = []
            for o in group:
                all_sources.extend(o.sources)

            actionable = avg_confidence >= 0.5 and total_chunks > 0

            insight = Insight(
                title=f"Erkenntnisse zu '{topic_key}'",
                description=(
                    f"{len(group)} Lernaufgaben abgeschlossen. "
                    f"Durchschnittliche Konfidenz: {avg_confidence:.1%}. "
                    f"{total_chunks} Chunks erstellt aus {len(set(all_sources))} Quellen."
                ),
                supporting_outcomes=[o.id for o in group],
                actionable=actionable,
            )
            insights.append(insight)

        self._insights.extend(insights)
        log.info("knowledge_synthesized", insights=len(insights))
        return insights

    # ------------------------------------------------------------------
    # 5. Improvement proposals
    # ------------------------------------------------------------------

    async def propose_improvements(
        self,
        insights: list[Insight],
    ) -> list[ImprovementProposal]:
        """Generate self-improvement proposals from synthesised insights."""
        proposals: list[ImprovementProposal] = []

        for insight in insights:
            if not insight.actionable:
                continue

            # Decide proposal type based on insight content
            desc_lower = insight.description.lower()
            if "chunks erstellt" in desc_lower:
                proposal_type = ProposalType.KNOWLEDGE_EXPANSION
                title = f"Wissensbereich erweitern: {insight.title}"
                description = (
                    f"Basierend auf {insight.description} sollte das Wissen "
                    f"in diesem Bereich weiter vertieft werden."
                )
                impact = 0.6
            elif "konfidenz" in desc_lower and "100" not in desc_lower:
                proposal_type = ProposalType.WORKFLOW_OPTIMIZATION
                title = f"Recherche-Qualitaet verbessern: {insight.title}"
                description = (
                    "Die Konfidenz der Ergebnisse kann durch bessere "
                    "Quellenauswahl und tiefere Recherche verbessert werden."
                )
                impact = 0.4
            else:
                proposal_type = ProposalType.NEW_SKILL
                title = f"Neuen Skill erstellen: {insight.title}"
                description = (
                    f"Ein dedizierter Skill koennte die Erkenntnisse aus "
                    f"{insight.title} direkt nutzbar machen."
                )
                impact = 0.7

            proposals.append(
                ImprovementProposal(
                    title=title,
                    proposal_type=proposal_type,
                    description=description,
                    insight_ids=[insight.id],
                    estimated_impact=impact,
                )
            )

        # Sort by impact descending
        proposals.sort(key=lambda p: -p.estimated_impact)
        self._proposals.extend(proposals)

        log.info("improvements_proposed", count=len(proposals))
        return proposals

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def outcomes(self) -> list[LearningOutcome]:
        return list(self._outcomes)

    @property
    def insights(self) -> list[Insight]:
        return list(self._insights)

    @property
    def proposals(self) -> list[ImprovementProposal]:
        return list(self._proposals)

    def stats(self) -> dict[str, Any]:
        return {
            "outcomes": len(self._outcomes),
            "insights": len(self._insights),
            "proposals": len(self._proposals),
            "avg_confidence": (
                sum(o.confidence for o in self._outcomes) / len(self._outcomes)
                if self._outcomes
                else 0.0
            ),
        }
