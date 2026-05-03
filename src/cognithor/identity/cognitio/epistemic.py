"""
cognitio/epistemic.py

Wittgenstein's epistemic certainty concept — a "how confident am I" score per topic.

EpistemicMap:
    - Maintains a 0.0–1.0 confidence score per topic
    - Updated based on memory addition/contradiction outcomes
    - Includes uncertain topics in the LLM context
    - "Knowing what I know" capacity — intellectual humility
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognithor.identity.cognitio.memory import MemoryRecord


class EpistemicMap:
    """
    Per-topic epistemic confidence map.

    Outcome types and their effects:
        'added':        +0.05  — new information, confidence increase
        'reinforced':   +0.08  — reinforcement, more confident
        'contradicted': -0.10  — conflict, confidence erosion
        'ambivalent':   -0.03  — ambiguity, mild erosion

    Parameters:
        default_confidence: Starting value for unknown topics
    """

    # Outcome → delta mapping
    OUTCOME_DELTAS: dict[str, float] = {
        "added": 0.05,
        "reinforced": 0.08,
        "contradicted": -0.10,
        "ambivalent": -0.03,
    }

    UNCERTAIN_THRESHOLD = 0.35  # Below this → uncertain
    CONFIDENT_THRESHOLD = 0.75  # Above this → confident

    def __init__(self, default_confidence: float = 0.5) -> None:
        self._confidence: dict[str, float] = {}
        self._evidence_count: dict[str, int] = {}
        self._default_confidence = default_confidence

    def update(self, topic: str, outcome: str) -> None:
        """
        Update the confidence score for a topic.

        Parameters:
            topic: Topic name (normalized, lowercase)
            outcome: 'added' | 'reinforced' | 'contradicted' | 'ambivalent'
        """
        topic = topic.lower().strip()
        if not topic:
            return

        current = self._confidence.get(topic, self._default_confidence)
        delta = self.OUTCOME_DELTAS.get(outcome, 0.0)

        new_score = max(0.0, min(1.0, current + delta))
        self._confidence[topic] = new_score
        self._evidence_count[topic] = self._evidence_count.get(topic, 0) + 1

    def update_from_memory(self, memory: MemoryRecord, outcome: str) -> None:
        """
        Update the confidence map based on a memory record's tags.

        Parameters:
            memory: Memory record to process
            outcome: 'added' | 'reinforced' | 'contradicted' | 'ambivalent'
        """
        # Use tags as topics
        for tag in memory.tags:
            self.update(tag, outcome)

        # Also add memory_type as a topic
        self.update(memory.memory_type.value, outcome)

    def get_confidence(self, topic: str) -> float:
        """
        Confidence score for a specific topic.

        Parameters:
            topic: Topic name

        Returns:
            float: Confidence score (0.0–1.0, unknown → default)
        """
        return self._confidence.get(topic.lower().strip(), self._default_confidence)

    def get_uncertain_topics(self, threshold: float | None = None) -> list[str]:
        """
        List topics with low confidence (uncertain).

        Parameters:
            threshold: Uncertainty threshold (None → UNCERTAIN_THRESHOLD)

        Returns:
            list[str]: Uncertain topics sorted by confidence (lowest first)
        """
        limit = threshold if threshold is not None else self.UNCERTAIN_THRESHOLD
        uncertain = [(topic, score) for topic, score in self._confidence.items() if score <= limit]
        return [t for t, _ in sorted(uncertain, key=lambda x: x[1])]

    def get_confident_topics(self, threshold: float | None = None) -> list[str]:
        """
        List topics with high confidence.

        Parameters:
            threshold: Confidence threshold (None → CONFIDENT_THRESHOLD)

        Returns:
            list[str]: Confident topics sorted by confidence (highest first)
        """
        limit = threshold if threshold is not None else self.CONFIDENT_THRESHOLD
        confident = [(topic, score) for topic, score in self._confidence.items() if score >= limit]
        return [t for t, _ in sorted(confident, key=lambda x: x[1], reverse=True)]

    def get_summary(self, max_topics: int = 5) -> str:
        """
        Epistemic status summary to include in the LLM context.

        Only uncertain topics are listed. Returns empty string if none.

        Parameters:
            max_topics: Maximum number of topics to list

        Returns:
            str: Summary text or empty string
        """
        uncertain = self.get_uncertain_topics()[:max_topics]
        if not uncertain:
            return ""

        lines = ["I am not certain about the following topics:"]
        for topic in uncertain:
            score = self._confidence[topic]
            lines.append(f"  - {topic} (confidence: {score:.2f})")

        return "\n".join(lines)

    def topic_count(self) -> int:
        """Total number of topics."""
        return len(self._confidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {
            "confidence": dict(self._confidence),
            "evidence_count": dict(self._evidence_count),
            "default_confidence": self._default_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpistemicMap:
        """Construct an EpistemicMap from a dict."""
        em = cls(default_confidence=data.get("default_confidence", 0.5))
        em._confidence = dict(data.get("confidence", {}))
        em._evidence_count = dict(data.get("evidence_count", {}))
        return em
