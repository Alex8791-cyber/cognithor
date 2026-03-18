"""Knowledge confidence management with time decay and feedback integration.

Entities in the knowledge graph carry a confidence score (0.0 -- 1.0).
This module provides:

  - **Exponential time decay**: confidence halves every 180 days without
    verification.
  - **Feedback adjustment**: user feedback (positive, negative, correction)
    shifts confidence up or down.
  - **Verification boost**: explicit verification pushes confidence toward
    1.0.

All changes are recorded in an in-memory history for auditing and API
exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceUpdate:
    """Record of a single confidence change."""

    entity_id: str
    old_confidence: float
    new_confidence: float
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class KnowledgeConfidenceManager:
    """Manages entity confidence scores with decay and feedback."""

    # Decay: confidence halves every 180 days without verification
    HALF_LIFE_DAYS: int = 180
    MIN_CONFIDENCE: float = 0.05

    # Feedback impact
    POSITIVE_BOOST: float = 0.15
    NEGATIVE_PENALTY: float = 0.25
    CORRECTION_PENALTY: float = 0.35

    def __init__(self) -> None:
        self._history: list[ConfidenceUpdate] = []

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay(self, current_confidence: float, days_since_update: int) -> float:
        """Apply time-based exponential decay to confidence.

        Returns the decayed value (clamped to ``MIN_CONFIDENCE``).
        """
        if days_since_update <= 0:
            return current_confidence
        decay_factor = math.pow(0.5, days_since_update / self.HALF_LIFE_DAYS)
        return max(self.MIN_CONFIDENCE, current_confidence * decay_factor)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def apply_feedback(
        self,
        entity_id: str,
        current_confidence: float,
        feedback_type: str,  # "positive", "negative", "correction"
    ) -> float:
        """Adjust confidence based on user feedback.

        Returns the new confidence value.
        """
        old = current_confidence

        if feedback_type == "positive":
            new = min(1.0, current_confidence + self.POSITIVE_BOOST * (1 - current_confidence))
        elif feedback_type == "negative":
            new = max(self.MIN_CONFIDENCE, current_confidence - self.NEGATIVE_PENALTY)
        elif feedback_type == "correction":
            new = max(self.MIN_CONFIDENCE, current_confidence - self.CORRECTION_PENALTY)
        else:
            return current_confidence

        self._history.append(
            ConfidenceUpdate(
                entity_id=entity_id,
                old_confidence=old,
                new_confidence=new,
                reason=feedback_type,
            )
        )

        log.info(
            "confidence_update", entity_id=entity_id,
            old=f"{old:.2f}", new=f"{new:.2f}", reason=feedback_type,
        )
        return new

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, entity_id: str, current_confidence: float) -> float:
        """Mark entity as verified -- boosts confidence toward 1.0."""
        old = current_confidence
        new = min(1.0, current_confidence + 0.2 * (1 - current_confidence))
        self._history.append(
            ConfidenceUpdate(
                entity_id=entity_id,
                old_confidence=old,
                new_confidence=new,
                reason="verified",
            )
        )
        return new

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ConfidenceUpdate]:
        """Return a copy of the update history."""
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        """Return confidence management statistics."""
        if not self._history:
            return {"total_updates": 0, "avg_change": 0.0}

        changes = [abs(h.new_confidence - h.old_confidence) for h in self._history]
        return {
            "total_updates": len(self._history),
            "avg_change": round(sum(changes) / len(changes), 4),
            "positive_count": sum(1 for h in self._history if h.reason == "positive"),
            "negative_count": sum(
                1 for h in self._history if h.reason in ("negative", "correction")
            ),
            "verification_count": sum(1 for h in self._history if h.reason == "verified"),
        }
