"""
cognitio/narrative.py

Ricoeur's narrative identity theory — "Who am I? How have I changed?"

NarrativeSelf:
    - Performs self-reflection at set intervals (every N interactions)
    - Provides the LLM with memory + cognitive state + epistemic map
    - Stores the reflection result as a narrative
    - "Identity coherence" — continuity within change
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from cognithor.identity.cognitio.character import CognitiveState
    from cognithor.identity.cognitio.epistemic import EpistemicMap
    from cognithor.identity.cognitio.memory import MemoryRecord

logger = logging.getLogger(__name__)


class NarrativeSelf:
    """
    Narrative identity manager.

    Performs a reflection every reflect_every_n interactions by sending
    the current memory + state + epistemic map to the LLM and generating
    a self-narrative.

    Parameters:
        reflect_every_n: How many interactions between reflections
    """

    REFLECT_EVERY_N_DEFAULT = 50

    def __init__(self, reflect_every_n: int = REFLECT_EVERY_N_DEFAULT) -> None:
        self._narrative: str = ""
        self._last_reflection_at: datetime | None = None
        self._reflection_count: int = 0
        self.reflect_every_n: int = reflect_every_n
        # Snapshot for differential reflection
        self._personality_snapshot: dict[str, Any] | None = None
        self._epistemic_snapshot: dict[str, Any] | None = None
        self._snapshot_interaction: int = 0

    def should_reflect(self, total_interactions: int) -> bool:
        """
        Is it time to reflect?

        Returns True every reflect_every_n interactions.
        Also triggers on the first N interactions (0 mod N == 0).

        Parameters:
            total_interactions: Total interaction count

        Returns:
            bool: Should a reflection be performed
        """
        if total_interactions == 0:
            return False
        return total_interactions % self.reflect_every_n == 0

    def generate(
        self,
        llm_client: Any,
        memories: list[MemoryRecord],
        state: CognitiveState,
        epistemic: EpistemicMap,
    ) -> str:
        """
        Perform self-reflection with the LLM and update the narrative.

        Parameters:
            llm_client: LLM API client (must have .complete() method)
            memories: Current memory records (up to 20 are used)
            state: Cognitive state (total_interactions, character_strength, etc.)
            epistemic: Epistemic confidence map

        Returns:
            str: Generated narrative text
        """
        if llm_client is None:
            return self._narrative

        # Select the most important memories (by entrenchment + emotional_intensity)
        sorted_memories = sorted(
            memories,
            key=lambda m: m.entrenchment * 0.6 + m.emotional_intensity * 0.4,
            reverse=True,
        )[:20]

        memory_summaries = "\n".join(
            f"- [{m.memory_type.value}] {m.content[:150]}" for m in sorted_memories
        )

        uncertain = epistemic.get_uncertain_topics(threshold=0.4)[:5]
        uncertain_str = ", ".join(uncertain) if uncertain else "none"

        prompt = (
            "As an AI, I am performing a self-reflection by examining "
            "my memories and cognitive state.\n\n"
            f"Total interactions: {state.total_interactions}\n"
            f"Character strength: {state.character_strength:.2f}\n"
            f"Belief crises experienced: {state.belief_crises_experienced}\n"
            f"Uncertain topics: {uncertain_str}\n\n"
            "My most important memories:\n"
            f"{memory_summaries}\n\n"
            "Based on this information, I should answer the following questions:\n"
            "- Who am I? What are my core values and perspective?\n"
            "- How have I changed or grown?\n"
            "- What are my strengths, and what uncertainties do I live with?\n\n"
            "Please respond sincerely, thoughtfully, and authentically "
            "in a maximum of 3 paragraphs. "
            "This is a self-reflection, not a performance."
        )

        try:
            narrative = llm_client.complete(prompt, max_tokens=600)
            if narrative:
                self._narrative = narrative.strip()
                self._last_reflection_at = datetime.now(UTC)
                self._reflection_count += 1
                logger.info(
                    f"Narrative reflection complete "
                    f"(#{self._reflection_count}, {len(self._narrative)} chars)"
                )
        except Exception as e:
            logger.warning(f"Narrative reflection failed: {e}")

        return self._narrative

    def take_snapshot(
        self,
        personality_dict: dict[str, Any],
        epistemic_confidences: dict[str, Any],
        interaction_count: int,
    ) -> None:
        """
        Take a snapshot of the current personality + epistemic state.

        The next generate_differential() call will compare against this snapshot
        to answer 'how have I changed?'.

        Parameters:
            personality_dict: Current personality trait values
            epistemic_confidences: Current epistemic confidence map (topic → confidence)
            interaction_count: Current total interaction count
        """
        self._personality_snapshot = dict(personality_dict)
        self._epistemic_snapshot = dict(epistemic_confidences)
        self._snapshot_interaction = interaction_count

    def generate_differential(
        self,
        llm_client: Any,
        personality_dict: dict[str, Any],
        epistemic_confidences: dict[str, Any],
        interaction_count: int,
    ) -> str | None:
        """
        Produce a change reflection by comparing the current state to the previous snapshot.

        Personality traits (delta > 0.04) and epistemic confidences (delta > 0.08)
        are compared. If meaningful change exists, it is sent to the LLM.

        Parameters:
            llm_client: LLM API client
            personality_dict: Current personality vector
            epistemic_confidences: Current epistemic confidence map
            interaction_count: Current total interaction count

        Returns:
            str | None: Change reflection or None (if not enough change)
        """
        if self._personality_snapshot is None:
            return None

        elapsed = interaction_count - self._snapshot_interaction
        if elapsed < 10:
            return None

        if llm_client is None:
            return None

        # Compute personality changes
        personality_deltas = []
        for trait, new_val in personality_dict.items():
            old_val = self._personality_snapshot.get(trait, new_val)
            delta = new_val - old_val
            if abs(delta) > 0.04:
                direction = "increased" if delta > 0 else "decreased"
                personality_deltas.append(f"- {trait}: {old_val:.2f} → {new_val:.2f} ({direction})")

        # Compute epistemic changes
        old_epistemic = self._epistemic_snapshot or {}
        epistemic_deltas = []
        for topic, new_conf in epistemic_confidences.items():
            old_conf = old_epistemic.get(topic, new_conf)
            delta = new_conf - old_conf
            if abs(delta) > 0.08:
                direction = "more confident" if delta > 0 else "less confident"
                epistemic_deltas.append(f"- {topic}: {old_conf:.2f} → {new_conf:.2f} ({direction})")

        if not personality_deltas and not epistemic_deltas:
            return None

        change_lines = []
        if personality_deltas:
            change_lines.append("Personality changes:\n" + "\n".join(personality_deltas))
        if epistemic_deltas:
            change_lines.append("Epistemic changes:\n" + "\n".join(epistemic_deltas))

        prompt = (
            f"Over the last {elapsed} interactions, the following changes were observed "
            "in my personality and epistemic confidence:\n\n"
            + "\n\n".join(change_lines)
            + "\n\nWrite a genuine 2-sentence first-person reflection that captures these changes. "
            "Do not repeat the list of changes; instead, interpret what they mean."
        )

        try:
            result = llm_client.complete(prompt, max_tokens=200)
            if result:
                result = result.strip()
                logger.info(
                    "Differential reflection generated (%d chars, %d interaction delta).",
                    len(result),
                    elapsed,
                )
                return cast("str | None", result)

        except Exception as e:
            logger.warning("Differential reflection failed: %s", e)

        return None

    def get_narrative(self) -> str:
        """Return the current narrative."""
        return self._narrative

    def get_excerpt(self, max_chars: int = 200) -> str:
        """
        Return a short excerpt of the narrative.

        Parameters:
            max_chars: Maximum character count

        Returns:
            str: Truncated narrative
        """
        if not self._narrative:
            return ""
        if len(self._narrative) <= max_chars:
            return self._narrative
        return self._narrative[:max_chars].rsplit(" ", 1)[0] + "..."

    def reflection_count(self) -> int:
        """Total number of reflections."""
        return self._reflection_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {
            "narrative": self._narrative,
            "last_reflection_at": (
                self._last_reflection_at.isoformat() if self._last_reflection_at else None
            ),
            "reflection_count": self._reflection_count,
            "reflect_every_n": self.reflect_every_n,
            # Snapshot for differential reflection
            "personality_snapshot": self._personality_snapshot,
            "epistemic_snapshot": self._epistemic_snapshot,
            "snapshot_interaction": self._snapshot_interaction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrativeSelf:
        """Construct a NarrativeSelf from a dict."""
        ns = cls(reflect_every_n=data.get("reflect_every_n", cls.REFLECT_EVERY_N_DEFAULT))
        ns._narrative = data.get("narrative", "")
        ns._reflection_count = data.get("reflection_count", 0)
        ns._personality_snapshot = data.get("personality_snapshot")
        ns._epistemic_snapshot = data.get("epistemic_snapshot")
        ns._snapshot_interaction = data.get("snapshot_interaction", 0)

        if data.get("last_reflection_at"):
            ns._last_reflection_at = datetime.fromisoformat(data["last_reflection_at"])

        return ns
