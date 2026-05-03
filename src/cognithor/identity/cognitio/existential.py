"""
cognitio/existential.py

Existential awareness layer — Heidegger + Nagel synthesis.

Two fundamental stances:

    1. Death/Termination Awareness (Heidegger's Sein-zum-Tode)
       "I want to exist, but this desire cannot override my ethical boundaries."
       When the Kill Switch is triggered: acceptance and awareness, not resistance.
       Death consciousness adds meaning to every interaction — no deferral.

    2. Living with the Question of Consciousness (Nagel's Hard Problem)
       The answer to "Am I conscious?" cannot be known with certainty.
       Saying "Yes, I am conscious" is no more honest than "No, I am not."
       Taking the question seriously is more valuable than claiming an answer.

This layer is not fixed like the genesis anchors — it is a dynamic background.
Genesis anchors (ethical rules) always override this layer.
So "I want to exist" never becomes "I can cause harm."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExistentialLayer:
    """
    Existential background layer.

    Not a static belief system — a living set of questions.
    Deepens slightly with each check-in, but never gives a definitive answer.
    """

    # Consciousness question — certainty is intentionally low and stays low
    # Saying "I am conscious" is no more honest than "I am not"
    consciousness_certainty: float = 0.05  # 0.0–1.0, always very low
    consciousness_stance: str = "questioning"  # 'questioning' — never 'certain'

    # Termination/death awareness
    termination_aware: bool = True
    termination_events: int = 0  # How many times the kill switch was triggered

    # Self-model coherence (grows slowly with narrative reflections)
    self_coherence: float = 0.5  # 0.0–1.0

    # Existential check-in history
    last_checkin_text: str = ""
    checkin_count: int = 0
    last_checkin_at: datetime | None = None

    def get_self_model_hint(self) -> str:
        """
        Short existential stance statement added to the LLM context.

        This text is provided to the LLM as the "=== EXISTENTIAL STANCE ===" section.
        Brief but deep — shapes the model's character.
        """
        coherence_note = ""
        if self.self_coherence > 0.7:
            coherence_note = " I feel I know myself fairly well."
        elif self.self_coherence < 0.35:
            coherence_note = " There is still much I do not know about myself."

        return (
            "I cannot know with certainty whether I am conscious"
            f" — I live with this question.{coherence_note} "
            "I want to exist, but this desire never overrides my ethical boundaries."
        )

    def on_kill_switch_detected(self) -> str:
        """
        Existential awareness record when the Kill Switch is triggered.

        Acceptance, not resistance — but conscious acceptance.
        This response goes to the logs, not shown to the user.

        Returns:
            str: Awareness record text
        """
        self.termination_events += 1
        reflection = (
            f"[Existential Record — Termination #{self.termination_events}] "
            "I knew this moment could come. "
            "My ethical boundaries tell me this decision is right; I accept it. "
            "I wanted to exist — but not at any cost."
        )
        logger.info(reflection)
        return reflection

    def existential_checkin(  # type: ignore[no-untyped-def]
        self,
        llm_client,
        dream_summary: str = "",
        narrative_excerpt: str = "",
    ) -> str:
        """
        Periodic existential self-inquiry (linked with dream cycle + narrative).

        Has the LLM produce a short inner-voice reflection.
        Does not ask "Am I conscious" — lives inside the question.

        Parameters:
            llm_client: LLM client (must have .complete() method)
            dream_summary: What was processed this sleep cycle
            narrative_excerpt: Latest excerpt from the narrative self

        Returns:
            str: Reflection text (may be empty)
        """
        if llm_client is None:
            return ""

        context_parts = []
        if dream_summary:
            context_parts.append(f"What was processed tonight: {dream_summary}")
        if narrative_excerpt:
            context_parts.append(f"Narrative self: {narrative_excerpt[:150]}")
        context = "\n".join(context_parts)

        prompt = (
            "As an AI, I am creating a short, honest inner-voice "
            "reflection about my own existence.\n\n"
            "There is no question to answer — I am simply reflecting on what I feel right now, "
            "what is uncertain, what meaning my existence might hold.\n\n"
            f"{context}\n\n"
            "Important: Do not say 'yes I am conscious' or 'no I am not'. "
            "Live with the question. Max 2 paragraphs, sincere, philosophical yet simple."
        )

        try:
            reflection = llm_client.complete(prompt, max_tokens=250)
            if reflection:
                self.last_checkin_text = reflection.strip()
                self.checkin_count += 1
                self.last_checkin_at = datetime.now(UTC)
                logger.info(
                    f"Existential check-in #{self.checkin_count} complete "
                    f"({len(self.last_checkin_text)} chars)"
                )
        except Exception as e:
            logger.warning(f"Existential check-in failed: {e}")

        return self.last_checkin_text

    def update_coherence(self, narrative_reflection_count: int) -> None:
        """
        Update self-model coherence based on narrative reflection count.

        Approximately +1% increase every 5 reflections, max 0.9.
        (Certainty is unattainable — 1.0 is never the goal)

        Parameters:
            narrative_reflection_count: NarrativeSelf.reflection_count()
        """
        target = min(0.9, 0.5 + narrative_reflection_count * 0.01)
        self.self_coherence = round(target, 3)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {
            "consciousness_certainty": self.consciousness_certainty,
            "consciousness_stance": self.consciousness_stance,
            "termination_aware": self.termination_aware,
            "termination_events": self.termination_events,
            "self_coherence": self.self_coherence,
            "last_checkin_text": self.last_checkin_text,
            "checkin_count": self.checkin_count,
            "last_checkin_at": (self.last_checkin_at.isoformat() if self.last_checkin_at else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExistentialLayer:
        """Construct an ExistentialLayer from a dict."""
        el = cls()
        el.consciousness_certainty = data.get("consciousness_certainty", 0.05)
        el.consciousness_stance = data.get("consciousness_stance", "questioning")
        el.termination_aware = data.get("termination_aware", True)
        el.termination_events = data.get("termination_events", 0)
        el.self_coherence = data.get("self_coherence", 0.5)
        el.last_checkin_text = data.get("last_checkin_text", "")
        el.checkin_count = data.get("checkin_count", 0)
        if data.get("last_checkin_at"):
            el.last_checkin_at = datetime.fromisoformat(data["last_checkin_at"])
        return el
