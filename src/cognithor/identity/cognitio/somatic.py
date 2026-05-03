"""
cognitio/somatic.py

Damasio's somatic marker hypothesis — digital body state.

SomaticState:
    - Tracks energy level (decreases under intense interaction, recovers at rest)
    - Provides somatic hints to the LLM (temperature, verbosity)
    - Integrates with ModelAdapter: fatigue → shorter, cooler responses
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class SomaticState:
    """
    Digital somatic state — energetic/normal/tired axis.

    Energy formula:
        Update:   energy -= intensity * 0.02  (min 0.1)
        Recovery: energy += elapsed_minutes * 0.01  (max 1.0)

    Classification:
        energetic: energy > 0.7
        normal:    0.4 <= energy <= 0.7
        tired:     energy < 0.4
    """

    energy_level: float = 1.0
    session_intensity_sum: float = 0.0
    interaction_count_this_session: int = 0
    last_rest_at: datetime | None = None

    def update(self, emotional_intensity: float) -> None:
        """
        Update energy after an interaction.

        Parameters:
            emotional_intensity: Emotional weight of the interaction (0.0–1.0)
        """
        # Intensity clamping
        intensity = max(0.0, min(1.0, emotional_intensity))
        self.energy_level = max(0.1, self.energy_level - intensity * 0.02)
        self.session_intensity_sum += intensity
        self.interaction_count_this_session += 1

    def recover(self, elapsed_minutes: float) -> None:
        """
        Regain energy after rest.

        Parameters:
            elapsed_minutes: Number of elapsed minutes
        """
        recovery = elapsed_minutes * 0.01
        self.energy_level = min(1.0, self.energy_level + recovery)
        self.last_rest_at = datetime.now(UTC)

    def classify(self) -> str:
        """
        Classify the current energy state.

        Returns:
            str: 'energetic' | 'normal' | 'tired'
        """
        if self.energy_level > 0.7:
            return "energetic"
        elif self.energy_level >= 0.4:
            return "normal"
        else:
            return "tired"

    def get_modifiers(self) -> dict[str, Any]:
        """
        Somatic modifiers to apply to LLM parameters.

        Tired:    temperature -0.15, verbosity "brief"
        Energetic: temperature +0.1, verbosity "detailed"
        Normal:   no change

        Returns:
            dict: {"temperature_delta": float, "verbosity": str | None}
        """
        state = self.classify()
        if state == "tired":
            return {"temperature_delta": -0.15, "verbosity": "brief"}
        elif state == "energetic":
            return {"temperature_delta": 0.1, "verbosity": "detailed"}
        else:
            return {"temperature_delta": 0.0, "verbosity": None}

    def get_context_hint(self) -> str:
        """
        Short somatic context text to include in the LLM prompt.

        Returns:
            str: Somatic state description
        """
        state = self.classify()
        energy_pct = int(self.energy_level * 100)

        hints = {
            "tired": (
                f"My energy level is low ({energy_pct}%). I will try to keep my responses concise."
            ),
            "normal": (
                f"My energy level is normal ({energy_pct}%). I am continuing in a balanced manner."
            ),
            "energetic": (
                f"My energy level is high ({energy_pct}%)."
                " I can provide detailed and thoughtful responses."
            ),
        }
        return hints[state]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {
            "energy_level": self.energy_level,
            "session_intensity_sum": self.session_intensity_sum,
            "interaction_count_this_session": self.interaction_count_this_session,
            "last_rest_at": self.last_rest_at.isoformat() if self.last_rest_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SomaticState:
        """Construct a SomaticState from a dict."""
        state = cls()
        state.energy_level = data.get("energy_level", 1.0)
        state.session_intensity_sum = data.get("session_intensity_sum", 0.0)
        state.interaction_count_this_session = data.get("interaction_count_this_session", 0)

        if data.get("last_rest_at"):
            state.last_rest_at = datetime.fromisoformat(data["last_rest_at"])

        return state
