"""Reward-Berechnung fuer Causal Learning.

Composite-Score aus mehreren Dimensionen:
  success * 0.4 + (1 - error_ratio) * 0.2 + efficiency * 0.2 + speed * 0.2

Komponenten:
  - success: Reflector-Score (0-1)
  - error_ratio: failed_tools / total_tools
  - efficiency: unique_tools / total_tool_calls (weniger Wiederholung = besser)
  - speed: 1.0 - min(duration / max_duration, 1.0)
"""

from __future__ import annotations

from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class RewardCalculator:
    """Berechnet zusammengesetzte Reward-Scores."""

    # Gewichte fuer die einzelnen Komponenten
    W_SUCCESS = 0.4
    W_ERROR = 0.2
    W_EFFICIENCY = 0.2
    W_SPEED = 0.2

    # Maximale Dauer bevor speed=0 wird (Sekunden)
    DEFAULT_MAX_DURATION = 300.0

    def __init__(self, max_duration: float = DEFAULT_MAX_DURATION) -> None:
        self._max_duration = max_duration

    def calculate_reward(
        self,
        success_score: float = 0.0,
        total_tools: int = 0,
        failed_tools: int = 0,
        unique_tools: int = 0,
        total_tool_calls: int = 0,
        duration_seconds: float = 0.0,
    ) -> float:
        """Berechnet den zusammengesetzten Reward.

        Args:
            success_score: Reflector-Score (0-1).
            total_tools: Gesamtzahl Tool-Aufrufe.
            failed_tools: Anzahl fehlgeschlagener Tool-Aufrufe.
            unique_tools: Anzahl unterschiedlicher Tools.
            total_tool_calls: Gesamtzahl Tool-Aufrufe (kann = total_tools sein).
            duration_seconds: Dauer der Session in Sekunden.

        Returns:
            Composite Reward Score (0-1).
        """
        # Clamp success to 0-1
        success = max(0.0, min(1.0, success_score))

        # Error ratio: 0 = alle erfolgreich, 1 = alle fehlgeschlagen
        if total_tools > 0:
            error_ratio = failed_tools / total_tools
        else:
            error_ratio = 0.0
        error_component = 1.0 - error_ratio

        # Efficiency: unique_tools / total_calls
        # Hoher Wert = wenig Wiederholung = gut
        if total_tool_calls > 0:
            efficiency = unique_tools / total_tool_calls
        else:
            efficiency = 1.0  # Keine Tool-Calls = kein Problem
        efficiency = max(0.0, min(1.0, efficiency))

        # Speed: schneller = besser
        if self._max_duration > 0:
            speed = 1.0 - min(duration_seconds / self._max_duration, 1.0)
        else:
            speed = 1.0

        reward = (
            self.W_SUCCESS * success
            + self.W_ERROR * error_component
            + self.W_EFFICIENCY * efficiency
            + self.W_SPEED * speed
        )

        return max(0.0, min(1.0, reward))

    def calculate_from_context(self, context: dict[str, Any]) -> float:
        """Berechnet Reward aus einem Session-Kontext-Dict.

        Erwartete Keys:
            success_score, total_tools, failed_tools,
            unique_tools, total_tool_calls, duration_seconds
        """
        return self.calculate_reward(
            success_score=context.get("success_score", 0.0),
            total_tools=context.get("total_tools", 0),
            failed_tools=context.get("failed_tools", 0),
            unique_tools=context.get("unique_tools", 0),
            total_tool_calls=context.get("total_tool_calls", 0),
            duration_seconds=context.get("duration_seconds", 0.0),
        )
