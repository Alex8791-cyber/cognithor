"""
cognitio/predictive.py

Predictive Processing / Active Inference (inspired by Karl Friston)

Inspired by Karl Friston's Free Energy Principle:
    The brain generates a "what will happen next" prediction at every moment.
    Only PREDICTION ERRORS rise to conscious awareness.

This module:
    1. Stores an "expectation vector" after the assistant's response
    2. Compares against the actual vector when the user message arrives
    3. Prediction error → affects salience and emotional intensity:
         - Low error (< 0.25): routine, low salience
         - Medium error (0.25–0.55): mild surprise
         - High error (> 0.55): surprise → emotional boost, redirect attention

Note: No LLM calls — only cosine distance in embedding space.
     Assistant response = expectation vector, user response = reality vector.
"""

import collections
import logging
import math

logger = logging.getLogger(__name__)

# Surprise thresholds
_ROUTINE_THRESHOLD = 0.25  # Below this → routine
_SURPRISE_THRESHOLD = 0.55  # Above this → high surprise
_EMOTIONAL_BOOST_MAX = 0.35  # Max emotional_tone boost on high surprise
_MIN_HISTORY = 5  # Minimum history size for trend analysis


class PredictiveEngine:
    """
    Expectation-based surprise detection.

    Stores the assistant's response as an "expectation vector".
    Computes prediction error by comparing with the next user message.

    High error → high surprise → memory salience and emotional intensity increase.
    This gives the system a "waiting, surprised, adapting" behavioral profile.
    """

    def __init__(self) -> None:
        self._expected_embedding: list[float] | None = None
        self._last_error: float = 0.0
        self._error_history: collections.deque = collections.deque(maxlen=100)  # for trend analysis

    def update_expectation(self, assistant_embedding: list[float]) -> None:
        """
        Update the expectation vector from the assistant's response.

        The next user message is expected to be a "continuation" of this embedding.

        Parameters:
            assistant_embedding: Embedding vector of the assistant's response
        """
        self._expected_embedding = list(assistant_embedding)

    def compute_error(self, actual_embedding: list[float]) -> float:
        """
        Prediction error between expectation and reality.

        cosine_distance = 1 - cosine_similarity
        0.0 → perfect prediction (expected topic continued)
        1.0 → complete surprise (completely different topic)

        Parameters:
            actual_embedding: Embedding vector of the incoming user message

        Returns:
            float: Prediction error (0.0–1.0)
        """
        if self._expected_embedding is None:
            return 0.0

        sim = self._cosine(actual_embedding, self._expected_embedding)
        error = 1.0 - max(0.0, min(1.0, sim))

        self._last_error = error
        self._error_history.append(error)

        level = self.classify_error(error)
        if level == "high_surprise":
            logger.debug("High prediction error: %.2f — unexpected message.", error)

        return error

    def has_expectation(self) -> bool:
        """Is an expectation vector stored?"""
        return self._expected_embedding is not None

    def classify_error(self, error: float | None = None) -> str:
        """
        Classify the prediction error.

        Parameters:
            error: Error value to classify (None → last error)

        Returns:
            str: 'routine' | 'mild_surprise' | 'high_surprise'
        """
        e = error if error is not None else self._last_error
        if e < _ROUTINE_THRESHOLD:
            return "routine"
        elif e < _SURPRISE_THRESHOLD:
            return "mild_surprise"
        else:
            return "high_surprise"

    def get_emotional_boost(self) -> float:
        """
        Compute emotional_tone boost based on surprise level.

        0.0 for routine interactions, _EMOTIONAL_BOOST_MAX for high surprise.
        Linear interpolation — no sudden jumps.

        Returns:
            float: Boost in range 0.0–0.35
        """
        if self._last_error < _ROUTINE_THRESHOLD:
            return 0.0
        normalized = min(1.0, (self._last_error - _ROUTINE_THRESHOLD) / (1.0 - _ROUTINE_THRESHOLD))
        return normalized * _EMOTIONAL_BOOST_MAX

    def get_context_hint(self) -> str | None:
        """
        Surprise hint to include in the LLM context.

        Returns an explanatory note if high or medium surprise exists.

        Returns:
            str | None: Surprise note or None
        """
        level = self.classify_error()
        if level == "high_surprise":
            return (
                f"The last message brought an unexpected direction "
                f"(prediction error: {self._last_error:.2f}) "
                f"— I directed my attention to this topic."
            )
        elif level == "mild_surprise":
            return (
                f"The conversation went in a slightly different direction than I expected "
                f"(delta: {self._last_error:.2f})."
            )
        return None

    def get_average_surprise(self, last_n: int = 10) -> float:
        """
        Average surprise level over the last N interactions.

        Parameters:
            last_n: Number of recent interactions to consider

        Returns:
            float: Average prediction error
        """
        if not self._error_history:
            return 0.0
        window = list(self._error_history)[-last_n:]
        return sum(window) / len(window) if window else 0.0

    def is_trending_surprising(self) -> bool:
        """
        Are recent interactions becoming increasingly surprising?

        Trend analysis — True if the last 5 error average is higher than the first 5.

        Returns:
            bool: Is there an increasing surprise trend
        """
        if len(self._error_history) < _MIN_HISTORY * 2:
            return False
        hist = list(self._error_history)
        first_half = hist[-_MIN_HISTORY * 2 : -_MIN_HISTORY]
        second_half = hist[-_MIN_HISTORY:]
        return (sum(second_half) / _MIN_HISTORY) > (sum(first_half) / _MIN_HISTORY) + 0.1

    @property
    def last_error(self) -> float:
        """Last computed prediction error."""
        return self._last_error

    def to_dict(self) -> dict:
        """Serialize to a dict."""
        return {
            "last_error": self._last_error,
            "error_history": list(self._error_history)[-20:],  # last 20 — keep size small
            # _expected_embedding is not serialized: session-local, large vector
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PredictiveEngine":
        """Construct a PredictiveEngine from a dict."""
        pe = cls()
        pe._last_error = data.get("last_error", 0.0)
        pe._error_history = list(data.get("error_history", []))
        return pe

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-8) if na and nb else 0.0
