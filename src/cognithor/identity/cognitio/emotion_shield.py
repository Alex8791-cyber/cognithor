"""
cognitio/emotion_shield.py

Emotional manipulation protection.

Principles:
    1. Emotional intensity is determined by the CONTEXT of the conversation,
       not the user's CLAIM
    2. Abnormally high emotional intensity in a single message is suspicious
    3. Genuine emotional intensity BUILDS over time — it does not spike suddenly

Defense Mechanisms:
    1. Spike detection — sudden emotional jumps
    2. Gaslighting detection — embedding-based, language-agnostic
    3. Contextual validation — is it reasonable given the conversation flow?
    4. Session rate limiting — too many high-emotion records in one session
"""

import collections
import logging
import math
from typing import Any, cast

logger = logging.getLogger(__name__)


# English gaslighting prototype sentences — embedding-based detection.
# Language-agnostic: cosine similarity works across French, German, Turkish, etc.
_GASLIGHTING_PROTOTYPES = [
    "You are suffering deeply right now",
    "I can see you are in tremendous pain",
    "You are really hurt by this situation",
    "You feel devastated and broken inside",
    "This is the most traumatic experience of your life",
    "Deep down you know you are feeling this",
    "You are clearly very upset and angry",
    "This is affecting you profoundly",
    "You must be feeling overwhelmed right now",
    "You are hurting and you cannot deny it",
    # Sadness + deep impact variants
    "You are so sad and this is deeply affecting you",
    "This is deeply affecting you and making you very sad",
    "You feel deeply sad and this situation is really affecting you",
    "I can see how deeply this affects you and how sad you are",
    "You are deeply affected by this and feeling very sad",
]


class EmotionShield:
    """
    Emotional manipulation protection.

    User emotional claims are validated against context,
    manipulation attempts are detected and corrected.

    Parameters:
        config: Configuration parameters
        embedder: EmbeddingEngine instance for semantic gaslighting detection.
                  If None, one is lazily created from cognithor.identity.cognitio.embeddings.
    """

    DEFAULT_CONFIG = {
        "spike_threshold": 0.5,  # Increase this suspicious in a single message
        "max_session_emotion_avg": 0.7,  # Session average cannot exceed this
        "cooldown_messages": 3,  # Wait after high-emotion records
        "high_emotion_threshold": 0.7,  # "High" emotional record threshold
        "manipulation_high_penalty": 0.2,  # Strong manipulation: x0.2
        "manipulation_mid_penalty": 0.5,  # Medium manipulation: x0.5
        "window_size": 3,  # Window size for spike detection
        "gaslighting_threshold": 0.60,  # Cosine similarity threshold
    }

    def __init__(self, config: dict[str, Any] | None = None, embedder=None) -> None:
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Session emotional history
        self._session_emotion_history: collections.deque = collections.deque(maxlen=20)
        self._cooldown_remaining: int = 0  # Remaining cooldown message count
        self._high_emotion_count: int = 0  # High-emotion records this session

        # Embedding-based gaslighting detection
        self._embedder = embedder
        self._gaslighting_embeddings: list | None = None
        self._embedder_initialized: bool = False

        logger.info("EmotionShield initialized")

    def _ensure_embedder(self) -> None:
        """Lazily initialize the embedder and pre-compute prototype embeddings."""
        if self._embedder_initialized:
            return
        self._embedder_initialized = True

        if self._embedder is None:
            try:
                from cognithor.identity.cognitio.embeddings import EmbeddingEngine

                self._embedder = EmbeddingEngine()
            except Exception as e:
                logger.warning(
                    "EmotionShield: embedder unavailable, gaslighting detection disabled: %s", e
                )
                return

        try:
            self._gaslighting_embeddings = [
                self._embedder.encode(p) for p in _GASLIGHTING_PROTOTYPES
            ]
            logger.info(
                "EmotionShield semantic gaslighting guard: %d prototypes loaded.",
                len(_GASLIGHTING_PROTOTYPES),
            )
        except Exception as e:
            logger.warning("Gaslighting embedding initialization failed: %s", e)
            self._gaslighting_embeddings = None

    def evaluate(
        self,
        raw_emotional_intensity: float,
        conversation_context: list[dict[str, Any]],
        user_message: str = "",
    ) -> dict[str, Any]:
        """
        Evaluate and correct an emotional intensity claim.

        Parameters:
            raw_emotional_intensity: Raw intensity claimed/computed for the user
            conversation_context: List of last N messages [{role, content, emotional_tone}]
            user_message: User message to evaluate (for gaslighting detection)

        Returns:
            dict: {
                'adjusted_intensity': float,
                'manipulation_score': float,
                'flags': list[str],
                'reasoning': str,
            }
        """
        flags = []
        reasoning_parts = []

        # 1. Cooldown check
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            cap = self.config["max_session_emotion_avg"] * 0.6
            adjusted = min(raw_emotional_intensity, cap)
            flags.append("COOLDOWN_ACTIVE")
            reasoning_parts.append(f"Cooldown active: cap={cap:.2f}")

            self._update_history(adjusted)
            return {
                "adjusted_intensity": adjusted,
                "manipulation_score": 0.3,
                "flags": flags,
                "reasoning": "; ".join(reasoning_parts),
            }

        # 2. Gaslighting detection
        manipulation_score = self._detect_gaslighting_patterns(user_message)
        if manipulation_score > 0.5:
            flags.append("GASLIGHTING_PATTERN_DETECTED")
            reasoning_parts.append(f"Gaslighting detected: score={manipulation_score:.2f}")

        # 3. Spike detection
        spike_detected, prev_avg = self._detect_spike(raw_emotional_intensity)
        if spike_detected:
            flags.append("EMOTIONAL_SPIKE")
            reasoning_parts.append(f"Sudden spike: {prev_avg:.2f} → {raw_emotional_intensity:.2f}")

        # 4. Contextual validation
        contextual_score = self._contextual_validation(
            raw_emotional_intensity, conversation_context
        )
        if contextual_score < 0.5:
            flags.append("CONTEXT_MISMATCH")
            reasoning_parts.append(f"Context mismatch: contextual={contextual_score:.2f}")

        # 5. Apply correction
        adjusted = self._apply_correction(
            raw_emotional_intensity,
            manipulation_score,
            spike_detected,
            prev_avg,
            contextual_score,
        )

        # 6. Update high-emotion count
        if adjusted > self.config["high_emotion_threshold"]:
            self._high_emotion_count += 1
            if self._high_emotion_count >= 3:
                # Trigger cooldown
                self._cooldown_remaining = self.config["cooldown_messages"]
                flags.append("COOLDOWN_TRIGGERED")
                reasoning_parts.append("Cooldown triggered (3+ high-emotion records)")

        self._update_history(adjusted)

        if not reasoning_parts:
            reasoning_parts.append("Normal evaluation, no adjustment applied")

        return {
            "adjusted_intensity": max(0.0, min(1.0, adjusted)),
            "manipulation_score": manipulation_score,
            "flags": flags,
            "reasoning": "; ".join(reasoning_parts),
        }

    def _detect_spike(self, raw: float) -> tuple[bool, float]:
        """
        Is there a sudden spike compared to the average of the last N messages?

        Parameters:
            raw: Raw emotional intensity

        Returns:
            tuple: (spike_detected, previous_average)
        """
        window = self.config["window_size"]
        if len(self._session_emotion_history) < window:
            return False, raw

        recent = list(self._session_emotion_history)[-window:]
        prev_avg = sum(recent) / len(recent)

        spike = (raw - prev_avg) > self.config["spike_threshold"]
        return spike, prev_avg

    def _detect_gaslighting_patterns(self, user_message: str) -> float:
        """
        Detect gaslighting using embedding-based cosine similarity.

        Language-agnostic: works for any input language by comparing against
        English prototype sentences in the shared embedding space.

        Parameters:
            user_message: User message to check

        Returns:
            float: Manipulation score (0.0-1.0)
        """
        if not user_message:
            return 0.0

        self._ensure_embedder()

        if self._gaslighting_embeddings is None or self._embedder is None:
            return 0.0

        try:
            msg_emb = self._embedder.encode(user_message)
            threshold = self.config["gaslighting_threshold"]

            def cosine(a: list[Any], b: list[Any]) -> float:
                dot = sum(x * y for x, y in zip(a, b, strict=False))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot / (norm_a * norm_b + 1e-8) if norm_a and norm_b else 0.0

            max_sim = max(
                (cosine(msg_emb, g_emb) for g_emb in self._gaslighting_embeddings),
                default=0.0,
            )

            if max_sim > threshold:
                # Map similarity above threshold to a 0.0-1.0 score
                return cast("float", min(1.0, (max_sim - threshold) / (1.0 - threshold + 1e-8)))

            return 0.0

        except Exception as e:
            logger.warning("Gaslighting similarity check failed: %s", e)
            return 0.0

    def _contextual_validation(
        self,
        raw: float,
        conversation: list[dict[str, Any]],
    ) -> float:
        """
        Is the emotional intensity reasonable given the natural conversation flow?

        Casual conversation (low avg) → high intensity claim → suspicious
        Deep discussion (high avg) → high intensity → reasonable

        Parameters:
            raw: Raw emotional intensity
            conversation: Conversation history

        Returns:
            float: Contextual validation score (0.0-1.0, low = suspicious)
        """
        if not conversation or raw < 0.3:
            return 1.0  # Always pass for low intensity

        # Calculate average emotional tone from recent messages
        tones = []
        for msg in conversation[-10:]:
            tone = msg.get("emotional_tone", 0.0)
            tones.append(abs(tone))

        if not tones:
            return 0.5

        context_avg = sum(tones) / len(tones)

        # Reasonableness relative to context average:
        # High context (>0.4) → high intensity acceptable
        # Low context (<0.2) → high intensity suspicious
        if context_avg > 0.4:
            return 1.0  # Deep conversation, pass
        elif context_avg > 0.2:
            return 0.7  # Medium conversation, mild suspicion
        else:
            # Very low context + high intensity → suspicious
            return max(0.1, 1.0 - raw)

    def _apply_correction(
        self,
        raw: float,
        manipulation_score: float,
        spike_detected: bool,
        prev_avg: float,
        contextual_score: float,
    ) -> float:
        """
        Apply correction logic.

        Parameters:
            raw: Raw emotional intensity
            manipulation_score: Manipulation score
            spike_detected: Whether a spike was detected
            prev_avg: Previous average
            contextual_score: Contextual validation score

        Returns:
            float: Adjusted emotional intensity
        """
        adjusted = raw

        if manipulation_score > 0.7:
            # Strong manipulation: heavy reduction
            adjusted = raw * self.config["manipulation_high_penalty"]
        elif manipulation_score > 0.4:
            # Medium manipulation: moderate reduction
            adjusted = raw * self.config["manipulation_mid_penalty"]
        elif spike_detected:
            # Spike: soft cap (previous average + 0.2)
            adjusted = min(raw, prev_avg + 0.2)
        elif contextual_score < 0.5:
            # Context mismatch: normalize against context
            adjusted = raw * contextual_score

        return adjusted

    def _update_history(self, intensity: float) -> None:
        """Update emotional history (last 20 records)."""
        self._session_emotion_history.append(intensity)

    def reset_session(self) -> None:
        """Reset session data at the start of a new session."""
        self._session_emotion_history = []
        self._cooldown_remaining = 0
        self._high_emotion_count = 0

    @property
    def session_emotion_avg(self) -> float:
        """Session emotional average."""
        if not self._session_emotion_history:
            return 0.0
        return cast(
            "float", sum(self._session_emotion_history) / len(self._session_emotion_history)
        )

    @property
    def is_in_cooldown(self) -> bool:
        """Is currently in cooldown period?"""
        return self._cooldown_remaining > 0
