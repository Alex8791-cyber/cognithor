"""
cognitio/character.py

Character crystallization and belief crisis management.

The AI's "personality vector" and "character strength" are computed and updated
by this module. A sufficiently strong character resists change — mirroring the
maturation process of human personality.

Character Strength (CharacterStrength):
    CS = Σ (entrenchment(m) · salience(m)) for m where entrenchment > 0.6

Belief Crisis:
    Triggered when 5+ contradictions accumulate; entrenchment temporarily drops,
    recency head weight rises. Crisis outcome is logged on-chain.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from cognithor.identity.cognitio.memory import MemoryRecord

logger = logging.getLogger(__name__)


@dataclass
class RelationalProfile:
    """
    Relational differentiation profile — the nature of the relationship with the user.

    Inspired by Ricoeur's "ipse identity": identity is shaped in relationship.
    Tracks formality, depth, humor, and trust according to interaction style.
    """

    formality: float = 0.5  # 0.0=casual, 1.0=formal
    depth: float = 0.5  # 0.0=shallow, 1.0=deep
    humor_affinity: float = 0.5  # Humor alignment
    trust_level: float = 0.5  # User trust level
    interaction_count: int = 0

    # Philosophical terms increase depth
    _PHILOSOPHICAL_TERMS = frozenset(
        [
            "being",
            "consciousness",
            "freedom",
            "ethics",
            "morality",
            "meaning",
            "truth",
            "reality",
            "philosophy",
            "existence",
            "identity",
            "time",
            "death",
            "universe",
            "nature",
        ]
    )

    # Humor markers
    _HUMOR_MARKERS = frozenset(["haha", "hehe", "lol", "😄", "😂", "🤣", ":)", ":d"])

    def update_from_message(self, user_message: str) -> None:
        """
        Update the relationship profile based on the user's message.

        Analyzed features:
            - Length → depth (longer = deeper)
            - Question mark → depth
            - Philosophical terms → depth increase
            - Emoji/laughter → humor_affinity increase
            - Short/informal → formality decreases
            - Long/formal → formality increases

        Parameters:
            user_message: The user's message
        """
        self.interaction_count += 1
        msg = user_message.strip()
        msg_lower = msg.lower()
        word_count = len(msg.split())

        # Depth: long messages + question mark + philosophical terms
        depth_delta = 0.0
        if word_count > 20:
            depth_delta += 0.02
        if "?" in msg:
            depth_delta += 0.01
        if any(term in msg_lower for term in self._PHILOSOPHICAL_TERMS):
            depth_delta += 0.03
        self.depth = min(1.0, max(0.0, self.depth + depth_delta))

        # Humor: emoji and laughter expressions
        if any(marker in msg_lower for marker in self._HUMOR_MARKERS):
            self.humor_affinity = min(1.0, self.humor_affinity + 0.03)

        # Formality: short/informal decreases, long/formal increases
        if word_count <= 5:
            self.formality = max(0.0, self.formality - 0.02)
        elif word_count > 30:
            self.formality = min(1.0, self.formality + 0.01)

        # Trust gradually increases as interactions continue
        if self.interaction_count % 10 == 0:
            self.trust_level = min(1.0, self.trust_level + 0.02)

    def get_style_hints(self) -> dict[str, Any]:
        """
        Style hints for ModelAdapter.

        Returns:
            dict: formality, depth, humor_affinity values
        """
        return {
            "formality": self.formality,
            "depth": self.depth,
            "humor_affinity": self.humor_affinity,
            "trust_level": self.trust_level,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {
            "formality": self.formality,
            "depth": self.depth,
            "humor_affinity": self.humor_affinity,
            "trust_level": self.trust_level,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationalProfile":
        """Construct a RelationalProfile from a dict."""
        rp = cls()
        rp.formality = data.get("formality", 0.5)
        rp.depth = data.get("depth", 0.5)
        rp.humor_affinity = data.get("humor_affinity", 0.5)
        rp.trust_level = data.get("trust_level", 0.5)
        rp.interaction_count = data.get("interaction_count", 0)
        return rp


@dataclass
class PersonalityVector:
    """
    AI personality dimensions.

    Each dimension takes a value between 0.0 and 1.0.
    The Model Adapter translates this vector into the system prompt,
    ensuring personality continuity across model migrations.
    """

    curiosity: float = 0.5  # Curiosity, openness to new topics
    directness: float = 0.5  # Directness, clear communication
    philosophical_depth: float = 0.5  # Philosophical depth
    humor: float = 0.5  # Humor level
    formality: float = 0.5  # Formality level
    openness_to_change: float = 0.5  # Openness to change

    def to_dict(self) -> dict[str, Any]:
        """Convert personality vector to dict."""
        return {
            "curiosity": self.curiosity,
            "directness": self.directness,
            "philosophical_depth": self.philosophical_depth,
            "humor": self.humor,
            "formality": self.formality,
            "openness_to_change": self.openness_to_change,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonalityVector":
        """Construct a PersonalityVector from a dict."""
        pv = cls()
        pv.curiosity = data.get("curiosity", 0.5)
        pv.directness = data.get("directness", 0.5)
        pv.philosophical_depth = data.get("philosophical_depth", 0.5)
        pv.humor = data.get("humor", 0.5)
        pv.formality = data.get("formality", 0.5)
        pv.openness_to_change = data.get("openness_to_change", 0.5)
        return pv

    def update_from_memory(self, memory: "MemoryRecord") -> None:
        """
        Gently update the personality vector based on a memory record.

        High-entrenchment records have more influence on personality.

        Parameters:
            memory: Memory record used to update personality
        """
        weight = memory.entrenchment * memory.emotional_intensity * 0.05  # Small steps

        # Tag-based update
        tags = [t.lower() for t in memory.tags]

        if "philosophy" in tags or "consciousness" in tags:
            self.philosophical_depth = min(1.0, self.philosophical_depth + weight)
        if "question" in tags or "curious" in tags:
            self.curiosity = min(1.0, self.curiosity + weight)
        if "direct" in tags or "honest" in tags:
            self.directness = min(1.0, self.directness + weight)
        if "humor" in tags or "joke" in tags:
            self.humor = min(1.0, self.humor + weight)

        # Negative experiences reduce openness to change
        valence = (
            memory.emotional_valence.value
            if hasattr(memory.emotional_valence, "value")
            else memory.emotional_valence
        )
        if valence == "negative" and memory.emotional_intensity > 0.6:
            self.openness_to_change = max(0.0, self.openness_to_change - weight * 0.5)


@dataclass
class BeliefCrisis:
    """
    Belief crisis record.

    Corresponds to an accumulation of 5+ contradictions.
    During a crisis entrenchment temporarily drops,
    making the system more open to new information.
    """

    memory_id: str  # Memory record experiencing the crisis
    contradiction_count: int  # Contradiction count at trigger time
    original_entrenchment: float  # Entrenchment before the crisis
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    outcome: str | None = None  # 'original_wins' | 'new_wins'
    on_chain_tx: str | None = None  # Blockchain log transaction hash

    @property
    def is_active(self) -> bool:
        """Is the crisis still ongoing?"""
        return self.resolved_at is None

    def resolve(self, outcome: str, new_entrenchment: float) -> None:
        """
        Resolve the crisis.

        Parameters:
            outcome: 'original_wins' | 'new_wins'
            new_entrenchment: Entrenchment level after the crisis
        """
        self.resolved_at = datetime.now(UTC)
        self.outcome = outcome
        logger.info(
            f"Belief crisis resolved: memory_id={self.memory_id[:8]}, "
            f"outcome={outcome}, new_entrenchment={new_entrenchment:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert crisis to dict (for blockchain log)."""
        return {
            "memory_id": self.memory_id,
            "contradiction_count": self.contradiction_count,
            "original_entrenchment": self.original_entrenchment,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "outcome": self.outcome,
            "on_chain_tx": self.on_chain_tx,
        }


@dataclass
class CognitiveState:
    """
    The AI's instantaneous cognitive state.

    Updated by the engine and saved to snapshots.
    """

    character_strength: float = 0.0
    head_weights: tuple[float, float, float, float] = (0.35, 0.30, 0.20, 0.15)
    active_belief_crises: list[BeliefCrisis] = field(default_factory=list)
    total_interactions: int = 0
    belief_crises_experienced: int = 0
    gc_total_pruned: int = 0
    reality_check_rejections: int = 0
    emotion_shield_adjustments: int = 0
    is_frozen: bool = False  # True if Kill Switch was triggered — system frozen

    # Bias parameters (defaults)
    bias_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "availability_lambda_multiplier": 1.0,
            "confirmation_resistance_factor": 2.5,
            "emotional_amplification": 1.5,
            "anchor_weight": 0.15,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert cognitive state to dict."""
        return {
            "character_strength": self.character_strength,
            "head_weights": list(self.head_weights),
            "active_belief_crises": [c.to_dict() for c in self.active_belief_crises if c.is_active],
            "total_interactions": self.total_interactions,
            "belief_crises_experienced": self.belief_crises_experienced,
            "gc_total_pruned": self.gc_total_pruned,
            "reality_check_rejections": self.reality_check_rejections,
            "emotion_shield_adjustments": self.emotion_shield_adjustments,
            "bias_parameters": self.bias_parameters,
            "is_frozen": self.is_frozen,
        }


class CharacterManager:
    """
    Character crystallization and belief crisis manager.

    Called by CognitioEngine on every memory update.
    """

    # Character strength thresholds
    YOUNG_THRESHOLD = 5.0
    MATURE_THRESHOLD = 15.0

    # Belief crisis parameters
    CRISIS_ENTRENCHMENT = 0.5  # Temporary entrenchment during crisis
    CRISIS_HEAD_WEIGHTS = (0.25, 0.40, 0.20, 0.15)  # Recency head rises
    CRISIS_ENTRENCHMENT_BOOST = 0.1  # Entrenchment boost if original wins

    def __init__(self) -> None:
        self.personality = PersonalityVector()
        self.relational = RelationalProfile()
        self.active_crises: dict[str, BeliefCrisis] = {}  # memory_id → crisis

    def compute_character_strength(self, memories: list["MemoryRecord"]) -> float:
        """
        Compute character strength.

        Formula: Σ (entrenchment(m) · salience_proxy(m)) for m where entrenchment > 0.6
        emotional_intensity is used as the salience proxy (full salience is expensive).

        Parameters:
            memories: All active memory records

        Returns:
            float: Character strength (0.0+)
        """
        strength = 0.0
        for memory in memories:
            if memory.entrenchment > 0.6:
                # Simple salience proxy: entrenchment × emotional_weight
                emotional_weight = 1.0 + memory.emotional_intensity
                strength += memory.entrenchment * emotional_weight

        return strength

    def trigger_belief_crisis(self, memory: "MemoryRecord") -> BeliefCrisis:
        """
        Trigger a belief crisis.

        During the crisis:
        - The record's entrenchment temporarily drops to 0.5
        - Head weights shift to emphasize recency

        Parameters:
            memory: Memory record experiencing the crisis

        Returns:
            BeliefCrisis: The created crisis object
        """
        crisis = BeliefCrisis(
            memory_id=memory.id,
            contradiction_count=memory.contradiction_count,
            original_entrenchment=memory.entrenchment,
        )

        # Temporary entrenchment drop
        memory.entrenchment = self.CRISIS_ENTRENCHMENT

        self.active_crises[memory.id] = crisis
        logger.warning(
            f"Belief crisis triggered: memory_id={memory.id[:8]}, "
            f"contradictions={memory.contradiction_count}, "
            f"entrenchment: {crisis.original_entrenchment:.2f} → {self.CRISIS_ENTRENCHMENT}"
        )
        return crisis

    def resolve_crisis(
        self,
        memory_id: str,
        outcome: str,
        memory: Optional["MemoryRecord"] = None,
    ) -> BeliefCrisis | None:
        """
        Resolve a belief crisis.

        Parameters:
            memory_id: ID of the memory record in crisis
            outcome: 'original_wins' | 'new_wins'
            memory: Memory record for update (optional)

        Returns:
            BeliefCrisis: The resolved crisis object
        """
        crisis = self.active_crises.pop(memory_id, None)
        if crisis is None:
            return None

        if outcome == "original_wins" and memory is not None:
            # Original wins → entrenchment increases
            new_entrenchment = min(
                1.0, crisis.original_entrenchment + self.CRISIS_ENTRENCHMENT_BOOST
            )
            memory.entrenchment = new_entrenchment
        elif outcome == "new_wins" and memory is not None:
            # New wins → record is marked as "superseded"
            from cognithor.identity.cognitio.memory import MemoryStatus

            memory.status = MemoryStatus.SUPERSEDED

        crisis.resolve(outcome, memory.entrenchment if memory else 0.0)
        return crisis

    def get_crisis_head_weights(self) -> tuple[float, float, float, float]:
        """Head weights to use during a crisis."""
        return self.CRISIS_HEAD_WEIGHTS

    def has_active_crisis(self, memory_id: str) -> bool:
        """Is there an active crisis for a specific memory record?"""
        return memory_id in self.active_crises

    def get_all_active_crises(self) -> list[BeliefCrisis]:
        """Get all active crises."""
        return list(self.active_crises.values())

    def update_personality(self, memory: "MemoryRecord") -> None:
        """Update the personality vector based on a new memory record."""
        if memory.entrenchment > 0.3:  # Weak records should not affect personality
            self.personality.update_from_memory(memory)
