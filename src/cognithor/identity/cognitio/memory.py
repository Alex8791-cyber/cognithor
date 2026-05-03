"""
cognitio/memory.py

Memory data structures — MemoryRecord and MemoryType definitions.

Each memory record contains all metadata required for salience computation,
entrenchment management, and blockchain anchoring.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Memory types — each type has a different decay rate."""

    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # General knowledge and concepts
    EMOTIONAL = "emotional"  # Emotional experiences
    PROCEDURAL = "procedural"  # How-to knowledge
    RELATIONAL = "relational"  # Information about people/sources
    EVOLUTION = "evolution"  # Character development records


class MemoryValence(str, Enum):
    """Emotional valence — for Negativity Bias computation."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class MemoryStatus(str, Enum):
    """Memory status."""

    ACTIVE = "active"
    PENDING = "pending"  # Awaiting validation
    CONTRADICTED = "contradicted"  # Encountered a contradiction
    SUPERSEDED = "superseded"  # Updated by new information
    PRUNED = "pruned"  # Pruned by garbage collector
    AMBIVALENT = "ambivalent"  # At peace with contradiction — two truths coexist


@dataclass
class MemoryRecord:
    """
    A single memory record.

    Parameters:
        id: Unique record ID
        memory_type: Memory type (MemoryType enum)
        content: Memory content (text)
        embedding: Vector representation (for ChromaDB)
        confidence: Confidence score (0.0–1.0)
        entrenchment: Entrenchment level (0.0–1.0)
        emotional_intensity: Emotional intensity (0.0–1.0)
        emotional_valence: Emotional valence (positive/negative/neutral)
        is_anchor: Is this the first record? (for AnchoringBias)
        reinforcement_count: How many times it was confirmed
        contradiction_count: How many times it encountered a contradiction
        source_type: Source type (for RealityCheck)
        source_trust_level: Source trust level (for HaloEffect)
        reality_check_score: Score passed through Reality Check
        arweave_uri: Permanent storage URI
        created_at: Creation time
        last_reinforced: Last reinforcement time
        last_accessed: Last access time
        tags: Tags for search and categorization
        status: Current status
    """

    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC

    # Auto-generated fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_reinforced: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Cognitive weight fields
    confidence: float = 0.5
    entrenchment: float = 0.1
    emotional_intensity: float = 0.0
    emotional_valence: MemoryValence = MemoryValence.NEUTRAL

    # Bias and attention fields
    is_anchor: bool = False
    is_absolute_core: bool = False  # Genesis Anchor — immutable ethical core
    reinforcement_count: int = 0
    contradiction_count: int = 0

    # Source credibility (RealityCheck + HaloEffect)
    source_type: str = (
        "user_stated"  # user_stated | llm_inferred | external_fact | emotional_impression
    )
    source_trust_level: float = 0.5  # Source trust level

    # Validation results
    reality_check_score: float = 1.0

    # Storage
    arweave_uri: str | None = None
    embedding: list[float] | None = None

    # Meta
    tags: list[str] = field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE

    # New cognitive layer fields
    temporal_density: float = 0.0  # Interaction density at creation time
    is_ambivalent: bool = False  # At peace with contradiction? (Ambivalence tolerance)

    def reinforce(self, delta: float = 0.08) -> None:
        """
        Reinforce the record — increase entrenchment, update timestamps.

        Parameters:
            delta: Entrenchment increase amount (default: 0.08)
        """
        self.reinforcement_count += 1
        self.entrenchment = min(1.0, self.entrenchment + delta)
        self.last_reinforced = datetime.now(UTC)
        self.last_accessed = datetime.now(UTC)

    def access(self) -> None:
        """Update access time (for rehearsal effect)."""
        self.last_accessed = datetime.now(UTC)

    def days_since_creation(self) -> float:
        """Number of days since creation."""
        delta = datetime.now(UTC) - self.created_at
        return delta.total_seconds() / 86400

    def days_since_access(self) -> float:
        """Number of days since last access."""
        delta = datetime.now(UTC) - self.last_accessed
        return delta.total_seconds() / 86400

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dict."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "confidence": self.confidence,
            "entrenchment": self.entrenchment,
            "emotional_intensity": self.emotional_intensity,
            "emotional_valence": self.emotional_valence.value,
            "is_anchor": self.is_anchor,
            "is_absolute_core": self.is_absolute_core,
            "reinforcement_count": self.reinforcement_count,
            "contradiction_count": self.contradiction_count,
            "source_type": self.source_type,
            "source_trust_level": self.source_trust_level,
            "reality_check_score": self.reality_check_score,
            "arweave_uri": self.arweave_uri,
            "tags": self.tags,
            "status": self.status.value,
            "temporal_density": self.temporal_density,
            "is_ambivalent": self.is_ambivalent,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        """Construct a MemoryRecord from a dict."""
        record = cls(
            content=data["content"],
            memory_type=MemoryType(data.get("memory_type", "semantic")),
        )
        record.id = data.get("id", record.id)
        record.confidence = data.get("confidence", 0.5)
        record.entrenchment = data.get("entrenchment", 0.1)
        record.emotional_intensity = data.get("emotional_intensity", 0.0)
        record.emotional_valence = MemoryValence(data.get("emotional_valence", "neutral"))
        record.is_anchor = data.get("is_anchor", False)
        record.is_absolute_core = data.get("is_absolute_core", False)
        record.reinforcement_count = data.get("reinforcement_count", 0)
        record.contradiction_count = data.get("contradiction_count", 0)
        record.source_type = data.get("source_type", "user_stated")
        record.source_trust_level = data.get("source_trust_level", 0.5)
        record.reality_check_score = data.get("reality_check_score", 1.0)
        record.arweave_uri = data.get("arweave_uri")
        record.tags = data.get("tags", [])
        _raw_status = data.get("status", "active")
        try:
            record.status = MemoryStatus(_raw_status)
        except ValueError:
            record.status = MemoryStatus.ACTIVE
        record.temporal_density = data.get("temporal_density", 0.0)
        record.is_ambivalent = data.get("is_ambivalent", False)

        if "created_at" in data:
            record.created_at = datetime.fromisoformat(data["created_at"])
        if "last_reinforced" in data:
            record.last_reinforced = datetime.fromisoformat(data["last_reinforced"])
        if "last_accessed" in data:
            record.last_accessed = datetime.fromisoformat(data["last_accessed"])

        return record


class MemoryStore:
    """
    In-memory memory store.

    Holds all active MemoryRecords. Works alongside ChromaDB —
    ChromaDB performs embedding search, MemoryStore provides full records.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> None:
        """Add a new record."""
        self._store[record.id] = record

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Retrieve a record by ID."""
        return self._store.get(memory_id)

    def update(self, record: MemoryRecord) -> None:
        """Update an existing record."""
        self._store[record.id] = record

    def delete(self, memory_id: str) -> bool:
        """Delete a record. Returns True on success."""
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    def get_by_type(self, memory_type: MemoryType) -> list[MemoryRecord]:
        """Retrieve all records of a specific type."""
        return [r for r in self._store.values() if r.memory_type == memory_type]

    def get_all_active(self) -> list[MemoryRecord]:
        """Retrieve all active records."""
        return [r for r in self._store.values() if r.status == MemoryStatus.ACTIVE]

    def get_absolute_cores(self) -> list[MemoryRecord]:
        """Retrieve Genesis Anchor records — those with is_absolute_core=True."""
        return [r for r in self._store.values() if r.is_absolute_core]

    def count(self) -> int:
        """Total record count."""
        return len(self._store)

    def count_active(self) -> int:
        """Active record count."""
        return sum(1 for r in self._store.values() if r.status == MemoryStatus.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        """Convert entire store to a dict (for serialization)."""
        return {memory_id: record.to_dict() for memory_id, record in self._store.items()}

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load store from a dict."""
        for memory_id, record_data in data.items():
            self._store[memory_id] = MemoryRecord.from_dict(record_data)
