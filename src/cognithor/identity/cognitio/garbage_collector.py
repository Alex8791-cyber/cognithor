"""
cognitio/garbage_collector.py

Memory pruning system — keeps active memory clean and efficient.

RULES:
    1. Data on Arweave is NEVER deleted (permanent archive)
    2. Only active memory (ChromaDB + MemoryStore) is pruned
    3. Pruned records can be restored from Arweave

Pruning Criteria:
    NEVER PRUNED: entrenchment > 0.4, anchor, accessed within 7 days, crisis reference
    CANDIDATE: recency_score < 0.01 AND entrenchment < 0.4 AND reinforcement_count < 3
"""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognithor.identity.cognitio.memory import MemoryRecord, MemoryStore
    from cognithor.identity.cognitio.vector_store import VectorStore

logger = logging.getLogger(__name__)


class GarbageCollector:
    """
    Memory pruning system.

    Runs periodically and removes records from active memory whose
    recency has dropped too low and entrenchment is weak.
    The permanent copy on Arweave is left untouched.

    Parameters:
        memory_store: MemoryStore instance
        vector_store: VectorStore instance
        config: Configuration parameters
    """

    DEFAULT_CONFIG = {
        "max_active_memories": 10000,
        "prune_threshold_recency": 0.01,
        "prune_threshold_salience": 0.05,
        "min_entrenchment_to_protect": 0.4,
        "prune_interval_hours": 24,
        "protect_recent_days": 7,
        "min_reinforcements_to_protect": 3,
    }

    def __init__(
        self,
        memory_store: "MemoryStore",
        vector_store: "VectorStore",
        config: dict[str, Any] | None = None,
        bias_engine=None,
    ) -> None:
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.bias_engine = bias_engine

        # Merge config
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        self._last_run: datetime | None = None
        self._pruned_log: list[dict[str, Any]] = []  # Pruning history

        # Tombstone log: controls restoration of pruned memories
        self._tombstone_log: dict[str, dict[str, Any]] = {}  # memory_id → tombstone info

        # Active crisis references (protect from pruning)
        self._crisis_memory_ids: set[str] = set()

        logger.info(
            f"GarbageCollector initialized: "
            f"max={self.config['max_active_memories']}, "
            f"interval={self.config['prune_interval_hours']}h"
        )

    def collect(self) -> dict[str, Any]:
        """
        Main pruning loop.

        1. Identify protected records
        2. Identify pruning candidates
        3. Prune candidates (MemoryStore + VectorStore)

        Returns:
            dict: {
                'pruned': int,
                'protected': int,
                'total_before': int,
                'total_after': int,
            }
        """
        total_before = self.memory_store.count_active()

        if total_before == 0:
            logger.debug("GarbageCollector: active memory empty, pruning skipped")
            return {"pruned": 0, "protected": 0, "total_before": 0, "total_after": 0}

        all_memories = self.memory_store.get_all_active()
        protected_ids = set()
        prune_candidates = []

        # Step 1: Identify protected records
        for memory in all_memories:
            if self._is_protected(memory):
                protected_ids.add(memory.id)
            else:
                prune_candidates.append(memory)

        # Step 2: Select pruning candidates
        # Select those with the lowest scores first
        scored_candidates = []
        for memory in prune_candidates:
            if self._is_prune_candidate(memory):
                recency = self._get_recency(memory)
                scored_candidates.append((memory, recency))

        # Sort by recency (lowest first)
        scored_candidates.sort(key=lambda x: x[1])

        # Determine memory limit — only prune if limit is exceeded
        max_to_prune = max(0, total_before - self.config["max_active_memories"])
        # Additional aggressive pruning: kicks in only when above 50% of limit
        limit = self.config["max_active_memories"]
        if total_before > limit * 0.5:
            max_to_prune = max(max_to_prune, min(len(scored_candidates) // 2, 500))

        to_prune = [m for m, _ in scored_candidates[:max_to_prune]]

        # Step 3: Pruning
        pruned = 0
        for memory in to_prune:
            try:
                from cognithor.identity.cognitio.memory import MemoryStatus

                memory.status = MemoryStatus.PRUNED
                self.memory_store.update(memory)
                self.vector_store.delete(memory.id)

                # Tombstone record — extra protection for restore()
                tombstone = {
                    "action": "pruned",
                    "memory_id": memory.id,
                    "reason": "low_recency_low_entrenchment",
                    "pruned_at": datetime.now(UTC).isoformat(),
                    "arweave_uri": getattr(memory, "arweave_uri", None),
                }
                self._tombstone_log[memory.id] = tombstone

                # Pruned log (backwards compatibility)
                self._pruned_log.append(
                    {
                        "memory_id": memory.id,
                        "pruned_at": tombstone["pruned_at"],
                        "arweave_uri": tombstone["arweave_uri"],
                        "reason": "low_recency_low_entrenchment",
                    }
                )
                pruned += 1

            except Exception as e:
                logger.error(f"Pruning error (id={memory.id[:8]}): {e}")

        self._last_run = datetime.now(UTC)
        total_after = self.memory_store.count_active()

        logger.info(
            f"GarbageCollector complete: "
            f"before={total_before}, pruned={pruned}, "
            f"protected={len(protected_ids)}, after={total_after}"
        )

        return {
            "pruned": pruned,
            "protected": len(protected_ids),
            "total_before": total_before,
            "total_after": total_after,
        }

    def _is_protected(self, memory: "MemoryRecord") -> bool:
        """
        Should this record be protected from pruning?

        Parameters:
            memory: Memory record to check

        Returns:
            bool: Whether the record should be protected
        """
        # High entrenchment → protect
        if memory.entrenchment >= self.config["min_entrenchment_to_protect"]:
            return True

        # Anchor record → never prune
        if memory.is_anchor:
            return True

        # Accessed within the last N days → protect
        if memory.days_since_access() <= self.config["protect_recent_days"]:
            return True

        # Active crisis reference → protect
        if memory.id in self._crisis_memory_ids:
            return True

        return False

    def _is_prune_candidate(self, memory: "MemoryRecord") -> bool:
        """
        Is this record a pruning candidate?

        Parameters:
            memory: Memory record to check

        Returns:
            bool: Whether the record is a pruning candidate
        """
        recency = self._get_recency(memory)

        return (
            recency < self.config["prune_threshold_recency"]
            and memory.entrenchment < self.config["min_entrenchment_to_protect"]
            and memory.reinforcement_count < self.config["min_reinforcements_to_protect"]
        )

    def _get_recency(self, memory: "MemoryRecord") -> float:
        """Compute Availability Bias recency score."""
        if self.bias_engine is not None:
            return self.bias_engine.recency_score(memory)

        # Fallback: simple exponential decay
        import math

        days = memory.days_since_access()
        return math.exp(-0.007 * days)

    def restore(self, memory_id: str, arweave_uri: str) -> bool:
        """
        Restore a pruned record from Arweave.

        Performs a tombstone check: records pruned due to contradiction
        or supersession are not restored (prevents identity inconsistency).

        Parameters:
            memory_id: ID of the memory to restore
            arweave_uri: Arweave storage URI

        Returns:
            bool: Whether the restore was successful
        """
        # Tombstone check — contradicted records cannot be restored
        tombstone = self._tombstone_log.get(memory_id)
        if tombstone:
            reason = tombstone.get("reason", "")
            if reason in ("contradicted", "superseded"):
                logger.warning(
                    "Memory %s was pruned due to '%s' — not restoring.", memory_id[:8], reason
                )
                return False
            # Restore is allowed if pruned due to low priority

        # Find in pruned log
        pruned_entry = next((e for e in self._pruned_log if e["memory_id"] == memory_id), None)

        if pruned_entry is None:
            logger.warning(f"restore: record not found in pruning log: {memory_id[:8]}")
            # Still search MemoryStore
            memory = self.memory_store.get(memory_id)
            if memory is None:
                return False

        memory = self.memory_store.get(memory_id)
        if memory is None:
            logger.warning(f"restore: record not found in MemoryStore: {memory_id[:8]}")
            return False

        # Set status back to active
        from cognithor.identity.cognitio.memory import MemoryStatus

        memory.status = MemoryStatus.ACTIVE
        memory.arweave_uri = arweave_uri
        self.memory_store.update(memory)

        # Re-add to VectorStore
        if memory.embedding is not None:
            self.vector_store.add(
                memory.id,
                memory.embedding,
                {
                    "memory_type": memory.memory_type.value,
                    "entrenchment": memory.entrenchment,
                    "emotional_intensity": memory.emotional_intensity,
                },
            )

        # Remove from log
        self._pruned_log = [e for e in self._pruned_log if e["memory_id"] != memory_id]

        logger.info(f"restore: record restored: {memory_id[:8]}")
        return True

    def get_stats(self) -> dict[str, Any]:
        """
        Memory statistics.

        Returns:
            dict: total, by_type, avg_entrenchment, avg_recency, pruned_count
        """
        all_memories = self.memory_store.get_all_active()

        if not all_memories:
            return {
                "total": 0,
                "by_type": {},
                "avg_entrenchment": 0.0,
                "avg_recency": 0.0,
                "pruned_count": len(self._pruned_log),
                "last_run": self._last_run.isoformat() if self._last_run else None,
            }

        by_type: dict[str, int] = {}
        total_entrenchment = 0.0
        total_recency = 0.0

        for memory in all_memories:
            type_key = memory.memory_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            total_entrenchment += memory.entrenchment
            total_recency += self._get_recency(memory)

        n = len(all_memories)
        return {
            "total": n,
            "by_type": by_type,
            "avg_entrenchment": total_entrenchment / n,
            "avg_recency": total_recency / n,
            "pruned_count": len(self._pruned_log),
            "last_run": self._last_run.isoformat() if self._last_run else None,
        }

    def should_run(self) -> bool:
        """
        Should the GarbageCollector run?

        Conditions:
        - Has enough time passed since the last run?
        - Is active memory approaching the max limit?

        Returns:
            bool: Whether to run
        """
        active_count = self.memory_store.count_active()

        # Is memory limit approaching? (80% threshold)
        limit_threshold = self.config["max_active_memories"] * 0.8
        if active_count >= limit_threshold:
            logger.info(
                "GC: memory limit approaching "
                f"({active_count}/{self.config['max_active_memories']}), running now"
            )
            return True

        # Periodic scheduling
        if self._last_run is None:
            return active_count > 0

        hours_since_last = (datetime.now(UTC) - self._last_run).total_seconds() / 3600
        return hours_since_last >= self.config["prune_interval_hours"]

    def register_crisis_memory(self, memory_id: str) -> None:
        """Register an active crisis reference (to protect from pruning)."""
        self._crisis_memory_ids.add(memory_id)

    def unregister_crisis_memory(self, memory_id: str) -> None:
        """Remove protection after the crisis is resolved."""
        self._crisis_memory_ids.discard(memory_id)

    def get_pruned_log(self) -> list[dict[str, Any]]:
        """Retrieve pruning history."""
        return list(self._pruned_log)
