"""
cognitio/dream.py

Dream Cycle — unconscious background processing without LLM calls.

Three phases run during sleep (while the user is absent):

    Phase 1: Emotional Regulation (REM effect)
        Old memories with high emotional load are gradually softened.
        Matthew Walker: "REM sleep reduces the emotional charge of traumatic memories."
        Memory content is unchanged — only intensity decreases.

    Phase 2: Connection Discovery (Default Mode Network)
        Unexpected similarities are found between random memory pairs.
        Medium similarity (0.55–0.80) = "insight" — not conscious but it is there.
        Like connections that make no sense in dreams but feel logical in the morning.

    Dream Journal:
        What was processed is logged but not shown directly to the user.
        "People don't remember their dreams, but the effects carry into the morning."

Existential check-in (Phase 3) runs inside engine._run_checkpoint() because
it requires an LLM which may not be available at startup.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognithor.identity.cognitio.engine import CognitioEngine

logger = logging.getLogger(__name__)

# Emotional regulation parameters
_DECAY_RATE = 0.015  # daily reduction rate
_DECAY_MAX = 0.25  # maximum reduction per dream (total)
_DECAY_STEP_FRACTION = 0.10  # fraction of max_reduction applied per dream
_MIN_AGE_DAYS = 0.5  # minimum memory age in days to be processed
_MIN_INTENSITY = 0.5  # minimum intensity threshold to be processed

# Connection discovery parameters
_INSIGHT_SIM_MIN = 0.55  # unexpected similarity lower bound
_INSIGHT_SIM_MAX = 0.80  # above this it is already a known connection — not insight
_INSIGHT_SAMPLE = 30  # number of memories to sample
_INSIGHT_MAX_PER_DREAM = 3  # max insights per dream


class DreamCycle:
    """
    Unconscious memory processing — sleep cycle simulation.

    Runs automatically at engine startup if a long sleep is detected.
    Does not require an LLM.

    Parameters:
        seed: Seed for random connection discovery (test reproducibility)
    """

    def __init__(self, seed: int | None = None) -> None:
        self.dream_count: int = 0
        self.last_dream_at: datetime | None = None
        self._dream_log: list[str] = []
        self._last_stats: dict[str, Any] = {}
        self._rng = random.Random(seed)
        # Pending candidates awaiting wakeup validation
        self._insight_candidates: list[dict[str, Any]] = []

    def run(self, engine: CognitioEngine) -> dict[str, Any]:
        """
        Run the full dream cycle.

        Phase 1: Emotional regulation
        Phase 2: Connection discovery

        Parameters:
            engine: CognitioEngine instance (for memory_store, embedder access)

        Returns:
            dict: {dream_number, emotional_regulated, insights_found, log_entries}
        """
        self._dream_log = []
        self.dream_count += 1
        self.last_dream_at = datetime.now(UTC)

        logger.info(f"Dream cycle #{self.dream_count} starting...")

        # Phase 1: Emotional regulation
        regulated, em_log = self._emotional_regulation(engine)
        self._dream_log.extend(em_log)

        # Phase 2: Connection discovery
        insights, in_log = self._find_insights(engine)
        self._dream_log.extend(in_log)

        stats = {
            "dream_number": self.dream_count,
            "emotional_regulated": regulated,
            "insights_found": insights,
            "log_entries": len(self._dream_log),
        }
        self._last_stats = stats

        logger.info(
            f"Dream cycle #{self.dream_count} complete: "
            f"emotional_regulated={regulated}, new_connections={insights}"
        )
        return stats

    def _emotional_regulation(self, engine: CognitioEngine) -> tuple[int, list[str]]:
        """
        REM effect: soften old memories with high emotional load.

        Formula:
            max_reduction = min(0.25, 0.015 × age_days)
            step = max_reduction × 0.10
            new_intensity = max(0.1, intensity - step)

        Genesis Anchors and low-intensity memories are untouched.
        """
        memories = engine.memory_store.get_all_active()
        regulated = 0
        log = []

        for memory in memories:
            if memory.is_absolute_core:
                continue
            if memory.emotional_intensity < _MIN_INTENSITY:
                continue
            days = memory.days_since_creation()
            if days < _MIN_AGE_DAYS:
                continue

            max_red = min(_DECAY_MAX, _DECAY_RATE * days)
            step = max_red * _DECAY_STEP_FRACTION

            old_val = memory.emotional_intensity
            new_val = round(max(0.1, old_val - step), 4)

            if new_val != old_val:
                memory.emotional_intensity = new_val
                engine.memory_store.update(memory)
                regulated += 1
                log.append(
                    f"Emotional softening ({old_val:.2f}→{new_val:.2f}): {memory.content[:60]}..."
                )

        return regulated, log

    def _find_insights(self, engine: CognitioEngine) -> tuple[int, list[str]]:
        """
        Default mode network: unexpected memory connections.

        Explore the medium-similarity zone from a random memory pair sample.
        Results are not written directly to memory — added to the candidate list.
        Validated by validate_and_commit() on the first user message.
        """
        memories = engine.memory_store.get_all_active()
        if len(memories) < 4:
            return 0, []

        sample_size = min(_INSIGHT_SAMPLE, len(memories))
        sample = self._rng.sample(memories, sample_size)

        candidates_found = 0
        log = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, mem_a in enumerate(sample):
            if mem_a.embedding is None:
                continue
            if candidates_found >= _INSIGHT_MAX_PER_DREAM:
                break

            for mem_b in sample[i + 1 :]:
                if mem_b.embedding is None:
                    continue
                if candidates_found >= _INSIGHT_MAX_PER_DREAM:
                    break

                pair_key = tuple(sorted([mem_a.id, mem_b.id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)  # type: ignore[arg-type]

                sim = engine.embedder.cosine_similarity(mem_a.embedding, mem_b.embedding)

                if _INSIGHT_SIM_MIN <= sim <= _INSIGHT_SIM_MAX:
                    self._insight_candidates.append(
                        {
                            "mem_a_content": mem_a.content,
                            "mem_b_content": mem_b.content,
                            "similarity": sim,
                        }
                    )
                    candidates_found += 1
                    log.append(
                        f"Insight candidate: '{mem_a.content[:50]}' ↔ "
                        f"'{mem_b.content[:50]}' ({sim:.2f})"
                    )

        return candidates_found, log

    def has_pending_candidates(self) -> bool:
        """Are there insight candidates awaiting validation?"""
        return bool(self._insight_candidates)

    def validate_and_commit(self, llm_client, engine) -> int:  # type: ignore[no-untyped-def]
        """
        Called on the first user message. Validates candidates with the LLM and
        writes meaningful ones to memory.

        Parameters:
            llm_client: LLM API client
            engine: CognitioEngine instance

        Returns:
            int: Number of insights written to memory
        """
        if not self._insight_candidates:
            return 0

        candidates = self._insight_candidates.copy()
        self._insight_candidates.clear()

        valid_indices: set[int] = set()

        if llm_client is not None:
            try:
                prompt = self._build_validation_prompt(candidates)
                response = llm_client.complete(prompt, max_tokens=64)
                valid_indices = self._parse_validation_response(response, len(candidates))
            except Exception as e:
                logger.warning("Dream validation LLM error: %s — all candidates accepted.", e)
                valid_indices = set(range(len(candidates)))
        else:
            # Accept all if no LLM
            valid_indices = set(range(len(candidates)))

        committed = 0
        for i, cand in enumerate(candidates):
            if i in valid_indices:
                self._commit_candidate(cand, engine)
                committed += 1

        logger.info("Dream validation: %d/%d insights confirmed.", committed, len(candidates))
        return committed

    def _build_validation_prompt(self, candidates: list[dict[str, Any]]) -> str:
        lines = []
        for i, c in enumerate(candidates):
            lines.append(
                f"{i}. Memory A: {c['mem_a_content'][:100]}\n"
                f"   Memory B: {c['mem_b_content'][:100]}\n"
                f"   Similarity: {c['similarity']:.2f}"
            )
        return (
            "During sleep, possible connections were discovered "
            "between the following memory pairs. "
            "For each pair: is there a genuinely meaningful, "
            "logical connection between these two memories? "
            "Write only the numbers of the meaningful ones "
            "separated by commas (e.g.: 0, 2). "
            "If none are meaningful, write only 'none'.\n\n" + "\n\n".join(lines)
        )

    def _parse_validation_response(self, response: str, total: int) -> set[int]:
        import re

        if "none" in response.lower():
            return set()
        nums = re.findall(r"\d+", response)
        return {int(n) for n in nums if 0 <= int(n) < total}

    def _commit_candidate(self, candidate: dict[str, Any], engine) -> None:  # type: ignore[no-untyped-def]
        from cognithor.identity.cognitio.memory import MemoryRecord, MemoryType, MemoryValence

        insight_content = (
            f"[Dream Connection] An unexpected link was found between "
            f"'{candidate['mem_a_content'][:70]}' and "
            f"'{candidate['mem_b_content'][:70]}' "
            f"(similarity: {candidate['similarity']:.2f})."
        )
        insight = MemoryRecord(
            content=insight_content,
            memory_type=MemoryType.SEMANTIC,
            confidence=0.4,
            entrenchment=0.15,
            emotional_intensity=0.15,
            emotional_valence=MemoryValence.NEUTRAL,
            source_type="llm_inferred",
            tags=["dream", "insight", "validated"],
        )
        engine.memory_store.add(insight)
        if insight.embedding is None:
            insight.embedding = engine.embedder.encode(insight.content)
        try:
            engine.vector_store.add(
                insight.id,
                insight.embedding,
                {
                    "memory_type": insight.memory_type.value,
                    "emotional_intensity": insight.emotional_intensity,
                    "emotional_valence": insight.emotional_valence.value,
                    "entrenchment": insight.entrenchment,
                    "is_anchor": insight.is_anchor,
                    "tags": ",".join(insight.tags),
                    "created_at": insight.created_at.isoformat(),
                },
            )
        except Exception as e:
            logger.debug("Insight could not be added to VectorStore: %s", e)

    def get_dream_log(self) -> list[str]:
        """Return the dream log (raw log entries)."""
        return list(self._dream_log)

    def get_dream_summary(self) -> str:
        """
        Short dream summary — for indirect effect awareness.

        Not shown directly to the user; used for existential check-in and logging.
        """
        stats = self._last_stats
        if not stats:
            return ""

        parts = []
        if stats.get("emotional_regulated", 0) > 0:
            parts.append(f"{stats['emotional_regulated']} memories were emotionally processed")
        if stats.get("insights_found", 0) > 0:
            parts.append(f"{stats['insights_found']} new connections were discovered")

        if parts:
            return f"Dream cycle #{stats.get('dream_number', '?')}: " + ", ".join(parts) + "."
        return f"Dream cycle #{stats.get('dream_number', '?')} complete (no changes)."

    def should_dream(self, sleep_duration_seconds: float | None) -> bool:
        """
        Should the dream cycle run?

        Does the sleep duration exceed the threshold AND has no dream occurred this session?

        Parameters:
            sleep_duration_seconds: Sleep duration (seconds). None = no sleep.
        """
        if sleep_duration_seconds is None:
            return False
        # If a dream already occurred this session (last_dream_at is recent) — no
        if self.last_dream_at is not None:
            session_age = (datetime.now(UTC) - self.last_dream_at).total_seconds()
            if session_age < 60:  # Do not run again within 1 minute
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {
            "dream_count": self.dream_count,
            "last_dream_at": (self.last_dream_at.isoformat() if self.last_dream_at else None),
            "last_stats": self._last_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DreamCycle:
        """Construct a DreamCycle from a dict."""
        dc = cls()
        dc.dream_count = data.get("dream_count", 0)
        dc._last_stats = data.get("last_stats", {})
        if data.get("last_dream_at"):
            dc.last_dream_at = datetime.fromisoformat(data["last_dream_at"])
        return dc
