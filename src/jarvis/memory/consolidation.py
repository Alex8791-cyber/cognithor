"""Memory Consolidation Pipeline — deduplicate, score, summarize, and gc.

Orchestrates the lifecycle of memory entries:
1. **Deduplication**: Merge semantically similar entries
2. **Scoring**: Rank entries by importance (relevance × recency × frequency)
3. **Summarization**: Compress old episodic entries into compact summaries
4. **Garbage Collection**: Archive/remove entries below score threshold
5. **Budget enforcement**: Keep total memory within configurable token limits

Architecture: §8.4 (Memory Consolidation Pipeline)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from jarvis.memory.scoring import ImportanceScorer
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


@dataclass
class DuplicateGroup:
    """A group of entries that are semantically equivalent."""

    canonical_id: str
    canonical_content: str
    duplicate_ids: list[str] = field(default_factory=list)
    similarity: float = 1.0


class ContentDeduplicator:
    """Detect and merge duplicate memory entries.

    Uses content hashing (exact) and n-gram overlap (fuzzy) to find
    duplicates without requiring embedding models.
    """

    # Maximum entries for O(N^2) fuzzy comparison.  Beyond this limit the
    # remaining entries are skipped to keep consolidation runtime bounded.
    # 500 entries → max 124 750 comparisons — still fast on any machine.
    MAX_FUZZY_ENTRIES: int = 500

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold
        self._hash_index: dict[str, str] = {}  # content_hash → entry_id
        self._merged_count: int = 0
        self._scanned_count: int = 0

    def content_hash(self, text: str) -> str:
        """Compute normalized content hash."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def ngram_similarity(self, a: str, b: str, n: int = 3) -> float:
        """Compute character n-gram Jaccard similarity between two texts."""
        if not a or not b:
            return 0.0
        a_lower = a.lower()
        b_lower = b.lower()
        a_ngrams = {a_lower[i : i + n] for i in range(len(a_lower) - n + 1)}
        b_ngrams = {b_lower[i : i + n] for i in range(len(b_lower) - n + 1)}
        if not a_ngrams or not b_ngrams:
            return 0.0
        intersection = len(a_ngrams & b_ngrams)
        union = len(a_ngrams | b_ngrams)
        return intersection / union if union > 0 else 0.0

    def find_duplicates(
        self,
        entries: list[dict[str, Any]],
    ) -> list[DuplicateGroup]:
        """Find groups of duplicate entries.

        Each entry dict should have: id, content.
        Returns groups where the first entry is the canonical (highest confidence).
        """
        self._scanned_count += len(entries)
        groups: list[DuplicateGroup] = []
        seen_ids: set[str] = set()

        # Phase 1: Exact hash duplicates
        hash_buckets: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            h = self.content_hash(entry.get("content", ""))
            hash_buckets.setdefault(h, []).append(entry)

        for _h, bucket in hash_buckets.items():
            if len(bucket) > 1:
                canonical = max(bucket, key=lambda e: e.get("confidence", 0.5))
                dups = [e["id"] for e in bucket if e["id"] != canonical["id"]]
                groups.append(
                    DuplicateGroup(
                        canonical_id=canonical["id"],
                        canonical_content=canonical.get("content", ""),
                        duplicate_ids=dups,
                        similarity=1.0,
                    )
                )
                seen_ids.update(e["id"] for e in bucket)

        # Phase 2: Fuzzy n-gram duplicates (skip already matched)
        remaining = [e for e in entries if e["id"] not in seen_ids]
        if len(remaining) > self.MAX_FUZZY_ENTRIES:
            log.warning(
                "dedup_fuzzy_batch_limited",
                total=len(remaining),
                limit=self.MAX_FUZZY_ENTRIES,
            )
            remaining = remaining[: self.MAX_FUZZY_ENTRIES]
        for i, entry_a in enumerate(remaining):
            if entry_a["id"] in seen_ids:
                continue
            content_a = entry_a.get("content", "")
            fuzzy_dups: list[str] = []
            for entry_b in remaining[i + 1 :]:
                if entry_b["id"] in seen_ids:
                    continue
                content_b = entry_b.get("content", "")
                sim = self.ngram_similarity(content_a, content_b)
                if sim >= self.similarity_threshold:
                    fuzzy_dups.append(entry_b["id"])
                    seen_ids.add(entry_b["id"])
            if fuzzy_dups:
                groups.append(
                    DuplicateGroup(
                        canonical_id=entry_a["id"],
                        canonical_content=content_a,
                        duplicate_ids=fuzzy_dups,
                        similarity=self.similarity_threshold,
                    )
                )
                seen_ids.add(entry_a["id"])

        self._merged_count += sum(len(g.duplicate_ids) for g in groups)
        return groups

    @property
    def duplicate_rate(self) -> float:
        """Fraction of entries that were duplicates."""
        if self._scanned_count == 0:
            return 0.0
        return self._merged_count / self._scanned_count

    def stats(self) -> dict[str, Any]:
        return {
            "scanned": self._scanned_count,
            "merged": self._merged_count,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "similarity_threshold": self.similarity_threshold,
        }


# ---------------------------------------------------------------------------
# Memory Budget
# ---------------------------------------------------------------------------


@dataclass
class TierBudget:
    """Token budget for a single memory tier."""

    tier: str
    max_tokens: int = 50000
    current_tokens: int = 0
    entry_count: int = 0

    @property
    def utilization(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return self.current_tokens / self.max_tokens

    @property
    def over_budget(self) -> bool:
        return self.current_tokens > self.max_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.current_tokens)


class MemoryBudgetManager:
    """Tracks and enforces per-tier token budgets."""

    def __init__(
        self,
        budgets: dict[str, int] | None = None,
    ) -> None:
        defaults = {
            "episodic": 100000,
            "semantic": 50000,
            "procedural": 30000,
        }
        self._budgets: dict[str, TierBudget] = {}
        for tier, max_tokens in (budgets or defaults).items():
            self._budgets[tier] = TierBudget(tier=tier, max_tokens=max_tokens)

    def update_usage(self, tier: str, tokens: int, entry_count: int) -> None:
        """Update current usage for a tier."""
        if tier in self._budgets:
            self._budgets[tier].current_tokens = tokens
            self._budgets[tier].entry_count = entry_count

    def get_budget(self, tier: str) -> TierBudget | None:
        return self._budgets.get(tier)

    def over_budget_tiers(self) -> list[str]:
        """Return tiers that exceed their budget."""
        return [t for t, b in self._budgets.items() if b.over_budget]

    def tokens_to_free(self, tier: str) -> int:
        """How many tokens must be freed to meet the budget."""
        budget = self._budgets.get(tier)
        if not budget or not budget.over_budget:
            return 0
        return budget.current_tokens - budget.max_tokens

    def stats(self) -> dict[str, Any]:
        return {
            tier: {
                "max_tokens": b.max_tokens,
                "current_tokens": b.current_tokens,
                "utilization": round(b.utilization, 4),
                "over_budget": b.over_budget,
                "entry_count": b.entry_count,
            }
            for tier, b in self._budgets.items()
        }


# ---------------------------------------------------------------------------
# Consolidation Result
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    timestamp: str = ""
    entries_scanned: int = 0
    duplicates_found: int = 0
    duplicates_merged: int = 0
    entries_archived: int = 0
    entries_summarized: int = 0
    tokens_freed: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "entries_scanned": self.entries_scanned,
            "duplicates_found": self.duplicates_found,
            "duplicates_merged": self.duplicates_merged,
            "entries_archived": self.entries_archived,
            "entries_summarized": self.entries_summarized,
            "tokens_freed": self.tokens_freed,
            "duration_ms": round(self.duration_ms, 1),
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Consolidation Pipeline
# ---------------------------------------------------------------------------


class ConsolidationPipeline:
    """Orchestrates memory deduplication, scoring, and garbage collection.

    .. deprecated:: 0.69.0
        Not wired into runtime. EpisodicCompressor handles compression
        via a daily background task in gateway.py instead.

    Usage::

        pipeline = ConsolidationPipeline()
        result = pipeline.run(entries, tier="episodic")
    """

    def __init__(
        self,
        scorer: ImportanceScorer | None = None,
        deduplicator: ContentDeduplicator | None = None,
        budget_manager: MemoryBudgetManager | None = None,
        archive_threshold: float = 0.15,
        summarize_age_days: float = 30.0,
    ) -> None:
        self.scorer = scorer or ImportanceScorer()
        self.deduplicator = deduplicator or ContentDeduplicator()
        self.budget_manager = budget_manager or MemoryBudgetManager()
        self.archive_threshold = archive_threshold
        self.summarize_age_days = summarize_age_days
        self._history: list[ConsolidationResult] = []

    def run(
        self,
        entries: list[dict[str, Any]],
        *,
        tier: str = "episodic",
    ) -> ConsolidationResult:
        """Run the full consolidation pipeline on a set of entries.

        Each entry dict should have:
            id, content, age_days, source_confidence, token_count (optional)

        Returns a ConsolidationResult with metrics.
        """
        start = time.monotonic()
        result = ConsolidationResult(
            timestamp=datetime.now(UTC).isoformat(),
            entries_scanned=len(entries),
        )

        if not entries:
            result.duration_ms = (time.monotonic() - start) * 1000
            self._history.append(result)
            return result

        # Phase 1: Deduplication
        dup_ids: set[str] = set()  # Init before try — used in Phase 4
        try:
            groups = self.deduplicator.find_duplicates(entries)
            result.duplicates_found = sum(len(g.duplicate_ids) for g in groups)
            for group in groups:
                dup_ids.update(group.duplicate_ids)
            result.duplicates_merged = len(dup_ids)
            # Remove duplicates from working set
            entries = [e for e in entries if e["id"] not in dup_ids]
        except Exception as exc:
            result.errors.append(f"Deduplication error: {exc}")

        # Phase 2: Scoring
        try:
            scores = self.scorer.score_batch(entries)
            below = self.scorer.find_below_threshold(scores, self.archive_threshold)
            archive_ids = {s.entry_id for s in below}
            result.entries_archived = len(archive_ids)
            # Estimate tokens freed
            for entry in entries:
                if entry["id"] in archive_ids:
                    result.tokens_freed += entry.get("token_count", 100)
        except Exception as exc:
            result.errors.append(f"Scoring error: {exc}")
            archive_ids = set()

        # Phase 3: Summarization candidates
        try:
            old_entries = [
                e
                for e in entries
                if e.get("age_days", 0) >= self.summarize_age_days and e["id"] not in archive_ids
            ]
            result.entries_summarized = len(old_entries)
        except Exception as exc:
            result.errors.append(f"Summarization error: {exc}")

        # Phase 4: Budget check
        try:
            remaining_tokens = sum(
                e.get("token_count", 100) for e in entries if e["id"] not in archive_ids
            )
            self.budget_manager.update_usage(
                tier,
                remaining_tokens,
                len(entries) - len(archive_ids),
            )
        except Exception as exc:
            result.errors.append(f"Budget error: {exc}")

        result.tokens_freed += sum(e.get("token_count", 100) for e in entries if e["id"] in dup_ids)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._history.append(result)

        log.info(
            "consolidation_complete",
            tier=tier,
            scanned=result.entries_scanned,
            merged=result.duplicates_merged,
            archived=result.entries_archived,
            tokens_freed=result.tokens_freed,
            duration_ms=round(result.duration_ms, 1),
        )

        return result

    @property
    def history(self) -> list[ConsolidationResult]:
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        total_scanned = sum(r.entries_scanned for r in self._history)
        total_merged = sum(r.duplicates_merged for r in self._history)
        total_archived = sum(r.entries_archived for r in self._history)
        total_freed = sum(r.tokens_freed for r in self._history)
        return {
            "runs": len(self._history),
            "total_scanned": total_scanned,
            "total_merged": total_merged,
            "total_archived": total_archived,
            "total_tokens_freed": total_freed,
            "dedup_stats": self.deduplicator.stats(),
            "scorer_stats": self.scorer.stats(),
            "budget_stats": self.budget_manager.stats(),
        }
