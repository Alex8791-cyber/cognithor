# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tactical-memory cache for synthesis results (spec §14).

Phase 1 ships an in-process cache with the right key derivation, TTL
buckets, and DSL-version invalidation. Tactical Memory is currently a
broader Cognithor subsystem; the wiring layer that swaps the dict-
backed implementation for the real one lives in Week 5's PGE adapter.

Cache key (spec §14.1)::

    sha256(spec.stable_hash() || dsl_version || budget_class.stable_hash())

Where ``budget_class`` is a coarse bucket, not the exact float-budget,
so similar searches share an entry instead of fragmenting the cache.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisResult,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.core.version import DSL_VERSION


class _Clock(Protocol):
    def time(self) -> float: ...


# TTL constants — spec §14.3.
TTL_SUCCESS_DAYS: float = 30.0
TTL_PARTIAL_DAYS: float = 7.0
TTL_NO_SOLUTION_DAYS: float = 1.0


def _ttl_for(status: SynthesisStatus) -> float:
    """Seconds. Success / partial / no-solution map to spec defaults."""
    if status == SynthesisStatus.SUCCESS:
        return TTL_SUCCESS_DAYS * 86400.0
    if status == SynthesisStatus.PARTIAL:
        return TTL_PARTIAL_DAYS * 86400.0
    if status == SynthesisStatus.NO_SOLUTION:
        return TTL_NO_SOLUTION_DAYS * 86400.0
    # Timeout / budget / sandbox / error are *not* cached — they reflect
    # the limits of this run, not a stable property of the spec.
    return 0.0


@dataclass(frozen=True)
class CacheEntry:
    """One stored synthesis outcome (spec §14.2)."""

    spec_hash: str
    dsl_version: str
    program_source: str | None
    program_hash: str | None
    status: SynthesisStatus
    score: float
    confidence: float
    cost_seconds: float
    created_at: float
    last_used_at: float
    use_count: int = 0


def _budget_bucket_hash(budget: Budget) -> str:
    """SHA-256 of the budget's coarse bucket class.

    Using the bucket (e.g. ``depth_3_wc_30s``) instead of the exact
    floats means searches with slightly different wall-clocks share a
    cache entry — the spec calls this out as essential for hit-rate.
    """
    return hashlib.sha256(budget.bucket_class().encode("utf-8")).hexdigest()


def cache_key(
    spec: TaskSpec,
    budget: Budget,
    dsl_version: str = DSL_VERSION,
) -> str:
    """Compute the canonical cache key for a (spec, budget, dsl) triple."""
    payload = b"||".join(
        [
            spec.stable_hash().encode("utf-8"),
            dsl_version.encode("utf-8"),
            _budget_bucket_hash(budget).encode("utf-8"),
        ]
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class PSECache:
    """In-memory cache of :class:`SynthesisResult` keyed by spec/budget/DSL.

    Thread-safety: not designed for concurrent writes. Phase 1 callers
    are single-threaded (the search engine is synchronous); Phase 2
    will swap this for the Tactical Memory wire-protocol.

    DSL-version invalidation: passing ``dsl_version`` (default
    :data:`DSL_VERSION`) means a major-bump in the DSL silently
    invalidates every cached entry without an explicit purge call —
    the new key won't match any existing entry.
    """

    def __init__(
        self,
        *,
        dsl_version: str = DSL_VERSION,
        clock: _Clock = time,
    ) -> None:
        self._dsl_version = dsl_version
        self._clock = clock
        self._entries: dict[str, CacheEntry] = {}

    # -- Public API --------------------------------------------------

    def get(self, spec: TaskSpec, budget: Budget) -> CacheEntry | None:
        """Return a fresh cache entry, or ``None`` if absent / expired.

        Refreshes ``last_used_at`` on a hit so the LRU-ish use_count
        and recency stay accurate.
        """
        key = cache_key(spec, budget, dsl_version=self._dsl_version)
        entry = self._entries.get(key)
        if entry is None:
            return None
        ttl = _ttl_for(entry.status)
        now = self._now()
        if ttl > 0 and now - entry.created_at > ttl:
            # Expired — drop it.
            del self._entries[key]
            return None
        # Refresh last_used_at + bump use_count.
        refreshed = CacheEntry(
            spec_hash=entry.spec_hash,
            dsl_version=entry.dsl_version,
            program_source=entry.program_source,
            program_hash=entry.program_hash,
            status=entry.status,
            score=entry.score,
            confidence=entry.confidence,
            cost_seconds=entry.cost_seconds,
            created_at=entry.created_at,
            last_used_at=now,
            use_count=entry.use_count + 1,
        )
        self._entries[key] = refreshed
        return refreshed

    def put(
        self,
        spec: TaskSpec,
        budget: Budget,
        result: SynthesisResult,
    ) -> None:
        """Store *result* unless its status is non-cacheable.

        Statuses that are NOT cached (TTL = 0): TIMEOUT, BUDGET_EXCEEDED,
        SANDBOX_VIOLATION, ERROR — they reflect the *run's* limits, not
        a stable property of the spec.
        """
        if _ttl_for(result.status) == 0.0:
            return
        key = cache_key(spec, budget, dsl_version=self._dsl_version)
        program_source: str | None = None
        program_hash: str | None = None
        if result.program is not None and hasattr(result.program, "to_source"):
            try:
                program_source = result.program.to_source()
            except Exception:
                program_source = None
        if result.program is not None and hasattr(result.program, "stable_hash"):
            try:
                program_hash = result.program.stable_hash()
            except Exception:
                program_hash = None
        now = self._now()
        self._entries[key] = CacheEntry(
            spec_hash=spec.stable_hash(),
            dsl_version=self._dsl_version,
            program_source=program_source,
            program_hash=program_hash,
            status=result.status,
            score=result.score,
            confidence=result.confidence,
            cost_seconds=result.cost_seconds,
            created_at=now,
            last_used_at=now,
            use_count=0,
        )

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._entries

    # -- Internals ---------------------------------------------------

    def _now(self) -> float:
        # ``self._clock`` defaults to the ``time`` module; tests can
        # inject a fake whose ``.time()`` returns a fixed value.
        return float(self._clock.time())


__all__ = [
    "TTL_NO_SOLUTION_DAYS",
    "TTL_PARTIAL_DAYS",
    "TTL_SUCCESS_DAYS",
    "CacheEntry",
    "PSECache",
    "cache_key",
]
