# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tactical-memory cache tests (spec §14)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    StageResult,
    SynthesisResult,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.integration.tactical_memory import (
    TTL_NO_SOLUTION_DAYS,
    TTL_PARTIAL_DAYS,
    TTL_SUCCESS_DAYS,
    PSECache,
    cache_key,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _spec() -> TaskSpec:
    return TaskSpec(
        examples=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
    )


def _ok_result(program=None) -> SynthesisResult:
    return SynthesisResult(
        status=SynthesisStatus.SUCCESS,
        program=program if program is not None else Program("rotate90", (InputRef(),), "Grid"),
        score=1.0,
        confidence=1.0,
        cost_seconds=0.5,
        cost_candidates=42,
        verifier_trace=(StageResult(stage="demo", passed=True),),
    )


# ---------------------------------------------------------------------------
# Fake clock (avoids time.sleep in TTL tests)
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self._t = t

    def time(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self) -> None:
        spec = _spec()
        b = Budget(max_depth=3, wall_clock_seconds=30.0)
        assert cache_key(spec, b) == cache_key(spec, b)

    def test_starts_with_sha256_prefix(self) -> None:
        spec = _spec()
        key = cache_key(spec, Budget())
        assert key.startswith("sha256:")
        assert len(key.split(":")[1]) == 64

    def test_changes_with_dsl_version(self) -> None:
        spec = _spec()
        b = Budget()
        a = cache_key(spec, b, dsl_version="1.2.0")
        c = cache_key(spec, b, dsl_version="2.0.0")
        assert a != c

    def test_changes_with_spec(self) -> None:
        a = cache_key(_spec(), Budget())
        other = TaskSpec(examples=((_g([[9]]), _g([[9]])),))
        assert a != cache_key(other, Budget())

    def test_bucket_class_groups_close_budgets(self) -> None:
        # bucket_class rounds the wall clock to the nearest int second,
        # so 30.0 and 30.4 share a bucket.
        a = cache_key(_spec(), Budget(wall_clock_seconds=30.0))
        b = cache_key(_spec(), Budget(wall_clock_seconds=30.4))
        assert a == b


# ---------------------------------------------------------------------------
# PSECache get/put
# ---------------------------------------------------------------------------


class TestPSECacheGetPut:
    def test_initial_empty(self) -> None:
        cache = PSECache()
        assert len(cache) == 0
        assert cache.get(_spec(), Budget()) is None

    def test_put_then_get_round_trip(self) -> None:
        cache = PSECache()
        spec = _spec()
        budget = Budget()
        cache.put(spec, budget, _ok_result())
        entry = cache.get(spec, budget)
        assert entry is not None
        assert entry.status == SynthesisStatus.SUCCESS
        assert entry.score == 1.0
        assert entry.confidence == 1.0
        assert entry.program_source == "rotate90(input)"
        assert entry.program_hash is not None
        assert entry.program_hash.startswith("sha256:")

    def test_get_increments_use_count(self) -> None:
        cache = PSECache()
        spec, budget = _spec(), Budget()
        cache.put(spec, budget, _ok_result())
        first = cache.get(spec, budget)
        second = cache.get(spec, budget)
        assert first is not None and second is not None
        assert second.use_count == first.use_count + 1

    def test_get_refreshes_last_used_at(self) -> None:
        clock = _FakeClock()
        cache = PSECache(clock=clock)
        spec, budget = _spec(), Budget()
        cache.put(spec, budget, _ok_result())
        first = cache.get(spec, budget)
        clock.advance(60.0)
        second = cache.get(spec, budget)
        assert first is not None and second is not None
        assert second.last_used_at > first.last_used_at

    def test_overwrite_preserves_dsl_version(self) -> None:
        cache = PSECache(dsl_version="1.2.3")
        spec, budget = _spec(), Budget()
        cache.put(spec, budget, _ok_result())
        entry = cache.get(spec, budget)
        assert entry is not None
        assert entry.dsl_version == "1.2.3"


# ---------------------------------------------------------------------------
# TTL behaviour (using fake clock)
# ---------------------------------------------------------------------------


class TestTTL:
    def test_success_30_day_ttl_constants(self) -> None:
        # Lock the spec §14.3 numeric defaults so a typo would surface.
        assert TTL_SUCCESS_DAYS == 30.0
        assert TTL_PARTIAL_DAYS == 7.0
        assert TTL_NO_SOLUTION_DAYS == 1.0

    def test_success_entry_expires_after_30_days(self) -> None:
        clock = _FakeClock()
        cache = PSECache(clock=clock)
        spec, budget = _spec(), Budget()
        cache.put(spec, budget, _ok_result())
        # Advance past 30 days.
        clock.advance(31.0 * 86400.0)
        entry = cache.get(spec, budget)
        assert entry is None

    def test_partial_expires_after_7_days(self) -> None:
        clock = _FakeClock()
        cache = PSECache(clock=clock)
        spec, budget = _spec(), Budget()
        partial = SynthesisResult(
            status=SynthesisStatus.PARTIAL,
            program=Program("rotate90", (InputRef(),), "Grid"),
            score=0.5,
            confidence=0.0,
            cost_seconds=1.0,
            cost_candidates=10,
        )
        cache.put(spec, budget, partial)
        clock.advance(8.0 * 86400.0)
        assert cache.get(spec, budget) is None

    def test_no_solution_expires_after_1_day(self) -> None:
        clock = _FakeClock()
        cache = PSECache(clock=clock)
        spec, budget = _spec(), Budget()
        no_sol = SynthesisResult(
            status=SynthesisStatus.NO_SOLUTION,
            program=None,
            score=0.0,
            confidence=0.0,
            cost_seconds=2.0,
            cost_candidates=999,
        )
        cache.put(spec, budget, no_sol)
        clock.advance(2.0 * 86400.0)
        assert cache.get(spec, budget) is None

    def test_timeout_not_cached(self) -> None:
        cache = PSECache()
        spec, budget = _spec(), Budget()
        timeout = SynthesisResult(
            status=SynthesisStatus.TIMEOUT,
            program=None,
            score=0.0,
            confidence=0.0,
            cost_seconds=30.0,
            cost_candidates=12345,
        )
        cache.put(spec, budget, timeout)
        assert len(cache) == 0

    def test_budget_exceeded_not_cached(self) -> None:
        cache = PSECache()
        spec, budget = _spec(), Budget()
        bx = SynthesisResult(
            status=SynthesisStatus.BUDGET_EXCEEDED,
            program=None,
            score=0.0,
            confidence=0.0,
            cost_seconds=15.0,
            cost_candidates=50_000,
        )
        cache.put(spec, budget, bx)
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# DSL-version invalidation (spec §14.4)
# ---------------------------------------------------------------------------


class TestDSLVersionInvalidation:
    def test_different_dsl_version_misses_cache(self) -> None:
        cache_a = PSECache(dsl_version="1.0.0")
        spec, budget = _spec(), Budget()
        cache_a.put(spec, budget, _ok_result())
        # Build a new cache claiming a different DSL version → key
        # differs → no cross-pollination of entries.
        cache_b = PSECache(dsl_version="2.0.0")
        assert cache_b.get(spec, budget) is None


# ---------------------------------------------------------------------------
# clear / __contains__
# ---------------------------------------------------------------------------


class TestCacheUtilities:
    def test_clear_removes_all_entries(self) -> None:
        cache = PSECache()
        cache.put(_spec(), Budget(), _ok_result())
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_contains_string_key(self) -> None:
        cache = PSECache()
        spec, budget = _spec(), Budget()
        cache.put(spec, budget, _ok_result())
        key = cache_key(spec, budget)
        assert key in cache
        assert "sha256:nope" not in cache

    def test_contains_rejects_non_string(self) -> None:
        cache = PSECache()
        assert (123 in cache) is False
        assert (None in cache) is False
