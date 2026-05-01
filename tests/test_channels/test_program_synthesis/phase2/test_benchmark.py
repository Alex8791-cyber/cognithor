# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 benchmark runner tests (Sprint-1 plan task 12, spec §12.3 / §12.4)."""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.datatypes import PartitionedBudget
from cognithor.channels.program_synthesis.synthesis.benchmark import (
    BenchmarkSummary,
    BenchmarkTask,
    BenchmarkTaskResult,
    run_benchmark,
)
from cognithor.channels.program_synthesis.synthesis.engine import (
    Phase2SynthesisResult,
)


def _budget() -> PartitionedBudget:
    return PartitionedBudget.from_spec_default()


def _task(task_id: str = "t0", *, wall_clock: float = 5.0) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        spec={"task": task_id},
        budget=_budget(),
        wall_clock_budget_seconds=wall_clock,
    )


def _result(
    *,
    score: float = 0.5,
    elapsed: float = 0.1,
    terminated_by: str = "search_exhausted",
    cache_hit: bool = False,
    refined: bool = False,
    refinement_path: tuple[str, ...] = (),
) -> Phase2SynthesisResult:
    return Phase2SynthesisResult(
        program=None,
        score=score,
        cache_hit=cache_hit,
        refined=refined,
        terminated_by=terminated_by,  # type: ignore[arg-type]
        elapsed_seconds=elapsed,
        candidates_evaluated=1,
        refinement_path=refinement_path,
    )


class _StubEngine:
    """Engine stub: hands back a queued list of results, one per task."""

    def __init__(self, results: list[Phase2SynthesisResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[Any, float, float]] = []

    async def synthesize(
        self,
        spec: Any,
        budget: PartitionedBudget,
        *,
        wall_clock_budget_seconds: float,
        current_alpha: float = 0.6,
    ) -> Phase2SynthesisResult:
        self.calls.append((spec, wall_clock_budget_seconds, current_alpha))
        if not self._results:
            raise AssertionError("stub engine ran out of queued results")
        return self._results.pop(0)


# ---------------------------------------------------------------------------
# Driver — runs every task, captures every row
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    @pytest.mark.asyncio
    async def test_runs_every_task_and_returns_summary(self) -> None:
        engine = _StubEngine(
            [
                _result(score=0.99, elapsed=0.1, terminated_by="search_success"),
                _result(score=0.4, elapsed=0.2, terminated_by="search_exhausted"),
                _result(score=0.97, elapsed=0.3, terminated_by="refined_success", refined=True),
            ]
        )
        tasks = [_task(f"t{i}") for i in range(3)]
        summary = await run_benchmark(engine, tasks)

        assert isinstance(summary, BenchmarkSummary)
        assert summary.n_tasks == 3
        # 2 of 3 cleared 0.95 → 2/3 success rate.
        assert abs(summary.success_rate - 2 / 3) < 1e-9
        # 1 of 3 was refined → 1/3 refined rate; uplift 1/1 = 1.0.
        assert abs(summary.refined_rate - 1 / 3) < 1e-9
        assert summary.refinement_uplift_rate == 1.0
        # Cache hit rate 0%.
        assert summary.cache_hit_rate == 0.0
        # All three rows captured.
        assert {r.task_id for r in summary.per_task_results} == {"t0", "t1", "t2"}
        # Every spec was forwarded with the per-task budget.
        assert engine.calls == [
            ({"task": "t0"}, 5.0, 0.6),
            ({"task": "t1"}, 5.0, 0.6),
            ({"task": "t2"}, 5.0, 0.6),
        ]

    @pytest.mark.asyncio
    async def test_passes_per_task_alpha(self) -> None:
        engine = _StubEngine([_result(score=0.5)])
        task = BenchmarkTask(
            task_id="custom",
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=2.0,
            current_alpha=0.42,
        )
        await run_benchmark(engine, [task])
        assert engine.calls[0][2] == 0.42

    @pytest.mark.asyncio
    async def test_empty_task_list_yields_zero_summary(self) -> None:
        engine = _StubEngine([])
        summary = await run_benchmark(engine, [])
        assert summary.n_tasks == 0
        assert summary.success_rate == 0.0
        assert summary.p50_seconds == 0.0
        assert summary.p95_seconds == 0.0


# ---------------------------------------------------------------------------
# Aggregation — success, cache, refinement, percentiles
# ---------------------------------------------------------------------------


class TestAggregation:
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self) -> None:
        engine = _StubEngine(
            [
                _result(score=0.99, cache_hit=True, terminated_by="cache_hit"),
                _result(score=0.5, cache_hit=False),
                _result(score=0.97, cache_hit=True, terminated_by="cache_hit"),
                _result(score=0.4, cache_hit=False),
            ]
        )
        tasks = [_task(f"t{i}") for i in range(4)]
        summary = await run_benchmark(engine, tasks)
        assert summary.cache_hit_rate == 0.5

    @pytest.mark.asyncio
    async def test_refinement_uplift_rate_excludes_non_refined(self) -> None:
        # 2 refined: 1 winning, 1 not. Uplift = 1/2 = 0.5.
        engine = _StubEngine(
            [
                _result(score=0.99, refined=True, terminated_by="refined_success"),
                _result(score=0.5, refined=True, terminated_by="search_exhausted"),
                _result(score=0.99, refined=False, terminated_by="search_success"),
            ]
        )
        tasks = [_task(f"t{i}") for i in range(3)]
        summary = await run_benchmark(engine, tasks)
        assert summary.refinement_uplift_rate == 0.5

    @pytest.mark.asyncio
    async def test_uplift_rate_zero_when_no_refinement(self) -> None:
        engine = _StubEngine([_result(score=0.99) for _ in range(3)])
        tasks = [_task(f"t{i}") for i in range(3)]
        summary = await run_benchmark(engine, tasks)
        assert summary.refinement_uplift_rate == 0.0

    @pytest.mark.asyncio
    async def test_percentiles_p50_p95(self) -> None:
        elapsed = [0.1, 0.5, 1.0, 2.0, 5.0]
        engine = _StubEngine([_result(elapsed=t) for t in elapsed])
        tasks = [_task(f"t{i}") for i in range(5)]
        summary = await run_benchmark(engine, tasks)
        # P50 of [0.1, 0.5, 1.0, 2.0, 5.0] = 1.0.
        assert abs(summary.p50_seconds - 1.0) < 1e-9
        # P95 lies between 2.0 and 5.0; with linear interp on rank 3.8:
        #   lo=3, hi=4 → 2.0·(1-0.8) + 5.0·0.8 = 4.4
        assert abs(summary.p95_seconds - 4.4) < 1e-9


# ---------------------------------------------------------------------------
# Errors — one task crashes, others continue
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_failing_task_recorded_in_errors(self) -> None:
        class _PartialEngine:
            def __init__(self) -> None:
                self._calls = 0

            async def synthesize(
                self,
                spec: Any,
                budget: PartitionedBudget,
                *,
                wall_clock_budget_seconds: float,
                current_alpha: float = 0.6,
            ) -> Phase2SynthesisResult:
                self._calls += 1
                if self._calls == 2:
                    raise RuntimeError("boom")
                return _result(score=0.99, terminated_by="search_success")

        engine = _PartialEngine()
        tasks = [_task(f"t{i}") for i in range(3)]
        summary = await run_benchmark(engine, tasks)
        # 2 successful + 1 errored.
        assert summary.n_tasks == 2
        assert len(summary.errors) == 1
        assert summary.errors[0] == ("t1", "RuntimeError")
        # Success rate = 2/2 = 1.0 (excludes errored tasks).
        assert summary.success_rate == 1.0


# ---------------------------------------------------------------------------
# Dataclass contract
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_task_with_hashable_spec(self) -> None:
        # Frozen dataclass — hashable when fields are hashable.
        t = BenchmarkTask(
            task_id="x",
            spec="some_string_spec",
            budget=_budget(),
            wall_clock_budget_seconds=5.0,
        )
        assert hash(t) == hash(t)

    def test_task_result_is_frozen(self) -> None:
        r = BenchmarkTaskResult(
            task_id="x",
            score=0.5,
            elapsed_seconds=0.1,
            terminated_by="search_exhausted",
            cache_hit=False,
            refined=False,
        )
        assert hash(r) == hash(r)
