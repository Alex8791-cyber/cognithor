# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Unit tests for the eval-suite metrics aggregator.

These run in *every* CI lane, even though the slow harness in
``test_arc_agi3_subset.py`` is skipped until the fixture set lands.
They pin the spec §18.3 metric definitions:

* ``Solved@30s`` / ``Solved@5s`` — count, not rate.
* ``Median-Time-Solved`` — median over solved tasks only.
* ``FP-Rate`` — fraction of *solved* tasks that pass demos but fail
  held-out.
* ``k1_threshold_met`` — spec §18.4 success-condition.
"""

from __future__ import annotations

import pytest

from tests.test_channels.test_program_synthesis.eval._metrics import (
    SubsetMetrics,
    TaskRunResult,
    aggregate,
    format_summary,
    k1_threshold_met,
)


def _r(
    task_id: str,
    *,
    solver: str = "pse",
    success: bool = True,
    cost_seconds: float = 0.5,
    demos_passed: bool = True,
    held_out_passed: bool = True,
) -> TaskRunResult:
    return TaskRunResult(
        task_id=task_id,
        solver=solver,
        success=success,
        cost_seconds=cost_seconds,
        demos_passed=demos_passed,
        held_out_passed=held_out_passed,
    )


class TestAggregateSolvedCounters:
    def test_solved_at_30s_includes_anything_under_30s(self) -> None:
        results = [
            _r("a", cost_seconds=4.9),
            _r("b", cost_seconds=29.99),
            _r("c", cost_seconds=30.0),  # boundary — counted
            _r("d", cost_seconds=30.01),  # over budget — not counted
            _r("e", success=False, cost_seconds=10.0),  # didn't solve
        ]
        m = aggregate(results, solver="pse", subset="train")
        assert m.solved_at_30s == 3
        assert m.solved_at_5s == 1
        assert m.n == 5

    def test_no_solved_returns_zero(self) -> None:
        results = [_r("a", success=False), _r("b", success=False)]
        m = aggregate(results, solver="pse", subset="train")
        assert m.solved_at_30s == 0
        assert m.solved_at_5s == 0


class TestMedianTimeSolved:
    def test_odd_count_returns_middle(self) -> None:
        results = [
            _r("a", cost_seconds=1.0),
            _r("b", cost_seconds=2.0),
            _r("c", cost_seconds=10.0),
        ]
        m = aggregate(results, solver="pse", subset="train")
        assert m.median_time_solved == 2.0

    def test_even_count_returns_average_of_middle_pair(self) -> None:
        results = [
            _r("a", cost_seconds=1.0),
            _r("b", cost_seconds=2.0),
            _r("c", cost_seconds=4.0),
            _r("d", cost_seconds=10.0),
        ]
        m = aggregate(results, solver="pse", subset="train")
        assert m.median_time_solved == 3.0

    def test_only_solved_tasks_count(self) -> None:
        results = [
            _r("a", cost_seconds=1.0, success=True),
            _r("b", cost_seconds=999.0, success=False),  # excluded
            _r("c", cost_seconds=3.0, success=True),
        ]
        m = aggregate(results, solver="pse", subset="train")
        assert m.median_time_solved == 2.0

    def test_no_solved_returns_none(self) -> None:
        results = [_r("a", success=False), _r("b", success=False)]
        m = aggregate(results, solver="pse", subset="train")
        assert m.median_time_solved is None


class TestFalsePositiveRate:
    def test_fp_rate_counts_demos_pass_but_held_out_fail(self) -> None:
        results = [
            _r("a", success=True, demos_passed=True, held_out_passed=False),
            _r("b", success=True, demos_passed=True, held_out_passed=True),
            _r("c", success=True, demos_passed=True, held_out_passed=False),
            _r("d", success=True, demos_passed=True, held_out_passed=True),
        ]
        m = aggregate(results, solver="pse", subset="held_out")
        assert m.fp_rate == 0.5  # 2 of 4 solved tasks were FPs

    def test_fp_rate_skips_unsolved_tasks(self) -> None:
        results = [
            _r("a", success=True, demos_passed=True, held_out_passed=False),
            _r("b", success=False),
            _r("c", success=False),
        ]
        m = aggregate(results, solver="pse", subset="held_out")
        # Only 1 solved task, all 1 was FP.
        assert m.fp_rate == 1.0

    def test_fp_rate_is_none_when_nothing_solved(self) -> None:
        results = [_r("a", success=False), _r("b", success=False)]
        m = aggregate(results, solver="pse", subset="held_out")
        assert m.fp_rate is None


class TestSolverFiltering:
    def test_aggregate_only_picks_results_for_named_solver(self) -> None:
        results = [
            _r("a", solver="pse", cost_seconds=2.0),
            _r("a", solver="baseline", cost_seconds=20.0),
            _r("b", solver="pse", cost_seconds=4.0),
            _r("b", solver="baseline", cost_seconds=10.0),
        ]
        pse = aggregate(results, solver="pse", subset="train")
        baseline = aggregate(results, solver="baseline", subset="train")
        assert pse.n == 2
        assert baseline.n == 2
        assert pse.median_time_solved == 3.0
        assert baseline.median_time_solved == 15.0


class TestK1Threshold:
    def _make(
        self,
        *,
        solver: str,
        solved30: int,
        solved5: int,
        subset: str = "train",
    ) -> SubsetMetrics:
        return SubsetMetrics(
            solver=solver,
            subset=subset,
            n=100,
            solved_at_30s=solved30,
            solved_at_5s=solved5,
            median_time_solved=2.0,
            fp_rate=None,
        )

    def test_pse_beats_baseline_by_5_passes(self) -> None:
        pse = self._make(solver="pse", solved30=50, solved5=20)
        baseline = self._make(solver="baseline", solved30=45, solved5=20)
        assert k1_threshold_met(pse, baseline)

    def test_pse_beats_baseline_by_4_fails(self) -> None:
        pse = self._make(solver="pse", solved30=49, solved5=20)
        baseline = self._make(solver="baseline", solved30=45, solved5=20)
        assert not k1_threshold_met(pse, baseline)

    def test_easy_task_regression_fails_threshold(self) -> None:
        # +5 on Solved@30s but -1 on Solved@5s = regression on easy tasks.
        pse = self._make(solver="pse", solved30=50, solved5=19)
        baseline = self._make(solver="baseline", solved30=45, solved5=20)
        assert not k1_threshold_met(pse, baseline)

    def test_subset_mismatch_raises(self) -> None:
        pse = self._make(solver="pse", solved30=50, solved5=20, subset="train")
        baseline = self._make(solver="baseline", solved30=45, solved5=20, subset="held_out")
        with pytest.raises(ValueError, match="subset mismatch"):
            k1_threshold_met(pse, baseline)


class TestFormatSummary:
    def test_format_summary_renders_full_table(self) -> None:
        pse = SubsetMetrics(
            solver="pse",
            subset="train",
            n=100,
            solved_at_30s=50,
            solved_at_5s=20,
            median_time_solved=2.5,
            fp_rate=0.05,
        )
        baseline = SubsetMetrics(
            solver="v0.78 NumPy solver",
            subset="train",
            n=100,
            solved_at_30s=45,
            solved_at_5s=20,
            median_time_solved=8.0,
            fp_rate=None,
        )
        out = format_summary(pse, baseline)
        assert "Subset: train (n=100)" in out
        assert "+5" in out  # Solved@30s delta
        assert "+0" in out  # Solved@5s delta
        assert "✅" in out
        assert "5.0%" in out  # FP rate

    def test_format_summary_subset_mismatch_raises(self) -> None:
        pse = SubsetMetrics(
            solver="pse",
            subset="train",
            n=1,
            solved_at_30s=0,
            solved_at_5s=0,
            median_time_solved=None,
            fp_rate=None,
        )
        baseline = SubsetMetrics(
            solver="baseline",
            subset="held_out",
            n=1,
            solved_at_30s=0,
            solved_at_5s=0,
            median_time_solved=None,
            fp_rate=None,
        )
        with pytest.raises(ValueError, match="subset mismatch"):
            format_summary(pse, baseline)
