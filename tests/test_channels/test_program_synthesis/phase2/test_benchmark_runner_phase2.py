# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-3 Track 1 — Phase-2 wiring smoke-tests for the benchmark runner."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.synthesis.benchmark_report import (
    load_summary,
)
from cognithor.channels.program_synthesis.synthesis.benchmark_runner import (
    _build_phase1_engine,
    _build_phase2_engine,
    _parse_args,
    _run_benchmark_async,
)

# ---------------------------------------------------------------------------
# Engine builders
# ---------------------------------------------------------------------------


class TestEngineBuilders:
    def test_phase1_builder_returns_engine(self) -> None:
        engine = _build_phase1_engine(success_threshold=0.95)
        assert engine is not None
        assert hasattr(engine, "synthesize")

    def test_phase2_builder_returns_wired_engine(self) -> None:
        engine = _build_phase2_engine()
        assert engine is not None
        assert hasattr(engine, "synthesize")

    def test_phase2_builder_accepts_refiner_min_score(self) -> None:
        # Just verify no exception on different thresholds.
        engine_strict = _build_phase2_engine(refiner_min_score=0.5)
        engine_loose = _build_phase2_engine(refiner_min_score=0.0)
        assert engine_strict is not None
        assert engine_loose is not None


# ---------------------------------------------------------------------------
# CLI smoke-tests — both engines run end-to-end without crashing
# ---------------------------------------------------------------------------


class TestCLISmoke:
    @pytest.mark.asyncio
    async def test_phase1_engine_runs_end_to_end(self, tmp_path: Path) -> None:
        out = tmp_path / "p1.json"
        args = _parse_args(
            [
                "--output",
                str(out),
                "--wall-clock-budget-seconds",
                "0.5",
            ]
        )
        exit_code = await _run_benchmark_async(args)
        assert exit_code == 0
        assert out.exists()
        summary = load_summary(out.read_text(encoding="utf-8"))
        assert summary.n_tasks == 20

    @pytest.mark.asyncio
    async def test_phase2_engine_runs_end_to_end(self, tmp_path: Path) -> None:
        out = tmp_path / "p2.json"
        args = _parse_args(
            [
                "--phase2",
                "--output",
                str(out),
                "--wall-clock-budget-seconds",
                "0.5",
            ]
        )
        exit_code = await _run_benchmark_async(args)
        assert exit_code == 0
        assert out.exists()
        summary = load_summary(out.read_text(encoding="utf-8"))
        assert summary.n_tasks == 20

    @pytest.mark.asyncio
    async def test_baseline_regression_check_passes_on_self_compare(self, tmp_path: Path) -> None:
        # Run once → use that as baseline → re-run; regression gate
        # passes (zero delta).
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        args = _parse_args(
            [
                "--output",
                str(first),
                "--wall-clock-budget-seconds",
                "0.5",
            ]
        )
        await _run_benchmark_async(args)
        args2 = _parse_args(
            [
                "--output",
                str(second),
                "--baseline",
                str(first),
                "--wall-clock-budget-seconds",
                "0.5",
            ]
        )
        exit_code = await _run_benchmark_async(args2)
        # Self-compare → no regression → exit 0.
        assert exit_code == 0


# ---------------------------------------------------------------------------
# A/B-test invariant: both engines see the same n_tasks
# ---------------------------------------------------------------------------


class TestABInvariant:
    @pytest.mark.asyncio
    async def test_both_engines_run_all_twenty_tasks(self, tmp_path: Path) -> None:
        # Sprint-3 directive's first invariant: regardless of uplift,
        # both engines must process every fixture (no silent drops).
        p1 = tmp_path / "p1.json"
        p2 = tmp_path / "p2.json"
        await _run_benchmark_async(
            _parse_args(["--output", str(p1), "--wall-clock-budget-seconds", "0.5"])
        )
        await _run_benchmark_async(
            _parse_args(
                [
                    "--phase2",
                    "--output",
                    str(p2),
                    "--wall-clock-budget-seconds",
                    "0.5",
                ]
            )
        )
        s1 = load_summary(p1.read_text(encoding="utf-8"))
        s2 = load_summary(p2.read_text(encoding="utf-8"))
        assert s1.n_tasks == 20
        assert s2.n_tasks == 20
        # Per-task IDs match (same fixture set).
        assert {r.task_id for r in s1.per_task_results} == {r.task_id for r in s2.per_task_results}


# ---------------------------------------------------------------------------
# Phase-2 reports a refined_rate field (even if 0.0)
# ---------------------------------------------------------------------------


class TestPhase2Reports:
    @pytest.mark.asyncio
    async def test_phase2_summary_includes_refinement_metrics(self, tmp_path: Path) -> None:
        out = tmp_path / "p2.json"
        args = _parse_args(
            [
                "--phase2",
                "--output",
                str(out),
                "--wall-clock-budget-seconds",
                "0.5",
            ]
        )
        await _run_benchmark_async(args)
        summary = load_summary(out.read_text(encoding="utf-8"))
        # The summary fields are always present, regardless of whether
        # refinement actually fired.
        assert hasattr(summary, "refined_rate")
        assert hasattr(summary, "refinement_uplift_rate")
        assert 0.0 <= summary.refined_rate <= 1.0
        assert 0.0 <= summary.refinement_uplift_rate <= 1.0


# ---------------------------------------------------------------------------
# Argparse contract
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_phase2_flag_default_false(self) -> None:
        args = _parse_args(["--output", "out.json"])
        assert args.phase2 is False

    def test_phase2_flag_set(self) -> None:
        args = _parse_args(["--phase2", "--output", "out.json"])
        assert args.phase2 is True

    def test_refiner_min_score_default_zero(self) -> None:
        args = _parse_args(["--output", "out.json"])
        assert args.refiner_min_score == 0.0

    def test_refiner_min_score_overridable(self) -> None:
        args = _parse_args(
            [
                "--output",
                "out.json",
                "--refiner-min-score",
                "0.42",
            ]
        )
        assert args.refiner_min_score == 0.42


# ---------------------------------------------------------------------------
# Single-task Phase-2 spot-check — the wiring actually runs the refiner path
# ---------------------------------------------------------------------------


class TestSingleTaskPhase2:
    @pytest.mark.asyncio
    async def test_phase2_engine_produces_score(self) -> None:
        # End-to-end: feed one fixture through the wired engine and
        # confirm we get back a score in [0, 1] (the actual value
        # depends on fixture difficulty).
        from cognithor.channels.program_synthesis.synthesis.leak_free_fixtures import (
            benchmark_tasks,
        )

        engine = _build_phase2_engine()
        tasks = list(benchmark_tasks(wall_clock_budget_seconds=0.5))
        result = await engine.synthesize(tasks[0].spec, tasks[0].budget)
        assert 0.0 <= result.final_score <= 1.0
        # Either Phase-1 succeeded immediately or refinement was attempted —
        # both paths are valid termination reasons.
        assert result.terminated_by in {
            "phase1_success",
            "refined_success",
            "refined_partial",
            "no_refinement_eligible",
            "no_solution",
        }


def test_module_importable() -> None:
    """Smoke: the runner module is importable without errors."""
    import cognithor.channels.program_synthesis.synthesis.benchmark_runner as m

    assert hasattr(m, "main")
    assert hasattr(m, "_build_phase1_engine")
    assert hasattr(m, "_build_phase2_engine")


# ---------------------------------------------------------------------------
# Helper to ensure asyncio doesn't leak between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_event_loop() -> None:
    # Each pytest-asyncio test gets its own loop; this fixture is just
    # a marker so we can extend isolation later if needed.
    asyncio.new_event_loop()
