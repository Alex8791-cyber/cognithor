# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Benchmark report serialisation + regression gate tests (Sprint-2 Track D)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.synthesis.benchmark import (
    BenchmarkSummary,
    BenchmarkTaskResult,
)
from cognithor.channels.program_synthesis.synthesis.benchmark_report import (
    RegressionVerdict,
    compare_to_baseline,
    dump_summary,
    load_summary,
    render_markdown,
    summary_from_json,
    summary_to_json,
)


def _summary(success: float = 0.5, *, p50: float = 0.5, p95: float = 1.0) -> BenchmarkSummary:
    return BenchmarkSummary(
        n_tasks=20,
        success_rate=success,
        cache_hit_rate=0.1,
        refined_rate=0.3,
        refinement_uplift_rate=0.5,
        p50_seconds=p50,
        p95_seconds=p95,
        per_task_results=(
            BenchmarkTaskResult(
                task_id="0001",
                score=0.97,
                elapsed_seconds=0.1,
                terminated_by="search_success",
                cache_hit=False,
                refined=False,
            ),
            BenchmarkTaskResult(
                task_id="0002",
                score=0.5,
                elapsed_seconds=0.3,
                terminated_by="search_exhausted",
                cache_hit=False,
                refined=True,
                refinement_path=("repair_full_llm",),
            ),
        ),
        errors=(("0003", "RuntimeError"),),
    )


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_summary_round_trip_preserves_all_fields(self) -> None:
        s = _summary()
        encoded = dump_summary(s, bundle_hash="sha256:abc")
        decoded = load_summary(encoded)
        # Top-level scalars match.
        assert decoded.n_tasks == s.n_tasks
        assert decoded.success_rate == s.success_rate
        assert decoded.cache_hit_rate == s.cache_hit_rate
        assert decoded.refined_rate == s.refined_rate
        assert decoded.refinement_uplift_rate == s.refinement_uplift_rate
        assert decoded.p50_seconds == s.p50_seconds
        assert decoded.p95_seconds == s.p95_seconds
        # Per-task rows + errors round-trip.
        assert len(decoded.per_task_results) == 2
        assert decoded.per_task_results[1].refinement_path == ("repair_full_llm",)
        assert decoded.errors == s.errors

    def test_unsupported_schema_raises(self) -> None:
        s = _summary()
        encoded = summary_to_json(s, schema_version=99)
        with pytest.raises(ValueError, match="schema_version"):
            summary_from_json(encoded)

    def test_bundle_hash_propagates_to_json(self) -> None:
        s = _summary()
        encoded = summary_to_json(s, bundle_hash="sha256:xyz")
        assert encoded["bundle_hash"] == "sha256:xyz"


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------


class TestRegressionGate:
    def test_no_regression_on_equal_scores(self) -> None:
        baseline = _summary(success=0.7)
        current = _summary(success=0.7)
        verdict = compare_to_baseline(baseline=baseline, current=current)
        assert verdict.regressed is False
        assert verdict.score_delta == 0.0

    def test_improvement_does_not_regress(self) -> None:
        baseline = _summary(success=0.5)
        current = _summary(success=0.8)
        verdict = compare_to_baseline(baseline=baseline, current=current)
        assert verdict.regressed is False
        assert verdict.score_delta == pytest.approx(0.3)

    def test_drop_within_tolerance_no_regression(self) -> None:
        # 0.7 → 0.65 drop = 0.05; tolerance 0.1 → no regression.
        baseline = _summary(success=0.7)
        current = _summary(success=0.65)
        verdict = compare_to_baseline(baseline=baseline, current=current, tolerance=0.1)
        assert verdict.regressed is False

    def test_drop_at_exactly_tolerance_no_regression(self) -> None:
        # Directive: "schlägt fehl bei Score-Regression > 10 %".
        # Strict > so a 10 % drop exactly does NOT fail.
        baseline = _summary(success=0.7)
        current = _summary(success=0.6)  # 10 pp drop
        verdict = compare_to_baseline(baseline=baseline, current=current, tolerance=0.1)
        assert verdict.regressed is False

    def test_drop_above_tolerance_regresses(self) -> None:
        # 0.7 → 0.55 drop = 0.15; tolerance 0.1 → regressed.
        baseline = _summary(success=0.7)
        current = _summary(success=0.55)
        verdict = compare_to_baseline(baseline=baseline, current=current, tolerance=0.1)
        assert verdict.regressed is True
        assert "REGRESSION" in verdict.messages[0]

    def test_invalid_tolerance_raises(self) -> None:
        baseline = _summary()
        current = _summary()
        with pytest.raises(ValueError, match="tolerance"):
            compare_to_baseline(baseline=baseline, current=current, tolerance=1.5)

    def test_p50_p95_deltas_recorded(self) -> None:
        baseline = _summary(success=0.7, p50=0.5, p95=1.0)
        current = _summary(success=0.7, p50=0.7, p95=1.5)
        verdict = compare_to_baseline(baseline=baseline, current=current)
        assert verdict.p50_delta == pytest.approx(0.2)
        assert verdict.p95_delta == pytest.approx(0.5)

    def test_messages_include_latency_lines(self) -> None:
        baseline = _summary(success=0.7)
        current = _summary(success=0.7)
        verdict = compare_to_baseline(baseline=baseline, current=current)
        joined = "\n".join(verdict.messages)
        assert "P50 latency" in joined
        assert "P95 latency" in joined


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestMarkdownRendering:
    def test_renders_aggregate_table(self) -> None:
        s = _summary(success=0.65)
        md = render_markdown(s, bundle_hash="sha256:test")
        assert "# PSE Phase-2 Benchmark Report" in md
        assert "| Metric | Value |" in md
        assert "65.0%" in md  # success rate
        assert "sha256:test" in md

    def test_renders_per_task_table(self) -> None:
        s = _summary()
        md = render_markdown(s)
        assert "| 0001 |" in md
        assert "| 0002 |" in md
        assert "repair_full_llm" in md

    def test_renders_errors_section(self) -> None:
        s = _summary()
        md = render_markdown(s)
        assert "## Errors" in md
        assert "0003" in md
        assert "RuntimeError" in md

    def test_renders_verdict_when_supplied(self) -> None:
        s = _summary()
        verdict = RegressionVerdict(
            regressed=True,
            score_delta=-0.2,
            p50_delta=0.0,
            p95_delta=0.0,
            messages=("REGRESSION test",),
        )
        md = render_markdown(s, verdict=verdict)
        assert "## Regression verdict" in md
        assert "REGRESSION test" in md

    def test_no_verdict_section_when_none(self) -> None:
        md = render_markdown(_summary(), verdict=None)
        assert "## Regression verdict" not in md


# ---------------------------------------------------------------------------
# Verdict dataclass contract
# ---------------------------------------------------------------------------


class TestVerdictDataclass:
    def test_is_frozen_and_hashable(self) -> None:
        v = RegressionVerdict(
            regressed=False,
            score_delta=0.0,
            p50_delta=0.0,
            p95_delta=0.0,
            messages=("ok",),
        )
        assert hash(v) == hash(v)
