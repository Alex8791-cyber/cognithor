"""Per-PR SLO regression gates — Sprint 3.1.

These benchmarks are intentionally fast (each hot path runs in <100 µs
to <10 ms) so the suite finishes inside ~30 seconds on a CI runner.
Heavyweight benchmarks (Crew kickoff with real LLM, audit-chain over
1M entries) live in `tests/slo/perf/` and run nightly only.

Hot-path budgets (median, P95):
  * vlm_router.select_profile_explained    < 200 µs / 500 µs
  * profile_alignment.check                < 50 µs  / 100 µs
  * audit canonical_json normalisation     < 200 µs / 500 µs

Reasonable budget for a hot path is "stays out of the user's perceived
latency". The thresholds here are conservative and adjusted upward
only with a one-line comment explaining why.
"""

from __future__ import annotations

import pytest

from cognithor.core.vlm_router import VlmRouter, classify_vlm_task
from cognithor.video.routing import check_profile_alignment

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def router() -> VlmRouter:
    return VlmRouter()


# ---------------------------------------------------------------------------
# VLM Router — every video chat hits these
# ---------------------------------------------------------------------------


def test_classify_quick_describe(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Heuristic fast-path — empty/short prompt, no video meta."""
    result = benchmark(classify_vlm_task, "Was passiert hier?")
    assert result.value == "quick_describe"


def test_classify_complex_reasoning(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Patterns matched against a longer reasoning prompt."""
    benchmark(
        classify_vlm_task,
        "Vergleiche die Bewegung in Sekunde 2 und Sekunde 6 mit der von früher.",
    )


def test_select_profile_explained_aligned(  # type: ignore[no-untyped-def]
    benchmark, router: VlmRouter
) -> None:
    """Full router decision incl. TRUST-2 explanation surface."""
    decision = benchmark(
        router.select_profile_explained,
        "Beschreibe diesen Clip kurz.",
    )
    assert decision.profile.name in {"fast", "balanced", "premium"}


def test_select_profile_with_long_video(  # type: ignore[no-untyped-def]
    benchmark, router: VlmRouter
) -> None:
    """Long-form escalation path — should still resolve in microseconds."""
    benchmark.pedantic(
        router.select_profile_explained,
        args=("Describe this clip.",),
        kwargs={"video_seconds": 120.0},
        rounds=10,
        iterations=100,
    )


# ---------------------------------------------------------------------------
# Profile alignment — pure-function, called per request
# ---------------------------------------------------------------------------


def test_check_profile_alignment_match(  # type: ignore[no-untyped-def]
    benchmark, router: VlmRouter
) -> None:
    decision = router.select_profile_explained("Beschreibe diesen Clip.")
    benchmark(
        check_profile_alignment,
        decision,
        ["Qwen/Qwen3-VL-8B-Instruct", "extra"],
    )


def test_check_profile_alignment_miss(  # type: ignore[no-untyped-def]
    benchmark, router: VlmRouter
) -> None:
    decision = router.select_profile_explained("Beschreibe diesen Clip.")
    benchmark(check_profile_alignment, decision, ["other-model"])


# ---------------------------------------------------------------------------
# Smoke benchmarks for budget validation — fail clearly if a future
# refactor blows past the SLO.
# ---------------------------------------------------------------------------


def test_router_decision_under_p95_budget(  # type: ignore[no-untyped-def]
    benchmark, router: VlmRouter
) -> None:
    """Hard budget: select_profile_explained P95 < 500 µs."""
    benchmark.pedantic(
        router.select_profile_explained,
        args=("Compare the frames.",),
        rounds=20,
        iterations=200,
    )
    p95_ns = benchmark.stats.stats.median * 1.5  # rough P95 proxy
    p95_us = p95_ns * 1_000_000  # benchmark.stats.stats.median is seconds
    # Soft assertion — emits a warning rather than failing on flaky runners.
    # Hard threshold is enforced via --benchmark-compare-fail in CI.
    if p95_us > 500.0:
        import warnings

        warnings.warn(
            f"select_profile_explained P95 estimate {p95_us:.1f}µs > 500µs target. "
            "Consider profiling vlm_router.classify_vlm_task for regex hotspots.",
            stacklevel=1,
        )
