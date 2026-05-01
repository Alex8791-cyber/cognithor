# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-2 Track D — Benchmark report serialisation + regression gate.

The benchmark driver (PR #260) returns a :class:`BenchmarkSummary`.
This module:

* serialises a :class:`BenchmarkSummary` to a stable JSON shape so
  the nightly CI workflow can persist results between runs;
* deserialises a previously-persisted run as the *baseline* for
  regression comparison;
* implements the **regression gate** the Sprint-2 directive
  specified — "schlägt fehl bei Score-Regression > 10 %";
* emits a compact human-readable Markdown report for review, no
  Streamlit dependency. (A separate Streamlit page can read the
  same JSON later — Sprint-2 ships the data layer; the
  visualisation is decoupled.)

The regression gate is intentionally conservative: only the
``success_rate`` is gated. P50/P95 latency drifts are *reported*
(so reviewers see them) but don't auto-fail CI — runner-noise
on shared GitHub-Actions hardware would make latency gating too
flaky.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cognithor.channels.program_synthesis.synthesis.benchmark import (
    BenchmarkSummary,
    BenchmarkTaskResult,
)

# ---------------------------------------------------------------------------
# JSON shape
# ---------------------------------------------------------------------------


def summary_to_json(
    summary: BenchmarkSummary,
    *,
    bundle_hash: str = "",
    schema_version: int = 1,
) -> dict[str, Any]:
    """Serialise a :class:`BenchmarkSummary` to a JSON-friendly dict.

    ``bundle_hash`` is the optional ``leak_free_set_hash()`` digest
    so the consumer can verify that the report was produced against
    the same fixture set the baseline used. ``schema_version`` is
    bumped on incompatible changes so old baselines fail loudly.
    """
    return {
        "schema_version": schema_version,
        "bundle_hash": bundle_hash,
        "n_tasks": summary.n_tasks,
        "success_rate": summary.success_rate,
        "cache_hit_rate": summary.cache_hit_rate,
        "refined_rate": summary.refined_rate,
        "refinement_uplift_rate": summary.refinement_uplift_rate,
        "p50_seconds": summary.p50_seconds,
        "p95_seconds": summary.p95_seconds,
        "errors": [{"task_id": tid, "error": err} for tid, err in summary.errors],
        "per_task_results": [
            {
                "task_id": r.task_id,
                "score": r.score,
                "elapsed_seconds": r.elapsed_seconds,
                "terminated_by": r.terminated_by,
                "cache_hit": r.cache_hit,
                "refined": r.refined,
                "refinement_path": list(r.refinement_path),
            }
            for r in summary.per_task_results
        ],
    }


def summary_from_json(data: dict[str, Any]) -> BenchmarkSummary:
    """Inverse of :func:`summary_to_json`. Validates schema version."""
    sv = int(data.get("schema_version", 0))
    if sv != 1:
        raise ValueError(f"benchmark_report: unsupported schema_version {sv}; expected 1")

    rows = tuple(
        BenchmarkTaskResult(
            task_id=str(r["task_id"]),
            score=float(r["score"]),
            elapsed_seconds=float(r["elapsed_seconds"]),
            terminated_by=str(r["terminated_by"]),
            cache_hit=bool(r["cache_hit"]),
            refined=bool(r["refined"]),
            refinement_path=tuple(str(s) for s in r.get("refinement_path", [])),
        )
        for r in data.get("per_task_results", [])
    )
    errors = tuple((str(e["task_id"]), str(e["error"])) for e in data.get("errors", []))
    return BenchmarkSummary(
        n_tasks=int(data["n_tasks"]),
        success_rate=float(data["success_rate"]),
        cache_hit_rate=float(data["cache_hit_rate"]),
        refined_rate=float(data["refined_rate"]),
        refinement_uplift_rate=float(data["refinement_uplift_rate"]),
        p50_seconds=float(data["p50_seconds"]),
        p95_seconds=float(data["p95_seconds"]),
        per_task_results=rows,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionVerdict:
    """Outcome of a baseline-vs-current comparison.

    ``regressed`` is ``True`` iff
    ``baseline.success_rate - current.success_rate > tolerance``.
    The directive's "Score-Regression > 10 %" maps to ``tolerance=0.1``
    on the success-rate scale (which is itself a fraction in [0, 1]).

    ``score_delta`` is ``current - baseline``: positive = improvement.

    ``messages`` carries human-readable lines for the CI log /
    Markdown report.
    """

    regressed: bool
    score_delta: float
    p50_delta: float
    p95_delta: float
    messages: tuple[str, ...]


def compare_to_baseline(
    *,
    baseline: BenchmarkSummary,
    current: BenchmarkSummary,
    tolerance: float = 0.1,
) -> RegressionVerdict:
    """Compare ``current`` to ``baseline``; gate by success-rate drift.

    Sprint-2 directive: "schlägt fehl bei Score-Regression > 10 %".
    A *strict* greater-than is used so a 10 % drop exactly does NOT
    fail (matches the directive wording — the gate triggers above 10 %).
    """
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError(f"tolerance must be in [0, 1]; got {tolerance}")
    score_delta = current.success_rate - baseline.success_rate
    p50_delta = current.p50_seconds - baseline.p50_seconds
    p95_delta = current.p95_seconds - baseline.p95_seconds
    regressed = score_delta < -tolerance

    messages: list[str] = []
    if regressed:
        messages.append(
            f"REGRESSION: success_rate dropped by {-score_delta:.1%} "
            f"(baseline {baseline.success_rate:.1%} → current {current.success_rate:.1%}; "
            f"tolerance {tolerance:.1%})"
        )
    else:
        messages.append(
            f"OK: success_rate {current.success_rate:.1%} "
            f"(Δ {score_delta:+.1%} vs baseline {baseline.success_rate:.1%}; "
            f"tolerance {tolerance:.1%})"
        )
    messages.append(
        f"P50 latency: {current.p50_seconds:.3f}s "
        f"(Δ {p50_delta:+.3f}s vs baseline {baseline.p50_seconds:.3f}s)"
    )
    messages.append(
        f"P95 latency: {current.p95_seconds:.3f}s "
        f"(Δ {p95_delta:+.3f}s vs baseline {baseline.p95_seconds:.3f}s)"
    )
    return RegressionVerdict(
        regressed=regressed,
        score_delta=score_delta,
        p50_delta=p50_delta,
        p95_delta=p95_delta,
        messages=tuple(messages),
    )


# ---------------------------------------------------------------------------
# Markdown rendering — Streamlit-free, decoupled visualisation
# ---------------------------------------------------------------------------


def render_markdown(
    summary: BenchmarkSummary,
    *,
    title: str = "PSE Phase-2 Benchmark Report",
    bundle_hash: str = "",
    verdict: RegressionVerdict | None = None,
) -> str:
    """Compact Markdown report — what nightly CI's PR comment posts."""
    lines: list[str] = [f"# {title}", ""]
    if bundle_hash:
        lines.append(f"Bundle hash: `{bundle_hash}`")
        lines.append("")
    lines.extend(
        [
            "## Aggregate",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Tasks | {summary.n_tasks} |",
            f"| Success rate | {summary.success_rate:.1%} |",
            f"| Cache-hit rate | {summary.cache_hit_rate:.1%} |",
            f"| Refined rate | {summary.refined_rate:.1%} |",
            f"| Refinement uplift | {summary.refinement_uplift_rate:.1%} |",
            f"| P50 latency | {summary.p50_seconds:.3f}s |",
            f"| P95 latency | {summary.p95_seconds:.3f}s |",
            f"| Errors | {len(summary.errors)} |",
            "",
        ]
    )
    if verdict is not None:
        lines.extend(["## Regression verdict", ""])
        for msg in verdict.messages:
            lines.append(f"- {msg}")
        lines.append("")
    if summary.per_task_results:
        lines.extend(
            [
                "## Per-task",
                "",
                "| Task | Score | Elapsed | Terminated by | Refined | Path |",
                "| --- | ---: | ---: | --- | :-: | --- |",
            ]
        )
        for row in summary.per_task_results:
            path = "+".join(row.refinement_path) if row.refinement_path else "-"
            lines.append(
                f"| {row.task_id} | {row.score:.2f} | {row.elapsed_seconds:.3f}s "
                f"| {row.terminated_by} | {'✓' if row.refined else ' '} | {path} |"
            )
        lines.append("")
    if summary.errors:
        lines.extend(["## Errors", ""])
        for tid, err in summary.errors:
            lines.append(f"- `{tid}`: {err}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: dump + load
# ---------------------------------------------------------------------------


def dump_summary(summary: BenchmarkSummary, *, bundle_hash: str = "") -> str:
    """JSON-encode a :class:`BenchmarkSummary` (single-line, stable keys)."""
    return json.dumps(
        summary_to_json(summary, bundle_hash=bundle_hash),
        sort_keys=True,
        separators=(",", ":"),
    )


def load_summary(payload: str) -> BenchmarkSummary:
    """Inverse of :func:`dump_summary`."""
    data = json.loads(payload)
    return summary_from_json(data)


__all__ = [
    "RegressionVerdict",
    "compare_to_baseline",
    "dump_summary",
    "load_summary",
    "render_markdown",
    "summary_from_json",
    "summary_to_json",
]
