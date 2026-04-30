# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-AGI-3 train + held-out evaluation harness (spec §17.5 + §18).

Phase-1 ships a *minimal* fixture set under
``cognithor_bench/arc_agi3/`` (8 train + 4 held-out tasks) plus a
frozen ``baseline_v0.78.json``. The harness runs the PSE channel
against every task, reads the baseline, and asserts the spec K1
success threshold.

The full 100/30 diverse curated set is a Phase-2 expansion (spec §18.1
calls it out as a manual selection task). Phase 1 closes D5 and D7
with a real-but-small benchmark — the same harness scales without code
changes once the larger set lands.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cognithor.channels.program_synthesis.core.types import Budget

# Load integration first to break the sandbox ⇄ integration import cycle.
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)
from tests.test_channels.test_program_synthesis.eval._loader import load_manifest
from tests.test_channels.test_program_synthesis.eval._metrics import (
    TaskRunResult,
    aggregate,
    format_summary,
    k1_threshold_met,
)

if TYPE_CHECKING:
    from tests.test_channels.test_program_synthesis.eval._loader import (
        EvalManifest,
    )

# ---------------------------------------------------------------------------
# Fixture-set discovery
# ---------------------------------------------------------------------------

# Resolve the manifest path relative to the repo root so the test runs
# from any cwd. The ``parents`` walk goes:
#   __file__/eval -> test_program_synthesis -> test_channels -> tests -> repo
ARC_FIXTURE_DIR = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3"
MANIFEST_PATH = ARC_FIXTURE_DIR / "manifest.json"
BASELINE_PATH = ARC_FIXTURE_DIR / "baselines" / "baseline_v0.78.json"


def _manifest_present() -> bool:
    return MANIFEST_PATH.is_file()


pytestmark = [
    pytest.mark.skipif(
        not _manifest_present(),
        reason=(
            "ARC-AGI-3 fixture set not committed (cognithor_bench/arc_agi3/manifest.json missing)."
        ),
    ),
]


@pytest.fixture(scope="module")
def manifest() -> EvalManifest:
    return load_manifest(MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _pse_run(
    subset_name: str,
    manifest: EvalManifest,
    *,
    channel: ProgramSynthesisChannel | None = None,
) -> list[TaskRunResult]:
    """Run PSE against every task in *subset_name* and return results."""
    ch = channel if channel is not None else ProgramSynthesisChannel()
    out: list[TaskRunResult] = []
    subset = next(s for s in manifest.subsets if s.name == subset_name)
    for task_id, spec in subset.tasks:
        t0 = time.monotonic()
        result = ch.synthesize(SynthesisRequest(spec=spec, budget=Budget(max_depth=4)))
        elapsed = time.monotonic() - t0
        success = result.status.value == "success"
        out.append(
            TaskRunResult(
                task_id=task_id,
                solver="pse",
                success=success,
                cost_seconds=elapsed,
                demos_passed=success,
                # Held-out check is the same as demos for this minimal
                # set: the synthesised program either matches or doesn't.
                # With a richer fixture set, the harness will withhold
                # one demo per task and re-evaluate against it.
                held_out_passed=success,
            )
        )
    return out


def _baseline_results_for(subset_name: str, _manifest: EvalManifest) -> list[TaskRunResult]:
    """Read the frozen baseline JSON committed alongside the manifest."""
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    rows = payload["results"][subset_name]
    return [
        TaskRunResult(
            task_id=task_id,
            solver="baseline",
            success=bool(row.get("success", False)),
            cost_seconds=float(row.get("cost_seconds", 0.0)),
            demos_passed=bool(row.get("success", False)),
            held_out_passed=bool(row.get("success", False)),
        )
        for task_id, row in rows.items()
    ]


# ---------------------------------------------------------------------------
# Smoke checks — fixture set integrity.
# ---------------------------------------------------------------------------


def test_manifest_loads(manifest: EvalManifest) -> None:
    assert manifest.version
    assert manifest.subsets, "manifest declares no subsets"


def test_each_subset_has_at_least_one_task(manifest: EvalManifest) -> None:
    for subset in manifest.subsets:
        assert subset.tasks, f"subset {subset.name!r} has no task files"


def test_baseline_file_present() -> None:
    assert BASELINE_PATH.is_file(), (
        "Baseline JSON missing — spec §18.2 requires baseline_v0.78.json "
        "frozen before any PSE run on the train subset."
    )


# ---------------------------------------------------------------------------
# Full pipeline — D5/K1 + D7 (cache hit-rate).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_subset_meets_k1_threshold(manifest: EvalManifest) -> None:
    pse_results = _pse_run("train", manifest)
    baseline_results = _baseline_results_for("train", manifest)
    pse_metrics = aggregate(pse_results, solver="pse", subset="train")
    baseline_metrics = aggregate(baseline_results, solver="baseline", subset="train")
    assert k1_threshold_met(pse_metrics, baseline_metrics), (
        f"K1 threshold not met: "
        f"PSE Solved@30s={pse_metrics.solved_at_30s} vs "
        f"baseline Solved@30s={baseline_metrics.solved_at_30s} "
        f"(spec §18.4 requires +5).\n"
        f"\n{format_summary(pse_metrics, baseline_metrics)}"
    )


@pytest.mark.slow
def test_held_out_subset_no_regression(manifest: EvalManifest) -> None:
    pse_results = _pse_run("held_out", manifest)
    baseline_results = _baseline_results_for("held_out", manifest)
    pse_metrics = aggregate(pse_results, solver="pse", subset="held_out")
    baseline_metrics = aggregate(baseline_results, solver="baseline", subset="held_out")
    # No regression: PSE solves at least as many held-out tasks as the
    # baseline. Spec §18.4 requires +5 on Solved@30s for *train*; on
    # held-out the contract is "no regression on easy tasks".
    assert pse_metrics.solved_at_5s >= baseline_metrics.solved_at_5s


@pytest.mark.slow
def test_cache_hit_rate_above_80_percent_on_rerun(manifest: EvalManifest) -> None:
    """Spec D7: tactical-memory cache demonstrably effective on reruns.

    Run the train subset twice through the same channel instance.
    Run #2 should hit the cache for every task that succeeded on
    Run #1; spec target is hit-rate ≥ 80 %.
    """
    channel = ProgramSynthesisChannel()
    first_pass = _pse_run("train", manifest, channel=channel)
    successful_first = sum(1 for r in first_pass if r.success)
    if successful_first == 0:
        pytest.skip("first-pass solved nothing; cache hit-rate undefined")

    # Re-run — same task IDs, same Budget bucket → should all hit cache.
    hits = 0
    for _task_id, spec in next(s for s in manifest.subsets if s.name == "train").tasks:
        result = channel.synthesize(SynthesisRequest(spec=spec, budget=Budget(max_depth=4)))
        if result.cache_hit:
            hits += 1

    rate = hits / successful_first
    assert rate >= 0.80, (
        f"D7: cache hit-rate {rate:.1%} below 80 % "
        f"({hits} hits across {successful_first} previously-solved tasks)."
    )
