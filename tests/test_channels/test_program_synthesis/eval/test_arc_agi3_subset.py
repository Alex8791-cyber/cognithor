# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-AGI-3 train + held-out evaluation harness (spec §17.5 + §18).

Status: **scaffold**. The test is skipped until
``cognithor_bench/arc_agi3/manifest.json`` lands. Once the fixture
files are committed, this test runs the full pipeline (PSE + frozen
baseline) and asserts the spec K1 success threshold.

The test is also marked ``slow`` so the regular CI suite does not
pull it in — it is intended for nightly runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_channels.test_program_synthesis.eval._loader import (
    EvalManifest,
    load_manifest,
)
from tests.test_channels.test_program_synthesis.eval._metrics import (
    SubsetMetrics,
    TaskRunResult,
    aggregate,
    k1_threshold_met,
)

# ---------------------------------------------------------------------------
# Fixture-set discovery
# ---------------------------------------------------------------------------

# Resolve the manifest path relative to the repo root so the test runs
# from any cwd. The ``parents`` walk goes:
#   __file__/eval -> test_program_synthesis -> test_channels -> tests -> repo
ARC_FIXTURE_DIR = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3"
MANIFEST_PATH = ARC_FIXTURE_DIR / "manifest.json"


def _manifest_present() -> bool:
    return MANIFEST_PATH.is_file()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _manifest_present(),
        reason=(
            "ARC-AGI-3 fixture set not committed yet "
            "(cognithor_bench/arc_agi3/manifest.json missing). "
            "This is the spec D5 deliverable — drop the manifest + "
            "task files in to switch the harness on."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Smoke check — no fixture data yet, just verify the harness wiring
# would resolve once the data lands.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> EvalManifest:
    return load_manifest(MANIFEST_PATH)


def test_manifest_loads(manifest: EvalManifest) -> None:
    assert manifest.version
    assert manifest.subsets, "manifest declares no subsets"


def test_each_subset_has_at_least_one_task(manifest: EvalManifest) -> None:
    for subset in manifest.subsets:
        assert subset.tasks, f"subset {subset.name!r} has no task files"


# ---------------------------------------------------------------------------
# Full pipeline — placeholder
# ---------------------------------------------------------------------------


def _pse_run(_subset_name: str, manifest: EvalManifest) -> list[TaskRunResult]:
    """Run PSE against every task in *_subset_name*.

    Phase-1 stub: not yet wired to the channel because the actual
    fixture set hasn't landed; the test that uses this is gated by
    the ``manifest_present`` skipif above. The wiring is straight
    ``ProgramSynthesisChannel().synthesize`` per task.
    """
    raise NotImplementedError("PSE eval driver lands with D5 alongside the fixture set.")


def _baseline_results_for(_subset_name: str, _manifest: EvalManifest) -> list[TaskRunResult]:
    """Read the frozen baseline JSON.

    Returns one ``TaskRunResult`` per task in the named subset.
    Not invoked until the manifest is committed.
    """
    raise NotImplementedError("baseline reader lands with D5 alongside baseline_v0.78.json.")


@pytest.mark.skip(
    reason=(
        "PSE eval driver + baseline reader land with D5; harness shape "
        "is committed so the implementation hooks have a stable home."
    )
)
def test_train_subset_meets_k1_threshold(manifest: EvalManifest) -> None:
    pse_results = _pse_run("train", manifest)
    baseline_results = _baseline_results_for("train", manifest)
    pse_metrics: SubsetMetrics = aggregate(pse_results, solver="pse", subset="train")
    baseline_metrics: SubsetMetrics = aggregate(baseline_results, solver="baseline", subset="train")
    assert k1_threshold_met(pse_metrics, baseline_metrics), (
        f"K1 threshold not met: PSE Solved@30s={pse_metrics.solved_at_30s} "
        f"vs baseline {baseline_metrics.solved_at_30s} "
        f"(spec §18.4 requires +5)."
    )
