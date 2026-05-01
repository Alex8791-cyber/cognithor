# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-9 — real ARC-AGI public corpus integration tests.

Verifies the full 800-task fchollet/ARC-AGI corpus committed at
``cognithor_bench/arc_agi3_real/`` loads correctly via the existing
``arc_corpus.load_corpus`` and produces the documented Sprint-9
reality-check baseline shape:
- 400 training tasks
- 400 evaluation tasks
- license file present + Apache-2.0 attribution

These tests are slow-ish (~3 s for full-corpus load) but stay under
the per-test budget. They run in CI to detect any future drift in the
corpus directory or manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.synthesis.arc_corpus import (
    ARCTask,
    corpus_benchmark_tasks,
    load_corpus,
    load_task_file,
)

REAL_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3_real"


def _corpus_present() -> bool:
    return (REAL_CORPUS_ROOT / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Bundle structure
# ---------------------------------------------------------------------------


class TestRealCorpusStructure:
    def test_corpus_directory_exists(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        assert REAL_CORPUS_ROOT.is_dir()

    def test_license_file_present(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        license_text = (REAL_CORPUS_ROOT / "LICENSE").read_text(encoding="utf-8")
        # Apache-2.0 attribution + upstream link.
        assert "Apache" in license_text
        assert "fchollet/ARC-AGI" in license_text

    def test_manifest_documents_apache_license(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        import json

        manifest = json.loads((REAL_CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source"]["license"] == "Apache-2.0"
        assert "fchollet/ARC-AGI" in manifest["source"]["repo"]


# ---------------------------------------------------------------------------
# Subset loading
# ---------------------------------------------------------------------------


class TestRealCorpusSubsets:
    def test_training_subset_loads_400(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        tasks = load_corpus(REAL_CORPUS_ROOT, subset="training")
        assert len(tasks) == 400
        for t in tasks[:5]:
            assert isinstance(t, ARCTask)
            assert len(t.examples) >= 1

    def test_evaluation_subset_loads_400(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        tasks = load_corpus(REAL_CORPUS_ROOT, subset="evaluation")
        assert len(tasks) == 400

    def test_training_and_evaluation_disjoint(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        train = {t.task_id for t in load_corpus(REAL_CORPUS_ROOT, subset="training")}
        eval_ = {t.task_id for t in load_corpus(REAL_CORPUS_ROOT, subset="evaluation")}
        assert train.isdisjoint(eval_), "training and evaluation subsets must not overlap"

    def test_unknown_subset_raises(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        with pytest.raises(KeyError, match="not in manifest"):
            load_corpus(REAL_CORPUS_ROOT, subset="all_50")


# ---------------------------------------------------------------------------
# Task content sanity
# ---------------------------------------------------------------------------


class TestTaskContent:
    def test_first_training_task_has_train_and_test(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        # 007bbfb7 is the alphabetically first training task in the
        # canonical fchollet/ARC-AGI corpus.
        path = REAL_CORPUS_ROOT / "tasks" / "training" / "007bbfb7.json"
        task = load_task_file(path)
        assert task.task_id == "007bbfb7"
        # Every ARC training task has at least one demo and one test.
        assert len(task.examples) >= 1
        assert len(task.test) >= 1

    def test_grid_values_in_arc_palette(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        # ARC colour palette is 0..9. Sample 10 random training tasks.
        tasks = load_corpus(REAL_CORPUS_ROOT, subset="training")[:10]
        for t in tasks:
            for inp, out in t.examples:
                assert int(inp.min()) >= 0 and int(inp.max()) <= 9, t.task_id
                assert int(out.min()) >= 0 and int(out.max()) <= 9, t.task_id


# ---------------------------------------------------------------------------
# BenchmarkTask projection
# ---------------------------------------------------------------------------


class TestBenchmarkTaskProjection:
    def test_corpus_benchmark_tasks_yields_400_for_training(self) -> None:
        if not _corpus_present():
            pytest.skip(f"real corpus not present at {REAL_CORPUS_ROOT}")
        tasks = corpus_benchmark_tasks(REAL_CORPUS_ROOT, subset="training")
        assert len(tasks) == 400
        first = tasks[0]
        # Default budget is the spec-default partition.
        assert abs(first.budget.mcts - 0.70) < 1e-9
