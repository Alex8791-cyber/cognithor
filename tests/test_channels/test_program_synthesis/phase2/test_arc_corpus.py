# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-4 — ARC-AGI-3 corpus loader tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.types import TaskSpec
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.synthesis.arc_corpus import (
    ARCTask,
    corpus_benchmark_tasks,
    corpus_hash,
    load_corpus,
    load_task_file,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture builder — write a temp ARC-AGI-3-shaped JSON
# ---------------------------------------------------------------------------


def _write_task(
    path: Path, *, examples: list[tuple[list, list]], test: list[tuple[list, list]] | None = None
) -> None:
    payload = {
        "examples": [{"input": i, "output": o} for i, o in examples],
    }
    if test is not None:
        payload["test"] = [{"input": i, "output": o} for i, o in test]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_corpus(root: Path, task_ids: list[str]) -> None:
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    for tid in task_ids:
        _write_task(
            root / "tasks" / f"{tid}.json",
            examples=[([[1, 2]], [[1, 2]])],
            test=[([[3, 4]], [[3, 4]])],
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "test",
                "subsets": {
                    "train": {
                        "n": len(task_ids),
                        "task_files": [f"tasks/{t}.json" for t in task_ids],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Single-task loader
# ---------------------------------------------------------------------------


class TestLoadTaskFile:
    def test_loads_examples_key(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _write_task(path, examples=[([[1, 2]], [[3, 4]])])
        task = load_task_file(path)
        assert task.task_id == "t"
        assert len(task.examples) == 1
        assert np.array_equal(task.examples[0][0], np.array([[1, 2]], dtype=np.int8))
        assert np.array_equal(task.examples[0][1], np.array([[3, 4]], dtype=np.int8))

    def test_loads_train_key_legacy_arc_format(self, tmp_path: Path) -> None:
        # Original ARC schema uses 'train' instead of 'examples'.
        path = tmp_path / "t.json"
        path.write_text(
            json.dumps(
                {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]], "output": [[4]]}],
                }
            ),
            encoding="utf-8",
        )
        task = load_task_file(path)
        assert len(task.examples) == 1
        assert np.array_equal(task.examples[0][1], np.array([[2]], dtype=np.int8))

    def test_test_section_optional(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _write_task(path, examples=[([[1]], [[2]])], test=None)
        task = load_task_file(path)
        assert task.test == ()

    def test_int8_value_clamping(self, tmp_path: Path) -> None:
        # ARC values are 0-9; verify they fit in int8.
        path = tmp_path / "t.json"
        _write_task(path, examples=[([[0, 9]], [[5, 5]])])
        task = load_task_file(path)
        assert task.examples[0][0].dtype == np.int8


# ---------------------------------------------------------------------------
# Corpus loader (manifest-driven and full-directory)
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    def test_subset_filtering_via_manifest(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, ["a", "b", "c"])
        tasks = load_corpus(tmp_path, subset="train")
        assert len(tasks) == 3
        assert {t.task_id for t in tasks} == {"a", "b", "c"}

    def test_no_subset_loads_all(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, ["a", "b"])
        # Add an extra task not in any subset; full-load should pick it up.
        _write_task(
            tmp_path / "tasks" / "z.json",
            examples=[([[1]], [[1]])],
        )
        tasks = load_corpus(tmp_path, subset=None)
        assert {t.task_id for t in tasks} == {"a", "b", "z"}

    def test_unknown_subset_raises(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, ["a"])
        with pytest.raises(KeyError, match="not in manifest"):
            load_corpus(tmp_path, subset="held_out")

    def test_missing_manifest_raises_when_subset_requested(self, tmp_path: Path) -> None:
        (tmp_path / "tasks").mkdir()
        with pytest.raises(FileNotFoundError, match="manifest"):
            load_corpus(tmp_path, subset="train")

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tasks").mkdir()
        with pytest.raises(FileNotFoundError, match="No ARC"):
            load_corpus(tmp_path, subset=None)


# ---------------------------------------------------------------------------
# TaskSpec / BenchmarkTask projection
# ---------------------------------------------------------------------------


class TestProjections:
    def test_to_task_spec_yields_taskspec(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _write_task(
            path,
            examples=[([[1, 2]], [[1, 2]])],
            test=[([[3, 4]], [[3, 4]])],
        )
        task = load_task_file(path)
        spec = task.to_task_spec()
        assert isinstance(spec, TaskSpec)
        assert len(spec.examples) == 1
        assert len(spec.held_out) == 1

    def test_corpus_benchmark_tasks_helper(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, ["a", "b"])
        tasks = corpus_benchmark_tasks(tmp_path, subset="train")
        assert len(tasks) == 2
        assert all(t.wall_clock_budget_seconds == 5.0 for t in tasks)


# ---------------------------------------------------------------------------
# Hash stability — pinning the corpus
# ---------------------------------------------------------------------------


class TestCorpusHash:
    def test_same_corpus_yields_same_hash(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, ["a", "b"])
        h1 = corpus_hash(load_corpus(tmp_path, subset="train"))
        h2 = corpus_hash(load_corpus(tmp_path, subset="train"))
        assert h1 == h2

    def test_different_examples_yield_different_hash(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, ["a"])
        h1 = corpus_hash(load_corpus(tmp_path, subset="train"))
        # Modify one task's output and re-hash.
        _write_task(
            tmp_path / "tasks" / "a.json",
            examples=[([[1]], [[99]])],
            test=[([[3, 4]], [[3, 4]])],
        )
        h2 = corpus_hash(load_corpus(tmp_path, subset="train"))
        assert h1 != h2


# ---------------------------------------------------------------------------
# Real corpus integration test — read the committed cognithor_bench/arc_agi3
# ---------------------------------------------------------------------------


class TestCommittedCorpus:
    def test_train_subset_loads_eight_tasks(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3"
        if not root.exists():
            pytest.skip(f"cognithor_bench/arc_agi3 not found at {root}")
        tasks = load_corpus(root, subset="train")
        assert len(tasks) == 8
        for t in tasks:
            assert isinstance(t, ARCTask)
            assert len(t.examples) >= 1

    def test_held_out_subset_loads_four_tasks(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3"
        if not root.exists():
            pytest.skip(f"cognithor_bench/arc_agi3 not found at {root}")
        tasks = load_corpus(root, subset="held_out")
        assert len(tasks) == 4

    def test_hard_subset_loads_eight_tasks(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3"
        if not root.exists():
            pytest.skip(f"cognithor_bench/arc_agi3 not found at {root}")
        tasks = load_corpus(root, subset="hard")
        assert len(tasks) == 8
        # Every hard task has at least one example.
        for t in tasks:
            assert len(t.examples) >= 1
