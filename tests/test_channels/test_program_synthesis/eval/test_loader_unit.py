# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Unit tests for the eval-suite manifest loader.

The slow harness in ``test_arc_agi3_subset.py`` is skipped until the
real fixture set lands, but the loader is exercised here against
synthetic manifests built in ``tmp_path`` so schema errors surface
in regular CI.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_channels.test_program_synthesis.eval._loader import (
    EvalManifestError,
    load_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_task(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "examples": [
                    {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(root: Path, *, task_files: list[str], baseline_path: str | None = None) -> Path:
    body: dict = {
        "version": "1.2.0",
        "subsets": {
            "train": {"n": len(task_files), "task_files": task_files},
        },
        "baseline": {"name": "v0.78 NumPy solver"},
    }
    if baseline_path is not None:
        body["baseline"]["results_path"] = baseline_path
    p = root / "manifest.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


class TestLoadManifest:
    def test_minimal_manifest_round_trips(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_path = tasks_dir / "0001.json"
        _write_task(task_path)
        manifest_path = _write_manifest(tmp_path, task_files=["tasks/0001.json"])

        manifest = load_manifest(manifest_path)

        assert manifest.version == "1.2.0"
        assert len(manifest.subsets) == 1
        train = manifest.subsets[0]
        assert train.name == "train"
        assert train.expected_n == 1
        assert train.task_ids == ("0001",)
        loaded_id, loaded_spec = train.tasks[0]
        assert loaded_id == "0001"
        assert len(loaded_spec.examples) == 1
        inp, out = loaded_spec.examples[0]
        assert np.array_equal(inp, np.array([[1, 2], [3, 4]], dtype=np.int8))
        assert np.array_equal(out, np.array([[3, 1], [4, 2]], dtype=np.int8))

    def test_baseline_results_path_resolves(self, tmp_path: Path) -> None:
        (tmp_path / "tasks").mkdir()
        _write_task(tmp_path / "tasks" / "0001.json")
        (tmp_path / "baselines").mkdir()
        baseline_path = tmp_path / "baselines" / "baseline_v0.78.json"
        baseline_path.write_text("{}", encoding="utf-8")
        manifest_path = _write_manifest(
            tmp_path,
            task_files=["tasks/0001.json"],
            baseline_path="baselines/baseline_v0.78.json",
        )
        manifest = load_manifest(manifest_path)
        assert manifest.baseline_results_path is not None
        assert manifest.baseline_results_path.is_file()

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvalManifestError, match="does not exist"):
            load_manifest(tmp_path / "manifest.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "manifest.json"
        bad.write_text("{ not json", encoding="utf-8")
        with pytest.raises(EvalManifestError, match="invalid JSON"):
            load_manifest(bad)

    def test_missing_subsets_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "manifest.json"
        bad.write_text(
            json.dumps(
                {
                    "version": "1.2.0",
                    "subsets": {},
                    "baseline": {"name": "v0.78"},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(EvalManifestError, match="non-empty object"):
            load_manifest(bad)

    def test_missing_task_file_raises(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, task_files=["tasks/missing.json"])
        with pytest.raises(EvalManifestError, match="does not exist"):
            load_manifest(manifest_path)

    def test_path_escape_raises(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "evil.json"
        outside.write_text("{}", encoding="utf-8")
        manifest_path = _write_manifest(tmp_path, task_files=["../evil.json"])
        with pytest.raises(EvalManifestError, match="escapes manifest root"):
            load_manifest(manifest_path)

    def test_task_examples_must_be_2d(self, tmp_path: Path) -> None:
        (tmp_path / "tasks").mkdir()
        bad = tmp_path / "tasks" / "0001.json"
        bad.write_text(
            json.dumps({"examples": [{"input": [1, 2], "output": [3, 4]}]}),
            encoding="utf-8",
        )
        manifest_path = _write_manifest(tmp_path, task_files=["tasks/0001.json"])
        with pytest.raises(EvalManifestError, match="grids must be 2D"):
            load_manifest(manifest_path)
