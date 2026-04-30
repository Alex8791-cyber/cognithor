# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Manifest + task loader for the ARC-AGI-3 eval suite (spec §17.5).

Reads ``cognithor_bench/arc_agi3/manifest.json`` (when it exists) and
the task JSON files it references, returning typed Phase-1
:class:`TaskSpec` objects ready to feed into the channel.

The loader is **deliberately strict** — D5's success depends on a
clean, byte-for-byte reproducible fixture set, so every schema
violation raises :class:`EvalManifestError` with the offending key
and file path. Silent fallbacks would defeat the purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cognithor.channels.program_synthesis.core.types import TaskSpec

if TYPE_CHECKING:
    from pathlib import Path


class EvalManifestError(ValueError):
    """Raised when the manifest or a task file violates the schema."""


@dataclass(frozen=True)
class SubsetSpec:
    """One named subset (train | held_out) from the manifest."""

    name: str
    expected_n: int
    task_ids: tuple[str, ...]
    tasks: tuple[tuple[str, TaskSpec], ...]


@dataclass(frozen=True)
class EvalManifest:
    """Top-level manifest with both subsets and the baseline reference."""

    version: str
    subsets: tuple[SubsetSpec, ...]
    baseline_name: str
    baseline_results_path: Path | None


def _resolve(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    # Refuse paths that escape the manifest root.
    try:
        p.relative_to(root.resolve())
    except ValueError as exc:
        raise EvalManifestError(f"task path {rel!r} escapes manifest root {root!s}") from exc
    return p


def _load_task(path: Path) -> TaskSpec:
    """Parse one task JSON into a :class:`TaskSpec`.

    Schema matches what ``cognithor pse run`` already accepts — the
    eval harness reuses the same shape so a task file is editable in
    isolation with the CLI before being added to the manifest.
    """
    if not path.is_file():
        raise EvalManifestError(f"task file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalManifestError(f"task {path}: invalid JSON: {exc}") from exc

    examples_raw = payload.get("examples")
    if not isinstance(examples_raw, list) or len(examples_raw) < 1:
        raise EvalManifestError(f"task {path}: 'examples' must be a non-empty list")
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    for i, ex in enumerate(examples_raw):
        if not isinstance(ex, dict):
            raise EvalManifestError(f"task {path}: example {i} is not an object")
        inp = ex.get("input")
        out = ex.get("output")
        if not isinstance(inp, list) or not isinstance(out, list):
            raise EvalManifestError(f"task {path}: example {i} missing input/output")
        try:
            inp_arr = np.array(inp, dtype=np.int8)
            out_arr = np.array(out, dtype=np.int8)
        except (ValueError, TypeError) as exc:
            raise EvalManifestError(f"task {path}: example {i} not a 2D int grid: {exc}") from exc
        if inp_arr.ndim != 2 or out_arr.ndim != 2:
            raise EvalManifestError(f"task {path}: example {i} grids must be 2D")
        examples.append((inp_arr, out_arr))
    return TaskSpec(examples=tuple(examples))


def load_manifest(manifest_path: Path) -> EvalManifest:
    """Load and validate the eval manifest at *manifest_path*.

    Raises :class:`EvalManifestError` on any schema violation. The
    returned object is fully-typed and frozen — pass it to the
    harness as-is.
    """
    if not manifest_path.is_file():
        raise EvalManifestError(f"manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalManifestError(f"manifest invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise EvalManifestError("manifest root must be an object")
    version = payload.get("version")
    if not isinstance(version, str):
        raise EvalManifestError("manifest 'version' must be a string")
    subsets_raw = payload.get("subsets")
    if not isinstance(subsets_raw, dict) or not subsets_raw:
        raise EvalManifestError("manifest 'subsets' must be a non-empty object")
    baseline_raw = payload.get("baseline")
    if not isinstance(baseline_raw, dict):
        raise EvalManifestError("manifest 'baseline' must be an object")
    baseline_name = baseline_raw.get("name")
    if not isinstance(baseline_name, str):
        raise EvalManifestError("manifest 'baseline.name' must be a string")
    baseline_results_rel = baseline_raw.get("results_path")
    baseline_results_path: Path | None = None
    if isinstance(baseline_results_rel, str):
        baseline_results_path = _resolve(manifest_path.parent, baseline_results_rel)

    subsets: list[SubsetSpec] = []
    for name, subset_payload in subsets_raw.items():
        if not isinstance(subset_payload, dict):
            raise EvalManifestError(f"manifest subset {name!r} must be an object")
        n = subset_payload.get("n")
        if not isinstance(n, int) or n <= 0:
            raise EvalManifestError(f"manifest subset {name!r}: 'n' must be a positive int")
        task_files_raw = subset_payload.get("task_files")
        if not isinstance(task_files_raw, list):
            raise EvalManifestError(f"manifest subset {name!r}: 'task_files' must be a list")
        loaded: list[tuple[str, TaskSpec]] = []
        ids: list[str] = []
        for rel in task_files_raw:
            if not isinstance(rel, str):
                raise EvalManifestError(
                    f"manifest subset {name!r}: task_file entries must be strings"
                )
            task_path = _resolve(manifest_path.parent, rel)
            spec = _load_task(task_path)
            ids.append(task_path.stem)
            loaded.append((task_path.stem, spec))
        subsets.append(
            SubsetSpec(
                name=name,
                expected_n=n,
                task_ids=tuple(ids),
                tasks=tuple(loaded),
            )
        )

    return EvalManifest(
        version=version,
        subsets=tuple(subsets),
        baseline_name=baseline_name,
        baseline_results_path=baseline_results_path,
    )
