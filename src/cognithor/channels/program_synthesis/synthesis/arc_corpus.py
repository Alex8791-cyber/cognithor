# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-4 — ARC-AGI-3 corpus loader (reality-check infrastructure).

Loads ARC-AGI-3-shaped JSON tasks from a directory tree (typically
``cognithor_bench/arc_agi3/tasks/``) and converts them into
:class:`TaskSpec` instances the Phase-2 benchmark driver consumes.

The ARC-AGI-3 task format (per Chollet's official repo) is::

    {
      "examples": [
        {"input": [[..]], "output": [[..]]},
        ...
      ],
      "test": [
        {"input": [[..]], "output": [[..]]}
      ]
    }

Some legacy fixtures use ``train`` instead of ``examples`` (matching
the original ARC schema); both are accepted.

Usage::

    from cognithor.channels.program_synthesis.synthesis.arc_corpus import (
        load_corpus,
        corpus_benchmark_tasks,
    )
    tasks = corpus_benchmark_tasks("cognithor_bench/arc_agi3", subset="train")

The ``subset`` keyword reads the manifest's ``subsets`` map and
returns only the listed tasks. Pass ``subset=None`` to load every
``.json`` file in ``tasks/``.

Sprint-4 acceptance: load 8 training + 4 held-out tasks from the
existing ``cognithor_bench/arc_agi3`` corpus; run Phase-1 enumerative
search through the benchmark driver; persist the baseline score in
``.ci/arc_agi3_phase1_baseline.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.core.types import Example, TaskSpec
from cognithor.channels.program_synthesis.phase2.datatypes import PartitionedBudget
from cognithor.channels.program_synthesis.synthesis.benchmark import BenchmarkTask

if TYPE_CHECKING:
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ARCTask:
    """One ARC-AGI-3 task loaded from JSON.

    ``task_id`` is the filename stem (e.g. ``0001_rotate90``).
    ``examples`` are the train pairs the synthesizer learns from;
    ``test`` are held-out pairs the benchmark scores against (if
    present in the JSON).

    ``source_path`` is kept for telemetry / debugging — points to
    the file the task was loaded from.
    """

    task_id: str
    examples: tuple[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], ...]
    test: tuple[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], ...]
    source_path: Path

    def to_task_spec(self) -> TaskSpec:
        """Project to a :class:`TaskSpec` the engine consumes."""
        examples = tuple(_typed_example(p) for p in self.examples)
        held_out = tuple(_typed_example(p) for p in self.test)
        return TaskSpec(examples=examples, held_out=held_out)

    def to_benchmark_task(
        self,
        *,
        budget: PartitionedBudget,
        wall_clock_budget_seconds: float,
        current_alpha: float = 0.6,
    ) -> BenchmarkTask:
        return BenchmarkTask(
            task_id=self.task_id,
            spec=self.to_task_spec(),
            budget=budget,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
            current_alpha=current_alpha,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_task_file(path: Path) -> ARCTask:
    """Load one ARC-AGI-3 JSON task file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Accept both 'examples' (cognithor) and 'train' (canonical ARC).
    examples_raw = raw.get("examples") or raw.get("train") or []
    test_raw = raw.get("test", [])
    examples = tuple(_pair_from_dict(p) for p in examples_raw)
    test = tuple(_pair_from_dict(p) for p in test_raw)
    return ARCTask(
        task_id=path.stem,
        examples=examples,
        test=test,
        source_path=path,
    )


def load_corpus(
    corpus_root: Path | str,
    *,
    subset: str | None = None,
) -> tuple[ARCTask, ...]:
    """Load every ARC-AGI-3 task in ``corpus_root``.

    ``corpus_root`` is the directory containing ``manifest.json`` and
    a ``tasks/`` subdirectory. Pass ``subset="train"`` /
    ``subset="held_out"`` to filter via the manifest; pass ``subset=None``
    to load every ``tasks/*.json``.
    """
    root = Path(corpus_root)
    if subset is not None:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"corpus subset {subset!r} requested but {manifest_path} not found"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        subsets = manifest.get("subsets", {})
        if subset not in subsets:
            raise KeyError(f"subset {subset!r} not in manifest; available: {sorted(subsets)}")
        files = [root / rel for rel in subsets[subset]["task_files"]]
    else:
        files = sorted((root / "tasks").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No ARC-AGI-3 tasks found under {root!r}")
    return tuple(load_task_file(p) for p in files)


def corpus_benchmark_tasks(
    corpus_root: Path | str,
    *,
    subset: str | None = None,
    budget: PartitionedBudget | None = None,
    wall_clock_budget_seconds: float = 5.0,
    current_alpha: float = 0.6,
) -> tuple[BenchmarkTask, ...]:
    """Convenience: load a corpus + project every task to a BenchmarkTask."""
    if budget is None:
        budget = PartitionedBudget.from_spec_default()
    tasks = load_corpus(corpus_root, subset=subset)
    return tuple(
        t.to_benchmark_task(
            budget=budget,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
            current_alpha=current_alpha,
        )
        for t in tasks
    )


def corpus_hash(tasks: Iterable[ARCTask]) -> str:
    """SHA-256 over the canonical (id, examples, test) serialisation.

    Pinned by tests so any drift in the loaded corpus fails CI.
    """
    canonical: list[dict[str, Any]] = []
    for t in tasks:
        canonical.append(
            {
                "id": t.task_id,
                "examples": [
                    [
                        np.asarray(inp, dtype=np.int8).tolist(),
                        np.asarray(out, dtype=np.int8).tolist(),
                    ]
                    for inp, out in t.examples
                ],
                "test": [
                    [
                        np.asarray(inp, dtype=np.int8).tolist(),
                        np.asarray(out, dtype=np.int8).tolist(),
                    ]
                    for inp, out in t.test
                ],
            }
        )
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _pair_from_dict(d: dict[str, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    inp = np.asarray(d["input"], dtype=np.int8)
    out = np.asarray(d["output"], dtype=np.int8)
    return (inp, out)


def _typed_example(pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]) -> Example:
    inp, out = pair
    return (np.asarray(inp, dtype=np.int8), np.asarray(out, dtype=np.int8))


__all__ = [
    "ARCTask",
    "corpus_benchmark_tasks",
    "corpus_hash",
    "load_corpus",
    "load_task_file",
]
