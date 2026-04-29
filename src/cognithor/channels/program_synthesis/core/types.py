# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PSE core data types (spec §6).

All structures are frozen / immutable. Mutation is forbidden; equality is
structural; ``stable_hash`` returns a SHA-256 digest over a canonical
serialization and is used as the cache key (see spec §14.1).

The TaskSpec / Budget / SynthesisResult triple forms the public boundary
of the channel — every other module operates on these.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# 2D numpy grid with values 0..9 (ARC convention). int8 keeps the cache
# fingerprint compact and matches the existing NumPy solver in cognithor.arc.
type Grid = NDArray[np.int8]

# Color is a small int 0..9; alias kept for readability of signatures.
type Color = int

# Demo pair: (input grid, expected output grid).
type Example = tuple[Grid, Grid]


def _grid_to_canonical(grid: Grid) -> list[list[int]]:
    return [[int(v) for v in row] for row in np.asarray(grid).tolist()]


def _examples_canonical(examples: tuple[Example, ...]) -> list[list[Any]]:
    return [[_grid_to_canonical(inp), _grid_to_canonical(out)] for inp, out in examples]


class TaskDomain(str, Enum):
    """Origin of a TaskSpec — drives downstream classification."""

    ARC_AGI_3 = "arc_agi_3"
    ARC_AGI_2 = "arc_agi_2"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Constraint:
    """A constraint a valid program must satisfy (spec §6.1)."""

    kind: Literal["size_preserving", "color_preserving", "monotonic_size", "custom"]
    payload: tuple[tuple[str, Any], ...] = ()

    def to_canonical(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}


@dataclass(frozen=True)
class TaskSpec:
    """A unique description of a synthesis task.

    ``stable_hash`` is the cache key (spec §14.1). Held-out examples are
    excluded from the search and used by the verifier to detect overfit
    programs.
    """

    examples: tuple[Example, ...]
    held_out: tuple[Example, ...] = ()
    test_input: Grid | None = None
    constraints: tuple[Constraint, ...] = ()
    domain: TaskDomain = TaskDomain.ARC_AGI_3
    annotations: tuple[tuple[str, Any], ...] = ()

    def stable_hash(self) -> str:
        canonical = {
            "examples": _examples_canonical(self.examples),
            "held_out": _examples_canonical(self.held_out),
            "test_input": (
                _grid_to_canonical(self.test_input) if self.test_input is not None else None
            ),
            "constraints": [c.to_canonical() for c in self.constraints],
            "domain": self.domain.value,
            "annotations": dict(self.annotations),
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Budget:
    """Compute budget for one synthesis call (spec §6.1).

    Values are *hard* limits — the search engine MUST short-circuit when
    any of them is reached.
    """

    max_depth: int = 4
    max_candidates: int = 50_000
    wall_clock_seconds: float = 30.0
    max_memory_mb: int = 1024
    per_candidate_ms: int = 100
    cache_lookup: bool = True

    def stable_hash(self) -> str:
        canonical = {
            "max_depth": self.max_depth,
            "max_candidates": self.max_candidates,
            "wall_clock_seconds": round(float(self.wall_clock_seconds), 3),
            "max_memory_mb": self.max_memory_mb,
            "per_candidate_ms": self.per_candidate_ms,
            "cache_lookup": self.cache_lookup,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def bucket_class(self) -> str:
        """Coarse bucket used for cache-key classification.

        See spec §14.1 — exact float budgets would explode the cache.
        """
        return f"depth_{self.max_depth}_wc_{round(self.wall_clock_seconds)}s"


class SynthesisStatus(str, Enum):
    """Outcome of a synthesis call."""

    SUCCESS = "success"
    PARTIAL = "partial"
    NO_SOLUTION = "no_solution"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget"
    SANDBOX_VIOLATION = "sandbox"
    ERROR = "error"


@dataclass(frozen=True)
class StageResult:
    """One verifier stage outcome (spec §10)."""

    stage: Literal["syntax", "type", "demo", "property", "held_out"]
    passed: bool
    detail: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class SynthesisResult:
    """Public result of a synthesis call.

    ``program`` is intentionally typed as ``Any`` here to break a circular
    import with ``search.candidate.Program``. The ``program`` attribute is
    a :class:`~cognithor.channels.program_synthesis.search.candidate.Program`
    or ``None`` when no solution was found.
    """

    status: SynthesisStatus
    program: Any
    score: float
    confidence: float
    cost_seconds: float
    cost_candidates: int
    verifier_trace: tuple[StageResult, ...] = ()
    cache_hit: bool = False
    annotations: tuple[tuple[str, Any], ...] = ()
