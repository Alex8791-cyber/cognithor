# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PSE core-types unit tests (Week 1 gate)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    Constraint,
    StageResult,
    SynthesisResult,
    SynthesisStatus,
    TaskDomain,
    TaskSpec,
)


def _grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _task() -> TaskSpec:
    return TaskSpec(
        examples=(
            (_grid([[1, 2], [3, 4]]), _grid([[3, 1], [4, 2]])),
            (_grid([[0, 0], [1, 1]]), _grid([[1, 0], [1, 0]])),
        ),
        held_out=((_grid([[5, 5]]), _grid([[5], [5]])),),
        test_input=_grid([[7, 8]]),
        constraints=(Constraint(kind="size_preserving"),),
        domain=TaskDomain.ARC_AGI_3,
        annotations=(("symmetry_hint", "rotation"),),
    )


class TestTaskSpec:
    def test_stable_hash_is_deterministic(self) -> None:
        a, b = _task(), _task()
        assert a.stable_hash() == b.stable_hash()

    def test_stable_hash_changes_on_examples(self) -> None:
        a = _task()
        b = TaskSpec(
            examples=((_grid([[9, 9]]), _grid([[9], [9]])),),
        )
        assert a.stable_hash() != b.stable_hash()

    def test_stable_hash_changes_on_test_input(self) -> None:
        a = _task()
        b = TaskSpec(
            examples=a.examples,
            held_out=a.held_out,
            test_input=_grid([[0, 0]]),
            constraints=a.constraints,
            domain=a.domain,
            annotations=a.annotations,
        )
        assert a.stable_hash() != b.stable_hash()

    def test_stable_hash_starts_with_sha256_prefix(self) -> None:
        h = _task().stable_hash()
        assert h.startswith("sha256:")
        assert len(h.split(":")[1]) == 64

    def test_taskspec_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        spec = _task()
        with pytest.raises(FrozenInstanceError):
            spec.examples = ()  # type: ignore[misc]


class TestBudget:
    def test_defaults_match_spec(self) -> None:
        b = Budget()
        assert b.max_depth == 4
        assert b.max_candidates == 50_000
        assert b.wall_clock_seconds == 30.0
        assert b.max_memory_mb == 1024
        assert b.per_candidate_ms == 100
        assert b.cache_lookup is True

    def test_bucket_class_normalises_floats(self) -> None:
        a = Budget(wall_clock_seconds=30.0)
        b = Budget(wall_clock_seconds=30.4)
        assert a.bucket_class() == b.bucket_class()

    def test_bucket_class_distinguishes_depth(self) -> None:
        assert Budget(max_depth=3).bucket_class() != Budget(max_depth=4).bucket_class()

    def test_stable_hash_is_deterministic(self) -> None:
        assert Budget().stable_hash() == Budget().stable_hash()


class TestSynthesisResult:
    def test_no_solution_default_program_is_none(self) -> None:
        result = SynthesisResult(
            status=SynthesisStatus.NO_SOLUTION,
            program=None,
            score=0.0,
            confidence=0.0,
            cost_seconds=1.5,
            cost_candidates=1234,
        )
        assert result.program is None
        assert result.verifier_trace == ()
        assert result.cache_hit is False

    def test_partial_carries_stage_trace(self) -> None:
        trace = (
            StageResult(stage="syntax", passed=True, duration_ms=0.1),
            StageResult(stage="type", passed=True, duration_ms=0.2),
            StageResult(stage="demo", passed=False, detail="2/3", duration_ms=42.0),
        )
        result = SynthesisResult(
            status=SynthesisStatus.PARTIAL,
            program="<placeholder>",
            score=0.66,
            confidence=0.5,
            cost_seconds=2.0,
            cost_candidates=400,
            verifier_trace=trace,
        )
        assert len(result.verifier_trace) == 3
        assert result.verifier_trace[2].passed is False
