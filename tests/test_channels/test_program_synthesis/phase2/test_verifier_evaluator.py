# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""End-to-end Verifier evaluator tests (Sprint-2 plan task 4, spec §7.3.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.config import Phase2Config
from cognithor.channels.program_synthesis.phase2.verifier_evaluator import (
    VerifierEvaluation,
    VerifierEvaluator,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
    ProgramNode,
)
from cognithor.channels.program_synthesis.search.executor import (
    InProcessExecutor,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


class _SyntheticSpec:
    """Minimal duck-typed TaskSpec — only `examples` is read."""

    def __init__(self, examples: list[tuple[Any, Any]]) -> None:
        self.examples = tuple(examples)


def _identity_program() -> Program:
    return Program(primitive="identity", children=(InputRef(),), output_type="Grid")


# ---------------------------------------------------------------------------
# Output bound — score in [0, 1] by construction (acceptance criterion)
# ---------------------------------------------------------------------------


class TestScoreBoundedInUnitInterval:
    def test_perfect_program_scores_high(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        spec = _SyntheticSpec([(_g([[1, 2]]), _g([[1, 2]]))])
        # identity(input) returns input → demo_pass=1.0
        result = evaluator.evaluate(_identity_program(), spec)
        assert isinstance(result, VerifierEvaluation)
        assert 0.0 <= result.final_score <= 1.0
        assert result.inputs.demo_pass_rate == 1.0

    def test_failing_program_scores_low(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        # rotate90 on a non-square grid — actual won't equal expected.
        spec = _SyntheticSpec([(_g([[1, 2, 3]]), _g([[1, 2, 3]]))])
        program = Program(primitive="rotate90", children=(InputRef(),), output_type="Grid")
        result = evaluator.evaluate(program, spec)
        assert 0.0 <= result.final_score <= 1.0
        assert result.inputs.demo_pass_rate == 0.0

    def test_score_always_in_unit_interval_across_random_demos(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        rng = np.random.default_rng(seed=42)
        for _ in range(20):
            shape = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
            inp = rng.integers(0, 10, size=shape, dtype=np.int8)
            expected = rng.integers(0, 10, size=shape, dtype=np.int8)
            spec = _SyntheticSpec([(inp, expected)])
            result = evaluator.evaluate(_identity_program(), spec)
            assert 0.0 <= result.final_score <= 1.0


# ---------------------------------------------------------------------------
# Sub-score wiring
# ---------------------------------------------------------------------------


class TestSubScoreWiring:
    def test_demo_pass_rate_proportional(self) -> None:
        # Two demos, identity gets one right, one wrong.
        evaluator = VerifierEvaluator(InProcessExecutor())
        spec = _SyntheticSpec(
            [
                (_g([[1, 2]]), _g([[1, 2]])),  # identity correct
                (_g([[1, 2]]), _g([[3, 4]])),  # identity wrong
            ]
        )
        result = evaluator.evaluate(_identity_program(), spec)
        assert result.inputs.demo_pass_rate == 0.5

    def test_failed_execution_marks_ok_false(self) -> None:
        # rotate90 on a 1-D grid → executor raises.
        class _BrokenExec:
            def execute(self, program: ProgramNode, input_grid: Any) -> Any:
                from cognithor.channels.program_synthesis.search.executor import (
                    ExecutionResult,
                )

                return ExecutionResult(ok=False, error="StubError")

        evaluator = VerifierEvaluator(_BrokenExec())  # type: ignore[arg-type]
        spec = _SyntheticSpec([(_g([[1]]), _g([[1]]))])
        result = evaluator.evaluate(_identity_program(), spec)
        assert result.ok_per_demo == (False,)
        assert result.actual_outputs == (None,)
        # demo_pass_rate is 0; everything else valid.
        assert result.inputs.demo_pass_rate == 0.0
        assert 0.0 <= result.final_score <= 1.0

    def test_actual_outputs_recorded_per_demo(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        spec = _SyntheticSpec([(_g([[1, 2]]), _g([[3, 4]]))])
        result = evaluator.evaluate(_identity_program(), spec)
        # identity returns the input grid.
        assert len(result.actual_outputs) == 1
        assert np.array_equal(result.actual_outputs[0], _g([[1, 2]]))


# ---------------------------------------------------------------------------
# Aggregation honours config weights (Plan-Task 8 acceptance kept)
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_uses_default_weights(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        spec = _SyntheticSpec([(_g([[1, 2]]), _g([[1, 2]]))])
        result = evaluator.evaluate(_identity_program(), spec)
        # demo_pass = 1.0 → contributes 0.55 to final.
        assert result.final_score >= 0.55  # other terms only add

    def test_custom_weights_override(self) -> None:
        # All weight on demo_pass_rate.
        from cognithor.channels.program_synthesis.phase2.config import (
            VerifierScoreWeights,
        )

        cfg = Phase2Config(
            verifier_score_weights=VerifierScoreWeights(
                demo_pass_rate=1.0,
                partial_pixel_match=0.0,
                invariants_satisfied=0.0,
                triviality_score=0.0,
                suspicion_score=0.0,
            )
        )
        evaluator = VerifierEvaluator(InProcessExecutor(), config=cfg)
        spec = _SyntheticSpec([(_g([[1, 2]]), _g([[1, 2]]))])
        result = evaluator.evaluate(_identity_program(), spec)
        assert abs(result.final_score - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Optional invariants hook
# ---------------------------------------------------------------------------


class TestInvariantsHook:
    def test_default_invariants_returns_one(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        spec = _SyntheticSpec([(_g([[1, 2]]), _g([[1, 2]]))])
        result = evaluator.evaluate(_identity_program(), spec)
        assert result.inputs.invariants_satisfied == 1.0

    def test_custom_invariants_callable_used(self) -> None:
        seen: list[tuple[Any, ...]] = []

        def my_invariant(
            _program: ProgramNode,
            _spec: Any,
            actual: tuple[Any, ...],
            expected: tuple[Any, ...],
        ) -> float:
            seen.append((actual, expected))
            return 0.5

        evaluator = VerifierEvaluator(InProcessExecutor(), invariants_check=my_invariant)
        spec = _SyntheticSpec([(_g([[1, 2]]), _g([[1, 2]]))])
        result = evaluator.evaluate(_identity_program(), spec)
        assert result.inputs.invariants_satisfied == 0.5
        assert len(seen) == 1

    def test_out_of_range_invariants_raises(self) -> None:
        def bad(*_args: Any, **_kw: Any) -> float:
            return 1.5

        evaluator = VerifierEvaluator(InProcessExecutor(), invariants_check=bad)
        spec = _SyntheticSpec([(_g([[1, 2]]), _g([[1, 2]]))])
        with pytest.raises(ValueError, match="invariants_check"):
            evaluator.evaluate(_identity_program(), spec)


# ---------------------------------------------------------------------------
# Empty examples corner-case
# ---------------------------------------------------------------------------


class TestEmptyExamples:
    def test_zero_examples_yields_well_defined_score(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        spec = _SyntheticSpec([])
        result = evaluator.evaluate(_identity_program(), spec)
        # demo_pass_rate = 0.0; partial_pixel = 0.0; triviality = 1.0;
        # invariants = 1.0 (default); suspicion remains in [0, 1].
        assert 0.0 <= result.final_score <= 1.0
        assert result.inputs.demo_pass_rate == 0.0


# ---------------------------------------------------------------------------
# VerifierEvaluation dataclass contract
# ---------------------------------------------------------------------------


class TestEvaluationDataclass:
    def test_is_frozen(self) -> None:
        evaluator = VerifierEvaluator(InProcessExecutor())
        spec = _SyntheticSpec([(_g([[1]]), _g([[1]]))])
        result = evaluator.evaluate(_identity_program(), spec)
        # Frozen → immutable.
        with pytest.raises(Exception):  # noqa: B017
            result.final_score = 0.0  # type: ignore[misc]
