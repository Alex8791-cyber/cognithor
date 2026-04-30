# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""CEGIS-loop tests (Sprint-1 plan task 9 slice, spec §6.5.3)."""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import Phase2Config
from cognithor.channels.program_synthesis.refiner import (
    CEGISLoop,
    CEGISResult,
    CounterExample,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)


def _prog(label: str) -> Program:
    return Program(primitive=label, children=(InputRef(),), output_type="Grid")


# ---------------------------------------------------------------------------
# Termination paths
# ---------------------------------------------------------------------------


class TestCEGISLoop:
    def test_initial_program_passes_immediately(self) -> None:
        # Evaluator returns no counter-examples → loop exits at iter 0.
        loop = CEGISLoop(
            synthesizer=lambda *_: pytest.fail("synthesizer should not be called"),
            evaluator=lambda _p, _d: [],
        )
        result = loop.run(
            _prog("rotate90"),
            demos=[(0, 0)],
            wall_clock_budget_seconds=10.0,
        )
        assert isinstance(result, CEGISResult)
        assert result.terminated_by == "all_demos_pass"
        assert result.iterations == 0

    def test_converges_after_two_iterations(self) -> None:
        # Synthesizer's nth call returns a fresh program. The
        # evaluator returns failing demos until iter 2.
        synthesizer_calls = {"n": 0}

        def synthesizer(_program: Any, _ce: list[CounterExample], _budget: float) -> Program:
            synthesizer_calls["n"] += 1
            return _prog(f"candidate_v{synthesizer_calls['n']}")

        eval_count = {"n": 0}

        def evaluator(_p: Any, demos: list[tuple[Any, Any]]) -> list[CounterExample]:
            eval_count["n"] += 1
            if eval_count["n"] >= 3:  # 1 initial + 2 post-synthesis
                return []
            return [
                CounterExample(
                    input_grid=demos[0][0], expected_output=demos[0][1], actual_output=None
                )
            ]

        loop = CEGISLoop(synthesizer=synthesizer, evaluator=evaluator)
        result = loop.run(
            _prog("rotate90"),
            demos=[(0, 1)],
            wall_clock_budget_seconds=10.0,
        )
        assert result.terminated_by == "all_demos_pass"
        assert result.iterations == 2
        assert isinstance(result.program, Program)
        assert result.program.primitive == "candidate_v2"

    def test_max_iterations_terminates(self) -> None:
        # Synthesizer always returns a fresh candidate; evaluator
        # always reports failing demos. Loop hits max_iterations.
        config = Phase2Config(cegis_max_iterations=3)

        def synthesizer(_p: Any, _ce: Any, _b: float) -> Program:
            return _prog("never_succeeds")

        def evaluator(_p: Any, demos: list[tuple[Any, Any]]) -> list[CounterExample]:
            return [
                CounterExample(
                    input_grid=demos[0][0], expected_output=demos[0][1], actual_output=None
                )
            ]

        loop = CEGISLoop(synthesizer=synthesizer, evaluator=evaluator, config=config)
        result = loop.run(
            _prog("rotate90"),
            demos=[(0, 1)],
            wall_clock_budget_seconds=10.0,
        )
        assert result.terminated_by == "max_iterations"
        assert result.iterations == 3

    def test_synthesizer_gives_up(self) -> None:
        # Synthesizer returns None on first call → loop exits early.
        def synthesizer(_p: Any, _ce: Any, _b: float) -> None:
            return None

        def evaluator(_p: Any, demos: list[tuple[Any, Any]]) -> list[CounterExample]:
            return [
                CounterExample(
                    input_grid=demos[0][0], expected_output=demos[0][1], actual_output=None
                )
            ]

        loop = CEGISLoop(synthesizer=synthesizer, evaluator=evaluator)
        result = loop.run(
            _prog("rotate90"),
            demos=[(0, 1)],
            wall_clock_budget_seconds=10.0,
        )
        assert result.terminated_by == "synthesizer_gave_up"
        assert result.iterations == 1

    def test_budget_exhausted_terminates(self) -> None:
        # Inject a clock that advances 100s per call. Budget=10s →
        # the second deadline check trips.
        clock_t = {"t": 0.0}

        def fake_clock() -> float:
            clock_t["t"] += 50.0
            return clock_t["t"]

        def synthesizer(_p: Any, _ce: Any, _b: float) -> Program:
            return _prog("v")

        def evaluator(_p: Any, demos: list[tuple[Any, Any]]) -> list[CounterExample]:
            return [
                CounterExample(
                    input_grid=demos[0][0], expected_output=demos[0][1], actual_output=None
                )
            ]

        loop = CEGISLoop(
            synthesizer=synthesizer,
            evaluator=evaluator,
            clock=fake_clock,
        )
        result = loop.run(
            _prog("rotate90"),
            demos=[(0, 1)],
            wall_clock_budget_seconds=10.0,
        )
        assert result.terminated_by == "budget_exhausted"
        assert result.iterations < 5

    def test_invalid_budget_raises(self) -> None:
        loop = CEGISLoop(synthesizer=lambda *_: None, evaluator=lambda *_: [])
        with pytest.raises(ValueError, match="must be > 0"):
            loop.run(
                _prog("rotate90"),
                demos=[(0, 1)],
                wall_clock_budget_seconds=0.0,
            )


# ---------------------------------------------------------------------------
# Sub-budget propagation + history
# ---------------------------------------------------------------------------


class TestSubBudgetAndHistory:
    def test_synthesizer_receives_sub_budget(self) -> None:
        # cegis_sub_budget_per_iter_fraction default = 0.33 →
        # 30s wall budget × 0.33 = 9.9s sub-budget per call.
        observed: list[float] = []

        def synthesizer(_p: Any, _ce: Any, sub_budget: float) -> Program:
            observed.append(sub_budget)
            return _prog("v")

        def evaluator(_p: Any, demos: list[tuple[Any, Any]]) -> list[CounterExample]:
            return [
                CounterExample(
                    input_grid=demos[0][0], expected_output=demos[0][1], actual_output=None
                )
            ]

        loop = CEGISLoop(
            synthesizer=synthesizer,
            evaluator=evaluator,
            config=Phase2Config(cegis_max_iterations=2),
        )
        loop.run(
            _prog("rotate90"),
            demos=[(0, 1)],
            wall_clock_budget_seconds=30.0,
        )
        assert all(abs(b - 9.9) < 0.001 for b in observed)
        assert len(observed) == 2

    def test_history_records_counter_examples(self) -> None:
        # Each iteration's counter-examples land in the history tuple.
        synthesizer_calls = {"n": 0}

        def synthesizer(_p: Any, _ce: Any, _b: float) -> Program:
            synthesizer_calls["n"] += 1
            return _prog(f"v{synthesizer_calls['n']}")

        def evaluator(_p: Any, demos: list[tuple[Any, Any]]) -> list[CounterExample]:
            if synthesizer_calls["n"] >= 2:
                return []
            return [
                CounterExample(
                    input_grid="i",
                    expected_output="e",
                    actual_output=f"got_{synthesizer_calls['n']}",
                )
            ]

        loop = CEGISLoop(synthesizer=synthesizer, evaluator=evaluator)
        result = loop.run(
            _prog("rotate90"),
            demos=[("i", "e")],
            wall_clock_budget_seconds=10.0,
        )
        assert len(result.counter_examples_history) == 2  # 2 iter × 1 ce each
        # First iter saw "got_0" CE, second saw "got_1".
        assert result.counter_examples_history[0][0].actual_output == "got_0"
        assert result.counter_examples_history[1][0].actual_output == "got_1"


# ---------------------------------------------------------------------------
# Phase2Config invariants for CEGIS
# ---------------------------------------------------------------------------


class TestCEGISConfigInvariants:
    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="cegis_max_iterations"):
            Phase2Config(cegis_max_iterations=0)

    def test_eligibility_score_in_unit(self) -> None:
        with pytest.raises(ValueError, match="cegis_eligibility_score_min"):
            Phase2Config(cegis_eligibility_score_min=1.5)
        with pytest.raises(ValueError, match="cegis_eligibility_score_min"):
            Phase2Config(cegis_eligibility_score_min=-0.1)

    def test_sub_budget_fraction_must_be_strictly_positive(self) -> None:
        with pytest.raises(ValueError, match="cegis_sub_budget"):
            Phase2Config(cegis_sub_budget_per_iter_fraction=0.0)
        with pytest.raises(ValueError, match="cegis_sub_budget"):
            Phase2Config(cegis_sub_budget_per_iter_fraction=1.5)
