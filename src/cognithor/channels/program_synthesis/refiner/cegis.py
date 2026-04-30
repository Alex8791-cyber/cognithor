# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.5.3 — CEGIS Refiner stage (Sprint-1 plan task 9 slice).

Counter-Example-Guided Inductive Synthesis (CEGIS) is the Refiner's
last-line repair: when the candidate's verifier score is at least
``cegis_eligibility_score_min`` (default 0.5) but the program still
fails *some* demos, the loop iterates:

1. Identify the failing demos as **counter-examples**.
2. Hand them to a **constrained synthesizer** (callable injected by
   the caller — Sprint-1 leaves the actual constraint compiler as a
   Phase-2-spec §6.5.3 detail to be filled in by the search-engine
   integration).
3. Replace the candidate with the synthesizer's output and re-evaluate.
4. Stop when:
   * every demo passes (success),
   * ``cegis_max_iterations`` is reached (budget — default 5),
   * the wall-clock budget is exhausted, or
   * the synthesizer returns ``None`` (no further candidate).

The loop is the **driver**: the search-side "given these
counter-examples find a program" step is dependency-injected so this
module is testable without a live search engine.

Plan acceptance criterion (task 9): *CEGIS terminiert garantiert in
≤ 5 Iter UND Budget-Limit* — verified in the test suite below.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterExample:
    """One failing demo the synthesizer must satisfy on the next pass.

    ``input_grid`` and ``expected_output`` come from the demo set;
    ``actual_output`` is what the current candidate produced. The
    triple lets the constrained synthesizer see *what went wrong*,
    not just "this demo failed".
    """

    input_grid: Any
    expected_output: Any
    actual_output: Any


@dataclass(frozen=True)
class CEGISResult:
    """Outcome of one :class:`CEGISLoop.run` call.

    ``program`` is the best-found candidate (or the original if the
    loop made no progress). ``terminated_by`` is the reason the loop
    stopped: one of ``"all_demos_pass"``, ``"max_iterations"``,
    ``"budget_exhausted"``, or ``"synthesizer_gave_up"``.

    ``iterations`` is the number of synthesizer calls actually made.
    ``elapsed_seconds`` is the wall-clock duration of the loop.
    """

    program: ProgramNode
    terminated_by: str
    iterations: int
    elapsed_seconds: float
    counter_examples_history: tuple[tuple[CounterExample, ...], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


# A `demo_evaluator` evaluates the candidate against every demo and
# returns the list of failing demos as CounterExamples. An empty list
# means "all demos pass".
DemoEvaluator = Callable[
    ["ProgramNode", list[tuple[Any, Any]]],
    list[CounterExample],
]

# The constrained synthesizer takes the current candidate plus the
# counter-examples accumulated so far and returns a fresh candidate,
# or `None` if it gives up.
ConstrainedSynthesizer = Callable[
    ["ProgramNode", list[CounterExample], float],
    "ProgramNode | None",
]


class CEGISLoop:
    """Spec §6.5.3 — driver for the CEGIS refinement loop.

    Stateless across runs — caller constructs once, calls :meth:`run`
    per refinement attempt. The :class:`Phase2Config` controls the
    max-iterations cap, eligibility threshold (caller-checked), and
    per-iteration sub-budget fraction.
    """

    def __init__(
        self,
        synthesizer: ConstrainedSynthesizer,
        evaluator: DemoEvaluator,
        *,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._synthesizer = synthesizer
        self._evaluator = evaluator
        self._config = config
        self._clock = clock

    def run(
        self,
        initial_program: ProgramNode,
        demos: list[tuple[Any, Any]],
        *,
        wall_clock_budget_seconds: float,
    ) -> CEGISResult:
        """Run the loop until termination — see module docstring."""
        if wall_clock_budget_seconds <= 0:
            raise ValueError(
                f"wall_clock_budget_seconds must be > 0; got {wall_clock_budget_seconds!r}"
            )
        start = self._clock()
        deadline = start + wall_clock_budget_seconds
        sub_budget = wall_clock_budget_seconds * self._config.cegis_sub_budget_per_iter_fraction
        program = initial_program
        history: list[tuple[CounterExample, ...]] = []

        # Initial check — maybe the candidate already passes.
        counter_examples = self._evaluator(program, demos)
        if not counter_examples:
            return CEGISResult(
                program=program,
                terminated_by="all_demos_pass",
                iterations=0,
                elapsed_seconds=self._clock() - start,
                counter_examples_history=tuple(history),
            )

        for iteration in range(1, self._config.cegis_max_iterations + 1):
            now = self._clock()
            if now >= deadline:
                return CEGISResult(
                    program=program,
                    terminated_by="budget_exhausted",
                    iterations=iteration - 1,
                    elapsed_seconds=now - start,
                    counter_examples_history=tuple(history),
                )

            history.append(tuple(counter_examples))
            candidate = self._synthesizer(program, list(counter_examples), sub_budget)
            if candidate is None:
                return CEGISResult(
                    program=program,
                    terminated_by="synthesizer_gave_up",
                    iterations=iteration,
                    elapsed_seconds=self._clock() - start,
                    counter_examples_history=tuple(history),
                )

            program = candidate
            counter_examples = self._evaluator(program, demos)
            if not counter_examples:
                return CEGISResult(
                    program=program,
                    terminated_by="all_demos_pass",
                    iterations=iteration,
                    elapsed_seconds=self._clock() - start,
                    counter_examples_history=tuple(history),
                )

        return CEGISResult(
            program=program,
            terminated_by="max_iterations",
            iterations=self._config.cegis_max_iterations,
            elapsed_seconds=self._clock() - start,
            counter_examples_history=tuple(history),
        )


__all__ = [
    "CEGISLoop",
    "CEGISResult",
    "ConstrainedSynthesizer",
    "CounterExample",
    "DemoEvaluator",
]
