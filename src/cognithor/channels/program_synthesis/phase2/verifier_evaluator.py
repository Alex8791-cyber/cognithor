# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §7.3.3 — End-to-end Verifier evaluator (Sprint-2 plan task 4).

Wires every individual Phase-2 verifier sub-score into a single
synchronous evaluator. Given a candidate :class:`Program` plus a
:class:`TaskSpec`, the evaluator computes:

* ``demo_pass_rate`` — fraction of demo examples the program
  produces correctly (the dominant signal).
* ``partial_pixel_match`` — average graduated pixel-level match
  across demos (shipped in :mod:`phase2.pixel_match`).
* ``invariants_satisfied`` — pass-through hook for the Sprint-3
  property-based invariant tests; defaults to ``1.0`` (no
  invariants) so Sprint-2 callers can wire it later.
* ``triviality_score`` — high when the program is non-trivial
  (shipped in :mod:`phase2.triviality`).
* ``suspicion_score`` — high when the (program, score) pair is
  not suspicious (shipped via :func:`compute_suspicion`).

The five values feed :func:`aggregate_verifier_score` to produce
the final score in ``[0, 1]``, with weights from
``Phase2Config.verifier_score_weights``.

The evaluator is executor-agnostic: the caller injects any
:class:`Executor` (e.g. ``InProcessExecutor`` for tests, sandboxed
worker for production). Spec §7.3.3 acceptance: end-to-end Verifier
evaluation produces a score in ``[0, 1]`` by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)
from cognithor.channels.program_synthesis.phase2.pixel_match import (
    average_partial_pixel_match,
)
from cognithor.channels.program_synthesis.phase2.scoring import (
    VerifierScoreInputs,
    aggregate_verifier_score,
)
from cognithor.channels.program_synthesis.phase2.triviality import (
    triviality_score,
)
from cognithor.channels.program_synthesis.phase2.verifier import (
    compute_suspicion,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )
    from cognithor.channels.program_synthesis.search.executor import (
        Executor,
    )


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierEvaluation:
    """Outcome of one :meth:`VerifierEvaluator.evaluate` call.

    ``final_score`` is the aggregated score in ``[0, 1]`` per spec
    §7.2 / §7.3.3. ``inputs`` carries every sub-score so callers
    can introspect *why* the score landed where it did (telemetry,
    A/B-test tags, refinement decisions).

    ``actual_outputs`` is the per-demo program output, parallel to
    ``spec.examples`` order. ``ok_per_demo`` records whether each
    individual execution succeeded — useful for the Refiner pipeline
    to mine partial-failure context.
    """

    final_score: float
    inputs: VerifierScoreInputs
    actual_outputs: tuple[Any, ...]
    ok_per_demo: tuple[bool, ...]


# ---------------------------------------------------------------------------
# Optional invariants hook
# ---------------------------------------------------------------------------


# Property-based invariants are a Sprint-3 deliverable; the hook
# here lets earlier sprints wire stub callables when the test suite
# exercises invariant-driven scoring. The callable receives the same
# arguments as :meth:`evaluate` and returns a fraction in ``[0, 1]``.
InvariantsCheck = Callable[
    ["ProgramNode", "Any", tuple[Any, ...], tuple[Any, ...]],
    float,
]


def _no_invariants(
    _program: ProgramNode,
    _spec: Any,
    _actual_outputs: tuple[Any, ...],
    _expected_outputs: tuple[Any, ...],
) -> float:
    """Default: report 1.0 (full pass) so the formula reduces to the four other terms."""
    return 1.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class VerifierEvaluator:
    """End-to-end Phase-2 Verifier — wires every sub-score into one number.

    Stateless across calls. The executor is fully injectable so
    tests can pin determinism without booting the sandbox; production
    wires the subprocess sandbox.

    The default ``invariants_check`` returns ``1.0`` (no invariants
    failing); callers that ship Sprint-3 property tests pass a real
    callable.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        invariants_check: InvariantsCheck = _no_invariants,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._executor = executor
        self._invariants_check = invariants_check
        self._config = config

    def evaluate(
        self,
        program: ProgramNode,
        spec: Any,
    ) -> VerifierEvaluation:
        """Compute every sub-score, aggregate, return the result.

        ``spec`` must expose an ``examples`` attribute that is an
        iterable of ``(input, output)`` pairs. The evaluator does
        not require a full :class:`TaskSpec` — duck-typing keeps
        this module decoupled from ``core.types``.
        """
        examples = list(spec.examples)
        actual_outputs: list[Any] = []
        ok_per_demo: list[bool] = []
        demo_correct = 0

        for inp, expected in examples:
            result = self._executor.execute(program, inp)
            ok_per_demo.append(result.ok)
            if not result.ok:
                actual_outputs.append(None)
                continue
            actual_outputs.append(result.value)
            if _outputs_equal(result.value, expected):
                demo_correct += 1

        n_demos = len(examples)
        demo_pass_rate = demo_correct / n_demos if n_demos else 0.0

        # partial_pixel_match — only for demos where execution succeeded
        # AND outputs are array-shaped. Failed demos count as 0.0
        # contribution; non-array outputs (e.g. a Color value) are
        # likewise neutral 0.0.
        partial_match_pairs: list[tuple[Any, Any]] = []
        for actual, expected in zip(actual_outputs, (e for _i, e in examples), strict=False):
            if actual is None:
                # Use a deliberately mismatched array so the pixel
                # match contribution is zero for this demo.
                partial_match_pairs.append((np.zeros((1, 1), dtype=np.int8), expected))
            else:
                partial_match_pairs.append((actual, expected))
        if partial_match_pairs:
            partial_pixel = average_partial_pixel_match(
                actual_grids=[a for a, _ in partial_match_pairs],
                expected_grids=[e for _, e in partial_match_pairs],
            )
        else:
            partial_pixel = 0.0

        # Triviality — operates on the actuals + inputs + expecteds.
        inputs_arr = [_as_array(inp) for inp, _ in examples]
        expecteds_arr = [_as_array(e) for _, e in examples]
        actuals_arr = [
            _as_array(actual) if actual is not None else _as_array(inp)
            for actual, (inp, _) in zip(actual_outputs, examples, strict=False)
        ]
        triv = triviality_score(actuals_arr, expecteds_arr, inputs_arr) if examples else 1.0

        # Suspicion — uses partial_pixel as the "partial score" feed.
        # Spec v1.4 §7.3.2: suspicion = partial · (1 - syntactic_complexity).
        # We lift the suspicion *score* (1 - suspicion-penalty) so the
        # weighted-sum formula stays "high = good" for every term.
        susp = compute_suspicion(program, partial_score=partial_pixel, config=self._config)
        suspicion_score = 1.0 - max(0.0, min(1.0, susp.value))

        # Invariants — optional hook.
        invariants = self._invariants_check(
            program,
            spec,
            tuple(actual_outputs),
            tuple(expected for _i, expected in examples),
        )
        if not 0.0 <= invariants <= 1.0:
            raise ValueError(f"invariants_check returned {invariants}; must be in [0, 1]")

        inputs_obj = VerifierScoreInputs(
            demo_pass_rate=demo_pass_rate,
            partial_pixel_match=partial_pixel,
            invariants_satisfied=invariants,
            triviality_score=triv,
            suspicion_score=suspicion_score,
        )
        final = aggregate_verifier_score(inputs_obj, config=self._config)
        return VerifierEvaluation(
            final_score=final,
            inputs=inputs_obj,
            actual_outputs=tuple(actual_outputs),
            ok_per_demo=tuple(ok_per_demo),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outputs_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        try:
            return bool(np.array_equal(actual, expected))
        except (TypeError, ValueError):
            return False
    try:
        return bool(actual == expected)
    except Exception:
        return False


def _as_array(x: Any) -> np.ndarray[Any, Any]:
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x, dtype=np.int8)
    except (TypeError, ValueError):
        return np.zeros((1, 1), dtype=np.int8)


__all__ = [
    "InvariantsCheck",
    "VerifierEvaluation",
    "VerifierEvaluator",
]
