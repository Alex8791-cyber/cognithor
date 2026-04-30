# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §7.3.1 — rule-based triviality detection (plan task 8 slice).

Phase-2 verifier deflates a candidate's score if it looks suspiciously
trivial — e.g. it just returns the input unchanged, or always returns
the same constant grid regardless of demo. Spec §7.3.1 calls for "5
regelbasierte Tests" against a 50-program adversarial corpus
("triviality ≤ 0.3").

This module implements those 5 rules. Each rule consumes the demo
outputs the candidate produced plus the corresponding inputs/expecteds
and returns a *trivial fraction* in ``[0, 1]`` — 1.0 means the rule
fired (the candidate looks trivial *for that rule*), 0.0 means it
didn't.

The aggregate :func:`triviality_score` returns ``1 - max(rules)`` so
that:

* 1.0 = non-trivial across every rule (verifier rewards).
* 0.0 = at least one rule fully fired (verifier deflates).

The five rules:

* :func:`r1_output_equals_input` — a candidate that just returns the
  input is the textbook trivial case.
* :func:`r2_output_is_constant` — every demo output is a single-value
  grid; the candidate ignored the input structure.
* :func:`r3_single_pixel_diff` — output differs from input by at
  most one cell on every demo (clearly not solving the task).
* :func:`r4_near_identity` — the output is *almost* the input on
  every demo (95 %+ pixel match).
* :func:`r5_output_unchanged_across_demos` — every demo got the
  same output regardless of input (constant-output regression).

Inputs and expecteds carry the "is this even on the right track"
context; a rule fires only when *every* demo exhibits the trivial
behaviour, so a candidate that solves three of four demos and only
matches the input on one is *not* flagged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Individual rules — each returns a trivial-fraction in [0, 1]
# ---------------------------------------------------------------------------


def r1_output_equals_input(
    actuals: list[np.ndarray[Any, Any]],
    inputs: list[np.ndarray[Any, Any]],
) -> float:
    """Fires (1.0) iff every actual output equals its corresponding input."""
    if not actuals or len(actuals) != len(inputs):
        return 0.0
    for actual, inp in zip(actuals, inputs, strict=True):
        if actual.shape != inp.shape or not np.array_equal(actual, inp):
            return 0.0
    return 1.0


def r2_output_is_constant(actuals: list[np.ndarray[Any, Any]]) -> float:
    """Fires (1.0) iff every actual output has only one unique value."""
    if not actuals:
        return 0.0
    for actual in actuals:
        if actual.size == 0:
            return 0.0
        unique = np.unique(actual)
        if unique.size != 1:
            return 0.0
    return 1.0


def r3_single_pixel_diff(
    actuals: list[np.ndarray[Any, Any]],
    inputs: list[np.ndarray[Any, Any]],
    *,
    max_diff_pixels: int = 1,
) -> float:
    """Fires (1.0) iff every actual differs from its input by at most ``max_diff_pixels``."""
    if not actuals or len(actuals) != len(inputs):
        return 0.0
    for actual, inp in zip(actuals, inputs, strict=True):
        if actual.shape != inp.shape:
            return 0.0
        diff_count = int(np.count_nonzero(actual != inp))
        if diff_count > max_diff_pixels:
            return 0.0
    return 1.0


def r4_near_identity(
    actuals: list[np.ndarray[Any, Any]],
    inputs: list[np.ndarray[Any, Any]],
    *,
    threshold: float = 0.95,
) -> float:
    """Fires (1.0) iff every actual matches its input on ``≥ threshold`` of cells."""
    if not actuals or len(actuals) != len(inputs):
        return 0.0
    for actual, inp in zip(actuals, inputs, strict=True):
        if actual.shape != inp.shape or actual.size == 0:
            return 0.0
        match = float(np.count_nonzero(actual == inp)) / float(actual.size)
        if match < threshold:
            return 0.0
    return 1.0


def r5_output_unchanged_across_demos(
    actuals: list[np.ndarray[Any, Any]],
) -> float:
    """Fires (1.0) iff every actual output is identical across demos.

    Distinct from r2: r5 catches the case where the candidate emits
    the *same* output regardless of input (could be non-constant grid
    but invariant). r2 catches single-value grids per demo.
    """
    if len(actuals) < 2:
        # With fewer than 2 demos we can't compare — don't fire.
        return 0.0
    first = actuals[0]
    for actual in actuals[1:]:
        if actual.shape != first.shape:
            return 0.0
        if not np.array_equal(actual, first):
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Aggregate — spec §7.3.1 score
# ---------------------------------------------------------------------------


def triviality_score(
    actuals: list[np.ndarray[Any, Any]],
    expecteds: list[np.ndarray[Any, Any]],
    inputs: list[np.ndarray[Any, Any]],
) -> float:
    """Spec §7.3.1 — non-trivial-ness score in ``[0, 1]``.

    1.0 = non-trivial on every rule (verifier rewards).
    0.0 = at least one rule fully fired (verifier deflates).

    The function ignores ``expecteds`` for now — the five Sprint-1
    rules are demo-vs-input + cross-demo checks. Spec §7.3.1 reserves
    expecteds for future "did it converge to the right answer in a
    suspiciously trivial way" rules; the parameter stays in the
    signature so callers don't have to refactor when later sprints
    add those.
    """
    if not actuals:
        return 1.0
    rule_scores = (
        r1_output_equals_input(actuals, inputs),
        r2_output_is_constant(actuals),
        r3_single_pixel_diff(actuals, inputs),
        r4_near_identity(actuals, inputs),
        r5_output_unchanged_across_demos(actuals),
    )
    trivial_fraction = max(rule_scores)
    # Suppress unused-arg lint for ``expecteds`` — it's a placeholder
    # for spec-reserved future rules.
    _ = expecteds
    return 1.0 - trivial_fraction


__all__ = [
    "r1_output_equals_input",
    "r2_output_is_constant",
    "r3_single_pixel_diff",
    "r4_near_identity",
    "r5_output_unchanged_across_demos",
    "triviality_score",
]
