# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 verifier final-score aggregation (spec §7.2).

Reduces five sub-scores to a single ``[0, 1]`` value the search
engine reads off as the candidate's reward signal:

    final = w_demo · demo_pass_rate
          + w_partial · partial_pixel_match
          + w_inv · invariants_satisfied
          + w_triv · triviality_score
          + w_susp · suspicion_score

Weights live on :class:`VerifierScoreWeights` (a flat field on
:class:`Phase2Config`) so the spec §17 D-criterion "12 % + 12 %"
binding for triviality + suspicion can be A/B-validated in Sprint-2
without touching code.

Sprint-1 plan acceptance criterion (task 8):
*Triviality + Suspicion gewichten korrekt im Final-Score (12 % + 12 %)*.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)


@dataclass(frozen=True)
class VerifierScoreInputs:
    """The five Phase-2 verifier sub-scores."""

    demo_pass_rate: float
    partial_pixel_match: float
    invariants_satisfied: float
    triviality_score: float
    suspicion_score: float

    def __post_init__(self) -> None:
        for name, value in (
            ("demo_pass_rate", self.demo_pass_rate),
            ("partial_pixel_match", self.partial_pixel_match),
            ("invariants_satisfied", self.invariants_satisfied),
            ("triviality_score", self.triviality_score),
            ("suspicion_score", self.suspicion_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"VerifierScoreInputs.{name} must be in [0, 1]; got {value}")


def aggregate_verifier_score(
    inputs: VerifierScoreInputs,
    *,
    config: Phase2Config = DEFAULT_PHASE2_CONFIG,
) -> float:
    """Compute the spec §7.2 weighted-sum final score in ``[0, 1]``.

    The result is in the closed unit interval *by construction*: each
    input is in ``[0, 1]`` (validated on construction) and the weights
    sum to 1.0 (validated on Phase2Config construction).
    """
    w = config.verifier_score_weights
    return (
        w.demo_pass_rate * inputs.demo_pass_rate
        + w.partial_pixel_match * inputs.partial_pixel_match
        + w.invariants_satisfied * inputs.invariants_satisfied
        + w.triviality_score * inputs.triviality_score
        + w.suspicion_score * inputs.suspicion_score
    )


__all__ = [
    "VerifierScoreInputs",
    "aggregate_verifier_score",
]
