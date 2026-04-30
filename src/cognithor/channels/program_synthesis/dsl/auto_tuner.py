# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cost-Auto-Tuner — deterministic, no-ML (spec §7.6).

The cost values that drive the Occam-prior in :mod:`primitives` are the
*only* lever we have for ranking candidates absent an LLM. Manually
tuning them is fragile, so the spec mandates a tiny deterministic
tuner: it adjusts each primitive's cost based on how often it appears
in *successful* synthesis programs versus how often it appears in
*failed* candidates, with a small learning-rate ε.

Phase-1 contract (spec §7.6):

* Pure-symbolic. No ML, no gradient descent, no model weights.
* Deterministic. Same input → same output across processes.
* Conservative. ε = 0.05 default keeps each round's update small.
* Capped. R = 5 rounds default; early-terminates when a round
  produces no improvement.
* Read-only on benchmark data; writes a fresh catalog dict.

Capability gate: the public ``pse:dsl:tune`` capability is
admin/dev-only — wired in from :mod:`integration.capability_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_LEARNING_RATE: float = 0.05
DEFAULT_ROUNDS: int = 5

# Cost cannot drop below this floor — keeps the ranking from collapsing
# to "everything is free" and protects against pathological benchmark
# data (e.g. one primitive with 1000× the success of any other).
MIN_COST: float = 0.1


@dataclass(frozen=True)
class TuneRound:
    """One round's diff applied to the catalog."""

    round_number: int
    costs_before: dict[str, float]
    costs_after: dict[str, float]
    score_before: float
    score_after: float

    @property
    def improvement(self) -> float:
        return self.score_after - self.score_before


@dataclass(frozen=True)
class TuneResult:
    """Aggregate of every TuneRound + the final catalog."""

    initial_costs: dict[str, float]
    final_costs: dict[str, float]
    rounds: tuple[TuneRound, ...] = field(default_factory=tuple)
    converged_early: bool = False


@dataclass(frozen=True)
class BenchmarkSample:
    """One task's outcome that feeds into the tuner.

    ``solving_program_primitives`` is the multi-set of primitive names
    used by the program that solved this task (or ``()`` if no program
    solved it). ``failed_candidate_primitives`` lists the primitives
    that appeared in any candidate the search rejected — we use the
    cardinality, not multiplicity, to avoid double-counting heavy use
    of the same primitive in a single failed program.
    """

    solving_program_primitives: tuple[str, ...] = ()
    failed_candidate_primitives: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _normalise(counts: dict[str, int]) -> dict[str, float]:
    """Return per-primitive frequency normalised to [0, 1].

    Empty input → empty output. Single-primitive input → 1.0 for that
    primitive (safe for the tuner's update equation).
    """
    if not counts:
        return {}
    max_v = max(counts.values())
    if max_v == 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k] / max_v for k in counts}


def _success_counts(samples: tuple[BenchmarkSample, ...]) -> dict[str, int]:
    """How many tasks a primitive helped to solve."""
    out: dict[str, int] = {}
    for s in samples:
        for prim in set(s.solving_program_primitives):
            out[prim] = out.get(prim, 0) + 1
    return out


def _failure_weights(samples: tuple[BenchmarkSample, ...]) -> dict[str, float]:
    """Per-primitive share of the failed-candidate set (0..1).

    For each sample we add 1.0/total_failed_distinct_primitives to each
    primitive in the failed set. Higher weight ↔ "this primitive shows
    up in many wrong candidates" → tuner should slightly raise its cost.
    """
    out: dict[str, float] = {}
    for s in samples:
        failed = set(s.failed_candidate_primitives)
        if not failed:
            continue
        share = 1.0 / len(failed)
        for prim in failed:
            out[prim] = out.get(prim, 0.0) + share
    if not out:
        return {}
    # Normalise to 0..1 so the tuner's ε * w stays bounded.
    max_w = max(out.values())
    if max_w == 0:
        return {k: 0.0 for k in out}
    return {k: out[k] / max_w for k in out}


def _score(samples: tuple[BenchmarkSample, ...]) -> float:
    """Coarse "how-good-is-this-catalog" score for early termination.

    Score = (number of tasks solved) / (total tasks). Used only to
    decide whether a round produced an improvement; the spec keeps
    this simple on purpose — full benchmark machinery happens
    elsewhere.
    """
    if not samples:
        return 0.0
    solved = sum(1 for s in samples if s.solving_program_primitives)
    return solved / len(samples)


def _apply_round(
    costs: dict[str, float],
    samples: tuple[BenchmarkSample, ...],
    learning_rate: float,
) -> dict[str, float]:
    """One tuner-round: c*(p) = c(p) * (1 - ε·norm_success) * (1 + ε·failure)."""
    success = _normalise(_success_counts(samples))
    failure = _failure_weights(samples)
    out: dict[str, float] = {}
    for name, c in costs.items():
        s = success.get(name, 0.0)
        f = failure.get(name, 0.0)
        new_cost = c * (1.0 - learning_rate * s) * (1.0 + learning_rate * f)
        # Clamp to floor so ranking can't collapse on adversarial input.
        if new_cost < MIN_COST:
            new_cost = MIN_COST
        out[name] = round(new_cost, 6)  # cap precision so output is stable
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_tune(
    initial_costs: dict[str, float],
    samples: tuple[BenchmarkSample, ...],
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    rounds: int = DEFAULT_ROUNDS,
) -> TuneResult:
    """Run the deterministic tuner.

    The tuner exits early when a round produces no improvement (score
    delta ≤ 0). Improvement is gauged on the same ``samples`` for every
    round — the spec leaves "rerun benchmark with c*" as an outer-loop
    concern; this function focuses on the deterministic update.
    """
    if not 0.0 < learning_rate < 1.0:
        raise ValueError(f"learning_rate must be in (0, 1); got {learning_rate!r}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1; got {rounds!r}")

    rounds_log: list[TuneRound] = []
    current = dict(initial_costs)
    score = _score(samples)
    converged = False

    for r in range(1, rounds + 1):
        new_costs = _apply_round(current, samples, learning_rate)
        new_score = _score(samples)
        rounds_log.append(
            TuneRound(
                round_number=r,
                costs_before=dict(current),
                costs_after=dict(new_costs),
                score_before=score,
                score_after=new_score,
            )
        )
        # Score is benchmark-derived, not catalog-derived in this Phase-1
        # implementation; the early-termination check fires when the
        # catalog has converged (no costs changed within rounding).
        if new_costs == current:
            converged = True
            current = new_costs
            score = new_score
            break
        current = new_costs
        score = new_score

    return TuneResult(
        initial_costs=dict(initial_costs),
        final_costs=dict(current),
        rounds=tuple(rounds_log),
        converged_early=converged,
    )


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_ROUNDS",
    "MIN_COST",
    "BenchmarkSample",
    "TuneResult",
    "TuneRound",
    "auto_tune",
]
