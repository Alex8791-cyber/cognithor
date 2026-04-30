# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 α-mixer + sample-size-dampening (spec v1.4 §4.4 / §4.4.4).

The Dual-Prior in Module A combines an LLM-derived prior π_llm with a
symbolic-heuristic prior π_symbolic via a multiplicative-adaptive α::

    α_entropy ∈ [0.5, 0.85]
    α_performance ∈ [0.5, 1.0]
    α = α_entropy · α_performance ∈ [0.25, 0.85]

The product is clamped to the Search-α-bounds derived from
:class:`Phase2Config`. The default config gives the spec's [0.25, 0.85]
range.

The Symbolic-Prior side carries a ``Sample-Size-Dämpfung`` so a heuristic
that has only seen a handful of demos can't over-confidently dominate
the α-mix. Spec §4.4 (referenced as "unverändert ggü. v1.3") expects
``effective_confidence = base · (n / (n + n0))`` shape; the constant n0
is in :class:`Phase2Config` as ``sample_size_dampening_n0`` and defaults
to 4.

Both helpers are pure, side-effect free, fully Phase2Config-overridable
— Sprint-1 contract.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)


def mix_alpha(
    alpha_entropy: float,
    alpha_performance: float,
    *,
    config: Phase2Config = DEFAULT_PHASE2_CONFIG,
) -> float:
    """Spec v1.4 §4.4.4 — multiplicative-adaptive α mixing.

    Both inputs are clamped to their config-defined intervals before
    multiplication, so a caller passing values outside the spec range
    (e.g. an over-confident performance tracker reporting 1.2) gets a
    safe in-bounds α back rather than an exception.
    """
    e = _clamp(alpha_entropy, config.alpha_entropy_lower, config.alpha_entropy_upper)
    p = _clamp(
        alpha_performance,
        config.alpha_performance_lower,
        config.alpha_performance_upper,
    )
    return e * p


def alpha_bounds(*, config: Phase2Config = DEFAULT_PHASE2_CONFIG) -> tuple[float, float]:
    """The min/max α a multiplicative-adaptive mix can return.

    Returns ``(lower * lower, upper * upper)`` of the two factor
    intervals. With spec defaults: ``(0.25, 0.85)``.
    """
    lo = config.alpha_entropy_lower * config.alpha_performance_lower
    hi = config.alpha_entropy_upper * config.alpha_performance_upper
    return lo, hi


def apply_sample_size_dampening(
    base_confidence: float,
    n_samples: int,
    *,
    config: Phase2Config = DEFAULT_PHASE2_CONFIG,
) -> float:
    """Spec §4.4 — dampen *base_confidence* by sample size.

    ``effective = base · (n / (n + n0))``

    At ``n = 0`` the result is ``0.0`` (no observations → no signal).
    At ``n = n0`` the dampening factor is exactly 0.5.
    At ``n → ∞`` the factor approaches 1.0.

    ``base_confidence`` must be in ``[0, 1]``; ``n_samples`` must be
    ``≥ 0``. Both are validated explicitly because the symbolic-prior
    aggregator is the canonical caller and must surface a typed
    ValueError on bad input rather than producing a quiet NaN.
    """
    if not 0.0 <= base_confidence <= 1.0:
        raise ValueError(
            f"apply_sample_size_dampening: base_confidence must be "
            f"in [0, 1]; got {base_confidence!r}"
        )
    if n_samples < 0:
        raise ValueError(f"apply_sample_size_dampening: n_samples must be >= 0; got {n_samples!r}")
    if n_samples == 0:
        return 0.0
    n0 = config.sample_size_dampening_n0
    factor = n_samples / (n_samples + n0)
    return base_confidence * factor


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


__all__ = [
    "alpha_bounds",
    "apply_sample_size_dampening",
    "mix_alpha",
]
