# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 configuration — typed, immutable, fully-overridable.

Spec v1.4 §16 (open questions 18-23) explicitly defers empirical
validation of every heuristic constant to Sprint-1. The external
reviewer's lone implementation note for Sprint-1 was:

    Die heuristischen Werte explizit als Config führen, nicht
    hardcoden: 3× (High-Impact), 1.5× (Structural-Abstraction),
    α-Schwellen 0.35/0.45, Hysterese-Window 3.

Every Phase-2 module reads from a :class:`Phase2Config` instance, so
the data-driven Round-5 review after Sprint 1 can A/B-test alternatives
by passing a different config — no source patches.

The defaults are the spec-anchored values. Subsequent sprints validate
or replace them, then update :data:`DEFAULT_PHASE2_CONFIG`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase2Config:
    """All Phase-2 heuristic constants in one place.

    Three groups:

    1. **F1 — Suspicion multipliers** (spec v1.4 §7.3.2): how much
       weight High-Impact / Structural-Abstraction primitives carry in
       ``compute_syntactic_complexity``.
    2. **F2 — Refiner mode thresholds** (spec v1.4 §6.5.2): the
       three-zone α boundaries plus the hysteresis window that pins a
       once-chosen mode to itself.
    3. **F3 — Reserve toggles** (spec v1.4 §22.4.2): switches for the
       two reserve mechanisms (Argument-Quality-Faktor,
       Few-Demos-Dampening). Off by default; flip on if the matching
       interaction tests in §12.2 fail.
    """

    # ── F1: Suspicion-Score multipliers ────────────────────────────
    high_impact_multiplier: float = 3.0
    structural_abstraction_multiplier: float = 1.5
    regular_primitive_multiplier: float = 1.0

    # ── F2: Refiner mode-selection thresholds ──────────────────────
    # Zone 1 (full LLM): α ≥ repair_alpha_zone1_lower
    # Zone 2 (hybrid):    repair_alpha_zone3_upper ≤ α < repair_alpha_zone1_lower
    # Zone 3 (symbolic):  α < repair_alpha_zone3_upper
    repair_alpha_zone1_lower: float = 0.45
    repair_alpha_zone3_upper: float = 0.35
    refiner_hysteresis_window: int = 3

    # ── F3: Phase-2 reserves (spec §22.4.2 — off until tests demand) ──
    enable_argument_quality_factor: bool = False
    enable_few_demos_dampening: bool = False

    # ── Search-α bounds (spec §4.4.4) ──────────────────────────────
    # Multiplicative-adaptive α = α_entropy · α_performance.
    # α_entropy ∈ [alpha_entropy_lower, alpha_entropy_upper]
    # α_performance ∈ [alpha_performance_lower, alpha_performance_upper]
    # → α ∈ [alpha_entropy_lower · alpha_performance_lower,
    #        alpha_entropy_upper · alpha_performance_upper]
    #   = [0.25, 0.85] with the spec defaults.
    alpha_entropy_lower: float = 0.5
    alpha_entropy_upper: float = 0.85
    alpha_performance_lower: float = 0.5
    alpha_performance_upper: float = 1.0

    # ── Sample-size dampening (Symbolic-Prior, spec §4.4) ──────────
    # effective_confidence = base_confidence · (n / (n + dampening_n0))
    # The default n0=4 means at n=4 demos the dampening factor is 0.5;
    # at n=12 it's 0.75; at n=∞ it's 1.0.
    sample_size_dampening_n0: int = 4

    def __post_init__(self) -> None:
        # Sanity: zones must form a non-empty graybereich.
        if not 0.0 < self.repair_alpha_zone3_upper < self.repair_alpha_zone1_lower < 1.0:
            raise ValueError(
                f"Phase2Config: repair α thresholds must satisfy "
                f"0 < zone3_upper < zone1_lower < 1; got "
                f"zone3_upper={self.repair_alpha_zone3_upper}, "
                f"zone1_lower={self.repair_alpha_zone1_lower}"
            )
        if self.refiner_hysteresis_window < 1:
            raise ValueError(
                f"Phase2Config: refiner_hysteresis_window must be >= 1; "
                f"got {self.refiner_hysteresis_window}"
            )
        # Multipliers must be ordered: regular ≤ structural-abstraction ≤ high-impact.
        # The spec rationale is that structural-abstraction is "leicht über
        # Standard, weit unter direkt-transformativen". Equality is allowed
        # so a Sprint-1 A/B can collapse the classes for comparison.
        if not (
            self.regular_primitive_multiplier
            <= self.structural_abstraction_multiplier
            <= self.high_impact_multiplier
        ):
            raise ValueError(
                f"Phase2Config: multipliers must be ordered "
                f"regular ({self.regular_primitive_multiplier}) "
                f"≤ structural ({self.structural_abstraction_multiplier}) "
                f"≤ high_impact ({self.high_impact_multiplier})."
            )
        # α-bounds (spec §4.4.4): each factor must be a valid
        # closed-interval inside [0, 1] with lower ≤ upper.
        for name, lo, hi in (
            ("alpha_entropy", self.alpha_entropy_lower, self.alpha_entropy_upper),
            (
                "alpha_performance",
                self.alpha_performance_lower,
                self.alpha_performance_upper,
            ),
        ):
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError(
                    f"Phase2Config: {name} interval [{lo}, {hi}] must "
                    f"satisfy 0 ≤ lower ≤ upper ≤ 1."
                )
        if self.sample_size_dampening_n0 < 1:
            raise ValueError(
                f"Phase2Config: sample_size_dampening_n0 must be >= 1; "
                f"got {self.sample_size_dampening_n0}."
            )


DEFAULT_PHASE2_CONFIG = Phase2Config()
"""The spec v1.4 defaults. Replace per Sprint-1 review if data demands."""


__all__ = ["DEFAULT_PHASE2_CONFIG", "Phase2Config"]
