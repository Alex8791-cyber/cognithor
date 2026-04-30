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

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VerifierScoreWeights:
    """Spec §7.2 — five-factor verifier score weights.

    The final verifier score is a weighted sum of:

    * ``demo_pass_rate`` — fraction of demos the program produces
      correctly (the dominant signal).
    * ``partial_pixel_match`` — Phase-2 graduated pixel-level match.
    * ``invariants_satisfied`` — fraction of property-invariant tests
      that hold.
    * ``triviality_score`` — high when the program is non-trivial
      (rule-based check, spec §7.3.1).
    * ``suspicion_score`` — high when the (program, score) pair is
      not suspicious (spec §7.3.2 — F1 multipliers feed into this).

    The five weights MUST sum to 1.0 — enforced at construction.
    """

    demo_pass_rate: float = 0.55
    partial_pixel_match: float = 0.13
    invariants_satisfied: float = 0.08
    triviality_score: float = 0.12
    suspicion_score: float = 0.12

    def __post_init__(self) -> None:
        for name, value in (
            ("demo_pass_rate", self.demo_pass_rate),
            ("partial_pixel_match", self.partial_pixel_match),
            ("invariants_satisfied", self.invariants_satisfied),
            ("triviality_score", self.triviality_score),
            ("suspicion_score", self.suspicion_score),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"VerifierScoreWeights.{name} must be in [0, 1]; got {value}")
        total = (
            self.demo_pass_rate
            + self.partial_pixel_match
            + self.invariants_satisfied
            + self.triviality_score
            + self.suspicion_score
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"VerifierScoreWeights must sum to 1.0; got {total}")


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

    # ── α-Controller hysteresis & sliding window (spec §4.4.4) ─────
    # alpha_hysteresis_iterations gates how many consecutive
    # observations of "the LLM is unreliable" the AlphaController
    # needs to see before it permanently lowers α_performance.
    # alpha_performance_window is the sliding-window size the
    # PriorPerformanceTracker uses to compute the current
    # α_performance value.
    # alpha_cold_start is the value the controller returns when no
    # observations have accumulated yet.
    alpha_hysteresis_iterations: int = 5
    alpha_performance_window: int = 10
    alpha_cold_start: float = 0.85

    # ── CEGIS Refiner stage (spec §6.5.3) ──────────────────────────
    # Counter-Example-Guided Inductive Synthesis runs *after* the
    # Drei-Zonen-Refiner if the candidate's score is in the eligible
    # band. The loop terminates on max_iterations OR budget timeout
    # OR all-demos-pass.
    cegis_max_iterations: int = 5
    cegis_eligibility_score_min: float = 0.5
    cegis_sub_budget_per_iter_fraction: float = 0.33

    # ── Module A — LLM-Prior over vLLM (spec §4.2 / §4.3 / §4.7) ────
    # Backend: vLLM exposing OpenAI-compat /v1/chat/completions.
    # Default model is the spec-anchored Qwen3.6-27B-Instruct on the
    # RTX 5090 (32 GB VRAM); Q5_K_M is the default quantisation,
    # Q4_K_M is the fallback for tighter VRAM budgets.
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model_name: str = "Qwen/Qwen3.6-27B-Instruct"
    llm_fallback_model_name: str = "Qwen/Qwen3.6-27B-Instruct-AWQ"
    # Two-Stage prompting (spec §4.7): Stage-1 free-form CoT, Stage-2
    # constrained JSON. Different temperatures because Stage-1 wants
    # exploration (default 0.7) and Stage-2 wants determinism (0.1).
    llm_temperature_stage1: float = 0.7
    llm_temperature_stage2: float = 0.1
    # Spec §4.7: retry the JSON stage exactly once on parse failure.
    llm_json_max_retries: int = 1
    # Spec §4.5: top-K depth-dependent. K starts at this default for
    # depth-1 candidates; deeper levels narrow it via the search engine.
    llm_top_k_default: int = 5
    # Wall-clock cap per LLM call (seconds). Spec §13.3 budget for
    # repair stages is sub-second; 8 s is a safe outer bound.
    llm_call_timeout_seconds: float = 8.0

    # ── Verifier score weights (spec §7.2) ──────────────────────────
    # Five-factor weighted sum that reduces a Phase-2 verifier
    # evaluation to a single score in [0, 1]. Weights must sum to 1.0;
    # the nested dataclass enforces that at construction time so a
    # mis-tuned config blows up loudly.
    verifier_score_weights: VerifierScoreWeights = field(default_factory=VerifierScoreWeights)

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
        if self.alpha_hysteresis_iterations < 1:
            raise ValueError(
                f"Phase2Config: alpha_hysteresis_iterations must be >= 1; "
                f"got {self.alpha_hysteresis_iterations}."
            )
        if self.alpha_performance_window < 1:
            raise ValueError(
                f"Phase2Config: alpha_performance_window must be >= 1; "
                f"got {self.alpha_performance_window}."
            )
        # alpha_cold_start is NOT band-checked at construction — the
        # AlphaController clamps it at read time. That keeps
        # Phase2Config(alpha_entropy_upper=0.7) etc. constructible
        # even when the spec-default alpha_cold_start=0.85 sits
        # above a customised band.
        if self.cegis_max_iterations < 1:
            raise ValueError(
                f"Phase2Config: cegis_max_iterations must be >= 1; got {self.cegis_max_iterations}."
            )
        if not 0.0 <= self.cegis_eligibility_score_min <= 1.0:
            raise ValueError(
                f"Phase2Config: cegis_eligibility_score_min must be in "
                f"[0, 1]; got {self.cegis_eligibility_score_min}."
            )
        if not 0.0 < self.cegis_sub_budget_per_iter_fraction <= 1.0:
            raise ValueError(
                f"Phase2Config: cegis_sub_budget_per_iter_fraction must "
                f"be in (0, 1]; got {self.cegis_sub_budget_per_iter_fraction}."
            )
        # LLM prior validation.
        if not self.llm_model_name:
            raise ValueError("Phase2Config: llm_model_name must be non-empty.")
        if not 0.0 <= self.llm_temperature_stage1 <= 2.0:
            raise ValueError(
                f"Phase2Config: llm_temperature_stage1 must be in [0, 2]; "
                f"got {self.llm_temperature_stage1}."
            )
        if not 0.0 <= self.llm_temperature_stage2 <= 2.0:
            raise ValueError(
                f"Phase2Config: llm_temperature_stage2 must be in [0, 2]; "
                f"got {self.llm_temperature_stage2}."
            )
        if self.llm_json_max_retries < 0:
            raise ValueError(
                f"Phase2Config: llm_json_max_retries must be >= 0; got {self.llm_json_max_retries}."
            )
        if self.llm_top_k_default < 1:
            raise ValueError(
                f"Phase2Config: llm_top_k_default must be >= 1; got {self.llm_top_k_default}."
            )
        if self.llm_call_timeout_seconds <= 0.0:
            raise ValueError(
                f"Phase2Config: llm_call_timeout_seconds must be > 0; "
                f"got {self.llm_call_timeout_seconds}."
            )


DEFAULT_PHASE2_CONFIG = Phase2Config()
"""The spec v1.4 defaults. Replace per Sprint-1 review if data demands."""


__all__ = ["DEFAULT_PHASE2_CONFIG", "Phase2Config"]
