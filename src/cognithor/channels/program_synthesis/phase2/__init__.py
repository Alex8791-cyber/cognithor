# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 (Neuro-Symbolic Synthesis Layer) — Sprint 1 foundations.

Contains the typed configuration the rest of Phase 2 (Dual-Prior, MCTS,
Refiner, extended Verifier) reads from. Spec v1.4 §16 leaves several
heuristic constants open for empirical validation in Sprint 1 — every
one of them lives in :class:`Phase2Config` and is overridable.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.phase2.alpha_controller import (
    AlphaController,
    PriorObservation,
    PriorPerformanceTracker,
)
from cognithor.channels.program_synthesis.phase2.alpha_mixer import (
    alpha_bounds,
    apply_sample_size_dampening,
    mix_alpha,
)
from cognithor.channels.program_synthesis.phase2.classification import (
    HIGH_IMPACT_PRIMITIVES,
    STRUCTURAL_ABSTRACTION_PRIMITIVES,
    classify_primitive_name,
)
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
    VerifierScoreWeights,
)
from cognithor.channels.program_synthesis.phase2.config_loader import (
    DEFAULT_HEURISTICS_PATH,
    ConfigLoadError,
    LoadedHeuristics,
    load_heuristics,
)
from cognithor.channels.program_synthesis.phase2.datatypes import (
    FeatureWithConfidence,
    MCTSNode,
    MCTSState,
    MixedPolicy,
    PartitionedBudget,
)
from cognithor.channels.program_synthesis.phase2.dual_prior import (
    DualPriorMixer,
    DualPriorResult,
)
from cognithor.channels.program_synthesis.phase2.llm_prior import (
    LLMPrior,
    LLMPriorClient,
    LLMPriorError,
)
from cognithor.channels.program_synthesis.phase2.pixel_match import (
    average_partial_pixel_match,
    partial_pixel_match,
)
from cognithor.channels.program_synthesis.phase2.scoring import (
    VerifierScoreInputs,
    aggregate_verifier_score,
)
from cognithor.channels.program_synthesis.phase2.symbolic_prior import (
    SymbolicPrior,
    SymbolicPriorResult,
    UniformSymbolicPrior,
)
from cognithor.channels.program_synthesis.phase2.telemetry import (
    phase2_counters,
)
from cognithor.channels.program_synthesis.phase2.triviality import (
    triviality_score,
)
from cognithor.channels.program_synthesis.phase2.verifier import (
    SuspicionScore,
    compute_suspicion,
)

__all__ = [
    "DEFAULT_HEURISTICS_PATH",
    "DEFAULT_PHASE2_CONFIG",
    "HIGH_IMPACT_PRIMITIVES",
    "STRUCTURAL_ABSTRACTION_PRIMITIVES",
    "AlphaController",
    "ConfigLoadError",
    "DualPriorMixer",
    "DualPriorResult",
    "FeatureWithConfidence",
    "LLMPrior",
    "LLMPriorClient",
    "LLMPriorError",
    "LoadedHeuristics",
    "MCTSNode",
    "MCTSState",
    "MixedPolicy",
    "PartitionedBudget",
    "Phase2Config",
    "PriorObservation",
    "PriorPerformanceTracker",
    "SuspicionScore",
    "SymbolicPrior",
    "SymbolicPriorResult",
    "UniformSymbolicPrior",
    "VerifierScoreInputs",
    "VerifierScoreWeights",
    "aggregate_verifier_score",
    "alpha_bounds",
    "apply_sample_size_dampening",
    "average_partial_pixel_match",
    "classify_primitive_name",
    "compute_suspicion",
    "load_heuristics",
    "mix_alpha",
    "partial_pixel_match",
    "phase2_counters",
    "triviality_score",
]
