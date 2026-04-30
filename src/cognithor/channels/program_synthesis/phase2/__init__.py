# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 (Neuro-Symbolic Synthesis Layer) — Sprint 1 foundations.

Contains the typed configuration the rest of Phase 2 (Dual-Prior, MCTS,
Refiner, extended Verifier) reads from. Spec v1.4 §16 leaves several
heuristic constants open for empirical validation in Sprint 1 — every
one of them lives in :class:`Phase2Config` and is overridable.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.phase2.classification import (
    HIGH_IMPACT_PRIMITIVES,
    STRUCTURAL_ABSTRACTION_PRIMITIVES,
    classify_primitive_name,
)
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)
from cognithor.channels.program_synthesis.phase2.verifier import (
    SuspicionScore,
    compute_suspicion,
)

__all__ = [
    "DEFAULT_PHASE2_CONFIG",
    "HIGH_IMPACT_PRIMITIVES",
    "STRUCTURAL_ABSTRACTION_PRIMITIVES",
    "Phase2Config",
    "SuspicionScore",
    "classify_primitive_name",
    "compute_suspicion",
]
