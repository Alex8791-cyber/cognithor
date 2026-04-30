# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 typed datatypes (Sprint-1 plan task 2, spec §9).

The Sprint-1 plan calls for the full set of frozen dataclasses the
Phase-2 modules read. This module ships the ones that aren't already
covered by Phase-1 (TaskSpec, Demo, Program, ProgramState) or by
earlier Sprint-1 work (DSLPrimitive flags, LLMPrior, SymbolicPriorResult,
DualPriorResult, SuspicionScore):

* :class:`FeatureWithConfidence` — symbolic-prior input slot. Carries
  a value and the number of demos that produced it; its
  ``effective_confidence`` reads :class:`Phase2Config` to apply the
  spec §4.4 sample-size dampening.
* :class:`PartitionedBudget` — spec §13.4 budget split across the four
  pipeline stages (pre-processing, MCTS, refiner, CEGIS). Validates
  that the four fractions sum to 1.0 at construction time.
* :class:`MixedPolicy` — light alias around the
  ``(primitive_scores, alpha)`` tuple. The dual-prior mixer's
  :class:`DualPriorResult` carries strictly more (per-side priors for
  telemetry), but Module B's MCTS controller only needs the mixed view.
* :class:`MCTSNode` — mutable PUCT node. Tracks visit count, total
  value, prior, and parent/children edges. ``puct_score`` property
  follows spec §5.2.
* :class:`MCTSState` — top-level search state holding the root node
  plus the live budget snapshot.

Spec §9 (v1.4, "unverändert ggü v1.3") does not pin every field of
the MCTS types verbatim — the dataclasses below are the smallest
shape the Sprint-2 controller will read and write. They're typed,
documented, and Hypothesis-friendly (hashable where frozen).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from cognithor.channels.program_synthesis.phase2.alpha_mixer import (
    apply_sample_size_dampening,
)
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

# ---------------------------------------------------------------------------
# Symbolic-prior input slot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureWithConfidence:
    """One feature observation feeding the symbolic-prior catalog.

    ``name`` is a stable string label used by the heuristic catalog to
    look the feature up. ``value`` is the observed quantity (any
    immutable scalar — int / float / str / bool / tuple). ``n_demos``
    is how many demo pairs the observation rests on; the symbolic-
    prior dampens its confidence by ``n / (n + n0)`` per spec §4.4.

    The frozen dataclass is hashable so feature snapshots can be used
    as dict keys for the symbolic-prior cache. Sprint-1 plan acceptance
    criterion: *FeatureWithConfidence.confidence korrekt skaliert via
    Sample-Size.*
    """

    name: str
    value: object
    n_demos: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureWithConfidence.name must be non-empty")
        if self.n_demos < 0:
            raise ValueError(f"FeatureWithConfidence.n_demos must be >= 0; got {self.n_demos}")

    def effective_confidence(
        self,
        *,
        base_confidence: float = 1.0,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> float:
        """Spec §4.4 sample-size-dampened confidence in ``[0, 1]``.

        ``base_confidence`` defaults to 1.0 (the heuristic gives the
        full signal at saturation). The symbolic-prior catalog can
        pass a per-rule base when the rule itself only contributes a
        partial confidence even at infinite demos.
        """
        return apply_sample_size_dampening(
            base_confidence=base_confidence,
            n_samples=self.n_demos,
            config=config,
        )


# ---------------------------------------------------------------------------
# Budget partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionedBudget:
    """Spec §13.4 strict budget partition.

    Each field is the fraction of the total wall-clock budget assigned
    to a pipeline stage. The four fractions must sum to ``1.0`` at
    construction time — the engine reclaims unused fractions back to
    MCTS only at runtime, not at construction.
    """

    pre_processing: float
    mcts: float
    refiner: float
    cegis: float

    def __post_init__(self) -> None:
        for name, value in (
            ("pre_processing", self.pre_processing),
            ("mcts", self.mcts),
            ("refiner", self.refiner),
            ("cegis", self.cegis),
        ):
            if value < 0.0:
                raise ValueError(f"PartitionedBudget.{name} must be >= 0; got {value}")
        total = self.pre_processing + self.mcts + self.refiner + self.cegis
        # Allow a tiny floating-point slack so YAML round-trips don't
        # fail on the obvious 1.0 sum.
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"PartitionedBudget fractions must sum to 1.0; got {total}")

    @classmethod
    def from_spec_default(cls) -> PartitionedBudget:
        """Spec §13.4 defaults: 0.07 / 0.70 / 0.18 / 0.05."""
        return cls(pre_processing=0.07, mcts=0.70, refiner=0.18, cegis=0.05)


# ---------------------------------------------------------------------------
# Module-A output projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixedPolicy:
    """Spec §3.2 — what Module B's MCTS controller reads from Module A.

    ``primitive_scores`` is the convex-combined per-primitive
    distribution; ``alpha`` is the resolved Search-α (in the
    [0.25, 0.85] band by default). Drop-in projection of
    :class:`DualPriorResult` with the per-side telemetry stripped.
    """

    primitive_scores: tuple[tuple[str, float], ...]
    alpha: float

    @classmethod
    def from_dict(
        cls,
        primitive_scores: dict[str, float],
        *,
        alpha: float,
    ) -> MixedPolicy:
        """Build a frozen MixedPolicy from the mixer's dict output."""
        ordered = tuple(sorted(primitive_scores.items()))
        return cls(primitive_scores=ordered, alpha=alpha)

    def as_dict(self) -> dict[str, float]:
        return dict(self.primitive_scores)


# ---------------------------------------------------------------------------
# MCTS node + state
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """Mutable PUCT node — spec §5.2.

    Sprint-1 ships the smallest shape Module B's controller will need;
    the full-feature node (virtual-loss bookkeeping, equivalence
    fingerprint cache, diversity bonus tally) is a Sprint-2 PR that
    extends this dataclass without breaking the existing field set.

    ``visit_count`` and ``total_value`` increment as the search
    progresses; ``prior`` is the prior probability the parent Action
    taken to reach this node carried (read off the mixer). ``children``
    maps action label → child node, populated lazily on expansion.
    """

    primitive: str
    prior: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    parent: MCTSNode | None = None
    children: dict[str, MCTSNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, *, c_puct: float, parent_visit_count: int) -> float:
        """Spec §5.2 — UCB-style PUCT score the parent uses to pick a child.

        ``Q(s,a) + c_puct · prior · sqrt(N_parent) / (1 + N_self)``.
        ``c_puct`` is the spec exploration constant (default 3.5 per
        the heuristics YAML).
        """
        if parent_visit_count <= 0:
            # Square-root degenerates; fall back to the prior alone.
            exploration = self.prior
        else:
            exploration = c_puct * self.prior * sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.mean_value + exploration

    def record_visit(self, value: float) -> None:
        """Increment visit count + accumulate value (back-prop step)."""
        self.visit_count += 1
        self.total_value += value


@dataclass
class MCTSState:
    """Top-level search state."""

    root: MCTSNode
    budget: PartitionedBudget
    iteration: int = 0
    best_so_far: MCTSNode | None = None

    def step(self) -> None:
        """Advance the iteration counter — controller calls per loop."""
        self.iteration += 1


__all__ = [
    "FeatureWithConfidence",
    "MCTSNode",
    "MCTSState",
    "MixedPolicy",
    "PartitionedBudget",
]
