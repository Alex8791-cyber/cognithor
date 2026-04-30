# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Module A — Symbolic-Prior interface (spec v1.4 §4.4).

The Symbolic-Prior contributes a heuristic-derived per-primitive
distribution that the Dual-Prior mixer combines with the LLM side.
Spec §4.4 references "~20 Heuristik-Regeln mit Confidence-Multiplikator"
plus :class:`apply_sample_size_dampening` from
:mod:`phase2.alpha_mixer` already; the rules themselves carry over
from spec v1.3 and land in a follow-up sprint.

Sprint-1 ships:

* the abstract :class:`SymbolicPrior` ABC the mixer reads,
* a :class:`UniformSymbolicPrior` stub that hands back a flat
  distribution over the registered DSL primitives — useful for tests
  and as the Phase-2 default until the heuristic catalog lands.

Both pieces are :class:`Phase2Config`-driven where a constant matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class SymbolicPriorResult:
    """One symbolic-prior call's output.

    ``primitive_scores`` is keyed by primitive name and sums to ~1.0.
    ``effective_confidence`` is the Sample-Size-Dampened confidence
    the mixer can read into ``α_performance``; it lives in ``[0, 1]``
    and reflects how well-supported the heuristic verdict is by the
    number of demos.
    """

    primitive_scores: dict[str, float]
    effective_confidence: float


class SymbolicPrior(ABC):
    """Abstract interface every symbolic-prior implementation honours.

    The Dual-Prior mixer asks for a prior given a sequence of
    ``(input, output)`` example pairs. Concrete implementations may
    cache, sample-size-dampen, or short-circuit — the contract is a
    valid :class:`SymbolicPriorResult` whose primitive_scores keys
    are subset of the live DSL whitelist.
    """

    @abstractmethod
    def get_prior(
        self,
        examples: Iterable[tuple[Any, Any]],
    ) -> SymbolicPriorResult:
        """Return the per-primitive prior + a confidence in [0, 1]."""


class UniformSymbolicPrior(SymbolicPrior):
    """Sprint-1 stub — uniform over the live DSL whitelist.

    Useful for two scenarios:

    * unit tests of the mixer that don't want a real heuristic catalog
      driving the assertions;
    * early Phase-2 runs where the heuristic catalog hasn't landed yet
      — the mixer treats this stub as "no symbolic signal" and the LLM
      side dominates (subject to the α-band clamping in
      :class:`Phase2Config`).

    Confidence reports the canonical sample-size-dampened value via
    :func:`apply_sample_size_dampening` against a base of 1.0, so the
    α-mixer sees a curve from 0.0 (no demos) to 1.0 (n→∞).
    """

    def __init__(
        self,
        *,
        primitive_whitelist: list[str] | None = None,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._explicit_whitelist = primitive_whitelist
        self._config = config

    def get_prior(
        self,
        examples: Iterable[tuple[Any, Any]],
    ) -> SymbolicPriorResult:
        from cognithor.channels.program_synthesis.phase2.alpha_mixer import (
            apply_sample_size_dampening,
        )

        materialised = list(examples)
        whitelist = self._resolve_whitelist()
        if not whitelist:
            raise ValueError(
                "UniformSymbolicPrior: empty primitive whitelist; "
                "the live REGISTRY has no primitives or the explicit "
                "whitelist is empty."
            )
        scores = _uniform_distribution(whitelist)
        confidence = apply_sample_size_dampening(
            base_confidence=1.0,
            n_samples=len(materialised),
            config=self._config,
        )
        return SymbolicPriorResult(
            primitive_scores=scores,
            effective_confidence=confidence,
        )

    def _resolve_whitelist(self) -> list[str]:
        if self._explicit_whitelist is not None:
            return list(self._explicit_whitelist)
        # Lazy import — keeps the module loadable without booting the
        # full DSL.
        from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

        return list(REGISTRY.names())


def _uniform_distribution(names: list[str]) -> dict[str, float]:
    n = len(names)
    return {name: 1.0 / n for name in names}


__all__ = [
    "SymbolicPrior",
    "SymbolicPriorResult",
    "UniformSymbolicPrior",
]
