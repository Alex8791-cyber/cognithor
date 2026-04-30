# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Module A — Dual-Prior mixer (spec v1.4 §4 + §4.4.4).

Combines an :class:`LLMPriorClient` output with a :class:`SymbolicPrior`
output via the multiplicative-adaptive α from :mod:`phase2.alpha_mixer`:

    α = α_entropy · α_performance ∈ [0.25, 0.85]
    π_combined = α · π_llm + (1 − α) · π_symbolic

α_entropy comes from the LLM (its self-reported entropy hint).
α_performance comes from the Symbolic-Prior's effective_confidence
(sample-size-dampened, [0, 1]). Both are clamped to their config
bands inside :func:`mix_alpha`, so a misbehaving side cannot push α
outside the spec range.

The mixer is fully :class:`Phase2Config`-driven; the heuristic catalog
behind the SymbolicPrior is a separate concern (lands in a Sprint-2
PR once spec v1.3 detail is consulted). For now the
:class:`UniformSymbolicPrior` Stub is the typical injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.phase2.alpha_mixer import mix_alpha
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cognithor.channels.program_synthesis.phase2.llm_prior import (
        LLMPrior,
        LLMPriorClient,
    )
    from cognithor.channels.program_synthesis.phase2.symbolic_prior import (
        SymbolicPrior,
        SymbolicPriorResult,
    )


@dataclass(frozen=True)
class DualPriorResult:
    """One dual-prior call's combined output.

    ``primitive_scores`` is the mixed distribution; sums to ~1.0.
    ``alpha`` is the resolved α, in the configured Search-α band.
    ``llm_prior`` and ``symbolic_prior`` echo the inputs so callers
    can audit per-side contributions (the spec-mandated telemetry
    field ``alpha_entropy_hint`` lives on ``llm_prior``).
    """

    primitive_scores: dict[str, float]
    alpha: float
    llm_prior: LLMPrior
    symbolic_prior: SymbolicPriorResult


class DualPriorMixer:
    """Combines LLM-Prior and Symbolic-Prior into one distribution.

    The mixer is async because the LLM side is. Concrete LLM clients
    pass a :class:`LLMPriorClient` whose ``get_prior`` returns an
    :class:`LLMPrior`; concrete symbolic priors implement
    :class:`SymbolicPrior`.
    """

    def __init__(
        self,
        llm_client: LLMPriorClient,
        symbolic_prior: SymbolicPrior,
        *,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._llm = llm_client
        self._symbolic = symbolic_prior
        self._config = config

    async def get_prior(
        self,
        examples: Iterable[tuple[Any, Any]],
    ) -> DualPriorResult:
        """Run both priors and combine with the configured α."""
        materialised = list(examples)
        # Symbolic side first — it's pure and cheap. The LLM call
        # follows, dominating wall-clock; running them sequentially
        # keeps the implementation simple and avoids the async-
        # parallel Pitfalls (event-loop politeness, error attribution).
        symbolic_result = self._symbolic.get_prior(materialised)
        llm_result = await self._llm.get_prior(materialised)
        alpha = mix_alpha(
            alpha_entropy=llm_result.alpha_entropy_hint,
            alpha_performance=symbolic_result.effective_confidence,
            config=self._config,
        )
        combined = _convex_mix(
            llm_result.primitive_scores,
            symbolic_result.primitive_scores,
            alpha=alpha,
        )
        return DualPriorResult(
            primitive_scores=combined,
            alpha=alpha,
            llm_prior=llm_result,
            symbolic_prior=symbolic_result,
        )


def _convex_mix(
    llm_scores: dict[str, float],
    symbolic_scores: dict[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    """Compute ``α · π_llm + (1 − α) · π_symbolic`` over the union of keys.

    Missing keys on one side are treated as zero — the contract is
    that the result has only keys that were in *some* input. The
    output is renormalised to sum to 1.0; if every entry collapses to
    zero (degenerate input), returns an empty dict — the caller is
    expected to treat that as "no prior signal" (typically falling
    back to a uniform).
    """
    keys = set(llm_scores) | set(symbolic_scores)
    raw = {
        k: alpha * llm_scores.get(k, 0.0) + (1.0 - alpha) * symbolic_scores.get(k, 0.0)
        for k in keys
    }
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


__all__ = [
    "DualPriorMixer",
    "DualPriorResult",
]
