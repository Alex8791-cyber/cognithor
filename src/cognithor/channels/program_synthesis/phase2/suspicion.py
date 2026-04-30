# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec v1.4 §7.3.2 — Suspicion-Score with two-class multipliers (F1).

Pure helper used by the Phase-2 verifier extension. Lives in the
``phase2`` sub-package so Phase-1 imports stay untouched until the
Phase-2 verifier wiring lands.

The suspicion score combines:

* a length factor (effective tokens vs a 12-token budget)
* a depth factor (composition depth vs a 6-deep budget)

The "effective tokens" sum is what the v1.4 split changes: each token
is weighted by its class multiplier from :class:`Phase2Config`, with
the spec defaults of 3× / 1.5× / 1×.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.phase2.classification import (
    classify_primitive_name,
)
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.search.candidate import ProgramNode


_LENGTH_BUDGET = 12.0
_DEPTH_BUDGET = 6.0


def _walk_primitive_names(node: ProgramNode) -> list[str]:
    """Collect every primitive name in the program tree, depth-first.

    InputRef and Const leaves contribute nothing — only Program nodes
    (the only ones with a ``primitive`` attribute) count toward the
    syntactic complexity. The Phase-2 spec's intuition is that
    "tokens" are primitive applications, not literals.
    """
    from cognithor.channels.program_synthesis.search.candidate import (
        Program as _Program,
    )

    names: list[str] = []
    if isinstance(node, _Program):
        names.append(node.primitive)
        for child in node.children:
            names.extend(_walk_primitive_names(child))
    return names


def _multiplier_for_class(
    cls: str,
    config: Phase2Config,
) -> float:
    if cls == "high_impact":
        return config.high_impact_multiplier
    if cls == "structural_abstraction":
        return config.structural_abstraction_multiplier
    return config.regular_primitive_multiplier


def effective_token_count(
    program: ProgramNode,
    *,
    config: Phase2Config = DEFAULT_PHASE2_CONFIG,
) -> float:
    """Sum of class-weighted token contributions for *program*.

    A 1-token ``tile(g)`` weighs 3.0 (spec default).
    A 1-token ``objects(g)`` weighs 1.5 (the F1 change).
    A 1-token ``recolor(g, 1, 5)`` weighs 1.0 — the regular default.
    """
    return sum(
        _multiplier_for_class(classify_primitive_name(name), config)
        for name in _walk_primitive_names(program)
    )


def compute_syntactic_complexity(
    program: ProgramNode,
    *,
    config: Phase2Config = DEFAULT_PHASE2_CONFIG,
) -> float:
    """Spec v1.4 §7.3.2 — graduated [0, 1] complexity score.

    ``0.6 · length_factor + 0.4 · depth_factor``

    where each factor is the relevant measure clamped to its budget.
    Returns ``0.0`` for an empty program (no Program nodes — pure
    InputRef/Const). The spec uses this score as one input to the
    suspicion calculation; lower scores mean "simpler, more
    suspicious" and trigger the Triviality penalty.
    """
    eff_len = effective_token_count(program, config=config)
    if eff_len == 0.0:
        return 0.0
    length_factor = min(eff_len / _LENGTH_BUDGET, 1.0)
    depth_factor = min(program.depth() / _DEPTH_BUDGET, 1.0)
    return 0.6 * length_factor + 0.4 * depth_factor


__all__ = [
    "compute_syntactic_complexity",
    "effective_token_count",
]
