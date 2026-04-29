# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""State-Graph-Navigator bridge (spec §15).

The bestehende SGN gives ARC-AGI-3 task hints (symmetry, dimension
ratio, dominant-color change). The PSE consumes those hints as
*task-specific cost multipliers* — no structural change to the search,
just a re-ordering of which primitives the enumerator tries first.

**Phase-1 guarantee:** Correctness MUST NOT depend on SGN being
available. Missing or empty SGN annotations degrade gracefully to
multiplier-1.0 (i.e. no preference change).
"""

from __future__ import annotations

from typing import Any

from cognithor.channels.program_synthesis.core.types import TaskSpec

# Hint keys that trigger a cost multiplier. Phase 1 keeps the set
# small and well-known so the bridge can be tested deterministically;
# Phase 2 may expand this when SGN grows new annotations.
SUPPORTED_HINT_KEYS: frozenset[str] = frozenset(
    {
        "mirror_horizontal",  # output is horizontal mirror of input
        "mirror_vertical",
        "rotate90",
        "rotate180",
        "rotate270",
        "scale_up_2x",
        "scale_up_3x",
        "scale_down_2x",
        "recolor_only",  # only colour mapping changes; geometry preserved
    }
)


# Cost multiplier applied to a primitive when SGN flags it as relevant.
# 0.5 means "half the cost" → the equivalence-pruner-friendly
# enumerator visits these candidates earlier without changing the set
# of candidates considered (Phase 1 doesn't yet implement priority
# search; multipliers are infrastructure for Phase 2).
PROMOTED_MULTIPLIER: float = 0.5
NEUTRAL_MULTIPLIER: float = 1.0


class StateGraphBridge:
    """Convert SGN annotations into cost multipliers consumable by search.

    The bridge is intentionally stateless — every call is a pure
    function of its arguments. Tests construct one instance per assertion
    and exercise the two public methods independently.
    """

    @staticmethod
    def annotate(spec: TaskSpec, sgn_result: dict[str, Any]) -> TaskSpec:
        """Return a copy of *spec* with SGN hints folded into ``annotations``.

        Unsupported keys are dropped silently — the spec calls this out
        as a robustness requirement so SGN protocol drift can't crash
        the search.
        """
        if not sgn_result:
            return spec
        kept: dict[str, Any] = {}
        for key, value in sgn_result.items():
            if key in SUPPORTED_HINT_KEYS:
                kept[key] = value
        if not kept:
            return spec
        merged = dict(spec.annotations)
        for k, v in kept.items():
            merged[f"sgn:{k}"] = v
        return TaskSpec(
            examples=spec.examples,
            held_out=spec.held_out,
            test_input=spec.test_input,
            constraints=spec.constraints,
            domain=spec.domain,
            annotations=tuple(sorted(merged.items())),
        )

    @staticmethod
    def cost_multipliers(annotations: dict[str, Any]) -> dict[str, float]:
        """Return ``{primitive_name: multiplier}`` for every recognised hint.

        Hints are namespaced with the ``sgn:`` prefix in :meth:`annotate`
        so unrelated annotations don't accidentally promote primitives.
        """
        out: dict[str, float] = {}
        for key, value in annotations.items():
            if not isinstance(key, str) or not key.startswith("sgn:"):
                continue
            primitive_name = key[len("sgn:") :]
            if primitive_name not in SUPPORTED_HINT_KEYS:
                continue
            # Truthy hints promote, falsy hints de-promote (rarely used
            # but cheap to support).
            out[primitive_name] = PROMOTED_MULTIPLIER if value else NEUTRAL_MULTIPLIER
        return out


__all__ = [
    "NEUTRAL_MULTIPLIER",
    "PROMOTED_MULTIPLIER",
    "SUPPORTED_HINT_KEYS",
    "StateGraphBridge",
]
