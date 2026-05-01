# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §4.4 — Symbolic-Prior heuristic catalog (Sprint-1 plan task 4).

The catalog maps observed features of the demo (input, output) pairs
to per-primitive scores. Each rule looks at the example pairs and
emits a small ``dict[primitive_name, score]`` — its vote on which
DSL primitives are likely useful. The aggregator sums every rule's
votes, normalises, and dampens by the demo count (spec §4.4
``n / (n + n0)`` formula already in :func:`apply_sample_size_dampening`).

Sprint-1 ships 20 rules grouped into four categories:

* **Shape** — exact / transposed / scaled / cropped / framed
* **Symmetry** — horizontal / vertical / diagonal flip + 90/180/270 rotation
* **Color** — palette preserved / changed / introduced / removed
* **Structure** — multi-object / background-dominant / identity

Every rule is a pure function ``rule(input, output) -> dict[str, float]``
keyed by primitive name. Scores live in ``[0, 1]`` and are produced
*per example pair*; the aggregator averages across pairs (so a rule
that fires on every pair carries more weight than a one-off).

The catalog is :class:`Phase2Config`-driven for the sample-size
dampening n0; rule firing is purely structural and needs no extra
configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.phase2.alpha_mixer import (
    apply_sample_size_dampening,
)
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)
from cognithor.channels.program_synthesis.phase2.symbolic_prior import (
    SymbolicPrior,
    SymbolicPriorResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


# Each rule takes one (input, output) pair and returns its votes.
HeuristicRule = Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], dict[str, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_array(x: Any) -> np.ndarray[Any, Any]:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.int8)


def _shape_equal(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> bool:
    return bool(a.shape == b.shape)


def _palette(grid: np.ndarray[Any, Any]) -> frozenset[int]:
    if grid.size == 0:
        return frozenset()
    return frozenset(int(v) for v in np.unique(grid).tolist())


# ---------------------------------------------------------------------------
# Shape rules
# ---------------------------------------------------------------------------


def r_shape_equal(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    """Same shape → identity, recolor, mask_*, swap_colors all plausible."""
    if not _shape_equal(inp, out):
        return {}
    return {
        "identity": 0.4,
        "recolor": 0.6,
        "swap_colors": 0.5,
        "replace_background": 0.4,
        "mask_eq": 0.3,
        "mask_ne": 0.3,
    }


def r_shape_transposed(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    """``out.shape == reversed(in.shape)`` → 90° rotations + transpose."""
    if inp.ndim != 2 or out.ndim != 2:
        return {}
    if out.shape != inp.shape[::-1]:
        return {}
    return {
        "rotate90": 0.7,
        "rotate270": 0.7,
        "transpose": 0.8,
        "mirror_diagonal": 0.5,
        "mirror_antidiagonal": 0.5,
    }


def r_shape_scaled_up_2x(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if inp.ndim != 2 or out.ndim != 2:
        return {}
    if inp.shape == (0, 0) or inp.size == 0:
        return {}
    if out.shape != (inp.shape[0] * 2, inp.shape[1] * 2):
        return {}
    return {"scale_up_2x": 0.9, "tile_2x": 0.6}


def r_shape_scaled_up_3x(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if inp.ndim != 2 or out.ndim != 2:
        return {}
    if inp.size == 0:
        return {}
    if out.shape != (inp.shape[0] * 3, inp.shape[1] * 3):
        return {}
    return {"scale_up_3x": 0.9}


def r_shape_scaled_down_2x(
    inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]
) -> dict[str, float]:
    if inp.ndim != 2 or out.ndim != 2:
        return {}
    if out.size == 0:
        return {}
    if inp.shape != (out.shape[0] * 2, out.shape[1] * 2):
        return {}
    return {"scale_down_2x": 0.9}


def r_output_smaller(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    """Output strictly smaller → crop / bounding-box / largest-object."""
    if inp.ndim != 2 or out.ndim != 2:
        return {}
    if not (out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]):
        return {}
    if out.shape == inp.shape:
        return {}
    return {
        "crop_bbox": 0.7,
        "bounding_box": 0.5,
        "largest_object": 0.4,
        "smallest_object": 0.3,
    }


def r_output_larger(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    """Output strictly larger → pad / frame / tile / stack."""
    if inp.ndim != 2 or out.ndim != 2:
        return {}
    if not (out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]):
        return {}
    if out.shape == inp.shape:
        return {}
    return {
        "pad_with": 0.6,
        "frame": 0.4,
        "tile_2x": 0.4,
        "stack_horizontal": 0.4,
        "stack_vertical": 0.4,
        "overlay": 0.3,
    }


# ---------------------------------------------------------------------------
# Symmetry rules
# ---------------------------------------------------------------------------


def r_horizontal_flip(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if not _shape_equal(inp, out):
        return {}
    if inp.size == 0:
        return {}
    if np.array_equal(np.fliplr(inp), out):
        return {"mirror_horizontal": 0.95}
    return {}


def r_vertical_flip(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if not _shape_equal(inp, out):
        return {}
    if inp.size == 0:
        return {}
    if np.array_equal(np.flipud(inp), out):
        return {"mirror_vertical": 0.95}
    return {}


def r_diagonal_flip(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if inp.ndim != 2 or out.ndim != 2:
        return {}
    if inp.size == 0:
        return {}
    if out.shape == inp.shape[::-1] and np.array_equal(inp.T, out):
        return {"transpose": 0.9, "mirror_diagonal": 0.85}
    return {}


def r_rotation_90(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if inp.ndim != 2 or out.ndim != 2 or inp.size == 0:
        return {}
    # DSL rotate90 is clockwise (= np.rot90 with k=-1).
    if out.shape == inp.shape[::-1] and np.array_equal(np.rot90(inp, k=-1), out):
        return {"rotate90": 0.95}
    return {}


def r_rotation_180(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if not _shape_equal(inp, out) or inp.size == 0:
        return {}
    if np.array_equal(np.rot90(inp, k=2), out):
        return {"rotate180": 0.95}
    return {}


def r_rotation_270(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if inp.ndim != 2 or out.ndim != 2 or inp.size == 0:
        return {}
    # DSL rotate270 = np.rot90 with k=1.
    if out.shape == inp.shape[::-1] and np.array_equal(np.rot90(inp, k=1), out):
        return {"rotate270": 0.95}
    return {}


# ---------------------------------------------------------------------------
# Color rules
# ---------------------------------------------------------------------------


def r_palette_preserved(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if _palette(inp) != _palette(out):
        return {}
    return {"identity": 0.3, "swap_colors": 0.4, "shift": 0.3, "wrap_shift": 0.3}


def r_palette_changed(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if _palette(inp) == _palette(out):
        return {}
    return {"recolor": 0.7, "replace_background": 0.4, "swap_colors": 0.3}


def r_new_colors_introduced(
    inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]
) -> dict[str, float]:
    introduced = _palette(out) - _palette(inp)
    if not introduced:
        return {}
    return {"recolor": 0.6, "render_objects": 0.5, "frame": 0.4, "pad_with": 0.4}


def r_colors_removed(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    removed = _palette(inp) - _palette(out)
    if not removed:
        return {}
    return {"recolor": 0.5, "mask_eq": 0.4, "mask_apply": 0.4, "replace_background": 0.4}


# ---------------------------------------------------------------------------
# Structure rules
# ---------------------------------------------------------------------------


def r_objects_present(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    """Multiple non-background objects in the input → object-level primitives."""
    if inp.size == 0:
        return {}
    palette = _palette(inp)
    if len(palette) <= 2:
        # Hard to call "many objects" with one or two colors total.
        return {}
    return {
        "objects_of_color": 0.5,
        "filter_objects": 0.5,
        "largest_object": 0.4,
        "smallest_object": 0.3,
        "object_count": 0.3,
        "sort_objects": 0.3,
        "map_objects": 0.4,
        "connected_components_4": 0.4,
        "connected_components_8": 0.3,
    }


def r_background_dominant(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    """One color covers ≥ 70 % of the input → background-related primitives."""
    if inp.size == 0:
        return {}
    counts = np.bincount(inp.ravel().astype(np.int64), minlength=10)
    if counts.size == 0 or counts.max() < 0.7 * inp.size:
        return {}
    return {
        "replace_background": 0.7,
        "most_common_color": 0.5,
        "least_common_color": 0.3,
        "mask_ne": 0.3,
    }


def r_identity_match(inp: np.ndarray[Any, Any], out: np.ndarray[Any, Any]) -> dict[str, float]:
    if _shape_equal(inp, out) and np.array_equal(inp, out):
        return {"identity": 0.95}
    return {}


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------


# 20 rules total — order is presentation-only; firing is independent.
DEFAULT_RULES: tuple[HeuristicRule, ...] = (
    # Shape (7)
    r_shape_equal,
    r_shape_transposed,
    r_shape_scaled_up_2x,
    r_shape_scaled_up_3x,
    r_shape_scaled_down_2x,
    r_output_smaller,
    r_output_larger,
    # Symmetry (6)
    r_horizontal_flip,
    r_vertical_flip,
    r_diagonal_flip,
    r_rotation_90,
    r_rotation_180,
    r_rotation_270,
    # Color (4)
    r_palette_preserved,
    r_palette_changed,
    r_new_colors_introduced,
    r_colors_removed,
    # Structure (3)
    r_objects_present,
    r_background_dominant,
    r_identity_match,
)


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class HeuristicSymbolicPrior(SymbolicPrior):
    """Sprint-1 heuristic catalog — runs 20 rules over each demo pair.

    Aggregation:

    1. For each ``(input, output)`` pair, run every rule and collect
       its vote dict.
    2. Sum votes per primitive across the rule outputs for that pair
       (one rule firing twice on the same primitive in different
       categories is intended — multiple signals reinforce).
    3. Average pair-level dicts so a rule that fires on every pair
       carries more weight than one that fires once.
    4. Filter to the live whitelist; normalise to a distribution.
    5. Effective confidence = sample-size dampening of the *non-empty
       rule fraction* (so two firing rules out of 20 yield lower
       confidence than fifteen firing rules).

    Falls back to a uniform distribution over the whitelist when no
    rule fires — keeps the dual-prior mixer well-defined.
    """

    def __init__(
        self,
        *,
        primitive_whitelist: list[str] | None = None,
        rules: tuple[HeuristicRule, ...] = DEFAULT_RULES,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._explicit_whitelist = primitive_whitelist
        self._rules = rules
        self._config = config

    def get_prior(
        self,
        examples: Iterable[tuple[Any, Any]],
    ) -> SymbolicPriorResult:
        materialised = [(_as_array(i), _as_array(o)) for i, o in examples]
        whitelist = self._resolve_whitelist()
        if not whitelist:
            raise ValueError(
                "HeuristicSymbolicPrior: empty primitive whitelist; the "
                "live REGISTRY has no primitives or the explicit whitelist "
                "is empty."
            )
        if not materialised:
            return SymbolicPriorResult(
                primitive_scores=_uniform(whitelist),
                effective_confidence=0.0,
            )

        per_pair_scores: list[dict[str, float]] = []
        per_pair_fire_counts: list[int] = []
        for inp, out in materialised:
            scores: dict[str, float] = {}
            fired = 0
            for rule in self._rules:
                vote = rule(inp, out)
                if not vote:
                    continue
                fired += 1
                for prim, value in vote.items():
                    scores[prim] = scores.get(prim, 0.0) + float(value)
            per_pair_scores.append(scores)
            per_pair_fire_counts.append(fired)

        # Average across pairs.
        averaged: dict[str, float] = {}
        n_pairs = len(materialised)
        for scores in per_pair_scores:
            for prim, value in scores.items():
                averaged[prim] = averaged.get(prim, 0.0) + value / n_pairs
        # Filter to whitelist + drop zero entries.
        filtered = {k: v for k, v in averaged.items() if k in set(whitelist) and v > 0.0}
        if not filtered:
            scores_out = _uniform(whitelist)
        else:
            scores_out = _normalise(filtered)

        # Confidence = fraction of rules that fired (averaged across pairs)
        # × sample-size dampening.
        avg_fire_fraction = (
            sum(per_pair_fire_counts) / (len(per_pair_fire_counts) * len(self._rules))
            if self._rules
            else 0.0
        )
        confidence = apply_sample_size_dampening(
            base_confidence=avg_fire_fraction,
            n_samples=n_pairs,
            config=self._config,
        )
        return SymbolicPriorResult(
            primitive_scores=scores_out,
            effective_confidence=confidence,
        )

    def _resolve_whitelist(self) -> list[str]:
        if self._explicit_whitelist is not None:
            return list(self._explicit_whitelist)
        from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

        return list(REGISTRY.names())


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _uniform(names: list[str]) -> dict[str, float]:
    n = len(names) or 1
    return {name: 1.0 / n for name in names}


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0.0:
        return _uniform(list(scores))
    return {k: v / total for k, v in scores.items()}


__all__ = [
    "DEFAULT_RULES",
    "HeuristicRule",
    "HeuristicSymbolicPrior",
    "r_background_dominant",
    "r_colors_removed",
    "r_diagonal_flip",
    "r_horizontal_flip",
    "r_identity_match",
    "r_new_colors_introduced",
    "r_objects_present",
    "r_output_larger",
    "r_output_smaller",
    "r_palette_changed",
    "r_palette_preserved",
    "r_rotation_90",
    "r_rotation_180",
    "r_rotation_270",
    "r_shape_equal",
    "r_shape_scaled_down_2x",
    "r_shape_scaled_up_2x",
    "r_shape_scaled_up_3x",
    "r_shape_transposed",
    "r_vertical_flip",
]
