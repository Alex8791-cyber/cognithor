# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec v1.4 §7.3.2 — DSL primitive classification (F1).

Two mutually-exclusive classes the suspicion-score reads:

* :data:`HIGH_IMPACT_PRIMITIVES` — produce the program output directly.
  Carry a ``high_impact_multiplier`` (3× by default).
* :data:`STRUCTURAL_ABSTRACTION_PRIMITIVES` — produce intermediate
  representations (object lists, masks, bboxes) rather than outputs.
  Carry a ``structural_abstraction_multiplier`` (1.5× by default).

Reclassification of ``objects`` from High-Impact to
Structural-Abstraction is the central F1 change in v1.4: a 1-token
``objects(g)`` produces a list, not a final grid, so the 3× boost was
distorting suspicion. The 1.5× tier captures "useful abstraction
without output" exactly.

The whitelists are frozensets so they can be used as drop-in lookup
tables. :func:`classify_primitive_name` maps a name string to the
class; absent names are 'regular' (1× multiplier).
"""

from __future__ import annotations

from typing import Literal

# Spec v1.4 §7.3.2 — primitives that produce program output directly.
HIGH_IMPACT_PRIMITIVES: frozenset[str] = frozenset(
    {
        "tile",
        "flood_fill",
        "mirror",
        "mirror_horizontal",
        "mirror_vertical",
        "rotate",
        "rotate90",
        "rotate180",
        "rotate270",
        "transpose",
        "compose_grid",
        "scale",
        "scale_up_2x",
        "scale_up_3x",
        "scale_down_2x",
    }
)

# Spec v1.4 §7.3.2 — primitives that produce intermediate
# representations (object sets, masks, bboxes), not outputs.
# ``objects()`` is the canonical example; this set widens it to every
# Phase-1 primitive that returns a non-Grid type meant to feed a
# downstream transformation.
STRUCTURAL_ABSTRACTION_PRIMITIVES: frozenset[str] = frozenset(
    {
        "objects",
        "connected_components_4",
        "connected_components_8",
        "filter_objects",
        "group_by_color",
        "find_pattern",
        "extract_bbox",
        "bounding_box",
        "largest_object",
        "smallest_object",
        "mask_eq",
        "mask_ne",
    }
)


PrimitiveClass = Literal["high_impact", "structural_abstraction", "regular"]


def classify_primitive_name(name: str) -> PrimitiveClass:
    """Return the spec v1.4 §7.3.2 class for a primitive name.

    Mutually-exclusive: a primitive belongs to at most one class.
    Names not in either whitelist are ``regular`` (1× multiplier).

    The function does not consult the live :class:`PrimitiveRegistry`
    — it purely reads the static whitelists, so it can be called
    without booting the registry (e.g. from a test fixture).
    """
    if name in HIGH_IMPACT_PRIMITIVES:
        return "high_impact"
    if name in STRUCTURAL_ABSTRACTION_PRIMITIVES:
        return "structural_abstraction"
    return "regular"


def _check_no_overlap() -> None:
    """Module import-time invariant: the two whitelists are disjoint.

    Spec v1.4 §18.2 guarantee. Triggered at module load so a future
    edit that accidentally puts the same name in both fails fast.
    """
    overlap = HIGH_IMPACT_PRIMITIVES & STRUCTURAL_ABSTRACTION_PRIMITIVES
    if overlap:
        raise AssertionError(
            f"classification.py: HIGH_IMPACT and STRUCTURAL_ABSTRACTION "
            f"whitelists must be disjoint; overlap: {sorted(overlap)}"
        )


_check_no_overlap()


__all__ = [
    "HIGH_IMPACT_PRIMITIVES",
    "STRUCTURAL_ABSTRACTION_PRIMITIVES",
    "PrimitiveClass",
    "classify_primitive_name",
]
