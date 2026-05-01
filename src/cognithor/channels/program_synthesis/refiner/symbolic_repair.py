# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.5.2 — Symbolic-Repair Advisor (Sprint-1 plan task 9 slice).

The Symbolic-Repair stage takes a candidate's actual output, the
demo's expected output, and the :class:`DiffReport` from
:mod:`refiner.diff_analyzer`, and returns prioritised
:class:`RepairSuggestion` items the (Sprint-2) repair driver can
turn into concrete program-tree mutations.

The five Sprint-1 rules (spec §6.5.2 narrative):

* **R1 ColorRepair** — fires when the diff has any missing/introduced
  colors. Suggests a ``recolor`` primitive.
* **R2 ScaleRepair** — fires on shape mismatch where one dimension is
  an integer multiple of the other. Suggests a ``scale_up``/``scale_down``.
* **R3 RotationRepair** — fires when expected equals a 90/180/270°
  rotation of actual. Suggests a ``rotate`` primitive.
* **R4 MirrorRepair** — fires when expected is the horizontal or
  vertical flip of actual. Suggests a ``mirror_horizontal`` /
  ``mirror_vertical`` primitive.
* **R5 LocalRepair** — fires when the pixel-diff is small (≤ 5 cells)
  with matching shape. Suggests delegating to the
  :class:`LocalEditMutator` for cell-level mutations.

Each rule has its own confidence in ``[0, 1]``; the advisor sorts
suggestions by confidence descending. Caller takes as many or as few
as the budget allows.

The module is pure-numpy; no Phase2Config dependency. The caller
gates the rule firing using :class:`DiffReport` flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.refiner.diff_analyzer import (
        DiffReport,
    )


RepairKind = Literal[
    "color_repair",
    "scale_repair",
    "rotation_repair",
    "mirror_repair",
    "local_repair",
]


@dataclass(frozen=True)
class RepairSuggestion:
    """One suggested repair direction.

    ``kind`` identifies which rule fired. ``primitive_hint`` is the
    DSL primitive name the rule recommends inserting (or ``None``
    for ``local_repair`` which delegates back to :class:`LocalEditMutator`).
    ``confidence`` is in ``[0, 1]`` — higher means more likely to
    succeed.

    ``detail`` is a free-form note for telemetry (e.g. "swap colors
    {1, 2}" for a color-repair).
    """

    kind: RepairKind
    primitive_hint: str | None
    confidence: float
    detail: str = ""


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def r1_color_repair(diff: DiffReport) -> RepairSuggestion | None:
    """Fires iff the diff has any missing OR introduced color."""
    if diff.identical:
        return None
    if not diff.colors.introduced and not diff.colors.missing:
        return None
    # Confidence scales with how localised the color disagreement is.
    # If only colors differ (no structural mismatch, all pixels are
    # off because of color) → high confidence.
    if diff.structure.shape_mismatch:
        confidence = 0.4  # color repair won't fix shape; lower
    elif not diff.colors.introduced or not diff.colors.missing:  # one-sided
        confidence = 0.6
    else:
        confidence = 0.8
    detail = f"introduced={sorted(diff.colors.introduced)}, missing={sorted(diff.colors.missing)}"
    return RepairSuggestion(
        kind="color_repair",
        primitive_hint="recolor",
        confidence=confidence,
        detail=detail,
    )


def r2_scale_repair(diff: DiffReport) -> RepairSuggestion | None:
    """Fires iff shapes differ AND one dim is integer-multiple of other."""
    if not diff.structure.shape_mismatch:
        return None
    a_shape = diff.structure.actual_shape
    e_shape = diff.structure.expected_shape
    if len(a_shape) != 2 or len(e_shape) != 2:
        return None
    if a_shape[0] == 0 or a_shape[1] == 0 or e_shape[0] == 0 or e_shape[1] == 0:
        return None
    # Integer up-scale: expected = k · actual.
    if e_shape[0] % a_shape[0] == 0 and e_shape[1] % a_shape[1] == 0:
        kr = e_shape[0] // a_shape[0]
        kc = e_shape[1] // a_shape[1]
        if kr == kc and kr in (2, 3):
            return RepairSuggestion(
                kind="scale_repair",
                primitive_hint=f"scale_up_{kr}x",
                confidence=0.85,
                detail=f"upscale {kr}x ({a_shape} → {e_shape})",
            )
    # Integer down-scale: actual = k · expected.
    if a_shape[0] % e_shape[0] == 0 and a_shape[1] % e_shape[1] == 0:
        kr = a_shape[0] // e_shape[0]
        kc = a_shape[1] // e_shape[1]
        if kr == kc and kr in (2,):
            return RepairSuggestion(
                kind="scale_repair",
                primitive_hint=f"scale_down_{kr}x",
                confidence=0.85,
                detail=f"downscale {kr}x ({a_shape} → {e_shape})",
            )
    return None


def r3_rotation_repair(
    actual: np.ndarray[Any, Any],
    expected: np.ndarray[Any, Any],
    diff: DiffReport,
) -> RepairSuggestion | None:
    """Fires iff expected equals 90/180/270 of actual."""
    if diff.identical:
        return None
    if diff.structure.shape_mismatch and actual.shape != expected.shape[::-1]:
        # 90 / 270 swap dimensions — that's still a candidate.
        # Only bail if neither shape match works.
        return None
    if actual.size == 0 or expected.size == 0:
        return None
    for k, primitive in ((1, "rotate90"), (2, "rotate180"), (3, "rotate270")):
        try:
            rotated = np.rot90(actual, k=k)
        except (ValueError, TypeError):
            continue
        if rotated.shape == expected.shape and np.array_equal(rotated, expected):
            return RepairSuggestion(
                kind="rotation_repair",
                primitive_hint=primitive,
                confidence=0.95,
                detail=f"k={k}",
            )
    return None


def r4_mirror_repair(
    actual: np.ndarray[Any, Any],
    expected: np.ndarray[Any, Any],
    diff: DiffReport,
) -> RepairSuggestion | None:
    """Fires iff expected is horizontal or vertical flip of actual."""
    if diff.identical:
        return None
    if diff.structure.shape_mismatch:
        return None
    if actual.size == 0 or expected.size == 0:
        return None
    flipped_h = np.fliplr(actual)
    if flipped_h.shape == expected.shape and np.array_equal(flipped_h, expected):
        return RepairSuggestion(
            kind="mirror_repair",
            primitive_hint="mirror_horizontal",
            confidence=0.95,
            detail="fliplr",
        )
    flipped_v = np.flipud(actual)
    if flipped_v.shape == expected.shape and np.array_equal(flipped_v, expected):
        return RepairSuggestion(
            kind="mirror_repair",
            primitive_hint="mirror_vertical",
            confidence=0.95,
            detail="flipud",
        )
    return None


def r5_local_repair(diff: DiffReport, *, max_diff_pixels: int = 5) -> RepairSuggestion | None:
    """Fires iff pixel diff is small (≤ max_diff_pixels) and shapes match."""
    if diff.identical:
        return None
    if diff.structure.shape_mismatch:
        return None
    if diff.pixels.count == 0 or diff.pixels.count > max_diff_pixels:
        return None
    return RepairSuggestion(
        kind="local_repair",
        primitive_hint=None,
        confidence=0.7,
        detail=f"{diff.pixels.count} pixel(s) differ",
    )


# ---------------------------------------------------------------------------
# Advisor — runs every rule, sorts by confidence
# ---------------------------------------------------------------------------


def advise_repairs(
    actual: np.ndarray[Any, Any],
    expected: np.ndarray[Any, Any],
    diff: DiffReport,
) -> list[RepairSuggestion]:
    """Run the 5 Sprint-1 symbolic-repair rules + sort by confidence.

    Rules that don't fire on the given diff are simply absent from
    the result. Caller iterates in order; high-confidence suggestions
    typically run first.
    """
    suggestions: list[RepairSuggestion] = []
    for sugg in (
        r1_color_repair(diff),
        r2_scale_repair(diff),
        r3_rotation_repair(actual, expected, diff),
        r4_mirror_repair(actual, expected, diff),
        r5_local_repair(diff),
    ):
        if sugg is not None:
            suggestions.append(sugg)
    # Stable sort: equal confidence preserves rule order so tests can
    # pin the canonical order R1-R5 within a confidence tier.
    suggestions.sort(key=lambda s: -s.confidence)
    return suggestions


__all__ = [
    "RepairKind",
    "RepairSuggestion",
    "advise_repairs",
    "r1_color_repair",
    "r2_scale_repair",
    "r3_rotation_repair",
    "r4_mirror_repair",
    "r5_local_repair",
]
