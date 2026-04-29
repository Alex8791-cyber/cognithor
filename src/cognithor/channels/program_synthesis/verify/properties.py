# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-1 property tests for the Verifier pipeline (spec §10.3).

Properties are simple invariants that any valid program output should
satisfy regardless of the specific demo. They run after the demo stage
and before the held-out stage, and they fail-fast on the first
violation.

Each property takes ``(actual_output, expected_output, demo_input)`` and
returns ``(passed, detail)``. Properties that don't apply to a value
type (e.g. ``output_grid_nonempty`` against a Color result) return
``(True, "n/a")``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# Signature: (actual, expected, demo_input) -> (passed, detail)
PropertyFn = Callable[[Any, Any, Any], tuple[bool, str]]


def output_grid_nonempty(actual: Any, expected: Any, demo_input: Any) -> tuple[bool, str]:
    """A grid output must have at least one cell."""
    if not isinstance(actual, np.ndarray):
        return True, "n/a (non-grid output)"
    if actual.size == 0:
        return False, f"empty grid {actual.shape}"
    return True, ""


def output_dimensions_match_inputs_or_constant(
    actual: Any, expected: Any, demo_input: Any
) -> tuple[bool, str]:
    """Output shape must equal expected shape (the spec's loosened version
    of "match input or be constant" — Phase 1 just enforces that the
    program produces the same shape the spec asks for)."""
    if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
        return True, "n/a (non-grid)"
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} != expected {expected.shape}"
    return True, ""


def output_colors_subset_of_input_colors_plus_const(
    actual: Any, expected: Any, demo_input: Any
) -> tuple[bool, str]:
    """Every color appearing in the output must appear in the input or
    in the expected output (Phase 1: the latter is the only way Const-
    introduced colors are allowed). Catches programs that hallucinate
    colors no demo could justify."""
    if (
        not isinstance(actual, np.ndarray)
        or not isinstance(expected, np.ndarray)
        or not isinstance(demo_input, np.ndarray)
    ):
        return True, "n/a"
    actual_colors = set(np.unique(actual).tolist())
    allowed = set(np.unique(demo_input).tolist()) | set(np.unique(expected).tolist())
    extra = actual_colors - allowed
    if extra:
        return False, f"colors {sorted(extra)} not in input ∪ expected"
    return True, ""


def no_nan_no_negative(actual: Any, expected: Any, demo_input: Any) -> tuple[bool, str]:
    """Grid pixel values must be 0..9. NaN can't occur on int8 but a
    negative cell would indicate corruption."""
    if not isinstance(actual, np.ndarray):
        return True, "n/a"
    if actual.size == 0:
        return True, "n/a (empty)"
    if int(actual.min()) < 0:
        return False, f"negative pixel: min={int(actual.min())}"
    if int(actual.max()) > 9:
        return False, f"out-of-range pixel: max={int(actual.max())}"
    return True, ""


# Default property set used by the Verifier when none is supplied.
DEFAULT_PROPERTIES: tuple[tuple[str, PropertyFn], ...] = (
    ("output_grid_nonempty", output_grid_nonempty),
    ("output_dimensions_match_inputs_or_constant", output_dimensions_match_inputs_or_constant),
    (
        "output_colors_subset_of_input_colors_plus_const",
        output_colors_subset_of_input_colors_plus_const,
    ),
    ("no_nan_no_negative", no_nan_no_negative),
)
