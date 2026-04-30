# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 graduated pixel-match metric (spec §7.2 + plan task 8).

Phase-1 verifies a candidate program by checking whether its output
*exactly* equals the expected demo output. Phase-2 introduces a
graduated signal: ``partial_pixel_match`` returns the fraction of
cells that match, in ``[0, 1]``. The verifier weighs this with the
``partial_pixel_match`` weight from :class:`VerifierScoreWeights`
(default 0.13) when computing the final score.

Behaviour:

* Both inputs must be 2-D ``numpy.ndarray``. Anything else raises
  :class:`TypeError`.
* Size mismatch returns ``0.0``. The spec's intuition: a program
  that produces the wrong-sized output is no closer to the answer
  than one that produces a totally wrong same-sized output, but the
  caller may want to apply a separate "size correctness" property.
* Empty grids (zero cells) return ``0.0`` rather than NaN — division
  by zero would otherwise pollute the verifier score.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def partial_pixel_match(actual: np.ndarray[Any, Any], expected: np.ndarray[Any, Any]) -> float:
    """Return the fraction of matching cells in ``[0, 1]``.

    * ``1.0`` iff both grids are pixel-identical.
    * ``0.0`` if the shapes differ or either grid is empty.
    * Otherwise: ``num_matching_cells / total_cells``.
    """
    if not isinstance(actual, np.ndarray):
        raise TypeError(
            f"partial_pixel_match: actual must be numpy.ndarray, got {type(actual).__name__}"
        )
    if not isinstance(expected, np.ndarray):
        raise TypeError(
            f"partial_pixel_match: expected must be numpy.ndarray, got {type(expected).__name__}"
        )
    if actual.ndim != 2 or expected.ndim != 2:
        raise TypeError(
            f"partial_pixel_match: both grids must be 2-D; got "
            f"actual.ndim={actual.ndim}, expected.ndim={expected.ndim}"
        )
    if actual.shape != expected.shape:
        return 0.0
    if actual.size == 0:
        return 0.0
    matches = np.count_nonzero(actual == expected)
    return float(matches) / float(actual.size)


def average_partial_pixel_match(
    actual_grids: list[np.ndarray[Any, Any]],
    expected_grids: list[np.ndarray[Any, Any]],
) -> float:
    """Average :func:`partial_pixel_match` across paired grid lists.

    Useful when the verifier checks the candidate program against
    every demo example: the per-demo pixel-match average becomes the
    aggregate signal the score weighter consumes.

    Returns ``0.0`` for an empty input list (the verifier should
    always pass at least one demo, but the function is total).
    Raises :class:`ValueError` if the two lists have different lengths.
    """
    if len(actual_grids) != len(expected_grids):
        raise ValueError(
            f"average_partial_pixel_match: length mismatch — "
            f"{len(actual_grids)} actual vs {len(expected_grids)} expected"
        )
    if not actual_grids:
        return 0.0
    total = sum(
        partial_pixel_match(a, e) for a, e in zip(actual_grids, expected_grids, strict=True)
    )
    return total / len(actual_grids)


__all__ = [
    "average_partial_pixel_match",
    "partial_pixel_match",
]
