# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Domain types for the ARC-DSL: Object, ObjectSet, Mask.

These structures wrap the raw numpy data that the search engine operates
on. They are all immutable so they can be safely stored in observational-
equivalence fingerprints and hashed for cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterator

# A boolean per-pixel mask. Always 2-D, shape matches the source grid.
type Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class Object:
    """A single connected component on a grid.

    ``color`` is the fill value (0..9). ``cells`` is the canonical set of
    (row, col) coordinates the object occupies, sorted lexicographically
    so equality is structural and hashing is deterministic.
    """

    color: int
    cells: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not 0 <= self.color <= 9:
            raise ValueError(f"Object.color {self.color} out of ARC range 0..9")
        # Canonicalise: sort the cells tuple. We do this here rather than
        # at every call site so equality on `Object` is well-defined.
        if list(self.cells) != sorted(self.cells):
            object.__setattr__(self, "cells", tuple(sorted(self.cells)))

    @property
    def size(self) -> int:
        """Number of pixels in this object."""
        return len(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Bounding box as ``(r0, r1, c0, c1)`` — half-open on r1/c1."""
        if not self.cells:
            return (0, 0, 0, 0)
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (min(rs), max(rs) + 1, min(cs), max(cs) + 1)

    def is_rectangle(self) -> bool:
        """True iff every cell inside the bbox is part of the object."""
        if not self.cells:
            return False
        r0, r1, c0, c1 = self.bbox
        expected = (r1 - r0) * (c1 - c0)
        return self.size == expected

    def is_square(self) -> bool:
        if not self.is_rectangle():
            return False
        r0, r1, c0, c1 = self.bbox
        return (r1 - r0) == (c1 - c0)


@dataclass(frozen=True)
class ObjectSet:
    """An ordered, immutable collection of :class:`Object` instances.

    Order matches discovery order from the source primitive (e.g. raster
    scan for connected_components). The order is part of the structural
    identity — search-engine-side observational equivalence relies on it.
    """

    objects: tuple[Object, ...] = ()

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[Object]:
        return iter(self.objects)

    def __getitem__(self, index: int) -> Object:
        return self.objects[index]

    def is_empty(self) -> bool:
        return len(self.objects) == 0
