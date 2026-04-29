# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Type signatures for DSL primitives (spec §7.4).

Phase 1 uses string type tags rather than Python types to keep signatures
JSON-serialisable for the catalog and for cache keys. Allowed type tags:

    Grid, Color, Mask, Object, ObjectSet, Int, Bool, Predicate, Lambda,
    AlignMode, SortKey
"""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_TYPES: frozenset[str] = frozenset(
    {
        "Grid",
        "Color",
        "Mask",
        "Object",
        "ObjectSet",
        "Int",
        "Bool",
        "Predicate",
        "Lambda",
        "AlignMode",
        "SortKey",
    }
)


@dataclass(frozen=True)
class Signature:
    """Static input/output type signature of a primitive.

    Phase 1 explicitly forbids generics and parametric types. ``inputs``
    and ``output`` are tuples / strings of type tags from
    :data:`ALLOWED_TYPES`.
    """

    inputs: tuple[str, ...]
    output: str

    def __post_init__(self) -> None:
        for t in (*self.inputs, self.output):
            if t not in ALLOWED_TYPES:
                raise ValueError(f"Unknown type tag {t!r}; allowed: {sorted(ALLOWED_TYPES)}")

    def matches(self, args: tuple[str, ...]) -> bool:
        """Return True iff ``args`` satisfies the input signature."""
        return self.inputs == args

    @property
    def arity(self) -> int:
        return len(self.inputs)
