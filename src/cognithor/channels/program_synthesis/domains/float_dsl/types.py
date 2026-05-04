"""Float-precision domain type-tags."""

from __future__ import annotations

FLOAT_TYPE_TAGS: frozenset[str] = frozenset(
    {
        "Float",
        "FloatList",
        "Epsilon",
    }
)
