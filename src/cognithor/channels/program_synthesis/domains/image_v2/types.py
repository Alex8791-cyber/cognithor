"""Image V2 domain type-tags."""

from __future__ import annotations

IMAGE_V2_TYPE_TAGS: frozenset[str] = frozenset(
    {
        "Grid",
        "Color",
        "Mask",
        "Anchor",
        "Object",
        "Bbox",
    }
)
