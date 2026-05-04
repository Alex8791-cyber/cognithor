"""BinaryData domain type-tags."""

from __future__ import annotations

BYTES_TYPE_TAGS: frozenset[str] = frozenset(
    {
        "Bytes",
        "BitField",
        "Endianness",
        "HashDigest",
    }
)
