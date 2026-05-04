"""JSON domain type-tags (Sprint-26.2)."""

from __future__ import annotations

JSON_TYPE_TAGS: frozenset[str] = frozenset(
    {
        "JsonValue",
        "JsonObject",
        "JsonArray",
        "JsonPath",
        "JsonPredicate",
    }
)
