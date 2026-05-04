"""Datetime domain type-tags (Sprint-26.3)."""

from __future__ import annotations

DATETIME_TYPE_TAGS: frozenset[str] = frozenset(
    {
        "Datetime",
        "Date",
        "Time",
        "Duration",
        "Timezone",
        "Weekday",
    }
)
