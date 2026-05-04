"""SQL type-tags + small enums (Sprint-26.2)."""

from __future__ import annotations

from enum import StrEnum

# Canonical type-tag strings the SQL domain introduces. Bridge
# whitelist (Owner-Decision D5) cares about these values: a
# ``datetime → sql_literal`` bridge converts a tagged Datetime into a
# tagged SqlLiteral so the synthesizer can compose cross-domain.
SQL_TYPE_TAGS: frozenset[str] = frozenset(
    {
        "Table",
        "Column",
        "SqlExpr",
        "SqlLiteral",
        "Predicate",
        "WindowSpec",
        "Aggregate",
        "JoinType",
        "OrderBy",
        "CTE",
    }
)


class JoinType(StrEnum):
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"


class SortDirection(StrEnum):
    ASC = "ASC"
    DESC = "DESC"
