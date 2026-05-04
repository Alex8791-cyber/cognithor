"""AST domain type-tags."""

from __future__ import annotations

AST_TYPE_TAGS: frozenset[str] = frozenset(
    {
        "Function",
        "FunctionBody",
        "PyExpr",
        "PyStmt",
        "TypeAnnotation",
    }
)
