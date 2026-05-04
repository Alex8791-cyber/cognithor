"""SQL synthesis domain (Sprint-26.2).

The SQL domain wires the Sprint-26.1 ``Domain`` protocol to a duckdb-
backed verifier and a 30-primitive catalog spanning SELECT, WHERE,
JOIN, GROUP BY, window functions, CTEs, set operations, and the
common date/string scalar functions Spider-easy queries need.

The verifier rejects programs whose result-set differs from the
example output. ``sqlglot`` is used for parse + format normalisation
so two semantically-identical queries don't fail the equality check
because of trivial whitespace.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.sql.catalog import (
    SQL_PRIMITIVE_NAMES,
    build_sql_catalog,
)
from cognithor.channels.program_synthesis.domains.sql.domain import (
    SqlDomain,
    register_sql_domain,
)
from cognithor.channels.program_synthesis.domains.sql.types import (
    SQL_TYPE_TAGS,
    JoinType,
    SortDirection,
)
from cognithor.channels.program_synthesis.domains.sql.verifier import (
    SqlVerifier,
    SqlVerifierError,
)

__all__ = [
    "SQL_PRIMITIVE_NAMES",
    "SQL_TYPE_TAGS",
    "JoinType",
    "SortDirection",
    "SqlDomain",
    "SqlVerifier",
    "SqlVerifierError",
    "build_sql_catalog",
    "register_sql_domain",
]
