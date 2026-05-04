"""SQL verifier — runs synthesised queries against duckdb in-memory.

Sprint-26.2 deliverable. The verifier accepts a synthesised SQL
query string + a list of ``{tables, expected}`` example records.
For each example it:

1. Parses the query with ``sqlglot`` (cheap, catches obvious syntax
   typos).
2. Spins up an in-process duckdb connection, materialises the example's
   tables, runs the query, and compares the result-set against the
   ``expected`` field.
3. Returns ``True`` only if every example matches.

``sqlglot`` and ``duckdb`` are imported lazily so that just importing
this module doesn't require either dependency — they only become
mandatory once a verifier is instantiated. This keeps the Sprint-26.1
foundation tests cheap to run on CI runners that don't have duckdb
installed yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class SqlVerifierError(Exception):
    """Raised when verification fails for a parseable but wrong query.

    Distinct from generic ``Exception`` so callers (the synthesis
    pipeline + audit log) can map it to a structured ``FailureMode``
    without sniffing message strings.
    """


# ---------------------------------------------------------------------------
# Result-set normalisation
# ---------------------------------------------------------------------------


def _normalise_row(row: object) -> tuple[Any, ...]:
    """Convert an arbitrary row representation into a hashable tuple."""
    if isinstance(row, tuple):
        return row
    if isinstance(row, list):
        return tuple(row)
    return (row,)


def _normalise_result(rows: Iterable[Any], *, ordered: bool) -> list[tuple[Any, ...]]:
    """Normalise a result-set for equality comparison.

    Ordered comparison preserves position; unordered uses sorted-tuple
    equality. Sprint-26.2 default: ordered, because Spider's expected
    outputs include ORDER BY semantics. Callers can pass
    ``ordered=False`` for queries where the order is irrelevant
    (aggregations, set ops without explicit ordering).
    """
    materialised = [_normalise_row(r) for r in rows]
    if ordered:
        return materialised
    # Ensure deterministic shape regardless of row content. Use string
    # representation as the secondary sort-key when types are mixed.
    return sorted(materialised, key=lambda r: tuple((repr(c), c) for c in r))


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SqlVerifier:
    """Verify synthesised SQL queries against example tables.

    Instances are cheap; the duckdb connection is created per
    ``verify`` call so independent examples never share state.
    """

    def __init__(self, *, ordered: bool = True) -> None:
        self._ordered = ordered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        query: str,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Return True when ``query`` reproduces every example.

        Each example is a mapping with two keys:

        * ``tables`` — dict of ``{table_name: list[row]}`` where
          ``row`` is a list of column values. The first row may
          optionally be the column-name header by passing
          ``columns`` as a sibling key.
        * ``expected`` — list of expected result rows.

        Returns True iff the query parses and every example's actual
        result matches ``expected``. Raises :class:`SqlVerifierError`
        with a structured explanation on the first mismatch — the
        synthesis pipeline catches and routes this to the audit log.
        """
        # Parse once — even if the query fails on every example,
        # surface the syntax issue early.
        self._parse_query(query)

        for index, example in enumerate(examples):
            actual = self._execute(query, example)
            expected = _normalise_result(
                example.get("expected", []),
                ordered=self._ordered,
            )
            if actual != expected:
                msg = f"Example {index}: query result {actual!r} != expected {expected!r}"
                raise SqlVerifierError(msg)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_query(query: str) -> None:
        try:
            import sqlglot  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "sqlglot is required for SQL verification. Install with `pip install sqlglot`."
            raise SqlVerifierError(msg) from exc
        try:
            sqlglot.parse_one(query, read="duckdb")
        except sqlglot.errors.ParseError as exc:
            raise SqlVerifierError(f"sqlglot parse error: {exc}") from exc

    def _execute(
        self,
        query: str,
        example: Mapping[str, Any],
    ) -> list[tuple[Any, ...]]:
        try:
            import duckdb  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "duckdb is required for SQL verification. Install with `pip install duckdb`."
            raise SqlVerifierError(msg) from exc

        conn = duckdb.connect(":memory:")
        try:
            tables = example.get("tables", {})
            for name, payload in tables.items():
                self._materialise_table(conn, name, payload)
            try:
                rows = conn.execute(query).fetchall()
            except Exception as exc:
                msg = f"duckdb execute error on example: {exc}"
                raise SqlVerifierError(msg) from exc
        finally:
            conn.close()
        return _normalise_result(rows, ordered=self._ordered)

    @staticmethod
    def _materialise_table(conn: Any, name: str, payload: Any) -> None:
        """Insert example rows into duckdb under ``name``.

        Two payload shapes are accepted:

        * ``{"columns": [...], "rows": [[...], ...]}`` — explicit schema
        * ``[[...], [...]]`` — rows only, columns auto-named c0..cN

        Either way duckdb infers types from the first row, which is
        sufficient for Sprint-26.2 Spider-easy coverage.
        """
        columns: list[str]
        rows: list[list[Any]]
        if isinstance(payload, Mapping):
            columns = list(payload.get("columns", []))
            rows = list(payload.get("rows", []))
        else:
            rows = list(payload)
            columns = [f"c{i}" for i in range(len(rows[0]))] if rows else []
        if not columns:
            return
        # Use a CREATE TABLE-from-VALUES idiom to let duckdb infer
        # types. For empty tables we fall back to TEXT columns so the
        # downstream query still binds.
        if not rows:
            placeholders = ", ".join(f'"{c}" TEXT' for c in columns)
            conn.execute(f'CREATE TABLE "{name}" ({placeholders})')
            return
        # duckdb supports tuples directly via .executemany.
        col_list = ", ".join(f'"{c}"' for c in columns)
        placeholder_row = "(" + ", ".join("?" for _ in columns) + ")"
        # Build CREATE TABLE from inferred types via a one-row INSERT.
        conn.execute(
            f'CREATE TABLE "{name}" AS SELECT * FROM (VALUES {placeholder_row}) AS t({col_list})',
            rows[0],
        )
        # Wipe the inferred row, then re-insert all rows.
        conn.execute(f'DELETE FROM "{name}"')
        for row in rows:
            conn.execute(
                f'INSERT INTO "{name}" ({col_list}) VALUES {placeholder_row}',
                row,
            )
