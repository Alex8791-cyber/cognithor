"""SQL primitive catalog (Sprint-26.2).

The SQL catalog is a flat list of string-builder primitives. Each
primitive accepts strings/tuples/ints and returns a SQL fragment;
composition happens at synthesis time. This is intentionally simpler
than the grid-DSL ``PrimitiveRegistry`` from
``cognithor.channels.program_synthesis.dsl``: SQL doesn't need the
strict type-tag system because the verifier runs the final query
through ``sqlglot`` + ``duckdb`` which catch shape errors at execute-
time.

The catalog exposes a stable ``SQL_PRIMITIVE_NAMES`` tuple so the
verifier, the cross-domain bridge, and the public scorecard can
reference primitives by canonical name without re-reading this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SqlPrimitive:
    """One SQL string-builder primitive.

    Frozen so the registry can hash entries for de-duplication; ``fn``
    is a stateless callable that returns a SQL fragment.
    """

    name: str
    fn: Callable[..., str]
    cost: float
    description: str = ""
    arity_min: int = 0
    arity_max: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            msg = f"Invalid SQL primitive name: {self.name!r}"
            raise ValueError(msg)
        if self.cost < 0:
            msg = f"SQL primitive cost must be >= 0, got {self.cost}"
            raise ValueError(msg)


class SqlCatalog:
    """Append-only catalog of :class:`SqlPrimitive` entries."""

    def __init__(self) -> None:
        self._entries: dict[str, SqlPrimitive] = {}

    def add(self, primitive: SqlPrimitive) -> None:
        if primitive.name in self._entries:
            msg = f"SQL primitive {primitive.name!r} already registered"
            raise ValueError(msg)
        self._entries[primitive.name] = primitive

    def get(self, name: str) -> SqlPrimitive:
        if name not in self._entries:
            msg = f"Unknown SQL primitive {name!r}"
            raise KeyError(msg)
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


# ---------------------------------------------------------------------------
# Primitive name list
# ---------------------------------------------------------------------------

SQL_PRIMITIVE_NAMES: tuple[str, ...] = (
    # SELECT + FROM
    "select",
    "select_distinct",
    "from_table",
    "from_subquery",
    # Predicates
    "eq",
    "neq",
    "lt",
    "gt",
    "between",
    "in_list",
    "like",
    "is_null",
    "and_",
    "or_",
    "not_",
    "where",
    # Joins
    "join",
    "left_join",
    "right_join",
    # Aggregations
    "sum_",
    "count_",
    "count_distinct",
    "avg_",
    "min_",
    "max_",
    "group_by",
    "having",
    # Window
    "row_number",
    "rank_",
    "dense_rank",
    "lag_",
    "lead_",
    "partition_by",
    # Order / limit
    "order_by",
    "limit_",
    # Set ops
    "union_",
    "intersect_",
    "except_",
    # CTE
    "with_cte",
    # Date / string scalars
    "date_trunc",
    "extract_part",
    "now_",
    "concat_",
    "lower_",
    "upper_",
    "substring_",
    "regexp_match",
)


# ---------------------------------------------------------------------------
# String-builder helpers
# ---------------------------------------------------------------------------


def _quote_ident(name: str) -> str:
    """Escape an identifier for duckdb (double-quoted)."""
    return '"' + name.replace('"', '""') + '"'


def _join_idents(cols: tuple[str, ...]) -> str:
    return ", ".join(_quote_ident(c) for c in cols)


def _ensure_str(value: Any, kind: str) -> str:
    if not isinstance(value, str):
        msg = f"{kind} must be a string, got {type(value).__name__}"
        raise TypeError(msg)
    return value


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------


def _select(columns: tuple[str, ...], source: str) -> str:
    cols = _join_idents(columns) if columns else "*"
    return f"SELECT {cols} FROM {source}"


def _select_distinct(columns: tuple[str, ...], source: str) -> str:
    cols = _join_idents(columns) if columns else "*"
    return f"SELECT DISTINCT {cols} FROM {source}"


def _from_table(name: str) -> str:
    return _quote_ident(_ensure_str(name, "table"))


def _from_subquery(query: str, alias: str) -> str:
    return f"({query}) AS {_quote_ident(alias)}"


def _eq(left: str, right: str) -> str:
    return f"({left} = {right})"


def _neq(left: str, right: str) -> str:
    return f"({left} <> {right})"


def _lt(left: str, right: str) -> str:
    return f"({left} < {right})"


def _gt(left: str, right: str) -> str:
    return f"({left} > {right})"


def _between(expr: str, lo: str, hi: str) -> str:
    return f"({expr} BETWEEN {lo} AND {hi})"


def _in_list(expr: str, values: tuple[str, ...]) -> str:
    if not values:
        return "FALSE"
    return f"({expr} IN ({', '.join(values)}))"


def _like(expr: str, pattern: str) -> str:
    return f"({expr} LIKE {pattern})"


def _is_null(expr: str) -> str:
    return f"({expr} IS NULL)"


def _and(left: str, right: str) -> str:
    return f"({left} AND {right})"


def _or(left: str, right: str) -> str:
    return f"({left} OR {right})"


def _not(expr: str) -> str:
    return f"(NOT {expr})"


def _where(query: str, predicate: str) -> str:
    return f"{query} WHERE {predicate}"


def _join(left: str, right: str, predicate: str) -> str:
    return f"{left} INNER JOIN {right} ON {predicate}"


def _left_join(left: str, right: str, predicate: str) -> str:
    return f"{left} LEFT JOIN {right} ON {predicate}"


def _right_join(left: str, right: str, predicate: str) -> str:
    return f"{left} RIGHT JOIN {right} ON {predicate}"


def _sum(expr: str) -> str:
    return f"SUM({expr})"


def _count(expr: str = "") -> str:
    return f"COUNT({expr})" if expr else "COUNT(*)"


def _count_distinct(expr: str) -> str:
    return f"COUNT(DISTINCT {expr})"


def _avg(expr: str) -> str:
    return f"AVG({expr})"


def _min(expr: str) -> str:
    return f"MIN({expr})"


def _max(expr: str) -> str:
    return f"MAX({expr})"


def _group_by(query: str, columns: tuple[str, ...]) -> str:
    return f"{query} GROUP BY {_join_idents(columns)}"


def _having(query: str, predicate: str) -> str:
    return f"{query} HAVING {predicate}"


def _row_number() -> str:
    return "ROW_NUMBER()"


def _rank() -> str:
    return "RANK()"


def _dense_rank() -> str:
    return "DENSE_RANK()"


def _lag(column: str, offset: int = 1) -> str:
    return f"LAG({_quote_ident(column)}, {offset})"


def _lead(column: str, offset: int = 1) -> str:
    return f"LEAD({_quote_ident(column)}, {offset})"


def _partition_by(window_fn: str, columns: tuple[str, ...]) -> str:
    return f"{window_fn} OVER (PARTITION BY {_join_idents(columns)})"


def _order_by(query: str, columns: tuple[str, ...], descending: bool = False) -> str:
    cols = _join_idents(columns)
    direction = "DESC" if descending else "ASC"
    return f"{query} ORDER BY {cols} {direction}"


def _limit(query: str, n: int) -> str:
    if n < 0:
        msg = "LIMIT must be non-negative"
        raise ValueError(msg)
    return f"{query} LIMIT {n}"


def _union(left: str, right: str) -> str:
    return f"({left}) UNION ({right})"


def _intersect(left: str, right: str) -> str:
    return f"({left}) INTERSECT ({right})"


def _except_(left: str, right: str) -> str:
    return f"({left}) EXCEPT ({right})"


def _with_cte(name: str, body: str, query: str) -> str:
    return f"WITH {_quote_ident(name)} AS ({body}) {query}"


def _date_trunc(unit: str, expr: str) -> str:
    return f"DATE_TRUNC('{unit}', {expr})"


def _extract_part(part: str, expr: str) -> str:
    return f"EXTRACT({part} FROM {expr})"


def _now() -> str:
    return "CURRENT_TIMESTAMP"


def _concat(left: str, right: str) -> str:
    return f"({left} || {right})"


def _lower(expr: str) -> str:
    return f"LOWER({expr})"


def _upper(expr: str) -> str:
    return f"UPPER({expr})"


def _substring(expr: str, start: int, length: int) -> str:
    return f"SUBSTRING({expr}, {start}, {length})"


def _regexp_match(expr: str, pattern: str) -> str:
    return f"REGEXP_MATCHES({expr}, {pattern})"


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_sql_catalog() -> SqlCatalog:
    """Return a fresh :class:`SqlCatalog` with all 30+ primitives.

    Cost values follow the Sprint-22 Occam-prior convention: cheap
    leaf primitives 0.1-0.3, medium combinators 0.4-0.6, heavy
    constructs (window functions, CTEs) 0.8-1.2.
    """
    cat = SqlCatalog()

    def add(
        name: str,
        fn: Callable[..., str],
        cost: float,
        description: str = "",
        arity_min: int = 0,
        arity_max: int | None = None,
    ) -> None:
        cat.add(
            SqlPrimitive(
                name=name,
                fn=fn,
                cost=cost,
                description=description,
                arity_min=arity_min,
                arity_max=arity_max,
            )
        )

    # SELECT
    add("select", _select, 0.4, "SELECT cols FROM source", 2, 2)
    add(
        "select_distinct",
        _select_distinct,
        0.5,
        "SELECT DISTINCT cols FROM source",
        2,
        2,
    )
    add("from_table", _from_table, 0.1, "Quoted table identifier", 1, 1)
    add(
        "from_subquery",
        _from_subquery,
        0.6,
        "(query) AS alias",
        2,
        2,
    )

    # Predicates
    for name, fn in (
        ("eq", _eq),
        ("neq", _neq),
        ("lt", _lt),
        ("gt", _gt),
    ):
        add(name, fn, 0.2, "binary comparison", 2, 2)
    add("between", _between, 0.3, "BETWEEN lo AND hi", 3, 3)
    add("in_list", _in_list, 0.3, "expr IN (literals)", 2, 2)
    add("like", _like, 0.3, "expr LIKE pattern", 2, 2)
    add("is_null", _is_null, 0.2, "expr IS NULL", 1, 1)
    add("and_", _and, 0.2, "AND combinator", 2, 2)
    add("or_", _or, 0.2, "OR combinator", 2, 2)
    add("not_", _not, 0.2, "NOT combinator", 1, 1)
    add("where", _where, 0.4, "query WHERE predicate", 2, 2)

    # Joins
    add("join", _join, 0.7, "INNER JOIN", 3, 3)
    add("left_join", _left_join, 0.7, "LEFT JOIN", 3, 3)
    add("right_join", _right_join, 0.7, "RIGHT JOIN", 3, 3)

    # Aggregations
    add("sum_", _sum, 0.4, "SUM(expr)", 1, 1)
    add("count_", _count, 0.4, "COUNT(expr) or COUNT(*)", 0, 1)
    add("count_distinct", _count_distinct, 0.4, "COUNT(DISTINCT expr)", 1, 1)
    add("avg_", _avg, 0.4, "AVG(expr)", 1, 1)
    add("min_", _min, 0.4, "MIN(expr)", 1, 1)
    add("max_", _max, 0.4, "MAX(expr)", 1, 1)
    add("group_by", _group_by, 0.5, "query GROUP BY cols", 2, 2)
    add("having", _having, 0.5, "query HAVING predicate", 2, 2)

    # Window
    add("row_number", _row_number, 0.6, "ROW_NUMBER()", 0, 0)
    add("rank_", _rank, 0.6, "RANK()", 0, 0)
    add("dense_rank", _dense_rank, 0.6, "DENSE_RANK()", 0, 0)
    add("lag_", _lag, 0.7, "LAG(col, offset)", 1, 2)
    add("lead_", _lead, 0.7, "LEAD(col, offset)", 1, 2)
    add(
        "partition_by",
        _partition_by,
        0.8,
        "window OVER (PARTITION BY cols)",
        2,
        2,
    )

    # Order / limit
    add("order_by", _order_by, 0.4, "query ORDER BY cols (asc/desc)", 2, 3)
    add("limit_", _limit, 0.3, "query LIMIT n", 2, 2)

    # Set ops
    add("union_", _union, 0.6, "UNION", 2, 2)
    add("intersect_", _intersect, 0.6, "INTERSECT", 2, 2)
    add("except_", _except_, 0.6, "EXCEPT", 2, 2)

    # CTE
    add("with_cte", _with_cte, 1.0, "WITH name AS (body) query", 3, 3)

    # Scalars
    add("date_trunc", _date_trunc, 0.4, "DATE_TRUNC(unit, expr)", 2, 2)
    add("extract_part", _extract_part, 0.4, "EXTRACT(part FROM expr)", 2, 2)
    add("now_", _now, 0.2, "CURRENT_TIMESTAMP", 0, 0)
    add("concat_", _concat, 0.3, "expr || expr", 2, 2)
    add("lower_", _lower, 0.2, "LOWER(expr)", 1, 1)
    add("upper_", _upper, 0.2, "UPPER(expr)", 1, 1)
    add("substring_", _substring, 0.4, "SUBSTRING(expr, start, length)", 3, 3)
    add("regexp_match", _regexp_match, 0.5, "REGEXP_MATCHES(expr, pattern)", 2, 2)

    return cat
