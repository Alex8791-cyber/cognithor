"""Tests for the SQL domain (Sprint-26.2)."""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.channels.program_synthesis.domains.registry import DomainRegistry
from cognithor.channels.program_synthesis.domains.sql import (
    SQL_PRIMITIVE_NAMES,
    SqlDomain,
    SqlVerifierError,
    build_sql_catalog,
    register_sql_domain,
)
from cognithor.channels.program_synthesis.domains.sql.catalog import (
    SqlCatalog,
    SqlPrimitive,
)

duckdb = pytest.importorskip("duckdb")
sqlglot = pytest.importorskip("sqlglot")


class TestSqlCatalog:
    def test_builds(self) -> None:
        cat = build_sql_catalog()
        assert isinstance(cat, SqlCatalog)
        # 30+ primitives — exact count locked via SQL_PRIMITIVE_NAMES.
        assert len(cat) == len(SQL_PRIMITIVE_NAMES)

    def test_all_canonical_names_registered(self) -> None:
        cat = build_sql_catalog()
        for name in SQL_PRIMITIVE_NAMES:
            assert name in cat, f"missing primitive {name!r}"

    def test_select_builds_well_formed_sql(self) -> None:
        cat = build_sql_catalog()
        sql = cat.get("select").fn(("id", "name"), '"orders"')
        assert sql.startswith("SELECT") and "orders" in sql

    def test_eq_predicate(self) -> None:
        cat = build_sql_catalog()
        out = cat.get("eq").fn("id", "1")
        assert out == "(id = 1)"

    def test_limit_rejects_negative(self) -> None:
        cat = build_sql_catalog()
        with pytest.raises(ValueError, match="non-negative"):
            cat.get("limit_").fn("query", -1)

    def test_unknown_primitive_raises(self) -> None:
        cat = build_sql_catalog()
        with pytest.raises(KeyError, match="Unknown SQL primitive"):
            cat.get("not_a_real_primitive")

    def test_double_register_raises(self) -> None:
        cat = SqlCatalog()
        cat.add(SqlPrimitive("p", lambda: "x", 0.1))
        with pytest.raises(ValueError, match="already registered"):
            cat.add(SqlPrimitive("p", lambda: "y", 0.1))

    def test_invalid_primitive_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid SQL primitive name"):
            SqlPrimitive(name="bad-name!", fn=lambda: "x", cost=0.1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            SqlPrimitive(name="p", fn=lambda: "x", cost=-1.0)


class TestSqlDomain:
    def test_metadata(self) -> None:
        d = SqlDomain()
        m = d.metadata
        assert m.name == "sql"
        assert m.benchmark_name == "spider-easy"
        assert m.benchmark_target == 0.30
        assert "Table" in m.type_tags

    def test_register_into_registry(self) -> None:
        reg = DomainRegistry()
        register_sql_domain(reg)
        assert "sql" in reg
        assert isinstance(reg.get("sql"), SqlDomain)

    def test_verify_aggregation(self) -> None:
        d = SqlDomain()
        ok = d.verify(
            "SELECT SUM(amount) FROM orders",
            [
                {
                    "tables": {
                        "orders": {
                            "columns": ["id", "amount"],
                            "rows": [[1, 10], [2, 20], [3, 30]],
                        }
                    },
                    "expected": [(60,)],
                }
            ],
        )
        assert ok

    def test_verify_dict_program_shape(self) -> None:
        d = SqlDomain()
        ok = d.verify(
            {"query": "SELECT COUNT(*) FROM orders"},
            [
                {
                    "tables": {
                        "orders": {
                            "columns": ["id"],
                            "rows": [[1], [2]],
                        }
                    },
                    "expected": [(2,)],
                }
            ],
        )
        assert ok

    def test_verify_mismatch_raises(self) -> None:
        d = SqlDomain()
        with pytest.raises(SqlVerifierError, match="!= expected"):
            d.verify(
                "SELECT COUNT(*) FROM orders",
                [
                    {
                        "tables": {
                            "orders": {
                                "columns": ["id"],
                                "rows": [[1], [2]],
                            }
                        },
                        "expected": [(99,)],
                    }
                ],
            )

    def test_verify_syntax_error_raises(self) -> None:
        d = SqlDomain()
        with pytest.raises(SqlVerifierError, match="parse error"):
            d.verify(
                "SELEKT FROM bla",
                [{"tables": {}, "expected": []}],
            )

    def test_empty_program_rejected(self) -> None:
        d = SqlDomain()
        with pytest.raises(SqlVerifierError, match="empty SQL"):
            d.verify("", [])

    def test_non_str_program_rejected(self) -> None:
        d = SqlDomain()
        with pytest.raises(SqlVerifierError, match="must be"):
            d.verify(42, [])  # type: ignore[arg-type]

    def test_dict_program_non_str_query_rejected(self) -> None:
        d = SqlDomain()
        with pytest.raises(SqlVerifierError, match="must be a string"):
            d.verify({"query": 42}, [])

    def test_verify_group_by(self) -> None:
        d = SqlDomain()
        ok = d.verify(
            "SELECT country, COUNT(*) AS n FROM users GROUP BY country ORDER BY country",
            [
                {
                    "tables": {
                        "users": {
                            "columns": ["id", "country"],
                            "rows": [[1, "DE"], [2, "DE"], [3, "AT"]],
                        }
                    },
                    "expected": [("AT", 1), ("DE", 2)],
                }
            ],
        )
        assert ok


class TestSqlPrimitiveCounts:
    def test_at_least_30_primitives(self) -> None:
        # Owner-Decision D7 floor — Sprint-26.2 commits "~30 primitives".
        assert len(SQL_PRIMITIVE_NAMES) >= 30

    def test_no_duplicate_names(self) -> None:
        names: tuple[str, ...] = SQL_PRIMITIVE_NAMES
        assert len(names) == len(set(names))


class TestSelectionShape:
    def test_select_no_columns_uses_star(self) -> None:
        cat = build_sql_catalog()
        sql = cat.get("select").fn((), '"t"')
        assert "*" in sql

    def test_in_list_empty_returns_false_predicate(self) -> None:
        cat = build_sql_catalog()
        out = cat.get("in_list").fn("x", ())
        assert out == "FALSE"

    def test_count_with_no_arg_uses_star(self) -> None:
        cat = build_sql_catalog()
        out = cat.get("count_").fn()
        assert out == "COUNT(*)"

    def test_count_with_arg(self) -> None:
        cat = build_sql_catalog()
        assert cat.get("count_").fn("id") == "COUNT(id)"


class TestVerifierUnordered:
    def test_unordered_match_after_normalisation(self) -> None:
        d = SqlDomain()
        # Default ordered=True — query with explicit ORDER BY needed.
        ok = d.verify(
            "SELECT id FROM t ORDER BY id",
            [
                {
                    "tables": {"t": {"columns": ["id"], "rows": [[2], [1]]}},
                    "expected": [(1,), (2,)],
                }
            ],
        )
        assert ok


class TestVerifierEmptyTable:
    def test_empty_table_is_creatable(self) -> None:
        d = SqlDomain()
        ok = d.verify(
            'SELECT COUNT(*) AS n FROM "users"',
            [
                {
                    "tables": {"users": {"columns": ["id"], "rows": []}},
                    "expected": [(0,)],
                }
            ],
        )
        assert ok


def _smoke_sample_table() -> dict[str, Any]:
    return {"columns": ["id", "amount"], "rows": [[1, 10.0], [2, 20.0]]}


class TestSqlEndToEnd:
    """Smoke: build a full SELECT-WHERE query from the catalog and verify."""

    def test_full_select_where(self) -> None:
        cat = build_sql_catalog()
        from_clause = cat.get("from_table").fn("orders")
        select_query = cat.get("select").fn(("id", "amount"), from_clause)
        predicate = cat.get("eq").fn('"id"', "1")
        final_query = cat.get("where").fn(select_query, predicate)
        d = SqlDomain()
        assert d.verify(
            final_query,
            [{"tables": {"orders": _smoke_sample_table()}, "expected": [(1, 10.0)]}],
        )
