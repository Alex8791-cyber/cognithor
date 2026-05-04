"""``SqlDomain`` — wires SQL catalog + verifier into the Sprint-26.1
``DomainRegistry`` (Owner-Decision D3 — SQL is first-priority Sprint-
26.2 because Spider is a recognised public benchmark)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.sql.catalog import (
    SqlCatalog,
    build_sql_catalog,
)
from cognithor.channels.program_synthesis.domains.sql.types import SQL_TYPE_TAGS
from cognithor.channels.program_synthesis.domains.sql.verifier import (
    SqlVerifier,
    SqlVerifierError,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


SQL_DOMAIN_METADATA = DomainMetadata(
    name="sql",
    display_name="SQL",
    description=(
        "Synthesises ANSI/duckdb-flavour SQL queries from "
        "(input_table, output_value) examples. "
        "Verifier runs the query in-memory and compares result-sets."
    ),
    capabilities=frozenset(
        {
            DomainCapability.SYNTHESISE,
            DomainCapability.EXECUTE,
            DomainCapability.PROPERTY,
            DomainCapability.BRIDGE,
        }
    ),
    type_tags=SQL_TYPE_TAGS,
    benchmark_name="spider-easy",
    benchmark_target=0.30,
    few_shot_bank_path="prompts/pse/sql/examples.jsonl",
)


class SqlDomain:
    """:class:`Domain` implementation for SQL synthesis."""

    def __init__(self) -> None:
        self._catalog: SqlCatalog = build_sql_catalog()
        self._verifier = SqlVerifier(ordered=True)

    @property
    def metadata(self) -> DomainMetadata:
        return SQL_DOMAIN_METADATA

    def primitives(self) -> SqlCatalog:
        return self._catalog

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Run the synthesised query against every example.

        Accepts either a raw query string or a dict
        ``{"query": "..."}`` (the shape the LLM-prior emits).
        """
        query = self._coerce_query(program)
        if not query:
            msg = "empty SQL query"
            raise SqlVerifierError(msg)
        return self._verifier.verify(query, examples)

    @staticmethod
    def _coerce_query(program: Any) -> str:
        if isinstance(program, str):
            return program.strip()
        if isinstance(program, Mapping):
            value = program.get("query", "")
            if not isinstance(value, str):
                msg = f"SQL program 'query' field must be a string, got {type(value).__name__}"
                raise SqlVerifierError(msg)
            return value.strip()
        msg = f"SQL program must be str or {{'query': str}}, got {type(program).__name__}"
        raise SqlVerifierError(msg)


def register_sql_domain(registry: DomainRegistry) -> None:
    """Register :class:`SqlDomain` with ``registry``.

    Sprint-26 sub-PRs invoke this from a top-level import so the
    canonical ``DOMAIN_REGISTRY`` is populated once per process.
    """
    registry.register(SQL_DOMAIN_METADATA, lambda: SqlDomain())
