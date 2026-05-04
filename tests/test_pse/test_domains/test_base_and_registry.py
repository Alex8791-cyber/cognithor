"""Tests for ``cognithor.channels.program_synthesis.domains.base|registry``.

Sprint-26.1 Foundation tests — DomainMetadata invariants, registry
register/get/lazy-instantiation, capability filtering.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cognithor.channels.program_synthesis.domains.base import (
    Domain,
    DomainCapability,
    DomainMetadata,
    domain_relative_path,
)
from cognithor.channels.program_synthesis.domains.registry import (
    DomainAlreadyRegisteredError,
    DomainRegistry,
    UnknownDomainError,
)


def _make_metadata(
    name: str = "sql",
    *,
    benchmark_target: float = 0.3,
    capabilities: frozenset[DomainCapability] | None = None,
) -> DomainMetadata:
    return DomainMetadata(
        name=name,
        display_name=name.upper(),
        description=f"{name} synthesis domain",
        capabilities=capabilities
        or frozenset({DomainCapability.SYNTHESISE, DomainCapability.EXECUTE}),
        type_tags=frozenset({"Table", "Column"}),
        benchmark_name="spider-easy",
        benchmark_target=benchmark_target,
    )


class _FakeDomain:
    def __init__(self, metadata: DomainMetadata) -> None:
        self._metadata = metadata
        self._primitives = MagicMock()

    @property
    def metadata(self) -> DomainMetadata:
        return self._metadata

    def primitives(self) -> Any:
        return self._primitives

    def verify(self, program: Any, examples: Any) -> bool:
        return True


class TestDomainMetadata:
    def test_valid_metadata(self) -> None:
        m = _make_metadata()
        assert m.name == "sql"
        assert m.benchmark_target == 0.3
        assert DomainCapability.SYNTHESISE in m.capabilities

    def test_rejects_uppercase_name(self) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            DomainMetadata(
                name="SQL",
                display_name="SQL",
                description="x",
                capabilities=frozenset(),
            )

    def test_rejects_invalid_chars(self) -> None:
        with pytest.raises(ValueError, match="Invalid domain name"):
            DomainMetadata(
                name="my-domain!",
                display_name="x",
                description="x",
                capabilities=frozenset(),
            )

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid domain name"):
            DomainMetadata(
                name="",
                display_name="x",
                description="x",
                capabilities=frozenset(),
            )

    def test_rejects_target_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="benchmark_target"):
            DomainMetadata(
                name="x",
                display_name="x",
                description="x",
                capabilities=frozenset(),
                benchmark_target=1.5,
            )

    def test_metadata_is_hashable(self) -> None:
        a = _make_metadata("a")
        b = _make_metadata("a")
        assert hash(a) == hash(b)

    def test_domain_relative_path(self) -> None:
        m = _make_metadata("datetime")
        assert (
            str(domain_relative_path(m, "system.md"))
            .replace("\\", "/")
            .endswith("prompts/pse/datetime/system.md")
        )

    def test_protocol_runtime_check(self) -> None:
        m = _make_metadata("sql")
        assert isinstance(_FakeDomain(m), Domain)


class TestDomainRegistry:
    def test_register_and_get(self) -> None:
        reg = DomainRegistry()
        m = _make_metadata("sql")
        reg.register(m, lambda: _FakeDomain(m))

        assert "sql" in reg
        assert reg.metadata("sql") is m
        domain = reg.get("sql")
        assert isinstance(domain, _FakeDomain)
        assert reg.get("sql") is domain  # cached

    def test_register_twice_raises(self) -> None:
        reg = DomainRegistry()
        m = _make_metadata("sql")
        reg.register(m, lambda: _FakeDomain(m))
        with pytest.raises(DomainAlreadyRegisteredError):
            reg.register(m, lambda: _FakeDomain(m))

    def test_get_unknown_raises(self) -> None:
        reg = DomainRegistry()
        with pytest.raises(UnknownDomainError, match="not registered"):
            reg.get("nope")

    def test_metadata_unknown_raises(self) -> None:
        reg = DomainRegistry()
        with pytest.raises(UnknownDomainError):
            reg.metadata("nope")

    def test_factory_called_lazily(self) -> None:
        reg = DomainRegistry()
        m = _make_metadata("sql")
        calls: list[int] = []

        def factory() -> _FakeDomain:
            calls.append(1)
            return _FakeDomain(m)

        reg.register(m, factory)
        assert not calls  # registration is cheap
        reg.get("sql")
        reg.get("sql")
        assert len(calls) == 1  # only first call materialised the domain

    def test_names_sorted(self) -> None:
        reg = DomainRegistry()
        for n in ("sql", "ast", "json"):
            m = _make_metadata(n)
            reg.register(m, lambda m=m: _FakeDomain(m))
        assert reg.names() == ["ast", "json", "sql"]
        assert len(reg) == 3

    def test_filter_by_capability(self) -> None:
        reg = DomainRegistry()
        sql = _make_metadata(
            "sql", capabilities=frozenset({DomainCapability.SYNTHESISE, DomainCapability.EXECUTE})
        )
        json_meta = _make_metadata("json", capabilities=frozenset({DomainCapability.SYNTHESISE}))
        reg.register(sql, lambda: _FakeDomain(sql))
        reg.register(json_meta, lambda: _FakeDomain(json_meta))

        synth_only = list(reg.filter_by_capability("synthesise"))
        assert len(synth_only) == 2

        exec_only = list(reg.filter_by_capability("execute"))
        assert len(exec_only) == 1
        assert exec_only[0].name == "sql"

        both = list(reg.filter_by_capability("synthesise", "execute"))
        assert len(both) == 1

    def test_list_metadata_returns_sorted_blocks(self) -> None:
        reg = DomainRegistry()
        for n in ("z_late", "a_early"):
            m = _make_metadata(n)
            reg.register(m, lambda m=m: _FakeDomain(m))
        names = [m.name for m in reg.list_metadata()]
        assert names == ["a_early", "z_late"]
