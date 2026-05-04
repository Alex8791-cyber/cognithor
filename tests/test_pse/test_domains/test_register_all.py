"""Tests for ``register_all_sprint26_domains``."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.domains import (
    SPRINT26_DOMAIN_NAMES,
    DomainRegistry,
    register_all_sprint26_domains,
)
from cognithor.channels.program_synthesis.domains.registry import (
    DomainAlreadyRegisteredError,
)


class TestRegisterAllSprint26Domains:
    def test_registers_all_seven(self) -> None:
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        assert len(reg) == 7
        for name in SPRINT26_DOMAIN_NAMES:
            assert name in reg, f"missing domain {name!r}"

    def test_owner_d3_priority_order(self) -> None:
        # Owner-Decision D3: SQL → JSON → Datetime → AST → BinaryData
        # → Float → Image-Boost. Stable ordering matters because the
        # public scorecard renders columns in this sequence.
        assert SPRINT26_DOMAIN_NAMES == (
            "sql",
            "json",
            "datetime",
            "ast",
            "bytes",
            "float",
            "image_v2",
        )

    def test_metadata_per_domain(self) -> None:
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        for name in SPRINT26_DOMAIN_NAMES:
            metadata = reg.metadata(name)
            assert metadata.name == name
            assert metadata.display_name
            assert metadata.benchmark_target >= 0.0

    def test_double_call_raises(self) -> None:
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        with pytest.raises(DomainAlreadyRegisteredError):
            register_all_sprint26_domains(reg)

    def test_lazy_factories(self) -> None:
        # Registration is cheap — domain instances aren't built until
        # the synthesis pipeline asks for them via reg.get(name).
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        # Materialise one to confirm the factory wires correctly.
        sql = reg.get("sql")
        assert sql.metadata.name == "sql"

    def test_capabilities_set_per_domain(self) -> None:
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        for name in SPRINT26_DOMAIN_NAMES:
            metadata = reg.metadata(name)
            assert metadata.capabilities, f"domain {name!r} declared no capabilities"

    def test_distinct_benchmark_names(self) -> None:
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        benchmarks = [reg.metadata(n).benchmark_name for n in SPRINT26_DOMAIN_NAMES]
        # Each domain must point at its own external benchmark — Sprint-26
        # public-scorecard contract.
        assert len(set(benchmarks)) == len(benchmarks)

    def test_partial_filter_by_capability(self) -> None:
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        execute_capable = list(reg.filter_by_capability("execute"))
        # SQL + AST run programs (D5 capabilities); the others verify
        # via interpreter or property-suite.
        names = {m.name for m in execute_capable}
        assert "sql" in names
        assert "ast" in names

    def test_each_domain_has_at_least_one_type_tag(self) -> None:
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        for name in SPRINT26_DOMAIN_NAMES:
            metadata = reg.metadata(name)
            assert metadata.type_tags, f"{name!r} has no type tags"
