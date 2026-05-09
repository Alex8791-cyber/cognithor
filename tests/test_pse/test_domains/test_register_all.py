"""Tests for ``register_all_sprint26_domains`` and the idempotent
``register_missing_sprint26_domains`` companion."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.domains import (
    SPRINT26_DOMAIN_NAMES,
    DomainRegistry,
    register_all_sprint26_domains,
    register_missing_sprint26_domains,
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


class TestRegisterMissingSprint26Domains:
    """Idempotent variant for gateway-boot wiring (set-based, never raises)."""

    def test_fresh_registry_registers_all_seven(self) -> None:
        reg = DomainRegistry()
        registered = register_missing_sprint26_domains(reg)
        assert registered == list(SPRINT26_DOMAIN_NAMES)
        assert len(reg) == 7

    def test_owner_d3_order_preserved(self) -> None:
        """Returned list of registered names must match Owner-D3 priority."""
        reg = DomainRegistry()
        registered = register_missing_sprint26_domains(reg)
        # Stable ordering — ScoreCard column order depends on this
        assert registered == [
            "sql",
            "json",
            "datetime",
            "ast",
            "bytes",
            "float",
            "image_v2",
        ]

    def test_idempotent_on_fully_populated_registry(self) -> None:
        """Calling twice on a complete registry must return [] and not raise."""
        reg = DomainRegistry()
        first = register_missing_sprint26_domains(reg)
        assert len(first) == 7
        second = register_missing_sprint26_domains(reg)
        assert second == [], (
            f"second call on fully-populated registry must register nothing — got {second}"
        )
        assert len(reg) == 7  # no duplicates / overrides

    def test_partial_registry_fills_only_gaps(self) -> None:
        """Pre-populate the registry with 2 of 7; helper fills the missing 5."""
        reg = DomainRegistry()
        # Use the bare register-functions to seed a partial state without
        # going through register_all (which would fill all 7).
        from cognithor.channels.program_synthesis.domains.json_dsl import (
            register_json_domain,
        )
        from cognithor.channels.program_synthesis.domains.sql import (
            register_sql_domain,
        )

        register_sql_domain(reg)
        register_json_domain(reg)
        assert len(reg) == 2

        registered = register_missing_sprint26_domains(reg)
        assert set(registered) == set(SPRINT26_DOMAIN_NAMES) - {"sql", "json"}
        assert len(reg) == 7
        # Owner-D3 order preserved among the gaps
        assert registered == ["datetime", "ast", "bytes", "float", "image_v2"]

    def test_does_not_raise_on_dupe(self) -> None:
        """Unlike ``register_all_sprint26_domains`` (which raises), this
        helper is the safe choice for boot paths that may run twice."""
        reg = DomainRegistry()
        register_all_sprint26_domains(reg)
        # The non-idempotent variant would raise DomainAlreadyRegisteredError
        with pytest.raises(DomainAlreadyRegisteredError):
            register_all_sprint26_domains(reg)
        # The idempotent variant must NOT raise
        register_missing_sprint26_domains(reg)
        register_missing_sprint26_domains(reg)
        register_missing_sprint26_domains(reg)
        assert len(reg) == 7

    def test_foreign_domains_in_registry_are_ignored(self) -> None:
        """A registry that already has non-Sprint-26 domains shouldn't
        confuse the helper — it only manages the Sprint-26 catalog."""
        reg = DomainRegistry()
        # Manually inject a fake domain name not in SPRINT26_DOMAIN_NAMES
        from cognithor.channels.program_synthesis.domains.base import (
            DomainCapability,
            DomainMetadata,
        )

        fake_meta = DomainMetadata(
            name="fake_external_domain",
            display_name="Fake",
            description="Test fixture",
            capabilities=frozenset({DomainCapability.PROPERTY}),
            type_tags=frozenset({"int"}),
            benchmark_name="fake_bench",
            benchmark_target=0.5,
        )
        reg.register(fake_meta, lambda: None)  # type: ignore[arg-type, return-value]
        assert "fake_external_domain" in reg

        registered = register_missing_sprint26_domains(reg)
        # All 7 sprint26 domains registered, foreign one untouched
        assert set(registered) == set(SPRINT26_DOMAIN_NAMES)
        assert "fake_external_domain" in reg  # still present
        assert len(reg) == 8  # 7 sprint26 + 1 foreign

    def test_return_type_is_list_for_immutability_in_logging(self) -> None:
        """Audit logs receive the returned names directly. A list (not
        a generator or set) is required so log handlers see a stable,
        repeatable, JSON-serialisable value."""
        reg = DomainRegistry()
        result = register_missing_sprint26_domains(reg)
        assert isinstance(result, list)
        # Re-iterable
        assert list(result) == list(result)
