# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Capability-token registration tests (spec §13)."""

from __future__ import annotations

from cognithor.channels.program_synthesis.integration.capability_tokens import (
    CapabilityRegistration,
    PSECapability,
    planned_registrations,
)


class TestPSECapability:
    def test_seven_capabilities_defined(self) -> None:
        # Spec §13 lists exactly seven capabilities.
        assert len(list(PSECapability)) == 7

    def test_capability_string_form(self) -> None:
        # All must use the "pse:" prefix and be lowercase with colons.
        for cap in PSECapability:
            assert cap.value.startswith("pse:")
            assert cap.value == cap.value.lower()

    def test_str_subclass_lets_value_pass_to_string_apis(self) -> None:
        # PSECapability inherits from str.
        cap = PSECapability.SYNTHESIZE
        assert isinstance(cap, str)
        assert cap.startswith("pse:")
        assert cap == "pse:synthesize"

    def test_specific_capability_values(self) -> None:
        # Lock the exact spelling against the spec table.
        assert PSECapability.SYNTHESIZE == "pse:synthesize"
        assert PSECapability.SYNTHESIZE_PRODUCTION == "pse:synthesize:production"
        assert PSECapability.EXECUTE == "pse:execute"
        assert PSECapability.CACHE_READ == "pse:cache:read"
        assert PSECapability.CACHE_WRITE == "pse:cache:write"
        assert PSECapability.DSL_EXTEND == "pse:dsl:extend"
        assert PSECapability.DSL_TUNE == "pse:dsl:tune"


class TestPlannedRegistrations:
    def test_seven_registrations_in_spec_order(self) -> None:
        regs = planned_registrations()
        assert len(regs) == 7
        # Order must match spec §13 verbatim — drift checks rely on it.
        capabilities = [r.capability for r in regs]
        assert capabilities == [
            PSECapability.SYNTHESIZE,
            PSECapability.SYNTHESIZE_PRODUCTION,
            PSECapability.EXECUTE,
            PSECapability.CACHE_READ,
            PSECapability.CACHE_WRITE,
            PSECapability.DSL_EXTEND,
            PSECapability.DSL_TUNE,
        ]

    def test_registrations_are_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        reg = planned_registrations()[0]
        assert isinstance(reg, CapabilityRegistration)
        with pytest.raises(FrozenInstanceError):
            reg.description = "tampered"  # type: ignore[misc]

    def test_admin_capabilities_default_to_admin_holder(self) -> None:
        regs = {r.capability: r for r in planned_registrations()}
        assert regs[PSECapability.DSL_EXTEND].default_holder == "admin"
        assert regs[PSECapability.DSL_TUNE].default_holder == "admin"

    def test_descriptions_non_empty(self) -> None:
        for reg in planned_registrations():
            assert reg.description, f"missing description for {reg.capability}"
