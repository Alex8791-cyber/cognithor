# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Capability-token constants for the PSE channel (spec §13).

Cognithor's Hashline-Guard validates Ed25519/HMAC-signed capability
tokens before forwarding requests to a channel. This module declares
the seven capabilities the PSE channel introduces, plus a helper for
the channel registration step at startup.

Phase 1 only defines the constants and a stub registrar — full
integration with the central Capability Registry happens when the
channel is wired into the PGE adapter (Week 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PSECapability(str, Enum):
    """The seven capabilities the PSE channel registers (spec §13).

    Inherits from ``str`` so values can be passed to existing token
    machinery that expects raw strings, while still benefiting from
    enum identity in code.
    """

    SYNTHESIZE = "pse:synthesize"
    """Permission to start a synthesis search."""

    SYNTHESIZE_PRODUCTION = "pse:synthesize:production"
    """Permission to use full wall-clock / memory budgets.

    Only granted on Linux native or Windows + WSL2 — the Research-Mode
    Windows fallback (spec §11.6) explicitly disables this capability.
    """

    EXECUTE = "pse:execute"
    """Permission to run a synthesized program against a test input."""

    CACHE_READ = "pse:cache:read"
    """Read access to the tactical-memory PSE cache."""

    CACHE_WRITE = "pse:cache:write"
    """Write access to the tactical-memory PSE cache."""

    DSL_EXTEND = "pse:dsl:extend"
    """Register a new primitive at runtime. Admin / dev only."""

    DSL_TUNE = "pse:dsl:tune"
    """Run the deterministic Cost-Auto-Tuner (spec §7.6). Admin / dev only."""


@dataclass(frozen=True)
class CapabilityRegistration:
    """Description of one capability registration call.

    Phase 1 keeps this trivial — the real Cognithor Capability Registry
    expects (token, holder, scope) records, but the PSE channel only
    needs to enumerate which capabilities exist; the full plumbing is
    handled by the central registry.
    """

    capability: PSECapability
    description: str
    default_holder: str


def planned_registrations() -> tuple[CapabilityRegistration, ...]:
    """Return the static registration plan for the PSE channel.

    Order matches spec §13's table verbatim so verify_readme_claims and
    other docs-vs-code drift checks can eyeball the alignment.
    """
    return (
        CapabilityRegistration(
            capability=PSECapability.SYNTHESIZE,
            description="Start a synthesis search.",
            default_holder="planner",
        ),
        CapabilityRegistration(
            capability=PSECapability.SYNTHESIZE_PRODUCTION,
            description="Full wall-clock + memory budgets (Linux/WSL2 only).",
            default_holder="planner",
        ),
        CapabilityRegistration(
            capability=PSECapability.EXECUTE,
            description="Run a synthesized program on a test input.",
            default_holder="executor",
        ),
        CapabilityRegistration(
            capability=PSECapability.CACHE_READ,
            description="Read the tactical-memory PSE cache.",
            default_holder="channel",
        ),
        CapabilityRegistration(
            capability=PSECapability.CACHE_WRITE,
            description="Write the tactical-memory PSE cache.",
            default_holder="channel",
        ),
        CapabilityRegistration(
            capability=PSECapability.DSL_EXTEND,
            description="Register a new primitive at runtime.",
            default_holder="admin",
        ),
        CapabilityRegistration(
            capability=PSECapability.DSL_TUNE,
            description="Run the deterministic Cost-Auto-Tuner.",
            default_holder="admin",
        ),
    )


__all__ = [
    "CapabilityRegistration",
    "PSECapability",
    "planned_registrations",
]
