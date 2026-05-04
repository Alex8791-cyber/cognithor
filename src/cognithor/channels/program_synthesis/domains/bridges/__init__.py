"""Cross-domain bridge layer (Sprint-26.2, Owner-Decision D5).

A *bridge* is a typed transformer that converts a value from one
domain's type-tag set to another's so the synthesizer can compose
programs across domain boundaries — e.g. parse a JSON field into a
datetime, format the datetime into a SQL literal, and feed it to a
SQL WHERE clause.

Sprint-26 ships only the **whitelisted 12 pairs** Owner-Decision D5
specifies. Anything outside the whitelist is rejected at registration
time. Sprint-28 will add a learning bridge-discovery layer once
real cross-domain task data is available.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.bridges.registry import (
    BRIDGE_REGISTRY,
    BridgeNotWhitelistedError,
    BridgeOperator,
    BridgeRegistry,
)
from cognithor.channels.program_synthesis.domains.bridges.whitelist import (
    SPRINT26_BRIDGE_WHITELIST,
    install_default_bridges,
)

__all__ = [
    "BRIDGE_REGISTRY",
    "SPRINT26_BRIDGE_WHITELIST",
    "BridgeNotWhitelistedError",
    "BridgeOperator",
    "BridgeRegistry",
    "install_default_bridges",
]
