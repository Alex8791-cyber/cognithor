# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cognithor-side integration: capabilities, cache, PGE adapter, SGN bridge.

Phase 1 lands the capability constants + the Tactical Memory cache.
The PGE adapter, SGN bridge, and NumPy-solver fast-path follow in
Week 5.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.integration.capability_tokens import (
    CapabilityRegistration,
    PSECapability,
    planned_registrations,
)
from cognithor.channels.program_synthesis.integration.tactical_memory import (
    TTL_NO_SOLUTION_DAYS,
    TTL_PARTIAL_DAYS,
    TTL_SUCCESS_DAYS,
    CacheEntry,
    PSECache,
    cache_key,
)

__all__ = [
    "TTL_NO_SOLUTION_DAYS",
    "TTL_PARTIAL_DAYS",
    "TTL_SUCCESS_DAYS",
    "CacheEntry",
    "CapabilityRegistration",
    "PSECache",
    "PSECapability",
    "cache_key",
    "planned_registrations",
]
