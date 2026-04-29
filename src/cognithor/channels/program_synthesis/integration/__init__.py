# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cognithor-side integration: capabilities, cache, PGE adapter, SGN bridge.

Phase 1 ships:
* Capability-token constants
* Tactical Memory cache
* PGE adapter + SynthesisRequest + is_synthesizable classifier
* State-Graph-Navigator bridge
* NumPy-solver fast-path bridge
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.integration.capability_tokens import (
    CapabilityRegistration,
    PSECapability,
    planned_registrations,
)
from cognithor.channels.program_synthesis.integration.numpy_solver_bridge import (
    NumpySolverBridge,
)
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
    is_synthesizable,
)
from cognithor.channels.program_synthesis.integration.state_graph_bridge import (
    NEUTRAL_MULTIPLIER,
    PROMOTED_MULTIPLIER,
    SUPPORTED_HINT_KEYS,
    StateGraphBridge,
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
    "NEUTRAL_MULTIPLIER",
    "PROMOTED_MULTIPLIER",
    "SUPPORTED_HINT_KEYS",
    "TTL_NO_SOLUTION_DAYS",
    "TTL_PARTIAL_DAYS",
    "TTL_SUCCESS_DAYS",
    "CacheEntry",
    "CapabilityRegistration",
    "NumpySolverBridge",
    "PSECache",
    "PSECapability",
    "ProgramSynthesisChannel",
    "StateGraphBridge",
    "SynthesisRequest",
    "cache_key",
    "is_synthesizable",
    "planned_registrations",
]
