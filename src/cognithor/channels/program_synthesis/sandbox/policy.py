# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sandbox policy constants (spec §11.2 + §11.6).

The actual setrlimit / AST-whitelist enforcement lands in a follow-up
PR with the full subprocess worker (a Linux-only piece). This module
holds the budget caps and the research-mode warning text that are
referenced by the strategy router.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxLimits:
    """Hard limits enforced by the sandbox layer.

    All three values are *hard* — exceeding any of them produces a
    SandboxTimeoutError / SandboxOOMError that the search engine
    short-circuits on.
    """

    wall_clock_seconds: float
    memory_mb: int
    per_candidate_ms: int


# Defaults from spec §11.2.
DEFAULT_LIMITS = SandboxLimits(
    wall_clock_seconds=30.0,
    memory_mb=256,
    per_candidate_ms=100,
)


# Reduced limits for Windows Research-Mode (spec §11.6).
# The capability ``pse:synthesize:production`` is also disabled in
# research mode — see the strategy router below.
RESEARCH_MODE_LIMITS = SandboxLimits(
    wall_clock_seconds=10.0,
    memory_mb=256,
    per_candidate_ms=100,
)


RESEARCH_MODE_WARNING = (
    "PSE running on Windows without WSL2. Reduced isolation. "
    "Research-Mode only. Install WSL2 + Ubuntu for production-grade "
    "sandbox."
)


__all__ = [
    "DEFAULT_LIMITS",
    "RESEARCH_MODE_LIMITS",
    "RESEARCH_MODE_WARNING",
    "SandboxLimits",
]
