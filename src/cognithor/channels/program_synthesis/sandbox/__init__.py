# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PSE sandbox layer.

Phase 1 ships the strategy router + platform detection + capability
gating. The real subprocess + AST-whitelist + setrlimit machinery
lives in a follow-up PR and slots into the same Strategy interface.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.sandbox.policy import (
    DEFAULT_LIMITS,
    RESEARCH_MODE_LIMITS,
    RESEARCH_MODE_WARNING,
    SandboxLimits,
)
from cognithor.channels.program_synthesis.sandbox.strategies import (
    LinuxSubprocessStrategy,
    StrategyInfo,
    WindowsResearchStrategy,
    WSL2WorkerStrategy,
    capabilities_for_strategy,
    select_sandbox_strategy,
)

__all__ = [
    "DEFAULT_LIMITS",
    "RESEARCH_MODE_LIMITS",
    "RESEARCH_MODE_WARNING",
    "LinuxSubprocessStrategy",
    "SandboxLimits",
    "StrategyInfo",
    "WSL2WorkerStrategy",
    "WindowsResearchStrategy",
    "capabilities_for_strategy",
    "select_sandbox_strategy",
]
