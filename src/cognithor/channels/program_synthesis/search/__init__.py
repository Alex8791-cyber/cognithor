# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Enumerative search engine for the PSE channel (spec §8).

Phase 1 ships a bottom-up enumerator with observational-equivalence
pruning. Week 3 of the roadmap lands the candidate types, the budget
allocator, and the engine itself.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
    ProgramNode,
)
from cognithor.channels.program_synthesis.search.equivalence import (
    ObservationalEquivalencePruner,
)
from cognithor.channels.program_synthesis.search.executor import (
    ExecutionResult,
    Executor,
    InProcessExecutor,
)

__all__ = [
    "Const",
    "ExecutionResult",
    "Executor",
    "InProcessExecutor",
    "InputRef",
    "ObservationalEquivalencePruner",
    "Program",
    "ProgramNode",
]
