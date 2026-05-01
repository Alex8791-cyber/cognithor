# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 top-level synthesis engine (spec §3.2).

Re-exports the engine + result types so callers can simply::

    from cognithor.channels.program_synthesis.synthesis import (
        Phase2SynthesisEngine,
        Phase2SynthesisResult,
    )
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.synthesis.benchmark import (
    BenchmarkSummary,
    BenchmarkTask,
    BenchmarkTaskResult,
    run_benchmark,
)
from cognithor.channels.program_synthesis.synthesis.engine import (
    Phase2SynthesisEngine,
    Phase2SynthesisResult,
)
from cognithor.channels.program_synthesis.synthesis.leak_free_fixtures import (
    LEAK_FREE_TASKS,
    LeakFreeTask,
    leak_free_set_hash,
)
from cognithor.channels.program_synthesis.synthesis.leak_free_fixtures import (
    benchmark_tasks as leak_free_benchmark_tasks,
)
from cognithor.channels.program_synthesis.synthesis.wired_engine import (
    WiredPhase2Engine,
    WiredSynthesisResult,
)

__all__ = [
    "LEAK_FREE_TASKS",
    "BenchmarkSummary",
    "BenchmarkTask",
    "BenchmarkTaskResult",
    "LeakFreeTask",
    "Phase2SynthesisEngine",
    "Phase2SynthesisResult",
    "WiredPhase2Engine",
    "WiredSynthesisResult",
    "leak_free_benchmark_tasks",
    "leak_free_set_hash",
    "run_benchmark",
]
