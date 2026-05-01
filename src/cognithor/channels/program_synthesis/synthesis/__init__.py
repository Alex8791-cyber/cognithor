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

__all__ = [
    "BenchmarkSummary",
    "BenchmarkTask",
    "BenchmarkTaskResult",
    "Phase2SynthesisEngine",
    "Phase2SynthesisResult",
    "run_benchmark",
]
