# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 Critic & Refiner sub-package.

Sprint-1 ships the mode-selection layer (F2): the three-zone refiner
mode controller with hysteresis. Hybrid-Repair, CEGIS, and Local-Edit
follow in subsequent sprints.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.refiner.cegis import (
    CEGISLoop,
    CEGISResult,
    CounterExample,
)
from cognithor.channels.program_synthesis.refiner.diff_analyzer import (
    ColorDiff,
    DiffReport,
    PixelDiff,
    StructureDiff,
    analyze_diff,
)
from cognithor.channels.program_synthesis.refiner.hybrid_repair import (
    CandidateOrigin,
    HybridRepairCandidate,
    HybridRepairResult,
    run_hybrid_repair,
)
from cognithor.channels.program_synthesis.refiner.llm_repair_two_stage import (
    LLMRepairError,
    LLMRepairResult,
    LLMRepairSuggestion,
    LLMRepairTwoStageClient,
)
from cognithor.channels.program_synthesis.refiner.local_edit import (
    LocalEditMutator,
)
from cognithor.channels.program_synthesis.refiner.mode_controller import (
    RefinerMode,
    RefinerModeController,
)
from cognithor.channels.program_synthesis.refiner.symbolic_repair import (
    RepairKind,
    RepairSuggestion,
    advise_repairs,
)
from cognithor.channels.program_synthesis.refiner.trace_replay import (
    TraceStep,
    find_divergence,
    find_first_failure,
    replay_trace,
)

__all__ = [
    "CEGISLoop",
    "CEGISResult",
    "CandidateOrigin",
    "ColorDiff",
    "CounterExample",
    "DiffReport",
    "HybridRepairCandidate",
    "HybridRepairResult",
    "LLMRepairError",
    "LLMRepairResult",
    "LLMRepairSuggestion",
    "LLMRepairTwoStageClient",
    "LocalEditMutator",
    "PixelDiff",
    "RefinerMode",
    "RefinerModeController",
    "RepairKind",
    "RepairSuggestion",
    "StructureDiff",
    "TraceStep",
    "advise_repairs",
    "analyze_diff",
    "find_divergence",
    "find_first_failure",
    "replay_trace",
    "run_hybrid_repair",
]
