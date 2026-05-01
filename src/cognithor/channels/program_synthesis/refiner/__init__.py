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

__all__ = [
    "CEGISLoop",
    "CEGISResult",
    "ColorDiff",
    "CounterExample",
    "DiffReport",
    "LLMRepairError",
    "LLMRepairResult",
    "LLMRepairSuggestion",
    "LLMRepairTwoStageClient",
    "LocalEditMutator",
    "PixelDiff",
    "RefinerMode",
    "RefinerModeController",
    "StructureDiff",
    "analyze_diff",
]
