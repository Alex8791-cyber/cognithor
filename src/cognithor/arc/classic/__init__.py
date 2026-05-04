"""``cognithor.arc.classic`` — DEPRECATED.

This subpackage shipped the original ARC-AGI-1/2 grid-puzzle solver
(``ArcSolver`` + DSL search + LLM fallback). The production ARC-AGI-3
path now lives under ``cognithor.channels.program_synthesis.arc_agi3``
and the higher-performance synthesis layer under
``cognithor.channels.program_synthesis.phase2``. ``classic`` has zero
production import sites; only its own test suite still references it.

Scheduled for deletion after one minor-version cycle.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cognithor.arc.classic is deprecated; use "
    "cognithor.channels.program_synthesis.* instead. "
    "This subpackage will be deleted in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
