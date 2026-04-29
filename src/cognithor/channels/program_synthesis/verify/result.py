# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Public result type for the Verifier pipeline (spec §10).

The Verifier returns a :class:`VerificationResult` summarising every
stage that ran. The enumerator can either fail-fast (drop the candidate
on the first non-passing stage) or run all stages and inspect the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.core.types import StageResult


@dataclass(frozen=True)
class VerificationResult:
    """Aggregate of every :class:`StageResult` produced by the pipeline.

    ``passed`` is True iff every stage that *runs* returns ``passed=True``.
    A stage may legitimately not run when the pipeline short-circuits on
    an earlier failure. ``confidence`` reflects how much of the held-out
    set the candidate also satisfies — 1.0 if held-out fully passes,
    less if it partially fails (still admitted, just lower-confidence).
    """

    passed: bool
    stages: tuple[StageResult, ...]
    confidence: float = 0.0
    detail: str = ""
    annotations: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def first_failed_stage(self) -> StageResult | None:
        for s in self.stages:
            if not s.passed:
                return s
        return None
