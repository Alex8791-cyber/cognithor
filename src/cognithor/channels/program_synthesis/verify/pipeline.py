# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verifier pipeline (spec §10).

Orchestrates the five stages, accumulates :class:`StageResult` entries
into a :class:`VerificationResult`, and computes a confidence score
based on the held-out outcome.

The pipeline is deterministic and stateless — running the same program
against the same spec produces an identical result. This is critical
for the cache layer (Week 4 day 3) and for K10 (replay reproducibility).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.dsl.registry import (
    REGISTRY,
)
from cognithor.channels.program_synthesis.search.executor import (
    InProcessExecutor,
)
from cognithor.channels.program_synthesis.verify.result import VerificationResult
from cognithor.channels.program_synthesis.verify.stages import (
    _StageContext,
    default_pipeline,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.core.types import StageResult, TaskSpec
    from cognithor.channels.program_synthesis.dsl.registry import PrimitiveRegistry
    from cognithor.channels.program_synthesis.search.candidate import ProgramNode
    from cognithor.channels.program_synthesis.search.executor import Executor
    from cognithor.channels.program_synthesis.verify.stages import Stage


class Verifier:
    """Drives a sequence of :class:`Stage` instances over one program.

    The default pipeline is the spec-mandated five-stage order; tests
    can inject a smaller pipeline (e.g. just SyntaxStage + DemoStage)
    when they want to focus on a single invariant.
    """

    def __init__(
        self,
        executor: Executor | None = None,
        registry: PrimitiveRegistry | None = None,
        stages: tuple[Stage, ...] | None = None,
    ) -> None:
        self._registry = registry if registry is not None else REGISTRY
        self._executor = (
            executor if executor is not None else InProcessExecutor(registry=self._registry)
        )
        self._stages: tuple[Stage, ...] = stages if stages is not None else default_pipeline()

    def verify(self, program: ProgramNode, spec: TaskSpec) -> VerificationResult:
        ctx = _StageContext(
            program=program,
            spec=spec,
            executor=self._executor,
            registry=self._registry,
        )
        results: list[StageResult] = []
        passed_overall = True
        held_out_score: float | None = None

        for stage in self._stages:
            result = stage.run(ctx)
            results.append(result)
            if not result.passed:
                passed_overall = False
                if stage.fail_fast:
                    return VerificationResult(
                        passed=False,
                        stages=tuple(results),
                        confidence=0.0,
                        detail=f"{stage.name} stage failed: {result.detail}",
                    )
            # Track held-out score for the confidence calculation.
            if stage.name == "held_out":
                held_out_score = self._held_out_fraction(result, spec)

        confidence = self._compute_confidence(passed_overall, held_out_score, spec)
        detail = "all stages passed" if passed_overall else "soft failure"
        return VerificationResult(
            passed=passed_overall,
            stages=tuple(results),
            confidence=confidence,
            detail=detail,
        )

    @staticmethod
    def _held_out_fraction(result: StageResult, spec: TaskSpec) -> float:
        """Parse the ``X/Y held-out pairs matched`` detail into a fraction."""
        if not spec.held_out:
            return 1.0
        # The detail string is well-formed because HeldOutStage always
        # emits ``f"{passed}/{total} held-out pairs matched"``.
        try:
            head = result.detail.split(" ", 1)[0]
            num, den = head.split("/", 1)
            return float(num) / float(den) if float(den) > 0 else 1.0
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _compute_confidence(
        passed_overall: bool, held_out_score: float | None, spec: TaskSpec
    ) -> float:
        """Confidence = held-out match fraction, or 1.0 if no held-out pairs.

        A program that passed every fail-fast stage but only partially
        passed held-out lands at the corresponding fraction; full pass
        gives 1.0; total failure gives 0.0. If the program failed any
        fail-fast stage we return 0.0 regardless.
        """
        if not passed_overall:
            return 0.0
        if not spec.held_out:
            return 1.0
        return held_out_score if held_out_score is not None else 1.0


__all__ = ["Verifier"]
