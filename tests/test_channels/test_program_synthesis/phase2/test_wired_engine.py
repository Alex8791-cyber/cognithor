# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Production-Wiring tests (Sprint-2 Track B)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    StageResult,
    SynthesisResult,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.verifier_evaluator import (
    VerifierEvaluator,
)
from cognithor.channels.program_synthesis.refiner.escalation import (
    RefinerEscalator,
)
from cognithor.channels.program_synthesis.refiner.mode_controller import (
    RefinerModeController,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
    ProgramNode,
)
from cognithor.channels.program_synthesis.search.executor import (
    InProcessExecutor,
)
from cognithor.channels.program_synthesis.synthesis.wired_engine import (
    WiredPhase2Engine,
    WiredSynthesisResult,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _identity_program() -> Program:
    return Program(primitive="identity", children=(InputRef(),), output_type="Grid")


def _spec() -> TaskSpec:
    return TaskSpec(
        examples=((_g([[1, 2]]), _g([[1, 2]])),),
    )


def _budget() -> Budget:
    return Budget(max_depth=2, wall_clock_seconds=5.0)


def _phase1_returning(
    *,
    status: SynthesisStatus,
    program: ProgramNode | None,
    score: float,
):
    def runner(_spec: TaskSpec, _budget: Budget) -> SynthesisResult:
        return SynthesisResult(
            status=status,
            program=program,
            score=score,
            confidence=score,
            cost_seconds=0.01,
            cost_candidates=10,
            verifier_trace=(StageResult(stage="demo", passed=score >= 0.95),),
        )

    return runner


def _no_op_runner(returner=None):
    async def runner(_p: ProgramNode) -> ProgramNode | None:
        return returner

    return runner


def _make_refiner(
    *,
    full_llm_candidate: ProgramNode | None = None,
    local_candidate: ProgramNode | None = None,
):
    async def verify(p: ProgramNode) -> float:
        if isinstance(p, Program):
            mapping = {"recolor": 0.97, "rotate90": 0.5, "identity": 0.97}
            return mapping.get(p.primitive, 0.0)
        return 0.0

    return RefinerEscalator(
        mode_controller=RefinerModeController(),
        local_edit=_no_op_runner(local_candidate),
        full_llm=_no_op_runner(full_llm_candidate),
        hybrid=_no_op_runner(None),
        symbolic=_no_op_runner(None),
        cegis=_no_op_runner(None),
        verifier=verify,
    )


# ---------------------------------------------------------------------------
# Stage 1 — Phase-1 short-circuits
# ---------------------------------------------------------------------------


class TestPhase1SuccessShortCircuit:
    @pytest.mark.asyncio
    async def test_phase1_success_skips_refiner(self) -> None:
        program = _identity_program()
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.SUCCESS, program=program, score=1.0
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            refiner=_make_refiner(
                full_llm_candidate=Program(
                    primitive="recolor", children=(InputRef(),), output_type="Grid"
                )
            ),
        )
        result = await engine.synthesize(_spec(), _budget())
        assert isinstance(result, WiredSynthesisResult)
        assert result.terminated_by == "phase1_success"
        assert result.refined is False
        assert result.program == program

    @pytest.mark.asyncio
    async def test_no_solution_returns_early(self) -> None:
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.NO_SOLUTION, program=None, score=0.0
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
        )
        result = await engine.synthesize(_spec(), _budget())
        assert result.terminated_by == "no_solution"
        assert result.program is None


# ---------------------------------------------------------------------------
# Stage 2 — Refiner gate
# ---------------------------------------------------------------------------


class TestRefinerGate:
    @pytest.mark.asyncio
    async def test_score_below_min_skips_refiner(self) -> None:
        # Phase-1 returned PARTIAL with score 0.2 < 0.3 default → no refiner.
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=_identity_program(),
                score=0.2,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            refiner=_make_refiner(),
        )
        result = await engine.synthesize(_spec(), _budget())
        assert result.terminated_by == "no_refinement_eligible"
        assert result.refined is False

    @pytest.mark.asyncio
    async def test_no_refiner_wired_skips_stage(self) -> None:
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=_identity_program(),
                score=0.5,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            refiner=None,
        )
        result = await engine.synthesize(_spec(), _budget())
        assert result.terminated_by == "no_refinement_eligible"


# ---------------------------------------------------------------------------
# Stage 3+4 — Refinement path
# ---------------------------------------------------------------------------


class TestRefinerSuccess:
    @pytest.mark.asyncio
    async def test_partial_promoted_via_full_llm_repair(self) -> None:
        rotated = Program(primitive="rotate90", children=(InputRef(),), output_type="Grid")
        recolored = Program(primitive="recolor", children=(InputRef(),), output_type="Grid")
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=rotated,
                score=0.5,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            refiner=_make_refiner(full_llm_candidate=recolored),
        )
        result = await engine.synthesize(_spec(), _budget())
        assert result.terminated_by == "refined_success"
        assert result.refined is True
        assert result.refinement_path == ("repair_full_llm",)
        assert result.program == recolored
        assert result.final_score >= 0.95

    @pytest.mark.asyncio
    async def test_refinement_helps_but_doesnt_win(self) -> None:
        rotated = Program(primitive="rotate90", children=(InputRef(),), output_type="Grid")
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=rotated,
                score=0.5,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            # No improvement candidate from any stage → refiner returns
            # the starting program with the original score.
            refiner=_make_refiner(),
        )
        result = await engine.synthesize(_spec(), _budget())
        assert result.refined is True
        assert result.terminated_by == "refined_partial"


# ---------------------------------------------------------------------------
# Search-α resolution via DualPriorMixer
# ---------------------------------------------------------------------------


class _StubMixer:
    """Hand-rolled DualPriorMixer that returns a canned alpha."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self.calls: list[Any] = []

    async def get_prior(self, examples: Any) -> Any:
        from cognithor.channels.program_synthesis.phase2.dual_prior import (
            DualPriorResult,
        )
        from cognithor.channels.program_synthesis.phase2.llm_prior import LLMPrior
        from cognithor.channels.program_synthesis.phase2.symbolic_prior import (
            SymbolicPriorResult,
        )

        self.calls.append(list(examples))
        return DualPriorResult(
            primitive_scores={"rotate90": 1.0},
            alpha=self._alpha,
            llm_prior=LLMPrior(primitive_scores={"rotate90": 1.0}, alpha_entropy_hint=self._alpha),
            symbolic_prior=SymbolicPriorResult(
                primitive_scores={"rotate90": 1.0}, effective_confidence=1.0
            ),
        )


class TestAlphaResolution:
    @pytest.mark.asyncio
    async def test_mixer_alpha_drives_refiner_zone(self) -> None:
        rotated = Program(primitive="rotate90", children=(InputRef(),), output_type="Grid")
        recolored = Program(primitive="recolor", children=(InputRef(),), output_type="Grid")
        # α=0.6 → full_llm zone; the refiner has only full_llm wired.
        mixer = _StubMixer(alpha=0.6)
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=rotated,
                score=0.5,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            dual_prior=mixer,  # type: ignore[arg-type]
            refiner=_make_refiner(full_llm_candidate=recolored),
        )
        result = await engine.synthesize(_spec(), _budget())
        assert result.current_alpha == 0.6
        assert mixer.calls  # mixer was consulted
        assert result.terminated_by == "refined_success"

    @pytest.mark.asyncio
    async def test_mixer_failure_falls_back_to_cold_start(self) -> None:
        rotated = Program(primitive="rotate90", children=(InputRef(),), output_type="Grid")

        class _BrokenMixer:
            async def get_prior(self, _examples: Any) -> Any:
                raise RuntimeError("mixer down")

        events: list[tuple[str, dict[str, Any]]] = []
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=rotated,
                score=0.5,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            dual_prior=_BrokenMixer(),  # type: ignore[arg-type]
            refiner=_make_refiner(),
            telemetry=lambda n, p: events.append((n, p)),
        )
        result = await engine.synthesize(_spec(), _budget())
        # Cold-start α from default config = 0.85.
        assert result.current_alpha == 0.85
        assert any(name == "wired.alpha_fallback" for name, _ in events)

    @pytest.mark.asyncio
    async def test_no_mixer_uses_cold_start(self) -> None:
        rotated = Program(primitive="rotate90", children=(InputRef(),), output_type="Grid")
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=rotated,
                score=0.5,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            dual_prior=None,
            refiner=_make_refiner(),
        )
        result = await engine.synthesize(_spec(), _budget())
        assert result.current_alpha == 0.85


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    @pytest.mark.asyncio
    async def test_phase1_done_event_fires(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.SUCCESS, program=_identity_program(), score=1.0
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            telemetry=lambda n, p: events.append((n, p)),
        )
        await engine.synthesize(_spec(), _budget())
        names = {n for n, _ in events}
        assert "wired.phase1_done" in names

    @pytest.mark.asyncio
    async def test_refined_event_fires_with_path(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        rotated = Program(primitive="rotate90", children=(InputRef(),), output_type="Grid")
        recolored = Program(primitive="recolor", children=(InputRef(),), output_type="Grid")
        engine = WiredPhase2Engine(
            phase1_search=_phase1_returning(
                status=SynthesisStatus.PARTIAL,
                program=rotated,
                score=0.5,
            ),
            verifier=VerifierEvaluator(InProcessExecutor()),
            refiner=_make_refiner(full_llm_candidate=recolored),
            telemetry=lambda n, p: events.append((n, p)),
        )
        await engine.synthesize(_spec(), _budget())
        names = {n for n, _ in events}
        assert "wired.refined" in names


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_success_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="success_threshold"):
            WiredPhase2Engine(
                phase1_search=_phase1_returning(
                    status=SynthesisStatus.NO_SOLUTION, program=None, score=0.0
                ),
                verifier=VerifierEvaluator(InProcessExecutor()),
                success_threshold=1.5,
            )

    def test_invalid_refiner_min_score_raises(self) -> None:
        with pytest.raises(ValueError, match="refiner_min_score"):
            WiredPhase2Engine(
                phase1_search=_phase1_returning(
                    status=SynthesisStatus.NO_SOLUTION, program=None, score=0.0
                ),
                verifier=VerifierEvaluator(InProcessExecutor()),
                refiner_min_score=-0.1,
            )
