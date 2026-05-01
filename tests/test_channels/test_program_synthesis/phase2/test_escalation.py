# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Refiner Escalation tests (Sprint-1 plan task 9 final slice, spec §6.6)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.refiner.escalation import (
    EscalationResult,
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


def _prog(primitive: str) -> Program:
    """Tiny helper: build a Program with a single InputRef child."""
    return Program(primitive=primitive, children=(InputRef(),), output_type="Grid")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _runner_returning(candidate: ProgramNode | None):
    async def run(_: ProgramNode) -> ProgramNode | None:
        return candidate

    return run


def _verifier_from_map(scores: dict[str, float]):
    """Verifier that looks up a score by program primitive (or 'input' for InputRef)."""

    async def verify(program: ProgramNode) -> float:
        if isinstance(program, Program):
            return scores.get(program.primitive, 0.0)
        return scores.get("input", 0.0)

    return verify


def _make_escalator(
    *,
    local_candidate: ProgramNode | None = None,
    full_llm_candidate: ProgramNode | None = None,
    hybrid_candidate: ProgramNode | None = None,
    symbolic_candidate: ProgramNode | None = None,
    cegis_candidate: ProgramNode | None = None,
    scores: dict[str, float],
    refiner_budget_exhausted: bool = False,
    cegis_budget_exhausted: bool = False,
    cegis_enabled: bool = True,
) -> RefinerEscalator:
    return RefinerEscalator(
        mode_controller=RefinerModeController(),
        local_edit=_runner_returning(local_candidate),
        full_llm=_runner_returning(full_llm_candidate),
        hybrid=_runner_returning(hybrid_candidate),
        symbolic=_runner_returning(symbolic_candidate),
        cegis=_runner_returning(cegis_candidate) if cegis_enabled else None,
        verifier=_verifier_from_map(scores),
        refiner_budget_exhausted=lambda: refiner_budget_exhausted,
        cegis_budget_exhausted=lambda: cegis_budget_exhausted,
    )


# ---------------------------------------------------------------------------
# Stage 1 — Local-Edit ALWAYS first
# ---------------------------------------------------------------------------


class TestStage1LocalAlways:
    @pytest.mark.asyncio
    async def test_local_short_circuits_on_success_threshold(self) -> None:
        starting = _prog("rotate90")
        local_winner = _prog("rotate270")
        esc = _make_escalator(
            local_candidate=local_winner,
            scores={"rotate90": 0.4, "rotate270": 0.97},  # local crosses 0.95
        )
        result = await esc.refine(starting, initial_score=0.4, current_alpha=0.7)
        assert isinstance(result, EscalationResult)
        assert result.program == local_winner
        assert result.final_score == 0.97
        assert result.terminated_by == "success"
        assert result.refinement_path == ("local",)

    @pytest.mark.asyncio
    async def test_local_improves_but_not_winning_advances_best(self) -> None:
        starting = _prog("rotate90")
        local_better = _prog("rotate180")
        esc = _make_escalator(
            local_candidate=local_better,
            full_llm_candidate=None,
            hybrid_candidate=None,
            symbolic_candidate=None,
            cegis_candidate=None,
            scores={"rotate90": 0.2, "rotate180": 0.4},
        )
        # initial_score 0.2 < 0.3 → mode-dispatch skipped, but local still improved best.
        result = await esc.refine(starting, initial_score=0.2, current_alpha=0.7)
        assert result.terminated_by == "exhausted"
        assert result.program == local_better
        assert result.final_score == 0.4
        assert result.refinement_path == ("local",)

    @pytest.mark.asyncio
    async def test_local_returns_none_skips_stage(self) -> None:
        starting = _prog("rotate90")
        esc = _make_escalator(
            local_candidate=None,
            scores={"rotate90": 0.2},
        )
        result = await esc.refine(starting, initial_score=0.2, current_alpha=0.7)
        assert result.program == starting
        assert result.refinement_path == ()


# ---------------------------------------------------------------------------
# Stage 2 — Mode-Dispatch (Drei-Zonen)
# ---------------------------------------------------------------------------


class TestStage2ModeDispatch:
    @pytest.mark.asyncio
    async def test_alpha_high_picks_full_llm(self) -> None:
        starting = _prog("rotate90")
        winner = _prog("recolor")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=winner,
            scores={"rotate90": 0.5, "recolor": 0.97},
        )
        result = await esc.refine(starting, initial_score=0.5, current_alpha=0.6)
        assert result.terminated_by == "success"
        assert result.refinement_path == ("repair_full_llm",)
        assert result.mode_selected == "full_llm"

    @pytest.mark.asyncio
    async def test_alpha_grey_picks_hybrid(self) -> None:
        starting = _prog("rotate90")
        winner = _prog("mirror_horizontal")
        esc = _make_escalator(
            local_candidate=None,
            hybrid_candidate=winner,
            scores={"rotate90": 0.4, "mirror_horizontal": 0.96},
        )
        result = await esc.refine(starting, initial_score=0.4, current_alpha=0.4)
        assert result.refinement_path == ("repair_hybrid",)
        assert result.mode_selected == "hybrid"

    @pytest.mark.asyncio
    async def test_alpha_low_picks_symbolic(self) -> None:
        starting = _prog("rotate90")
        winner = _prog("flip")
        esc = _make_escalator(
            local_candidate=None,
            symbolic_candidate=winner,
            scores={"rotate90": 0.4, "flip": 0.96},
        )
        result = await esc.refine(starting, initial_score=0.4, current_alpha=0.2)
        assert result.refinement_path == ("repair_symbolic",)
        assert result.mode_selected == "symbolic"

    @pytest.mark.asyncio
    async def test_score_below_min_skips_mode_dispatch(self) -> None:
        starting = _prog("rotate90")
        full_llm = _prog("recolor")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=full_llm,
            scores={"rotate90": 0.25, "recolor": 0.97},
        )
        # initial_score 0.25 < 0.3 → mode-dispatch skipped entirely.
        result = await esc.refine(starting, initial_score=0.25, current_alpha=0.7)
        assert result.refinement_path == ()
        assert result.mode_selected is None
        assert result.program == starting

    @pytest.mark.asyncio
    async def test_refiner_budget_exhausted_skips_mode_dispatch(self) -> None:
        starting = _prog("rotate90")
        full_llm = _prog("recolor")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=full_llm,
            scores={"rotate90": 0.5, "recolor": 0.97},
            refiner_budget_exhausted=True,
        )
        result = await esc.refine(starting, initial_score=0.5, current_alpha=0.7)
        # Mode-dispatch skipped; CEGIS would still be eligible (score=0.5)
        # but no cegis_candidate set so it adds nothing.
        assert result.refinement_path == ()
        assert result.mode_selected is None


# ---------------------------------------------------------------------------
# Stage 3 — CEGIS only when score >= 0.5
# ---------------------------------------------------------------------------


class TestStage3CEGIS:
    @pytest.mark.asyncio
    async def test_cegis_runs_when_score_above_threshold(self) -> None:
        starting = _prog("rotate90")
        # Local + mode-dispatch produce nothing; CEGIS fires.
        cegis_winner = _prog("compose")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=None,
            cegis_candidate=cegis_winner,
            scores={"rotate90": 0.6, "compose": 0.97},
        )
        result = await esc.refine(starting, initial_score=0.6, current_alpha=0.7)
        assert result.terminated_by == "success"
        assert result.refinement_path == ("cegis",)
        assert result.program == cegis_winner

    @pytest.mark.asyncio
    async def test_cegis_skipped_when_score_below_threshold(self) -> None:
        starting = _prog("rotate90")
        cegis_winner = _prog("compose")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=None,
            symbolic_candidate=None,
            cegis_candidate=cegis_winner,
            scores={"rotate90": 0.4, "compose": 0.97},
        )
        # initial 0.4 → mode-dispatch runs but produces nothing; 0.4 < 0.5 → CEGIS skipped.
        result = await esc.refine(starting, initial_score=0.4, current_alpha=0.7)
        assert result.refinement_path == ()
        assert result.terminated_by == "exhausted"
        # Best is still the starting program.
        assert result.program == starting

    @pytest.mark.asyncio
    async def test_cegis_budget_exhausted_skips_stage(self) -> None:
        starting = _prog("rotate90")
        cegis_winner = _prog("compose")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=None,
            cegis_candidate=cegis_winner,
            scores={"rotate90": 0.6, "compose": 0.97},
            cegis_budget_exhausted=True,
        )
        result = await esc.refine(starting, initial_score=0.6, current_alpha=0.7)
        assert result.refinement_path == ()
        assert result.program == starting

    @pytest.mark.asyncio
    async def test_cegis_disabled_with_none(self) -> None:
        starting = _prog("rotate90")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=None,
            cegis_enabled=False,
            scores={"rotate90": 0.6},
        )
        result = await esc.refine(starting, initial_score=0.6, current_alpha=0.7)
        assert result.terminated_by == "exhausted"
        assert result.refinement_path == ()


# ---------------------------------------------------------------------------
# Path composition — multiple stages contribute
# ---------------------------------------------------------------------------


class TestRefinementPathComposition:
    @pytest.mark.asyncio
    async def test_local_then_repair_then_cegis(self) -> None:
        starting = _prog("rotate90")
        local_better = _prog("rotate180")
        repair_better = _prog("recolor")
        cegis_winner = _prog("compose")
        esc = _make_escalator(
            local_candidate=local_better,
            full_llm_candidate=repair_better,
            cegis_candidate=cegis_winner,
            scores={
                "rotate90": 0.4,
                "rotate180": 0.55,
                "recolor": 0.7,
                "compose": 0.99,
            },
        )
        result = await esc.refine(starting, initial_score=0.4, current_alpha=0.7)
        assert result.terminated_by == "success"
        assert result.refinement_path == ("local", "repair_full_llm", "cegis")
        assert result.program == cegis_winner
        assert result.mode_selected == "full_llm"

    @pytest.mark.asyncio
    async def test_stage_run_but_no_improvement_omits_from_path(self) -> None:
        starting = _prog("rotate90")
        local_worse = _prog("rotate180")
        repair_winner = _prog("recolor")
        esc = _make_escalator(
            local_candidate=local_worse,
            full_llm_candidate=repair_winner,
            scores={
                "rotate90": 0.6,
                "rotate180": 0.5,  # local ran but didn't improve
                "recolor": 0.97,
            },
        )
        result = await esc.refine(starting, initial_score=0.6, current_alpha=0.7)
        # local omitted from path — it ran but didn't help.
        assert result.refinement_path == ("repair_full_llm",)
        assert result.program == repair_winner

    @pytest.mark.asyncio
    async def test_no_stage_helps_returns_starting_program(self) -> None:
        starting = _prog("rotate90")
        esc = _make_escalator(
            local_candidate=_prog("rotate180"),
            full_llm_candidate=_prog("recolor"),
            cegis_candidate=_prog("compose"),
            scores={
                "rotate90": 0.6,
                "rotate180": 0.5,  # worse
                "recolor": 0.55,  # worse than 0.6
                "compose": 0.4,  # worse
            },
        )
        result = await esc.refine(starting, initial_score=0.6, current_alpha=0.7)
        assert result.terminated_by == "exhausted"
        assert result.refinement_path == ()
        assert result.program == starting


# ---------------------------------------------------------------------------
# Mode controller hysteresis is honoured across calls
# ---------------------------------------------------------------------------


class TestModeControllerHysteresis:
    @pytest.mark.asyncio
    async def test_consecutive_calls_share_hysteresis_state(self) -> None:
        starting = _prog("rotate90")
        esc = _make_escalator(
            local_candidate=None,
            full_llm_candidate=None,
            hybrid_candidate=None,
            symbolic_candidate=None,
            scores={"rotate90": 0.5},
        )
        # First call at α=0.5 → full_llm.
        r1 = await esc.refine(starting, initial_score=0.5, current_alpha=0.5)
        assert r1.mode_selected == "full_llm"
        # Second call at α=0.4 (would be hybrid normally), but hysteresis
        # holds full_llm for 3 calls minimum.
        r2 = await esc.refine(starting, initial_score=0.5, current_alpha=0.4)
        assert r2.mode_selected == "full_llm"
        r3 = await esc.refine(starting, initial_score=0.5, current_alpha=0.4)
        assert r3.mode_selected == "full_llm"
        # 4th call: hysteresis released → hybrid.
        r4 = await esc.refine(starting, initial_score=0.5, current_alpha=0.4)
        assert r4.mode_selected == "hybrid"


# ---------------------------------------------------------------------------
# Result dataclass contract
# ---------------------------------------------------------------------------


class TestEscalationResultDataclass:
    def test_is_frozen(self) -> None:
        r = EscalationResult(
            program=_prog("rotate90"),
            final_score=0.5,
            terminated_by="exhausted",
        )
        # Default tuples preserved.
        assert r.refinement_path == ()
        assert r.mode_selected is None
        # Hashable.
        assert hash(r) == hash(r)
