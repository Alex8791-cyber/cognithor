# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 top-level engine tests (Sprint-1 plan task 10, spec §3.2)."""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.datatypes import (
    PartitionedBudget,
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
from cognithor.channels.program_synthesis.synthesis import (
    Phase2SynthesisEngine,
    Phase2SynthesisResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(primitive: str) -> Program:
    return Program(primitive=primitive, children=(InputRef(),), output_type="Grid")


def _budget() -> PartitionedBudget:
    return PartitionedBudget.from_spec_default()


def _scoring_verifier(scores: dict[str, float]):
    async def verify(program: ProgramNode) -> float:
        if isinstance(program, Program):
            return scores.get(program.primitive, 0.0)
        return 0.0

    return verify


def _search_returning(*candidates: tuple[ProgramNode, float]):
    async def runner(_spec: Any, _budget: float) -> list[tuple[ProgramNode, float]]:
        return list(candidates)

    return runner


def _no_op_runner(returner=None):
    async def runner(_p: ProgramNode) -> ProgramNode | None:
        return returner

    return runner


def _make_refiner(
    *,
    local_candidate: ProgramNode | None = None,
    full_llm_candidate: ProgramNode | None = None,
    cegis_candidate: ProgramNode | None = None,
    scores: dict[str, float],
) -> RefinerEscalator:
    return RefinerEscalator(
        mode_controller=RefinerModeController(),
        local_edit=_no_op_runner(local_candidate),
        full_llm=_no_op_runner(full_llm_candidate),
        hybrid=_no_op_runner(None),
        symbolic=_no_op_runner(None),
        cegis=_no_op_runner(cegis_candidate),
        verifier=_scoring_verifier(scores),
    )


# ---------------------------------------------------------------------------
# Stage 0 — Cache hit short-circuit
# ---------------------------------------------------------------------------


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit_short_circuits(self) -> None:
        cached = _prog("recolor")

        def reader(_spec: Any) -> ProgramNode | None:
            return cached

        engine = Phase2SynthesisEngine(
            search_runner=_search_returning(),  # would fail if reached
            verifier=_scoring_verifier({"recolor": 0.97}),
            cache_reader=reader,
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert isinstance(result, Phase2SynthesisResult)
        assert result.terminated_by == "cache_hit"
        assert result.cache_hit is True
        assert result.program == cached

    @pytest.mark.asyncio
    async def test_stale_cache_falls_through_to_search(self) -> None:
        # Cached entry no longer wins (score 0.6 < 0.95 threshold).
        cached = _prog("rotate90")
        searched = _prog("recolor")

        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((searched, 0.97)),
            verifier=_scoring_verifier({"rotate90": 0.6, "recolor": 0.97}),
            cache_reader=lambda _: cached,
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert result.cache_hit is False
        assert result.terminated_by == "search_success"
        assert result.program == searched


# ---------------------------------------------------------------------------
# Stage 1 — Search short-circuit
# ---------------------------------------------------------------------------


class TestSearchSuccess:
    @pytest.mark.asyncio
    async def test_search_picks_highest_scoring(self) -> None:
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning(
                (_prog("rotate90"), 0.4),
                (_prog("recolor"), 0.96),
                (_prog("transpose"), 0.7),
            ),
            verifier=_scoring_verifier({"recolor": 0.96}),
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert result.terminated_by == "search_success"
        assert isinstance(result.program, Program)
        assert result.program.primitive == "recolor"
        assert result.candidates_evaluated == 3

    @pytest.mark.asyncio
    async def test_no_candidates_terminates(self) -> None:
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning(),  # empty
            verifier=_scoring_verifier({}),
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert result.terminated_by == "no_candidates"
        assert result.program is None
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Stage 2 — Refinement
# ---------------------------------------------------------------------------


class TestRefinement:
    @pytest.mark.asyncio
    async def test_refinement_promotes_partial_to_success(self) -> None:
        starting = _prog("rotate90")
        refined = _prog("recolor")
        refiner = _make_refiner(
            full_llm_candidate=refined,
            scores={"rotate90": 0.5, "recolor": 0.97},
        )
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((starting, 0.5)),
            verifier=_scoring_verifier({"rotate90": 0.5, "recolor": 0.97}),
            refiner=refiner,
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
            current_alpha=0.7,
        )
        assert result.terminated_by == "refined_success"
        assert result.refined is True
        assert isinstance(result.program, Program)
        assert result.program.primitive == "recolor"
        assert result.refinement_path == ("repair_full_llm",)

    @pytest.mark.asyncio
    async def test_refinement_helps_but_doesnt_win(self) -> None:
        starting = _prog("rotate90")
        refined = _prog("recolor")
        refiner = _make_refiner(
            full_llm_candidate=refined,
            scores={"rotate90": 0.5, "recolor": 0.7},
        )
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((starting, 0.5)),
            verifier=_scoring_verifier({"rotate90": 0.5, "recolor": 0.7}),
            refiner=refiner,
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
            current_alpha=0.7,
        )
        # Improved score but didn't cross 0.95 → search_exhausted.
        assert result.terminated_by == "search_exhausted"
        assert result.refined is True
        assert result.score == 0.7

    @pytest.mark.asyncio
    async def test_refiner_min_score_gate(self) -> None:
        # Score 0.2 < 0.3 (default) → refiner is NOT called.
        starting = _prog("rotate90")
        refined = _prog("recolor")
        refiner = _make_refiner(
            full_llm_candidate=refined,
            scores={"rotate90": 0.2, "recolor": 0.97},
        )
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((starting, 0.2)),
            verifier=_scoring_verifier({"rotate90": 0.2, "recolor": 0.97}),
            refiner=refiner,
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        # Refiner skipped → original 0.2 is returned.
        assert result.refined is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_refiner_none_disables_stage(self) -> None:
        starting = _prog("rotate90")
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((starting, 0.5)),
            verifier=_scoring_verifier({"rotate90": 0.5}),
            refiner=None,  # disabled
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert result.refined is False
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# Cache writes
# ---------------------------------------------------------------------------


class TestCacheWrites:
    @pytest.mark.asyncio
    async def test_cache_written_on_search_success(self) -> None:
        writes: list[tuple[Any, ProgramNode, float]] = []

        def writer(spec: Any, program: ProgramNode, score: float) -> None:
            writes.append((spec, program, score))

        searched = _prog("recolor")
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((searched, 0.97)),
            verifier=_scoring_verifier({"recolor": 0.97}),
            cache_writer=writer,
        )
        await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert len(writes) == 1
        assert writes[0][1] == searched
        assert writes[0][2] == 0.97

    @pytest.mark.asyncio
    async def test_cache_written_on_refined_success(self) -> None:
        writes: list[tuple[Any, ProgramNode, float]] = []

        def writer(spec: Any, program: ProgramNode, score: float) -> None:
            writes.append((spec, program, score))

        refined = _prog("recolor")
        refiner = _make_refiner(
            full_llm_candidate=refined,
            scores={"rotate90": 0.5, "recolor": 0.97},
        )
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((_prog("rotate90"), 0.5)),
            verifier=_scoring_verifier({"rotate90": 0.5, "recolor": 0.97}),
            refiner=refiner,
            cache_writer=writer,
        )
        result = await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert result.terminated_by == "refined_success"
        assert len(writes) == 1
        assert writes[0][2] == 0.97

    @pytest.mark.asyncio
    async def test_no_cache_write_on_partial_result(self) -> None:
        writes: list[tuple[Any, ProgramNode, float]] = []

        def writer(spec: Any, program: ProgramNode, score: float) -> None:
            writes.append((spec, program, score))

        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((_prog("rotate90"), 0.5)),
            verifier=_scoring_verifier({"rotate90": 0.5}),
            cache_writer=writer,
        )
        await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert writes == []


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    @pytest.mark.asyncio
    async def test_telemetry_fires_on_search_success(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []

        engine = Phase2SynthesisEngine(
            search_runner=_search_returning((_prog("recolor"), 0.97)),
            verifier=_scoring_verifier({"recolor": 0.97}),
            telemetry=lambda name, payload: events.append((name, payload)),
        )
        await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        names = [n for n, _ in events]
        assert "engine.search_success" in names

    @pytest.mark.asyncio
    async def test_telemetry_fires_on_no_candidates(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning(),
            verifier=_scoring_verifier({}),
            telemetry=lambda n, p: events.append((n, p)),
        )
        await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert any(name == "engine.no_candidates" for name, _ in events)

    @pytest.mark.asyncio
    async def test_telemetry_fires_on_cache_hit(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        cached = _prog("recolor")
        engine = Phase2SynthesisEngine(
            search_runner=_search_returning(),
            verifier=_scoring_verifier({"recolor": 0.97}),
            cache_reader=lambda _: cached,
            telemetry=lambda n, p: events.append((n, p)),
        )
        await engine.synthesize(
            spec={"task": "x"},
            budget=_budget(),
            wall_clock_budget_seconds=10.0,
        )
        assert any(name == "engine.cache_hit" for name, _ in events)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_success_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="success_threshold"):
            Phase2SynthesisEngine(
                search_runner=_search_returning(),
                verifier=_scoring_verifier({}),
                success_threshold=0.0,
            )

    def test_invalid_refiner_min_score_raises(self) -> None:
        with pytest.raises(ValueError, match="refiner_min_score"):
            Phase2SynthesisEngine(
                search_runner=_search_returning(),
                verifier=_scoring_verifier({}),
                refiner_min_score=1.5,
            )


# ---------------------------------------------------------------------------
# Smoke: 5 ARC-shaped tasks (plan acceptance criterion)
# ---------------------------------------------------------------------------


class TestSmokeFiveTasks:
    @pytest.mark.asyncio
    async def test_five_tasks_complete_without_crashing(self) -> None:
        # Vary score per task so we touch every termination path:
        # task 0 → cache hit
        # task 1 → search success
        # task 2 → refined success
        # task 3 → search exhausted (partial)
        # task 4 → no_candidates
        results: list[Phase2SynthesisResult] = []
        for i in range(5):
            cached = _prog("recolor") if i == 0 else None
            search_outputs: list[tuple[ProgramNode, float]]
            if i == 1:
                search_outputs = [(_prog("recolor"), 0.97)]
            elif i in (2, 3):
                search_outputs = [(_prog("rotate90"), 0.5)]
            else:
                search_outputs = []

            scores = {"recolor": 0.97, "rotate90": 0.5}
            refiner = (
                _make_refiner(
                    full_llm_candidate=_prog("recolor") if i == 2 else None,
                    scores=scores,
                )
                if i in (2, 3)
                else None
            )

            engine = Phase2SynthesisEngine(
                search_runner=_search_returning(*search_outputs),
                verifier=_scoring_verifier(scores),
                cache_reader=(lambda _spec, c=cached: c),
                refiner=refiner,
            )
            result = await engine.synthesize(
                spec={"task": i},
                budget=_budget(),
                wall_clock_budget_seconds=10.0,
                current_alpha=0.7,
            )
            results.append(result)

        # Every task produced a result.
        assert len(results) == 5
        # We touched at least 4 distinct termination reasons.
        reasons = {r.terminated_by for r in results}
        assert len(reasons) >= 4
