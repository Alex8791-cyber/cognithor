# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Hybrid-Repair tests (Sprint-1 plan task 9 slice, spec §6.5.2 Zone-2)."""

from __future__ import annotations

import asyncio

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.refiner.hybrid_repair import (
    HybridRepairCandidate,
    HybridRepairResult,
    run_hybrid_repair,
)
from cognithor.channels.program_synthesis.refiner.llm_repair_two_stage import (
    LLMRepairResult,
    LLMRepairSuggestion,
)
from cognithor.channels.program_synthesis.refiner.symbolic_repair import (
    RepairSuggestion,
)


def _symbolic_only(*sugs: RepairSuggestion):
    def supply() -> list[RepairSuggestion]:
        return list(sugs)

    return supply


def _llm_only(*sugs: LLMRepairSuggestion):
    async def supply() -> LLMRepairResult:
        return LLMRepairResult(suggestions=tuple(sugs))

    return supply


def _scorer_from_map(scores: dict[str, float]):
    async def score(candidate: HybridRepairCandidate) -> float:
        return scores.get(candidate.source, 0.0)

    return score


# ---------------------------------------------------------------------------
# Happy path — best of both backends wins
# ---------------------------------------------------------------------------


class TestRunHybridRepair:
    @pytest.mark.asyncio
    async def test_picks_highest_scoring_candidate(self) -> None:
        sym = _symbolic_only(
            RepairSuggestion(
                kind="rotation_repair",
                primitive_hint="rotate90",
                confidence=0.9,
                detail="rotate",
            ),
        )
        llm = _llm_only(
            LLMRepairSuggestion(
                replacement_source="rotate270(input)",
                confidence=0.5,
                reasoning="other rotation",
            ),
        )
        scorer = _scorer_from_map(
            {
                "rotate90(input)": 0.6,
                "rotate270(input)": 0.95,  # LLM wins
            }
        )

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=llm,
            scorer=scorer,
        )

        assert isinstance(result, HybridRepairResult)
        assert result.winner is not None
        assert result.winner.source == "rotate270(input)"
        assert result.winner.origin == "llm"
        assert result.winner_score == 0.95
        # Both candidates were evaluated.
        assert len(result.candidates_evaluated) == 2

    @pytest.mark.asyncio
    async def test_threshold_filters_below_baseline(self) -> None:
        sym = _symbolic_only(
            RepairSuggestion(
                kind="color_repair",
                primitive_hint="recolor",
                confidence=0.8,
                detail="",
            ),
        )
        llm = _llm_only(
            LLMRepairSuggestion(
                replacement_source="rotate90(input)",
                confidence=0.7,
                reasoning="",
            ),
        )
        # All candidates score below 0.5 — nothing beats the threshold.
        scorer = _scorer_from_map({"recolor(input)": 0.3, "rotate90(input)": 0.4})

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=llm,
            scorer=scorer,
            threshold=0.5,
        )
        assert result.winner is None
        # But the candidates were still evaluated and recorded.
        assert {c.source for c in result.candidates_evaluated} == {
            "recolor(input)",
            "rotate90(input)",
        }

    @pytest.mark.asyncio
    async def test_symbolic_alone_wins_when_llm_empty(self) -> None:
        sym = _symbolic_only(
            RepairSuggestion(
                kind="rotation_repair",
                primitive_hint="rotate90",
                confidence=0.9,
                detail="",
            ),
        )
        # Empty LLMRepairResult.
        llm = _llm_only()
        scorer = _scorer_from_map({"rotate90(input)": 0.7})

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=llm,
            scorer=scorer,
        )
        assert result.winner is not None
        assert result.winner.origin == "symbolic"

    @pytest.mark.asyncio
    async def test_llm_alone_wins_when_symbolic_returns_nothing(self) -> None:
        sym = _symbolic_only()  # zero suggestions
        llm = _llm_only(
            LLMRepairSuggestion(
                replacement_source="recolor(input, 1, 5)",
                confidence=0.7,
                reasoning="",
            )
        )
        scorer = _scorer_from_map({"recolor(input, 1, 5)": 0.65})

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=llm,
            scorer=scorer,
        )
        assert result.winner is not None
        assert result.winner.origin == "llm"

    @pytest.mark.asyncio
    async def test_local_repair_suggestion_with_no_hint_dropped(self) -> None:
        # R5 local-edit suggestion has primitive_hint=None — orchestrator
        # drops it because the default lifter can't make a source.
        sym = _symbolic_only(
            RepairSuggestion(
                kind="local_repair",
                primitive_hint=None,
                confidence=0.7,
                detail="1 pixel diff",
            ),
            RepairSuggestion(
                kind="rotation_repair",
                primitive_hint="rotate90",
                confidence=0.9,
                detail="",
            ),
        )
        llm = _llm_only()
        scorer = _scorer_from_map({"rotate90(input)": 0.5})

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=llm,
            scorer=scorer,
        )
        # Only one candidate makes it through (the rotation one).
        assert len(result.candidates_evaluated) == 1
        assert result.candidates_evaluated[0].source == "rotate90(input)"


# ---------------------------------------------------------------------------
# Failure isolation — neither side can sink the other
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_symbolic_exception_records_tag_and_uses_llm(self) -> None:
        def bad_sym() -> list[RepairSuggestion]:
            raise RuntimeError("boom")

        llm = _llm_only(
            LLMRepairSuggestion(
                replacement_source="rotate90(input)",
                confidence=0.6,
                reasoning="",
            )
        )
        scorer = _scorer_from_map({"rotate90(input)": 0.55})

        result = await run_hybrid_repair(
            symbolic_supplier=bad_sym,
            llm_supplier=llm,
            scorer=scorer,
        )
        # LLM still wins.
        assert result.winner is not None
        assert result.winner.origin == "llm"
        assert result.symbolic_failed == "RuntimeError"
        assert result.llm_failed is None

    @pytest.mark.asyncio
    async def test_llm_exception_records_tag_and_uses_symbolic(self) -> None:
        sym = _symbolic_only(
            RepairSuggestion(
                kind="rotation_repair",
                primitive_hint="rotate90",
                confidence=0.9,
                detail="",
            )
        )

        async def bad_llm() -> LLMRepairResult:
            raise TimeoutError("vllm down")

        scorer = _scorer_from_map({"rotate90(input)": 0.75})

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=bad_llm,
            scorer=scorer,
        )
        assert result.winner is not None
        assert result.winner.origin == "symbolic"
        assert result.llm_failed == "TimeoutError"
        assert result.symbolic_failed is None

    @pytest.mark.asyncio
    async def test_both_backends_fail_returns_none_winner(self) -> None:
        def bad_sym() -> list[RepairSuggestion]:
            raise ValueError("nope")

        async def bad_llm() -> LLMRepairResult:
            raise TimeoutError("nope")

        result = await run_hybrid_repair(
            symbolic_supplier=bad_sym,
            llm_supplier=bad_llm,
            scorer=_scorer_from_map({}),
        )
        assert result.winner is None
        assert result.symbolic_failed == "ValueError"
        assert result.llm_failed == "TimeoutError"

    @pytest.mark.asyncio
    async def test_scorer_exception_skips_candidate(self) -> None:
        sym = _symbolic_only(
            RepairSuggestion(
                kind="rotation_repair",
                primitive_hint="rotate90",
                confidence=0.9,
                detail="",
            ),
            RepairSuggestion(
                kind="color_repair",
                primitive_hint="recolor",
                confidence=0.8,
                detail="",
            ),
        )

        async def picky_scorer(candidate: HybridRepairCandidate) -> float:
            if candidate.source == "rotate90(input)":
                raise ValueError("can't parse this source")
            return 0.6

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=_llm_only(),
            scorer=picky_scorer,
        )
        # Only recolor survived; it's the winner.
        assert result.winner is not None
        assert result.winner.source == "recolor(input)"


# ---------------------------------------------------------------------------
# Parallel execution — symbolic and LLM run concurrently
# ---------------------------------------------------------------------------


class TestParallelism:
    @pytest.mark.asyncio
    async def test_runs_backends_concurrently(self) -> None:
        # Symbolic supplier sleeps 50 ms, LLM sleeps 50 ms. Total wall-
        # clock should be ~50 ms (parallel), not ~100 ms (sequential).
        order: list[str] = []

        def slow_sym() -> list[RepairSuggestion]:
            import time

            time.sleep(0.05)
            order.append("sym_done")
            return [
                RepairSuggestion(
                    kind="rotation_repair",
                    primitive_hint="rotate90",
                    confidence=0.5,
                    detail="",
                )
            ]

        async def slow_llm() -> LLMRepairResult:
            await asyncio.sleep(0.05)
            order.append("llm_done")
            return LLMRepairResult(
                suggestions=(
                    LLMRepairSuggestion(replacement_source="rotate270(input)", confidence=0.5),
                )
            )

        scorer = _scorer_from_map({"rotate90(input)": 0.5, "rotate270(input)": 0.5})

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        result = await run_hybrid_repair(
            symbolic_supplier=slow_sym,
            llm_supplier=slow_llm,
            scorer=scorer,
        )
        elapsed = loop.time() - t0

        # Wall-clock < 0.09 s (well under sequential 0.1 s, with margin).
        assert elapsed < 0.09, f"hybrid was sequential? elapsed={elapsed:.3f}s"
        # Both candidates evaluated.
        assert {c.origin for c in result.candidates_evaluated} == {"symbolic", "llm"}


# ---------------------------------------------------------------------------
# Custom symbolic_to_source override
# ---------------------------------------------------------------------------


class TestSymbolicToSourceOverride:
    @pytest.mark.asyncio
    async def test_caller_can_supply_richer_lifter(self) -> None:
        sym = _symbolic_only(
            RepairSuggestion(
                kind="color_repair",
                primitive_hint="recolor",
                confidence=0.8,
                detail="",
            )
        )
        llm = _llm_only()

        def my_lifter(sug: RepairSuggestion) -> str:
            # Always lift recolor with placeholder color args.
            assert sug.primitive_hint == "recolor"
            return "recolor(input, 1, 5)"

        scorer = _scorer_from_map({"recolor(input, 1, 5)": 0.7})

        result = await run_hybrid_repair(
            symbolic_supplier=sym,
            llm_supplier=llm,
            scorer=scorer,
            symbolic_to_source=my_lifter,
        )
        assert result.winner is not None
        assert result.winner.source == "recolor(input, 1, 5)"


# ---------------------------------------------------------------------------
# Dataclass contract
# ---------------------------------------------------------------------------


class TestDataclassesAreFrozen:
    def test_candidate_is_hashable(self) -> None:
        c = HybridRepairCandidate(
            source="rotate90(input)",
            confidence=0.5,
            origin="symbolic",
        )
        assert hash(c) == hash(c)

    def test_result_is_hashable(self) -> None:
        r = HybridRepairResult(
            winner=None,
            winner_score=0.0,
            candidates_evaluated=(),
        )
        assert hash(r) == hash(r)
