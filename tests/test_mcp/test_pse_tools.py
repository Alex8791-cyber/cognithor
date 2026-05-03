# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 Track A — PSE MCP tool surface tests."""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.mcp.pse_tools import (
    _coerce_grid,
    _examples_from_json,
    handle_pse_is_synthesizable,
    handle_pse_status,
    handle_pse_synthesize,
    register_pse_tools,
)


class TestCoerceGrid:
    def test_valid_2d_int_grid_returns_int8_array(self) -> None:
        import numpy as np

        out = _coerce_grid([[0, 1, 2], [3, 4, 5]])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int8
        assert out.shape == (2, 3)

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _coerce_grid([])

    def test_empty_row_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty rows"):
            _coerce_grid([[]])

    def test_non_uniform_width_rejected(self) -> None:
        with pytest.raises(ValueError, match="uniform width"):
            _coerce_grid([[0, 1], [0, 1, 2]])

    def test_non_int_cell_rejected(self) -> None:
        with pytest.raises(ValueError, match="cells must be ints"):
            _coerce_grid([[0, "a"]])


class TestExamplesFromJson:
    def test_two_valid_examples_parse(self) -> None:
        out = _examples_from_json(
            [
                {"input": [[0]], "output": [[1]]},
                {"input": [[2]], "output": [[3]]},
            ]
        )
        assert len(out) == 2

    def test_non_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            _examples_from_json({"input": [], "output": []})

    def test_single_example_rejected_too_under_specified(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _examples_from_json([{"input": [[0]], "output": [[1]]}])

    def test_missing_input_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs both 'input' and 'output'"):
            _examples_from_json(
                [
                    {"output": [[1]]},
                    {"input": [[2]], "output": [[3]]},
                ]
            )

    def test_invalid_inner_grid_propagates_with_index(self) -> None:
        # Sprint-22: ``"not a grid"`` is now a *valid* string input, so
        # the coercion no longer rejects it per-example. Instead the
        # mixed grid+string payload triggers the homogeneity check.
        # PR#3: error message switched from "homogeneous" to the more
        # precise "must not mix grids with text-shaped values".
        with pytest.raises(ValueError, match="must not mix grids"):
            _examples_from_json(
                [
                    {"input": [[0]], "output": [[1]]},
                    {"input": "not a grid", "output": [[3]]},
                ]
            )

    def test_truly_invalid_value_still_propagates_per_example_with_index(self) -> None:
        """A payload that is neither a grid (2-D int list) nor a string
        nor an int is rejected by ``_coerce_value`` per-example, with
        the index in the error so the caller can locate the bad row.
        """
        # PR#3: ``int`` is now a valid input type via the Number-DSL
        # family, so the previous "42" test value is no longer rejected.
        # ``None`` and ``float`` are still genuinely unsupported.
        with pytest.raises(ValueError, match="example 1"):
            _examples_from_json(
                [
                    {"input": "abc", "output": "abc"},
                    {"input": None, "output": "x"},  # None is not a grid/str/int
                ]
            )


class TestHandleIsSynthesizable:
    @pytest.mark.asyncio
    async def test_routable_examples_return_yes(self) -> None:
        # Two-example, every example has both keys, first input is grid → yes.
        out = await handle_pse_is_synthesizable(
            examples=[
                {"input": [[0]], "output": [[1]]},
                {"input": [[2]], "output": [[3]]},
            ]
        )
        assert out == "yes"

    @pytest.mark.asyncio
    async def test_single_example_returns_no(self) -> None:
        out = await handle_pse_is_synthesizable(
            examples=[{"input": [[0]], "output": [[1]]}],
        )
        assert out == "no"

    @pytest.mark.asyncio
    async def test_missing_examples_returns_error(self) -> None:
        out = await handle_pse_is_synthesizable()
        assert out.startswith("Error:")


class TestHandleStatus:
    @pytest.mark.asyncio
    async def test_returns_engine_metadata(self) -> None:
        out = await handle_pse_status()
        # Format: "PSE_VERSION=..., DSL_VERSION=..., primitives=..."
        assert "PSE_VERSION=" in out
        assert "DSL_VERSION=" in out
        assert "primitives=" in out


class TestHandleSynthesize:
    @pytest.mark.asyncio
    async def test_missing_examples_returns_error_dict(self) -> None:
        out = await handle_pse_synthesize()
        assert "error" in out
        assert "examples" in out["error"]

    @pytest.mark.asyncio
    async def test_invalid_examples_propagate_error(self) -> None:
        out = await handle_pse_synthesize(examples="not a list")
        assert "error" in out
        assert "must be a list" in out["error"]

    @pytest.mark.asyncio
    async def test_invalid_budget_returns_error(self) -> None:
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[1]]},
                {"input": [[2]], "output": [[3]]},
            ],
            budget="not a dict",
        )
        assert "error" in out
        assert "budget" in out["error"]

    @pytest.mark.asyncio
    async def test_happy_path_returns_structured_result(self) -> None:
        # Simple 1×1 identity-grid examples — engine may not solve them
        # but the call MUST return a structured dict with a status key
        # (success / no_solution / partial / etc.) instead of crashing.
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
            ],
            budget={"max_depth": 2, "wall_clock_seconds": 5.0},
        )
        assert "status" in out
        assert "cost_seconds" in out
        assert isinstance(out["cost_seconds"], float)
        assert out["status"] in {
            "success",
            "partial",
            "no_solution",
            "timeout",
            "budget",
            "sandbox",
            "error",
        }
        # Sprint-22 A.4 hardening: response includes the new fields.
        assert "held_out_examples" in out
        assert "escalations" in out

    @pytest.mark.asyncio
    async def test_auto_held_out_promotes_last_example_when_three_or_more(self) -> None:
        """Sprint-22 A.4: ≥3 examples + auto_held_out=True (default) →
        last example is auto-split into held_out so the verifier has an
        anti-overfit gate.
        """
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
            ],
            budget={"max_depth": 2, "wall_clock_seconds": 5.0},
        )
        assert out["held_out_examples"] == 1

    @pytest.mark.asyncio
    async def test_auto_held_out_disabled_via_flag(self) -> None:
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
            ],
            auto_held_out=False,
            budget={"max_depth": 2, "wall_clock_seconds": 5.0},
        )
        assert out["held_out_examples"] == 0

    @pytest.mark.asyncio
    async def test_auto_held_out_skipped_below_three_examples(self) -> None:
        """Two examples is the engine's minimum demo set — splitting
        one off would break it. Auto-promotion must skip in that case.
        """
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
            ],
            budget={"max_depth": 2, "wall_clock_seconds": 5.0},
        )
        assert out["held_out_examples"] == 0

    @pytest.mark.asyncio
    async def test_explicit_held_out_overrides_auto_split(self) -> None:
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
            ],
            held_out=[
                {"input": [[5]], "output": [[5]]},
                {"input": [[6]], "output": [[6]]},
            ],
            budget={"max_depth": 2, "wall_clock_seconds": 5.0},
        )
        assert out["held_out_examples"] == 2  # uses explicit, not auto-split

    @pytest.mark.asyncio
    async def test_invalid_held_out_propagates_error(self) -> None:
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
            ],
            held_out=[{"only": "input missing"}, {"only": "input missing"}],
        )
        assert "error" in out
        assert "held_out" in out["error"]

    @pytest.mark.asyncio
    async def test_auto_escalate_default_off(self) -> None:
        """Sprint-22 A.5: ``auto_escalate`` is False by default — a single
        attempt runs even when budget exhausts.
        """
        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
            ],
            budget={"max_depth": 2, "wall_clock_seconds": 5.0},
        )
        assert out["escalations"] == 0

    @pytest.mark.asyncio
    async def test_auto_escalate_when_first_attempt_busts_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sprint-22 A.5: when ``auto_escalate=True`` AND the first
        attempt comes back BUDGET_EXCEEDED, the handler retries once
        with depth+1 and 2× the candidate cap.
        """
        from cognithor.channels.program_synthesis.core.types import (
            SynthesisResult,
            SynthesisStatus,
        )
        from cognithor.mcp import pse_tools

        attempts: list[dict[str, Any]] = []

        class _StubChannel:
            def synthesize(self, request: Any) -> SynthesisResult:
                attempts.append(
                    {
                        "max_depth": request.budget.max_depth,
                        "max_candidates": request.budget.max_candidates,
                    }
                )
                if len(attempts) == 1:
                    return SynthesisResult(
                        status=SynthesisStatus.BUDGET_EXCEEDED,
                        program=None,
                        score=0.0,
                        confidence=0.0,
                        cost_seconds=0.05,
                        cost_candidates=10,
                    )
                return SynthesisResult(
                    status=SynthesisStatus.SUCCESS,
                    program="<stub>",
                    score=1.0,
                    confidence=1.0,
                    cost_seconds=0.05,
                    cost_candidates=10,
                )

        monkeypatch.setattr(pse_tools, "_get_channel", lambda: _StubChannel())

        out = await handle_pse_synthesize(
            examples=[
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
            ],
            budget={
                "max_depth": 4,
                "max_candidates": 1000,
                "wall_clock_seconds": 5.0,
                "auto_escalate": True,
            },
        )
        # Two attempts: first busted, second succeeded with escalated budget.
        assert len(attempts) == 2
        assert attempts[1]["max_depth"] == 5  # depth + 1
        assert attempts[1]["max_candidates"] == 2000  # candidates × 2
        assert out["status"] == "success"
        assert out["escalations"] == 1
        # Aggregated cost across both attempts:
        assert out["cost_seconds"] == pytest.approx(0.1, rel=1e-6)
        assert out["cost_candidates"] == 20


class TestRegisterPseTools:
    def test_registers_all_four_tools(self) -> None:
        # Sprint-25 added ``pse_synthesize_refined``. The set is the
        # contract the gateway relies on.
        registered: list[dict[str, Any]] = []

        class _StubMCP:
            def register_tool(self, **kwargs: Any) -> None:
                registered.append(kwargs)

        register_pse_tools(_StubMCP())
        names = {r["name"] for r in registered}
        assert names == {
            "pse_is_synthesizable",
            "pse_status",
            "pse_synthesize",
            "pse_synthesize_refined",
        }

    def test_missing_register_tool_no_crash(self) -> None:
        # MCP clients without a register_tool callable must NOT crash the
        # gateway's tool-phase boot — log + return cleanly.
        class _BareClient:
            pass

        register_pse_tools(_BareClient())  # would raise AttributeError if buggy

    def test_llm_fn_is_stored_for_refinement(self) -> None:
        # Sprint-25: when the gateway passes an llm_fn, the module-level
        # holder picks it up so the tool handler can use it without
        # threading the callable through every MCP register call.
        from cognithor.mcp import pse_tools

        async def _fake_llm(_p: str) -> str:
            return ""

        class _StubMCP:
            def register_tool(self, **kwargs: Any) -> None:
                pass

        register_pse_tools(_StubMCP(), llm_fn=_fake_llm)
        assert pse_tools._refinement_llm_fn is _fake_llm

        # And clears back to None on subsequent registration without one.
        register_pse_tools(_StubMCP())
        assert pse_tools._refinement_llm_fn is None
