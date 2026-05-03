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
        with pytest.raises(ValueError, match="example 1"):
            _examples_from_json(
                [
                    {"input": [[0]], "output": [[1]]},
                    {"input": "not a grid", "output": [[3]]},
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


class TestRegisterPseTools:
    def test_registers_three_tools(self) -> None:
        registered: list[dict[str, Any]] = []

        class _StubMCP:
            def register_tool(self, **kwargs: Any) -> None:
                registered.append(kwargs)

        register_pse_tools(_StubMCP())
        names = {r["name"] for r in registered}
        assert names == {"pse_is_synthesizable", "pse_status", "pse_synthesize"}

    def test_missing_register_tool_no_crash(self) -> None:
        # MCP clients without a register_tool callable must NOT crash the
        # gateway's tool-phase boot — log + return cleanly.
        class _BareClient:
            pass

        register_pse_tools(_BareClient())  # would raise AttributeError if buggy
