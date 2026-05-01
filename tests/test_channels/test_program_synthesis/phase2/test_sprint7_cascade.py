# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-7 Track A1 — Cascade-Generalisation tests."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.search.executor import InProcessExecutor
from cognithor.channels.program_synthesis.synthesis.sprint7_cascade_runner import (
    _parse_args,
    _unary_grid_to_grid_primitives,
    cascade_repair,
    enumerate_recolor_variants,
    enumerate_unary_chains,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Unary primitive enumeration
# ---------------------------------------------------------------------------


class TestUnaryPrimitives:
    def test_returns_at_least_basic_geometry(self) -> None:
        prims = _unary_grid_to_grid_primitives()
        # Sanity — every basic geometric op must be unary Grid→Grid.
        for expected in ("rotate90", "rotate180", "rotate270", "transpose", "identity"):
            assert expected in prims

    def test_no_higher_arity_primitives(self) -> None:
        from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

        prims = _unary_grid_to_grid_primitives()
        for name in prims:
            spec = REGISTRY.get(name)
            assert spec.signature.arity == 1
            assert spec.signature.inputs == ("Grid",)
            assert spec.signature.output == "Grid"


# ---------------------------------------------------------------------------
# Chain enumeration
# ---------------------------------------------------------------------------


class TestEnumerateUnaryChains:
    def test_depth_one_yields_n_chains(self) -> None:
        prims = ["rotate90", "rotate180"]
        chains = list(enumerate_unary_chains(prims, max_depth=1))
        assert len(chains) == 2

    def test_depth_two_yields_n_squared_chains(self) -> None:
        prims = ["rotate90", "rotate180"]
        chains = list(enumerate_unary_chains(prims, max_depth=2))
        # depth-1: 2, depth-2: 4 → total 6.
        assert len(chains) == 6

    def test_chains_compose_correctly(self) -> None:
        prims = ["rotate90", "rotate180"]
        chains = list(enumerate_unary_chains(prims, base=InputRef(), max_depth=2))
        # The depth-2 chain rotate180(rotate90(input)) should be produced.
        sources = {c.to_source() for c in chains}
        assert "rotate90(input)" in sources
        assert "rotate180(input)" in sources
        assert "rotate180(rotate90(input))" in sources

    def test_base_is_used(self) -> None:
        # When we pass a custom base, the chain wraps it.
        base = Program("transpose", (InputRef(),), "Grid")
        chains = list(enumerate_unary_chains(["rotate90"], base=base, max_depth=1))
        assert len(chains) == 1
        assert chains[0].to_source() == "rotate90(transpose(input))"


# ---------------------------------------------------------------------------
# Recolor variants
# ---------------------------------------------------------------------------


class TestEnumerateRecolorVariants:
    def test_yields_no_recolor_first(self) -> None:
        chain = Program("rotate90", (InputRef(),), "Grid")
        variants = list(enumerate_recolor_variants(chain, [1, 2], [3]))
        assert variants[0] == chain

    def test_one_step_wrap(self) -> None:
        chain = Program("rotate90", (InputRef(),), "Grid")
        variants = list(enumerate_recolor_variants(chain, [1, 2], [3]))
        sources = {v.to_source() for v in variants}
        assert "recolor(rotate90(input), 1, 3)" in sources
        assert "recolor(rotate90(input), 2, 3)" in sources

    def test_two_step_cascade(self) -> None:
        chain = Program("identity", (InputRef(),), "Grid")
        variants = list(enumerate_recolor_variants(chain, [1, 2], [0], max_recolor_depth=2))
        sources = {v.to_source() for v in variants}
        # Cascade: recolor(recolor(identity(input), 1, 0), 2, 0)
        cascade_src = "recolor(recolor(identity(input), 1, 0), 2, 0)"
        assert cascade_src in sources

    def test_max_recolor_depth_one_no_cascade(self) -> None:
        chain = Program("identity", (InputRef(),), "Grid")
        variants = list(enumerate_recolor_variants(chain, [1, 2], [0], max_recolor_depth=1))
        # No 2-step cascade in the variants when max=1.
        sources = {v.to_source() for v in variants}
        assert all("recolor(recolor" not in s for s in sources)


# ---------------------------------------------------------------------------
# Cascade-repair end-to-end
# ---------------------------------------------------------------------------


class TestCascadeRepair:
    def test_already_winning_unchanged(self) -> None:
        program = Program("identity", (InputRef(),), "Grid")
        demos = [(_g([[1, 2]]), _g([[1, 2]]))]
        prog, score, refined = cascade_repair(program, demos, InProcessExecutor())
        assert score == 1.0
        assert refined is False

    def test_solves_2step_recolor_cascade(self) -> None:
        # Same as the 0202 sub-problem: remove colors {2, 3}.
        demos = [
            (
                _g([[1, 1, 0, 2], [1, 1, 0, 0], [0, 0, 0, 3]]),
                _g([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
            ),
            (_g([[5, 0, 0], [5, 5, 0], [5, 0, 0]]), _g([[5, 0, 0], [5, 5, 0], [5, 0, 0]])),
        ]
        # Phase-1's wrong base.
        phase1 = Program("mirror_vertical", (InputRef(),), "Grid")
        prog, score, refined = cascade_repair(phase1, demos, InProcessExecutor(), max_depth=2)
        assert score == 1.0
        assert refined is True

    def test_solves_3step_unary_chain(self) -> None:
        # Synthetic: rotate90 + mirror_horizontal + recolor 1->9.
        # Set up a single-demo case mirroring task 0208's first demo.
        demo = (_g([[1, 0], [0, 1]]), _g([[9, 0], [0, 9]]))
        # On a square symmetric input, rotate90 + mirror_horizontal
        # is identity, so this reduces to recolor 1->9.
        prog, score, refined = cascade_repair(InputRef(), [demo], InProcessExecutor(), max_depth=3)
        assert score == 1.0
        # Could be refined or not depending on enumeration order.

    def test_no_demos_returns_baseline(self) -> None:
        program = Program("identity", (InputRef(),), "Grid")
        prog, score, refined = cascade_repair(program, [], InProcessExecutor())
        assert score == 0.0
        assert refined is False


# ---------------------------------------------------------------------------
# Argparse contract
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_default_subset_is_hard(self) -> None:
        args = _parse_args([])
        assert args.subset == "hard"

    def test_default_chain_depth_is_three(self) -> None:
        args = _parse_args([])
        assert args.max_chain_depth == 3

    def test_chain_depth_overridable(self) -> None:
        args = _parse_args(["--max-chain-depth", "5"])
        assert args.max_chain_depth == 5
