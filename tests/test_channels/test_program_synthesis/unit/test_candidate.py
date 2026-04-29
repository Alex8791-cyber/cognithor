# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Search candidate-tree tests (spec §6.2)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)


class TestInputRef:
    def test_default_output_type_grid(self) -> None:
        ref = InputRef()
        assert ref.output_type == "Grid"

    def test_to_source(self) -> None:
        assert InputRef().to_source() == "input"

    def test_depth_size(self) -> None:
        ref = InputRef()
        assert ref.depth() == 0
        assert ref.size() == 1

    def test_frozen(self) -> None:
        ref = InputRef()
        with pytest.raises(FrozenInstanceError):
            ref.output_type = "Color"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert InputRef() == InputRef()
        assert InputRef() != InputRef(output_type="Color")


class TestConst:
    def test_int_to_source(self) -> None:
        assert Const(value=5, output_type="Color").to_source() == "5"

    def test_str_to_source(self) -> None:
        # Strings keep their quotes so source round-trips through eval().
        assert Const(value="center", output_type="AlignMode").to_source() == "'center'"

    def test_depth_size(self) -> None:
        c = Const(value=0, output_type="Color")
        assert c.depth() == 0
        assert c.size() == 1

    def test_frozen(self) -> None:
        c = Const(value=1, output_type="Int")
        with pytest.raises(FrozenInstanceError):
            c.value = 2  # type: ignore[misc]

    def test_equality_is_structural(self) -> None:
        a = Const(value=3, output_type="Color")
        b = Const(value=3, output_type="Color")
        assert a == b
        assert hash(a) == hash(b)


class TestProgram:
    def test_to_source_unary(self) -> None:
        p = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        assert p.to_source() == "rotate90(input)"

    def test_to_source_ternary_with_consts(self) -> None:
        p = Program(
            primitive="recolor",
            children=(
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=2, output_type="Color"),
            ),
            output_type="Grid",
        )
        assert p.to_source() == "recolor(input, 1, 2)"

    def test_to_source_nested(self) -> None:
        # mirror_horizontal(rotate90(input))
        inner = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        outer = Program(
            primitive="mirror_horizontal",
            children=(inner,),
            output_type="Grid",
        )
        assert outer.to_source() == "mirror_horizontal(rotate90(input))"

    def test_depth_for_leaf_primitive(self) -> None:
        # rotate90(input) has depth 1: one application above a depth-0 leaf.
        p = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        assert p.depth() == 1

    def test_depth_for_nested(self) -> None:
        inner = Program("rotate90", (InputRef(),), "Grid")
        middle = Program("mirror_horizontal", (inner,), "Grid")
        outer = Program("transpose", (middle,), "Grid")
        assert outer.depth() == 3

    def test_size_counts_all_nodes(self) -> None:
        # recolor(input, 1, 2): 1 program node + 1 InputRef + 2 Consts = 4
        p = Program(
            primitive="recolor",
            children=(
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=2, output_type="Color"),
            ),
            output_type="Grid",
        )
        assert p.size() == 4

    def test_zero_arity_primitive(self) -> None:
        p = Program(
            primitive="const_color_5",
            children=(),
            output_type="Color",
        )
        assert p.to_source() == "const_color_5()"
        assert p.depth() == 1
        assert p.size() == 1

    def test_stable_hash_deterministic(self) -> None:
        a = Program("rotate90", (InputRef(),), "Grid")
        b = Program("rotate90", (InputRef(),), "Grid")
        assert a.stable_hash() == b.stable_hash()
        assert a.stable_hash().startswith("sha256:")
        assert len(a.stable_hash().split(":")[1]) == 64

    def test_stable_hash_changes_on_primitive(self) -> None:
        a = Program("rotate90", (InputRef(),), "Grid")
        b = Program("rotate180", (InputRef(),), "Grid")
        assert a.stable_hash() != b.stable_hash()

    def test_stable_hash_changes_on_const_value(self) -> None:
        a = Program(
            "recolor",
            (InputRef(), Const(1, "Color"), Const(2, "Color")),
            "Grid",
        )
        b = Program(
            "recolor",
            (InputRef(), Const(1, "Color"), Const(3, "Color")),
            "Grid",
        )
        assert a.stable_hash() != b.stable_hash()

    def test_frozen(self) -> None:
        p = Program("rotate90", (InputRef(),), "Grid")
        with pytest.raises(FrozenInstanceError):
            p.primitive = "rotate180"  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        a = Program("rotate90", (InputRef(),), "Grid")
        b = Program("rotate90", (InputRef(),), "Grid")
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_on_child_order(self) -> None:
        # swap_colors(input, 1, 2) != swap_colors(input, 2, 1)
        a = Program(
            "swap_colors",
            (InputRef(), Const(1, "Color"), Const(2, "Color")),
            "Grid",
        )
        b = Program(
            "swap_colors",
            (InputRef(), Const(2, "Color"), Const(1, "Color")),
            "Grid",
        )
        assert a != b
        assert a.stable_hash() != b.stable_hash()
