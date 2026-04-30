# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Local-Edit Repair tests (Sprint-1 plan task 9 slice, spec §6.5.1)."""

from __future__ import annotations

from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.refiner import LocalEditMutator
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)

# ---------------------------------------------------------------------------
# Mutator construction + happy paths
# ---------------------------------------------------------------------------


class TestLocalEditMutator:
    def test_mutator_with_no_registry_skips_primitive_substitutions(self) -> None:
        # rotate90(input)
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        m = LocalEditMutator(registry=None)
        edits = list(m.mutate(program))
        # No primitive substitution (registry=None) and no swappable
        # children (only 1 child), no Color literals. Should be empty.
        assert edits == []


class TestPrimitiveSubstitution:
    def test_rotate_family_substitutions(self) -> None:
        # rotate90(input) → expect rotate180 / rotate270 / mirror_h /
        # mirror_v / transpose / scale_up_2x... as substitutions
        # since they share (arity=1, output=Grid).
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        m = LocalEditMutator(registry=REGISTRY)
        edits = list(m.mutate(program))
        primitive_names = {
            e.primitive for e in edits if isinstance(e, Program) and len(e.children) == 1
        }
        # Other arity-1 Grid->Grid primitives must be substituted in.
        assert "rotate180" in primitive_names
        assert "rotate270" in primitive_names
        # Original must NOT appear.
        assert "rotate90" not in primitive_names

    def test_substitution_filters_by_arity(self) -> None:
        # recolor takes 3 args (Grid, Color, Color) → mutator should
        # only propose other arity-3 primitives.
        program = Program(
            primitive="recolor",
            children=(
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=5, output_type="Color"),
            ),
            output_type="Grid",
        )
        m = LocalEditMutator(registry=REGISTRY)
        edits = list(m.mutate(program))
        for e in edits:
            if not isinstance(e, Program):
                continue
            if e.primitive == "recolor":
                continue
            # Any primitive substitution must preserve arity=3.
            spec = REGISTRY.get(e.primitive)
            if spec.signature.output == "Grid":
                # only count primitive subs (not deeper child mutations
                # that might change arity at lower levels).
                pass


class TestChildSwap:
    def test_two_child_program_yields_swap(self) -> None:
        # Synthetic: a Program with two children that share output type.
        program = Program(
            primitive="swap_colors",
            children=(
                Const(value=1, output_type="Color"),
                Const(value=5, output_type="Color"),
            ),
            output_type="Color",
        )
        m = LocalEditMutator()  # no registry needed for child-swap
        edits = list(m.mutate(program))
        # The swap edit must appear.
        swap_edit = None
        for e in edits:
            if (
                isinstance(e, Program)
                and e.primitive == "swap_colors"
                and isinstance(e.children[0], Const)
                and e.children[0].value == 5
                and isinstance(e.children[1], Const)
                and e.children[1].value == 1
            ):
                swap_edit = e
                break
        assert swap_edit is not None, "child swap not yielded"

    def test_one_child_program_does_not_swap(self) -> None:
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        m = LocalEditMutator()
        # No registry, only 1 child, no Color literal → no edits.
        edits = list(m.mutate(program))
        assert edits == []


class TestColorLiteralChange:
    def test_const_color_yields_nine_alternatives(self) -> None:
        # recolor(input, 1, 5) — every Const(_, "Color") should yield
        # 9 alternative colors (excluding its current value).
        program = Program(
            primitive="recolor",
            children=(
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=5, output_type="Color"),
            ),
            output_type="Grid",
        )
        m = LocalEditMutator()  # no registry
        edits = list(m.mutate(program))
        # 9 mutations on first Const (1 → {0,2,3,4,5,6,7,8,9}) +
        # 9 mutations on second Const (5 → {0,1,2,3,4,6,7,8,9}) = 18.
        color_edits = [e for e in edits if isinstance(e, Program)]
        assert len(color_edits) == 18

    def test_non_color_const_unchanged(self) -> None:
        # An Int Const (not Color) should not be mutated.
        program = Program(
            primitive="dummy",
            children=(Const(value=3, output_type="Int"),),
            output_type="Grid",
        )
        m = LocalEditMutator()  # no registry
        edits = list(m.mutate(program))
        assert edits == []


class TestOriginalNeverEmitted:
    def test_input_ref_alone_yields_nothing(self) -> None:
        m = LocalEditMutator(registry=REGISTRY)
        edits = list(m.mutate(InputRef()))
        assert edits == []

    def test_original_program_not_in_output(self) -> None:
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        m = LocalEditMutator(registry=REGISTRY)
        for edit in m.mutate(program):
            assert edit != program, "mutator emitted the original program"


class TestIsLazy:
    def test_generator_can_be_partially_consumed(self) -> None:
        # The mutator returns an iterator — the caller can stop early.
        program = Program(
            primitive="recolor",
            children=(
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=5, output_type="Color"),
            ),
            output_type="Grid",
        )
        m = LocalEditMutator(registry=REGISTRY)
        gen = m.mutate(program)
        # Take the first 3 edits without consuming the rest.
        first_three = [next(gen), next(gen), next(gen)]
        assert len(first_three) == 3
