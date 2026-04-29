# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verifier pipeline tests (spec §10)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.core.types import TaskSpec
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.verify import (
    DEFAULT_PROPERTIES,
    DemoStage,
    HeldOutStage,
    PropertyStage,
    SyntaxStage,
    TypeStage,
    Verifier,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Stage 1: Syntax
# ---------------------------------------------------------------------------


class TestSyntaxStage:
    def test_well_formed_program_passes(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(examples=((_g([[1]]), _g([[1]])),))
        v = Verifier(stages=(SyntaxStage(),))
        result = v.verify(prog, spec)
        assert result.passed
        assert result.stages[0].stage == "syntax"

    def test_unknown_primitive_fails(self) -> None:
        prog = Program("does_not_exist", (InputRef(),), "Grid")
        spec = TaskSpec(examples=((_g([[1]]), _g([[1]])),))
        v = Verifier(stages=(SyntaxStage(),))
        result = v.verify(prog, spec)
        assert not result.passed
        assert "unknown primitive" in result.stages[0].detail

    def test_arity_mismatch_fails(self) -> None:
        # rotate90 has arity 1; pass 0 children.
        prog = Program("rotate90", (), "Grid")
        spec = TaskSpec(examples=((_g([[1]]), _g([[1]])),))
        v = Verifier(stages=(SyntaxStage(),))
        result = v.verify(prog, spec)
        assert not result.passed
        assert "arity" in result.stages[0].detail


# ---------------------------------------------------------------------------
# Stage 2: Type
# ---------------------------------------------------------------------------


class TestTypeStage:
    def test_typed_correctly_passes(self) -> None:
        prog = Program(
            "recolor",
            (
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=2, output_type="Color"),
            ),
            "Grid",
        )
        spec = TaskSpec(examples=((_g([[1, 2]]), _g([[2, 2]])),))
        v = Verifier(stages=(TypeStage(),))
        result = v.verify(prog, spec)
        assert result.passed

    def test_wrong_output_type_fails(self) -> None:
        # rotate90 returns Grid, but we declared Color.
        prog = Program("rotate90", (InputRef(),), "Color")
        spec = TaskSpec(examples=((_g([[1]]), _g([[1]])),))
        v = Verifier(stages=(TypeStage(),))
        result = v.verify(prog, spec)
        assert not result.passed
        assert "output" in result.stages[0].detail

    def test_wrong_arg_type_fails(self) -> None:
        # recolor expects (Grid, Color, Color); pass (Grid, Grid, Color).
        prog = Program(
            "recolor",
            (
                InputRef(),
                InputRef(),  # wrong type — Grid where Color expected
                Const(value=2, output_type="Color"),
            ),
            "Grid",
        )
        spec = TaskSpec(examples=((_g([[1]]), _g([[1]])),))
        v = Verifier(stages=(TypeStage(),))
        result = v.verify(prog, spec)
        assert not result.passed
        assert "arg" in result.stages[0].detail


# ---------------------------------------------------------------------------
# Stage 3: Demo
# ---------------------------------------------------------------------------


class TestDemoStage:
    def test_correct_program_passes(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
        )
        v = Verifier(stages=(DemoStage(),))
        result = v.verify(prog, spec)
        assert result.passed

    def test_wrong_program_fails_with_index(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=(
                (_g([[1, 2]]), _g([[1, 2]])),  # demo 0 — rotate90 produces a column
            ),
        )
        v = Verifier(stages=(DemoStage(),))
        result = v.verify(prog, spec)
        assert not result.passed
        assert "demo 0" in result.stages[0].detail

    def test_no_demos_fails(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(examples=())
        v = Verifier(stages=(DemoStage(),))
        result = v.verify(prog, spec)
        assert not result.passed
        assert "no demo" in result.stages[0].detail


# ---------------------------------------------------------------------------
# Stage 4: Property
# ---------------------------------------------------------------------------


class TestPropertyStage:
    def test_well_behaved_program_passes(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
        )
        v = Verifier(stages=(PropertyStage(),))
        result = v.verify(prog, spec)
        assert result.passed
        assert "4 properties verified" in result.stages[0].detail

    def test_dimension_mismatch_fails(self) -> None:
        # rotate90 of 1x3 → 3x1, but expected 1x3 → demo would catch it,
        # however property stage runs independently. Use the property
        # check by constructing a program whose output shape differs
        # from expected.
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=(
                # Input 1x3 → rotate90 → 3x1; but expected = 1x3.
                # PropertyStage's dim-match should catch this.
                (_g([[1, 2, 3]]), _g([[1, 2, 3]])),
            ),
        )
        v = Verifier(stages=(PropertyStage(),))
        result = v.verify(prog, spec)
        assert not result.passed
        assert "dimensions_match" in result.stages[0].detail

    def test_custom_property_set(self) -> None:
        # Property set with a single always-pass property → must pass.
        always_pass = (("ok", lambda a, e, i: (True, "")),)
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1, 2]]), _g([[1], [2]])),),
        )
        v = Verifier(stages=(PropertyStage(properties=always_pass),))
        result = v.verify(prog, spec)
        assert result.passed
        assert "1 properties" in result.stages[0].detail


# ---------------------------------------------------------------------------
# Stage 5: Held-Out (soft)
# ---------------------------------------------------------------------------


class TestHeldOutStage:
    def test_no_held_out_passes(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1]]), _g([[1]])),),
            held_out=(),
        )
        v = Verifier(stages=(HeldOutStage(),))
        result = v.verify(prog, spec)
        assert result.passed
        assert "no held-out" in result.stages[0].detail

    def test_full_held_out_match_passes(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1]]), _g([[1]])),),
            held_out=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
        )
        v = Verifier(stages=(HeldOutStage(),))
        result = v.verify(prog, spec)
        assert result.passed

    def test_partial_held_out_match_fails_soft(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1]]), _g([[1]])),),
            held_out=(
                (_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),  # match
                (_g([[5]]), _g([[9]])),  # no match
            ),
        )
        v = Verifier(stages=(HeldOutStage(),))
        result = v.verify(prog, spec)
        # Stage didn't fully pass — but fail_fast=False so the pipeline
        # would continue. Here we only ran one stage so the overall
        # passed=False.
        assert not result.passed
        assert "1/2" in result.stages[0].detail


# ---------------------------------------------------------------------------
# Full pipeline + confidence calculation
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_correct_program_full_pipeline(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=(
                (_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),
                (_g([[5, 6, 7]]), _g([[5], [6], [7]])),
            ),
            held_out=((_g([[9, 8]]), _g([[9], [8]])),),
        )
        v = Verifier()
        result = v.verify(prog, spec)
        assert result.passed
        assert result.confidence == 1.0
        # 5 stages must all be present.
        assert {s.stage for s in result.stages} == {
            "syntax",
            "type",
            "demo",
            "property",
            "held_out",
        }

    def test_demo_failure_short_circuits_pipeline(self) -> None:
        # rotate90 on input where output != rotate90(input).
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=(
                (_g([[1, 2]]), _g([[7, 7]])),  # bogus expected
            ),
        )
        v = Verifier()
        result = v.verify(prog, spec)
        assert not result.passed
        assert result.confidence == 0.0
        # syntax + type passed, demo failed, property + held_out skipped.
        stage_names = [s.stage for s in result.stages]
        assert "syntax" in stage_names
        assert "type" in stage_names
        assert "demo" in stage_names
        assert "property" not in stage_names
        assert "held_out" not in stage_names

    def test_partial_held_out_lowers_confidence(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
            held_out=(
                (_g([[5, 6]]), _g([[5], [6]])),  # passes
                (_g([[7, 8]]), _g([[9], [9]])),  # fails
            ),
        )
        v = Verifier()
        result = v.verify(prog, spec)
        # Demo + property passed, held_out partially failed (1/2).
        # passed_overall is False because one stage didn't pass, but
        # fail_fast on held_out is False so the pipeline ran to the end.
        assert not result.passed  # held_out stage was the soft-fail
        # Soft failure → confidence is 0 because the pipeline returned
        # passed_overall=False (any non-passing stage forces 0).
        assert result.confidence == 0.0

    def test_no_held_out_confidence_one(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        spec = TaskSpec(
            examples=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
            held_out=(),
        )
        v = Verifier()
        result = v.verify(prog, spec)
        assert result.passed
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Properties module
# ---------------------------------------------------------------------------


class TestProperties:
    def test_default_properties_count(self) -> None:
        # Spec §10.3 lists 4 Phase-1 properties.
        assert len(DEFAULT_PROPERTIES) == 4

    def test_default_properties_are_callable(self) -> None:
        for name, fn in DEFAULT_PROPERTIES:
            assert isinstance(name, str)
            assert callable(fn)

    def test_grid_nonempty_rejects_empty(self) -> None:
        from cognithor.channels.program_synthesis.verify.properties import (
            output_grid_nonempty,
        )

        empty = np.zeros((0, 0), dtype=np.int8)
        ok, _ = output_grid_nonempty(empty, _g([[1]]), _g([[1]]))
        assert not ok

    def test_no_negative_rejects_negative(self) -> None:
        from cognithor.channels.program_synthesis.verify.properties import (
            no_nan_no_negative,
        )

        bad = np.array([[-1, 0]], dtype=np.int8)
        ok, _ = no_nan_no_negative(bad, _g([[0, 0]]), _g([[0, 0]]))
        assert not ok

    def test_color_subset_allows_input_or_expected_colors(self) -> None:
        from cognithor.channels.program_synthesis.verify.properties import (
            output_colors_subset_of_input_colors_plus_const,
        )

        # actual uses color 5; input has 5; expected has 5 → ok.
        actual = _g([[5, 5]])
        expected = _g([[5, 5]])
        demo_input = _g([[5, 5]])
        ok, _ = output_colors_subset_of_input_colors_plus_const(actual, expected, demo_input)
        assert ok

    def test_color_subset_rejects_hallucinated_color(self) -> None:
        from cognithor.channels.program_synthesis.verify.properties import (
            output_colors_subset_of_input_colors_plus_const,
        )

        # actual uses color 7; not in input or expected → rejected.
        actual = _g([[7, 7]])
        expected = _g([[1, 1]])
        demo_input = _g([[2, 2]])
        ok, detail = output_colors_subset_of_input_colors_plus_const(actual, expected, demo_input)
        assert not ok
        assert "7" in detail
