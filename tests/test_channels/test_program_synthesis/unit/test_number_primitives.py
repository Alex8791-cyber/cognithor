# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 Track B PR#3 — Number/Int-DSL family tests."""

from __future__ import annotations

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.dsl import number_primitives as np_dsl
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestNumberPrimitivesPure:
    def test_identity(self) -> None:
        assert np_dsl.int_identity(5) == 5
        assert np_dsl.int_identity(0) == 0
        assert np_dsl.int_identity(-3) == -3

    def test_increment(self) -> None:
        assert np_dsl.int_increment(0) == 1
        assert np_dsl.int_increment(-1) == 0
        assert np_dsl.int_increment(99) == 100

    def test_decrement(self) -> None:
        assert np_dsl.int_decrement(1) == 0
        assert np_dsl.int_decrement(0) == -1

    def test_double(self) -> None:
        assert np_dsl.int_double(0) == 0
        assert np_dsl.int_double(7) == 14
        assert np_dsl.int_double(-3) == -6

    def test_triple(self) -> None:
        assert np_dsl.int_triple(0) == 0
        assert np_dsl.int_triple(4) == 12
        assert np_dsl.int_triple(-2) == -6

    def test_half(self) -> None:
        assert np_dsl.int_half(10) == 5
        assert np_dsl.int_half(7) == 3  # floor division
        assert np_dsl.int_half(-1) == -1  # rounds toward -infinity

    def test_negate(self) -> None:
        assert np_dsl.int_negate(5) == -5
        assert np_dsl.int_negate(-7) == 7
        assert np_dsl.int_negate(0) == 0

    def test_abs(self) -> None:
        assert np_dsl.int_abs(5) == 5
        assert np_dsl.int_abs(-5) == 5
        assert np_dsl.int_abs(0) == 0

    def test_square(self) -> None:
        assert np_dsl.int_square(0) == 0
        assert np_dsl.int_square(3) == 9
        assert np_dsl.int_square(-4) == 16

    def test_int_to_string(self) -> None:
        assert np_dsl.int_to_string(0) == "0"
        assert np_dsl.int_to_string(42) == "42"
        assert np_dsl.int_to_string(-7) == "-7"

    def test_string_to_int(self) -> None:
        assert np_dsl.string_to_int("0") == 0
        assert np_dsl.string_to_int("42") == 42
        assert np_dsl.string_to_int("-7") == -7

    def test_string_length(self) -> None:
        assert np_dsl.string_length("") == 0
        assert np_dsl.string_length("hi") == 2
        assert np_dsl.string_length("hello") == 5


class TestNumberPrimitivesTypeChecking:
    """Every primitive raises on wrong input shape so the executor
    sandbox prunes the candidate."""

    def test_int_double_rejects_str(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            np_dsl.int_double("5")  # type: ignore[arg-type]

    def test_int_double_rejects_bool(self) -> None:
        # ``bool`` is a subclass of ``int`` in Python — we explicitly
        # reject it so the type filter cannot accept ``True`` / ``False``
        # as Int leaves.
        import pytest

        with pytest.raises(TypeError):
            np_dsl.int_double(True)  # type: ignore[arg-type]

    def test_string_to_int_rejects_non_str(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            np_dsl.string_to_int(5)  # type: ignore[arg-type]

    def test_string_to_int_raises_on_non_numeric(self) -> None:
        import pytest

        # Python's ``int(s)`` raises ValueError on a non-numeric string.
        # The sandbox catches that just like a TypeError and prunes the
        # candidate.
        with pytest.raises(ValueError):
            np_dsl.string_to_int("not a number")

    def test_string_length_rejects_non_str(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            np_dsl.string_length(5)  # type: ignore[arg-type]


class TestNumberPrimitivesRegistration:
    """All 12 number primitives are registered with the singleton REGISTRY
    on package import."""

    def test_all_12_are_registered(self) -> None:
        registered = {p.name for p in REGISTRY.all_primitives()}
        expected = {
            "int_identity",
            "int_increment",
            "int_decrement",
            "int_double",
            "int_triple",
            "int_half",
            "int_negate",
            "int_abs",
            "int_square",
            "int_to_string",
            "string_to_int",
            "string_length",
        }
        assert expected.issubset(registered)

    def test_int_signatures(self) -> None:
        spec = REGISTRY.get("int_double")
        assert spec.signature.inputs == ("Int",)
        assert spec.signature.output == "Int"

    def test_int_to_string_signature(self) -> None:
        spec = REGISTRY.get("int_to_string")
        assert spec.signature.inputs == ("Int",)
        assert spec.signature.output == "String"

    def test_string_to_int_signature(self) -> None:
        spec = REGISTRY.get("string_to_int")
        assert spec.signature.inputs == ("String",)
        assert spec.signature.output == "Int"


# ---------------------------------------------------------------------------
# End-to-end synthesis tests
# ---------------------------------------------------------------------------


def _synthesize(examples: tuple[tuple[object, object], ...]) -> object:
    """Helper: run the channel against a small spec — int, str, or mixed."""
    ch = ProgramSynthesisChannel(actor="number-test")
    spec = TaskSpec(examples=examples)
    return ch.synthesize(
        SynthesisRequest(
            spec=spec,
            budget=Budget(max_depth=3, wall_clock_seconds=5.0, cache_lookup=False),
        )
    )


class TestNumberSynthesisEndToEnd:
    """The engine routes int → int / str → int / int → str tasks
    through the same enumerative loop now that ``Int`` is a
    first-class input *and* demo-output type tag.
    """

    def test_synth_int_double(self) -> None:
        result = _synthesize(
            (
                (3, 6),
                (7, 14),
                (10, 20),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_double" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_int_square(self) -> None:
        result = _synthesize(
            (
                (3, 9),
                (4, 16),
                (5, 25),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_square" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_int_increment(self) -> None:
        result = _synthesize(
            (
                (0, 1),
                (5, 6),
                (99, 100),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_increment" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_int_to_string(self) -> None:
        result = _synthesize(
            (
                (5, "5"),
                (42, "42"),
                (-7, "-7"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_to_string" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_string_length(self) -> None:
        result = _synthesize(
            (
                ("hi", 2),
                ("hello", 5),
                ("foo", 3),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_length" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_string_to_int(self) -> None:
        result = _synthesize(
            (
                ("5", 5),
                ("42", 42),
                ("100", 100),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_to_int" in str(result.program)  # type: ignore[attr-defined]
