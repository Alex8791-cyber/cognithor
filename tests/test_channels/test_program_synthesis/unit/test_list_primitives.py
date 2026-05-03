# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 Track B PR#4 — List/Sequence-DSL family tests."""

from __future__ import annotations

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.dsl import list_primitives as lp
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.dsl.signatures import ALLOWED_TYPES
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests — StringList
# ---------------------------------------------------------------------------


class TestStringListPrimitivesPure:
    def test_reverse(self) -> None:
        assert lp.string_list_reverse(["a", "b", "c"]) == ["c", "b", "a"]
        assert lp.string_list_reverse([]) == []

    def test_unique(self) -> None:
        assert lp.string_list_unique(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
        assert lp.string_list_unique([]) == []

    def test_sort(self) -> None:
        assert lp.string_list_sort(["c", "a", "b"]) == ["a", "b", "c"]
        assert lp.string_list_sort([]) == []

    def test_first(self) -> None:
        assert lp.string_list_first(["a", "b"]) == "a"
        assert lp.string_list_first([]) == ""

    def test_last(self) -> None:
        assert lp.string_list_last(["a", "b", "c"]) == "c"
        assert lp.string_list_last([]) == ""

    def test_length(self) -> None:
        assert lp.string_list_length([]) == 0
        assert lp.string_list_length(["a", "b", "c"]) == 3

    def test_count_nonempty(self) -> None:
        assert lp.string_list_count_nonempty(["a", "", "b", ""]) == 2
        assert lp.string_list_count_nonempty([]) == 0


# ---------------------------------------------------------------------------
# Pure-function unit tests — IntList
# ---------------------------------------------------------------------------


class TestIntListPrimitivesPure:
    def test_reverse(self) -> None:
        assert lp.int_list_reverse([1, 2, 3]) == [3, 2, 1]
        assert lp.int_list_reverse([]) == []

    def test_sort(self) -> None:
        assert lp.int_list_sort([3, 1, 2]) == [1, 2, 3]
        assert lp.int_list_sort([]) == []

    def test_first(self) -> None:
        assert lp.int_list_first([5, 9, 1]) == 5

    def test_first_raises_on_empty(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            lp.int_list_first([])

    def test_sum(self) -> None:
        assert lp.int_list_sum([1, 2, 3]) == 6
        assert lp.int_list_sum([]) == 0  # empty sum is 0
        assert lp.int_list_sum([-1, 1]) == 0

    def test_max(self) -> None:
        assert lp.int_list_max([5, 1, 9]) == 9
        assert lp.int_list_max([-3, -1, -7]) == -1

    def test_max_raises_on_empty(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            lp.int_list_max([])

    def test_min(self) -> None:
        assert lp.int_list_min([5, 1, 9]) == 1
        assert lp.int_list_min([-3, -1, -7]) == -7

    def test_min_raises_on_empty(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            lp.int_list_min([])

    def test_length(self) -> None:
        assert lp.int_list_length([]) == 0
        assert lp.int_list_length([7]) == 1
        assert lp.int_list_length([1, 2, 3, 4, 5]) == 5


# ---------------------------------------------------------------------------
# Type-checking
# ---------------------------------------------------------------------------


class TestListPrimitivesTypeChecking:
    def test_string_list_reverse_rejects_non_list(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            lp.string_list_reverse("abc")  # type: ignore[arg-type]

    def test_int_list_sum_rejects_non_list(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            lp.int_list_sum(5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestListPrimitivesRegistration:
    def test_intlist_in_allowed_types(self) -> None:
        assert "IntList" in ALLOWED_TYPES

    def test_all_14_are_registered(self) -> None:
        registered = {p.name for p in REGISTRY.all_primitives()}
        expected = {
            "string_list_reverse",
            "string_list_unique",
            "string_list_sort",
            "string_list_first",
            "string_list_last",
            "string_list_length",
            "string_list_count_nonempty",
            "int_list_reverse",
            "int_list_sort",
            "int_list_first",
            "int_list_sum",
            "int_list_max",
            "int_list_min",
            "int_list_length",
        }
        assert expected.issubset(registered)

    def test_string_list_signatures(self) -> None:
        spec = REGISTRY.get("string_list_first")
        assert spec.signature.inputs == ("StringList",)
        assert spec.signature.output == "String"

    def test_int_list_signatures(self) -> None:
        spec = REGISTRY.get("int_list_max")
        assert spec.signature.inputs == ("IntList",)
        assert spec.signature.output == "Int"


# ---------------------------------------------------------------------------
# End-to-end synthesis tests
# ---------------------------------------------------------------------------


def _synthesize(examples: tuple[tuple[object, object], ...]) -> object:
    ch = ProgramSynthesisChannel(actor="list-test")
    spec = TaskSpec(examples=examples)
    return ch.synthesize(
        SynthesisRequest(
            spec=spec,
            budget=Budget(max_depth=3, wall_clock_seconds=5.0, cache_lookup=False),
        )
    )


class TestStringListSynthesisEndToEnd:
    def test_synth_reverse(self) -> None:
        result = _synthesize(
            (
                (["a", "b", "c"], ["c", "b", "a"]),
                (["x", "y"], ["y", "x"]),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_list_reverse" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_sort(self) -> None:
        result = _synthesize(
            (
                (["c", "a", "b"], ["a", "b", "c"]),
                (["d", "b", "c", "a"], ["a", "b", "c", "d"]),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_list_sort" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_first(self) -> None:
        result = _synthesize(
            (
                (["alpha", "beta", "gamma"], "alpha"),
                (["foo", "bar"], "foo"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_list_first" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_length(self) -> None:
        result = _synthesize(
            (
                (["a", "b"], 2),
                (["x", "y", "z"], 3),
                (["one"], 1),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_list_length" in str(result.program)  # type: ignore[attr-defined]


class TestIntListSynthesisEndToEnd:
    def test_synth_sum(self) -> None:
        result = _synthesize(
            (
                ([1, 2, 3], 6),
                ([10, 20, 30], 60),
                ([5, 5], 10),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_list_sum" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_max(self) -> None:
        result = _synthesize(
            (
                ([5, 1, 9], 9),
                ([3, 7, 4], 7),
                ([-1, -5, -2], -1),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_list_max" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_min(self) -> None:
        result = _synthesize(
            (
                ([5, 1, 9], 1),
                ([3, 7, 4], 3),
                ([-1, -5, -2], -5),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_list_min" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_length(self) -> None:
        result = _synthesize(
            (
                ([1, 2], 2),
                ([7, 8, 9], 3),
                ([42], 1),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_list_length" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_sort(self) -> None:
        result = _synthesize(
            (
                ([3, 1, 2], [1, 2, 3]),
                ([5, 9, 1], [1, 5, 9]),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "int_list_sort" in str(result.program)  # type: ignore[attr-defined]
