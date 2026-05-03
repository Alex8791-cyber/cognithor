# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 — String-DSL family tests.

Two test layers:

1. Unit tests for each primitive's pure-function behaviour.
2. End-to-end synthesis tests proving the engine + new ``InputRef``
   type-tagging + verifier extension actually find the right
   programs on real string demos.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.dsl import string_primitives as sp
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestStringPrimitivesPure:
    def test_identity(self) -> None:
        assert sp.string_identity("hi") == "hi"

    def test_lower(self) -> None:
        assert sp.string_lower("HELLO") == "hello"

    def test_upper(self) -> None:
        assert sp.string_upper("hello") == "HELLO"

    def test_capitalize(self) -> None:
        assert sp.string_capitalize("hello world") == "Hello world"

    def test_strip(self) -> None:
        assert sp.string_strip("  foo  ") == "foo"

    def test_split_space(self) -> None:
        assert sp.string_split_space("hello world") == ["hello", "world"]

    def test_split_comma(self) -> None:
        assert sp.string_split_comma("a,b,c") == ["a", "b", "c"]

    def test_join_space(self) -> None:
        assert sp.string_join_space(["hello", "world"]) == "hello world"

    def test_join_comma(self) -> None:
        assert sp.string_join_comma(["a", "b", "c"]) == "a, b, c"

    def test_replace_dash_with_space(self) -> None:
        assert sp.string_replace_dash_with_space("a-b-c") == "a b c"

    def test_replace_underscore_with_space(self) -> None:
        assert sp.string_replace_underscore_with_space("foo_bar") == "foo bar"

    def test_first_word(self) -> None:
        assert sp.string_first_word("quick brown fox") == "quick"
        assert sp.string_first_word("") == ""

    def test_last_word(self) -> None:
        assert sp.string_last_word("quick brown fox") == "fox"
        assert sp.string_last_word("") == ""

    def test_reverse(self) -> None:
        assert sp.string_reverse("abc") == "cba"


class TestStringPrimitivesTypeChecking:
    """Every primitive raises TypeError on non-str input so the executor
    sandbox marks the candidate as failed and the search engine prunes
    it instead of blowing up.
    """

    def test_lower_rejects_non_str(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            sp.string_lower(123)  # type: ignore[arg-type]

    def test_join_space_rejects_non_list(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            sp.string_join_space("not a list")  # type: ignore[arg-type]


class TestStringPrimitivesRegistration:
    """All 14 string primitives are registered with the singleton REGISTRY
    on package import."""

    def test_all_14_are_registered(self) -> None:
        registered = {p.name for p in REGISTRY.all_primitives()}
        expected = {
            "string_identity",
            "string_lower",
            "string_upper",
            "string_capitalize",
            "string_strip",
            "string_split_space",
            "string_split_comma",
            "string_join_space",
            "string_join_comma",
            "string_replace_dash_with_space",
            "string_replace_underscore_with_space",
            "string_first_word",
            "string_last_word",
            "string_reverse",
        }
        assert expected.issubset(registered)

    def test_signatures_use_string_type_tag(self) -> None:
        spec = REGISTRY.get("string_lower")
        assert spec.signature.inputs == ("String",)
        assert spec.signature.output == "String"

    def test_split_returns_string_list_type(self) -> None:
        spec = REGISTRY.get("string_split_space")
        assert spec.signature.inputs == ("String",)
        assert spec.signature.output == "StringList"


# ---------------------------------------------------------------------------
# End-to-end synthesis tests
# ---------------------------------------------------------------------------


def _synthesize(examples: tuple[tuple[str, str], ...]) -> object:
    """Helper: run the channel against a small string-string spec."""
    ch = ProgramSynthesisChannel(actor="string-test")
    spec = TaskSpec(examples=examples)
    return ch.synthesize(
        SynthesisRequest(
            spec=spec,
            budget=Budget(max_depth=3, wall_clock_seconds=5.0, cache_lookup=False),
        )
    )


class TestStringSynthesisEndToEnd:
    """The engine synthesises real programs over the string family —
    proves InputRef-type-tagging, verifier-extension, and primitive
    registration are all correctly wired.
    """

    def test_synth_lower(self) -> None:
        result = _synthesize(
            (
                ("HELLO", "hello"),
                ("WORLD", "world"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_lower" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_strip(self) -> None:
        # Inputs have inner whitespace so ``string_remove_spaces`` would
        # destroy it, plus digits/punct so ``string_keep_letters`` /
        # ``string_remove_punctuation`` would also fail. Only
        # ``string_strip`` survives.
        result = _synthesize(
            (
                ("  hello-12 world  ", "hello-12 world"),
                (" foo.45 bar ", "foo.45 bar"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_strip" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_reverse(self) -> None:
        result = _synthesize(
            (
                ("abc", "cba"),
                ("hi", "ih"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_reverse" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_upper(self) -> None:
        result = _synthesize(
            (
                ("hi", "HI"),
                ("foo", "FOO"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_upper" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_first_word(self) -> None:
        result = _synthesize(
            (
                ("quick brown fox", "quick"),
                ("hello world", "hello"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_first_word" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_replace_dash(self) -> None:
        result = _synthesize(
            (
                ("a-b-c", "a b c"),
                ("x-y", "x y"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_replace_dash_with_space" in str(result.program)  # type: ignore[attr-defined]

    def test_grid_synthesis_still_works_after_string_family(self) -> None:
        """Regression: the new family must not break the legacy grid path."""
        import numpy as np

        ch = ProgramSynthesisChannel(actor="grid-test")
        spec = TaskSpec(
            examples=(
                (
                    np.array([[1, 2], [3, 4]], dtype=np.int8),
                    np.array([[3, 1], [4, 2]], dtype=np.int8),
                ),
                (
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8),
                    np.array([[4, 1], [5, 2], [6, 3]], dtype=np.int8),
                ),
            )
        )
        result = ch.synthesize(
            SynthesisRequest(
                spec=spec,
                budget=Budget(max_depth=3, wall_clock_seconds=5.0, cache_lookup=False),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS
        assert "rotate90" in str(result.program)
