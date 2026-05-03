# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 Track B PR#2 — Regex/Pattern-DSL family tests."""

from __future__ import annotations

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.dsl import regex_primitives as rp
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestRegexPrimitivesPureKeep:
    def test_keep_digits(self) -> None:
        assert rp.string_keep_digits("abc123def456") == "123456"
        assert rp.string_keep_digits("no-digits") == ""

    def test_keep_letters(self) -> None:
        assert rp.string_keep_letters("a1b2c3") == "abc"
        assert rp.string_keep_letters("12345") == ""

    def test_keep_alphanumeric(self) -> None:
        assert rp.string_keep_alphanumeric("hello, world! 123") == "helloworld123"
        assert rp.string_keep_alphanumeric("@@@!!!") == ""


class TestRegexPrimitivesPureRemove:
    def test_remove_digits(self) -> None:
        assert rp.string_remove_digits("abc123def") == "abcdef"

    def test_remove_letters(self) -> None:
        assert rp.string_remove_letters("a1b2c3") == "123"

    def test_remove_punctuation(self) -> None:
        assert rp.string_remove_punctuation("hello, world!") == "hello world"
        assert rp.string_remove_punctuation("a.b,c;d") == "abcd"

    def test_remove_spaces(self) -> None:
        assert rp.string_remove_spaces("a b c\td\ne") == "abcde"


class TestRegexPrimitivesPureNormalise:
    def test_collapse_spaces(self) -> None:
        assert rp.string_collapse_spaces("a   b\t\tc") == "a b c"
        assert rp.string_collapse_spaces("  trailing  ") == " trailing "


class TestRegexPrimitivesPureRunExtract:
    def test_first_digit_run(self) -> None:
        assert rp.string_first_digit_run("abc123def456") == "123"
        assert rp.string_first_digit_run("no-digits") == ""

    def test_last_digit_run(self) -> None:
        assert rp.string_last_digit_run("abc123def456") == "456"
        assert rp.string_last_digit_run("no-digits") == ""


class TestRegexPrimitivesPurePatternExtract:
    def test_extract_email(self) -> None:
        assert rp.string_extract_email("contact me at foo@example.com please") == "foo@example.com"
        assert rp.string_extract_email("no email here") == ""

    def test_extract_url(self) -> None:
        assert rp.string_extract_url("see https://cognithor.ai for docs") == "https://cognithor.ai"
        assert rp.string_extract_url("see http://x.io now") == "http://x.io"
        assert rp.string_extract_url("no link here") == ""


class TestRegexPrimitivesPureSlug:
    def test_replace_space_with_dash(self) -> None:
        assert rp.string_replace_space_with_dash("hello world foo") == "hello-world-foo"

    def test_replace_space_with_underscore(self) -> None:
        assert rp.string_replace_space_with_underscore("foo bar baz") == "foo_bar_baz"


class TestRegexPrimitivesTypeChecking:
    """Every primitive raises TypeError on non-str input so the executor
    sandbox marks the candidate as failed and the search engine prunes
    it instead of blowing up."""

    def test_keep_digits_rejects_non_str(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            rp.string_keep_digits(123)  # type: ignore[arg-type]

    def test_extract_email_rejects_non_str(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            rp.string_extract_email(["a@b.com"])  # type: ignore[arg-type]


class TestRegexPrimitivesRegistration:
    """All 14 regex primitives are registered with the singleton REGISTRY
    on package import."""

    def test_all_14_are_registered(self) -> None:
        registered = {p.name for p in REGISTRY.all_primitives()}
        expected = {
            "string_keep_digits",
            "string_keep_letters",
            "string_keep_alphanumeric",
            "string_remove_digits",
            "string_remove_letters",
            "string_remove_punctuation",
            "string_remove_spaces",
            "string_collapse_spaces",
            "string_first_digit_run",
            "string_last_digit_run",
            "string_extract_email",
            "string_extract_url",
            "string_replace_space_with_dash",
            "string_replace_space_with_underscore",
        }
        assert expected.issubset(registered)

    def test_signatures_use_string_type_tag(self) -> None:
        spec = REGISTRY.get("string_keep_digits")
        assert spec.signature.inputs == ("String",)
        assert spec.signature.output == "String"


# ---------------------------------------------------------------------------
# End-to-end synthesis tests
# ---------------------------------------------------------------------------


def _synthesize(examples: tuple[tuple[str, str], ...]) -> object:
    """Helper: run the channel against a small string-string spec."""
    ch = ProgramSynthesisChannel(actor="regex-test")
    spec = TaskSpec(examples=examples)
    return ch.synthesize(
        SynthesisRequest(
            spec=spec,
            budget=Budget(max_depth=3, wall_clock_seconds=5.0, cache_lookup=False),
        )
    )


class TestRegexSynthesisEndToEnd:
    """The engine synthesises real programs over the regex family."""

    def test_synth_keep_digits(self) -> None:
        result = _synthesize(
            (
                ("abc123", "123"),
                ("x9y8z", "98"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_keep_digits" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_remove_punctuation(self) -> None:
        result = _synthesize(
            (
                ("hello, world!", "hello world"),
                ("a.b,c", "abc"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        # Either remove_punctuation or some equivalent — accept any program
        # that produces the right output. Engine may pick a cheaper path.
        assert result.program is not None  # type: ignore[attr-defined]

    def test_synth_first_digit_run(self) -> None:
        result = _synthesize(
            (
                ("order-123-abc-456", "123"),
                ("x9y8", "9"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_first_digit_run" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_replace_space_with_dash(self) -> None:
        result = _synthesize(
            (
                ("hello world", "hello-world"),
                ("foo bar baz", "foo-bar-baz"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_replace_space_with_dash" in str(result.program)  # type: ignore[attr-defined]

    def test_synth_extract_email(self) -> None:
        result = _synthesize(
            (
                ("write me at foo@bar.com soon", "foo@bar.com"),
                ("a@b.io is mine", "a@b.io"),
            )
        )
        assert result.status == SynthesisStatus.SUCCESS  # type: ignore[attr-defined]
        assert "string_extract_email" in str(result.program)  # type: ignore[attr-defined]
