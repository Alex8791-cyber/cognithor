"""Sprint-25 — pure unit tests for ``cognithor.core.pse_refinement``.

The refinement function is intentionally side-effect-free: it builds a
prompt, calls the supplied ``llm_fn``, and parses the response. These
tests pin every branch (parse-OK, missing fields, garbage JSON, the
``llm_fn`` raises, no ``llm_fn`` wired) so the wrapping MCP tool can
rely on a stable contract.
"""

from __future__ import annotations

import pytest

from cognithor.core.pse_refinement import (
    VERDICT_ACCEPT,
    VERDICT_CAUTION,
    VERDICT_NEUTRAL,
    VERDICT_REJECT,
    build_refinement_prompt,
    parse_refinement_response,
    refine_pse_program,
)

# ---------------------------------------------------------------------------
# build_refinement_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_contains_all_fields() -> None:
    p = build_refinement_prompt(
        program="rotate_90 ∘ flip_h",
        task_description="Rotate the grid 90° clockwise then flip horizontally.",
        n_examples=3,
        n_held_out=1,
        score=0.95,
    )
    assert "rotate_90 ∘ flip_h" in p
    assert "Rotate the grid 90° clockwise" in p
    assert "3" in p
    assert "1 held-out" in p
    assert "0.950" in p
    # The prompt must demand JSON to keep the parser deterministic.
    assert '"verdict"' in p
    assert '"explanation"' in p
    assert '"confidence"' in p


def test_build_prompt_truncates_huge_program() -> None:
    huge = "x" * 5_000
    p = build_refinement_prompt(
        program=huge,
        task_description="anything",
        n_examples=2,
        n_held_out=0,
        score=1.0,
    )
    # Truncated marker added when program exceeded the cap.
    assert "[... truncated ...]" in p
    # ... and the fully-rendered prompt is shorter than the raw program.
    assert len(p) < len(huge) + 2_000


def test_build_prompt_handles_empty_task_description() -> None:
    p = build_refinement_prompt(
        program="identity",
        task_description="",
        n_examples=2,
        n_held_out=0,
        score=1.0,
    )
    assert "(no task description provided)" in p


# ---------------------------------------------------------------------------
# parse_refinement_response
# ---------------------------------------------------------------------------


def test_parse_pure_json_accept() -> None:
    raw = '{"verdict": "accept", "explanation": "rotate matches the spec", "confidence": "high"}'
    v = parse_refinement_response(raw)
    assert v.verdict == VERDICT_ACCEPT
    assert v.confidence == "high"
    assert "rotate matches" in v.explanation


def test_parse_markdown_fence_caution() -> None:
    raw = (
        "Sure, here's my critique:\n"
        "```json\n"
        '{"verdict": "caution", "explanation": "edge cases unclear", '
        '"confidence": "medium"}\n'
        "```\n"
        "Hope this helps!"
    )
    v = parse_refinement_response(raw)
    assert v.verdict == VERDICT_CAUTION
    assert v.confidence == "medium"


def test_parse_inline_json_reject() -> None:
    raw = (
        "I think this is wrong. "
        '{"verdict": "reject", "explanation": "swap colours not '
        'rotation", "confidence": "high"} '
        "Definitely reject."
    )
    v = parse_refinement_response(raw)
    assert v.verdict == VERDICT_REJECT
    assert "swap colours" in v.explanation


def test_parse_invalid_json_returns_neutral() -> None:
    v = parse_refinement_response("totally not json")
    assert v.verdict == VERDICT_NEUTRAL
    assert v.confidence == "low"


def test_parse_empty_returns_neutral() -> None:
    v = parse_refinement_response("")
    assert v.verdict == VERDICT_NEUTRAL


def test_parse_unknown_verdict_returns_neutral() -> None:
    raw = '{"verdict": "maybe", "explanation": "?", "confidence": "high"}'
    v = parse_refinement_response(raw)
    assert v.verdict == VERDICT_NEUTRAL
    # The original LLM reply is preserved for diagnostics.
    assert "maybe" in v.raw_response


def test_parse_missing_confidence_defaults_low() -> None:
    raw = '{"verdict": "accept", "explanation": "ok"}'
    v = parse_refinement_response(raw)
    assert v.verdict == VERDICT_ACCEPT
    assert v.confidence == "low"


def test_parse_missing_explanation_uses_placeholder() -> None:
    raw = '{"verdict": "accept", "confidence": "high"}'
    v = parse_refinement_response(raw)
    assert v.verdict == VERDICT_ACCEPT
    assert "(LLM provided no explanation)" in v.explanation


def test_parse_truncates_long_explanation() -> None:
    long_text = "x" * 2_000
    raw = '{"verdict": "accept", "explanation": "' + long_text + '", "confidence": "low"}'
    v = parse_refinement_response(raw)
    # Hard-cap from the module — see _MAX_EXPLANATION_CHARS.
    assert len(v.explanation) <= 800


def test_parse_non_dict_json_returns_neutral() -> None:
    v = parse_refinement_response("[1, 2, 3]")
    assert v.verdict == VERDICT_NEUTRAL


# ---------------------------------------------------------------------------
# refine_pse_program (the async wrapper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refine_no_llm_fn_returns_neutral() -> None:
    v = await refine_pse_program(
        program="rotate_90",
        task_description="rotate",
        n_examples=2,
        n_held_out=0,
        score=1.0,
        llm_fn=None,
    )
    assert v.verdict == VERDICT_NEUTRAL
    assert "no llm_fn" in v.explanation


@pytest.mark.asyncio
async def test_refine_no_program_returns_neutral() -> None:
    async def _llm(_p: str) -> str:  # never called
        raise AssertionError("llm_fn should not be invoked when program is empty")

    v = await refine_pse_program(
        program="",
        task_description="rotate",
        n_examples=2,
        n_held_out=0,
        score=0.0,
        llm_fn=_llm,
    )
    assert v.verdict == VERDICT_NEUTRAL
    assert v.explanation == "no program to refine"


@pytest.mark.asyncio
async def test_refine_calls_llm_and_parses_accept() -> None:
    captured_prompt: list[str] = []

    async def _llm(p: str) -> str:
        captured_prompt.append(p)
        return '{"verdict": "accept", "explanation": "looks right", "confidence": "high"}'

    v = await refine_pse_program(
        program="rotate_90",
        task_description="Rotate 90° clockwise.",
        n_examples=3,
        n_held_out=1,
        score=0.95,
        llm_fn=_llm,
    )
    assert v.verdict == VERDICT_ACCEPT
    assert v.confidence == "high"
    # The captured prompt is the same one ``build_refinement_prompt`` produces.
    assert "Rotate 90° clockwise" in captured_prompt[0]
    assert "rotate_90" in captured_prompt[0]


@pytest.mark.asyncio
async def test_refine_llm_raises_returns_neutral() -> None:
    async def _llm(_p: str) -> str:
        raise RuntimeError("LLM down")

    v = await refine_pse_program(
        program="rotate_90",
        task_description="rotate",
        n_examples=2,
        n_held_out=0,
        score=1.0,
        llm_fn=_llm,
    )
    assert v.verdict == VERDICT_NEUTRAL
    assert "raised" in v.explanation


@pytest.mark.asyncio
async def test_refine_llm_returns_garbage_returns_neutral() -> None:
    async def _llm(_p: str) -> str:
        return "I don't speak JSON."

    v = await refine_pse_program(
        program="rotate_90",
        task_description="rotate",
        n_examples=2,
        n_held_out=0,
        score=1.0,
        llm_fn=_llm,
    )
    assert v.verdict == VERDICT_NEUTRAL


@pytest.mark.asyncio
async def test_refine_to_dict_round_trip() -> None:
    async def _llm(_p: str) -> str:
        return '{"verdict": "caution", "explanation": "may overfit", "confidence": "medium"}'

    v = await refine_pse_program(
        program="invert_colors",
        task_description="invert colours",
        n_examples=2,
        n_held_out=0,
        score=0.9,
        llm_fn=_llm,
    )
    d = v.to_dict()
    assert d == {
        "verdict": "caution",
        "explanation": "may overfit",
        "confidence": "medium",
    }
