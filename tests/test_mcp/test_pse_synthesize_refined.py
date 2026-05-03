"""Sprint-25 — integration tests for ``handle_pse_synthesize_refined``.

The handler glues two pieces together:

  1. ``handle_pse_synthesize``  — heavy, owns the engine + cache.
  2. ``refine_pse_program``     — light, pure refinement pass.

The pure refinement is unit-tested in
``tests/test_core/test_pse_refinement.py``. Here we exercise the
**wrapper** by monkey-patching ``handle_pse_synthesize`` (so the engine
never boots) and the module-level ``_refinement_llm_fn``.

That way the test stays cheap and focused on the integration contract:

  * ``task_description`` is consumed and not forwarded to the underlying
    synthesis call.
  * The ``refinement`` field on the result has the expected shape.
  * Refinement is skipped (``refinement = None``) when no program was
    synthesised — never when a successful program was found.
  * A flaky ``llm_fn`` (raises / returns garbage) does not lose the
    synthesis result; the refinement collapses to ``"neutral"``.
"""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.mcp import pse_tools
from cognithor.mcp.pse_tools import handle_pse_synthesize_refined


def _ok_synthesis(**_: Any) -> dict[str, Any]:
    """A fake successful pse_synthesize result for monkey-patching."""
    return {
        "status": "success",
        "program": "rotate_90 ∘ flip_h",
        "score": 1.0,
        "confidence": 0.95,
        "cost_seconds": 0.05,
        "cost_candidates": 42,
        "cache_hit": False,
        "held_out_examples": 1,
        "escalations": 0,
    }


def _no_program_synthesis(**_: Any) -> dict[str, Any]:
    """A fake no-solution pse_synthesize result."""
    return {
        "status": "no_solution",
        "program": None,
        "score": 0.0,
        "confidence": 0.0,
        "cost_seconds": 1.0,
        "cost_candidates": 100,
        "cache_hit": False,
        "held_out_examples": 0,
        "escalations": 0,
    }


@pytest.mark.asyncio
async def test_refined_strips_task_description_before_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``task_description`` must not leak into ``handle_pse_synthesize`` —
    the underlying handler does not know about it.
    """
    captured: list[dict[str, Any]] = []

    async def _fake(**kwargs: Any) -> dict[str, Any]:
        captured.append(dict(kwargs))
        return _ok_synthesis()

    async def _fake_llm(_p: str) -> str:
        return '{"verdict": "accept", "explanation": "ok", "confidence": "high"}'

    monkeypatch.setattr(pse_tools, "handle_pse_synthesize", _fake)
    monkeypatch.setattr(pse_tools, "_refinement_llm_fn", _fake_llm)

    out = await handle_pse_synthesize_refined(
        examples=[
            {"input": [[0, 1]], "output": [[1, 0]]},
            {"input": [[2, 3]], "output": [[3, 2]]},
        ],
        task_description="Swap the two cells in each row.",
    )

    assert "task_description" not in captured[0]
    # Synthesis fields preserved verbatim.
    assert out["status"] == "success"
    assert out["program"] == "rotate_90 ∘ flip_h"
    # And the new refinement field is populated.
    assert out["refinement"] == {
        "verdict": "accept",
        "explanation": "ok",
        "confidence": "high",
    }


@pytest.mark.asyncio
async def test_refined_returns_none_refinement_when_no_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No program means there's nothing to refine — refinement field is None."""

    async def _fake(**_: Any) -> dict[str, Any]:
        return _no_program_synthesis()

    monkeypatch.setattr(pse_tools, "handle_pse_synthesize", _fake)

    # Even with an llm_fn wired, refinement skips when there's no program.
    async def _llm(_p: str) -> str:  # never called
        raise AssertionError("llm_fn must not be invoked when no program was found")

    monkeypatch.setattr(pse_tools, "_refinement_llm_fn", _llm)

    out = await handle_pse_synthesize_refined(
        examples=[
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
        ],
        task_description="identity",
    )
    assert out["status"] == "no_solution"
    assert out["refinement"] is None


@pytest.mark.asyncio
async def test_refined_with_no_llm_returns_neutral_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no LLM is wired, refinement degrades to a neutral verdict —
    the synthesis result still rides through.
    """

    async def _fake(**_: Any) -> dict[str, Any]:
        return _ok_synthesis()

    monkeypatch.setattr(pse_tools, "handle_pse_synthesize", _fake)
    monkeypatch.setattr(pse_tools, "_refinement_llm_fn", None)

    out = await handle_pse_synthesize_refined(
        examples=[
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
        ],
        task_description="anything",
    )
    assert out["status"] == "success"
    assert out["refinement"] is not None
    assert out["refinement"]["verdict"] == "neutral"
    assert "no llm_fn" in out["refinement"]["explanation"]


@pytest.mark.asyncio
async def test_refined_llm_raises_keeps_synthesis_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM-side failures must not corrupt the synthesis side of the result."""

    async def _fake(**_: Any) -> dict[str, Any]:
        return _ok_synthesis()

    async def _broken_llm(_p: str) -> str:
        raise RuntimeError("LLM unreachable")

    monkeypatch.setattr(pse_tools, "handle_pse_synthesize", _fake)
    monkeypatch.setattr(pse_tools, "_refinement_llm_fn", _broken_llm)

    out = await handle_pse_synthesize_refined(
        examples=[
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
        ],
        task_description="anything",
    )
    # Synthesis result intact:
    assert out["status"] == "success"
    assert out["program"] == "rotate_90 ∘ flip_h"
    # Refinement collapsed to neutral:
    assert out["refinement"]["verdict"] == "neutral"
    assert "raised" in out["refinement"]["explanation"]


@pytest.mark.asyncio
async def test_refined_passes_correct_held_out_count_to_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refinement prompt must reflect the synthesis's actual held-out
    count (not the user's input count) — auto_held_out can promote one
    example into the held-out set.
    """
    captured_prompt: list[str] = []

    async def _fake(**_: Any) -> dict[str, Any]:
        # Synthesis claims 1 held-out example was used.
        return {
            **_ok_synthesis(),
            "held_out_examples": 1,
        }

    async def _fake_llm(p: str) -> str:
        captured_prompt.append(p)
        return '{"verdict": "accept", "explanation": "ok", "confidence": "high"}'

    monkeypatch.setattr(pse_tools, "handle_pse_synthesize", _fake)
    monkeypatch.setattr(pse_tools, "_refinement_llm_fn", _fake_llm)

    await handle_pse_synthesize_refined(
        # Caller passed 4 examples; synthesis auto-promoted 1 to held-out.
        examples=[
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
            {"input": [[2]], "output": [[2]]},
            {"input": [[3]], "output": [[3]]},
        ],
        task_description="identity-like",
    )

    # Prompt should mention the "3 examples + 1 held-out" split.
    assert "1 held-out" in captured_prompt[0]
