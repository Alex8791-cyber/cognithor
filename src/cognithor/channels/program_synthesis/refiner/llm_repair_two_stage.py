# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.5.2 Zone-1 — LLM-Repair Two-Stage with retry (Sprint-1 plan task 9).

When the Refiner mode controller picks ``full_llm`` (α ≥ 0.45), this
module runs the LLM-driven repair: a Two-Stage CoT→JSON prompt
that asks the model to (1) reason about *why* the candidate
program fails its demos, and (2) produce one or more replacement
sub-program source strings the caller can substitute into the
program tree.

Pipeline mirrors :mod:`cognithor.channels.program_synthesis.phase2.llm_prior`:

* Stage 1 (CoT): free-form prose. Six sentences max.
* Stage 2 (JSON): a list of ``LLMRepairSuggestion`` objects. The
  client retries Stage 2 once on parse failure (per spec §4.7's
  retry-once rule).

The actual mutation of the :class:`Program` tree is the *caller's*
job — this module returns the LLM's textual suggestions plus
confidence values, and the Refiner driver decides which to attempt
(typically by handing each source to the search-engine source
parser and re-verifying the resulting Program).

Sprint-1 ships this independently of an actual vLLM connection;
tests inject a fake :class:`LLMBackend` so CI never touches the
real model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cognithor.channels.program_synthesis.refiner.diff_analyzer import (
        DiffReport,
    )
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )
    from cognithor.core.llm_backend import LLMBackend


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMRepairSuggestion:
    """One candidate replacement the LLM proposed.

    ``replacement_source`` is the DSL source the LLM thinks would
    fix the failing demos — e.g. ``"rotate90(rotate90(input))"`` or
    ``"recolor(input, 1, 5)"``. The Refiner driver runs this through
    the search-engine source parser to lift it into a Program tree;
    if parsing fails, the suggestion is skipped.

    ``confidence`` is the LLM's self-reported confidence in [0, 1],
    clamped at parse time. ``reasoning`` is a free-form note —
    telemetry only, never read by the search engine.
    """

    replacement_source: str
    confidence: float
    reasoning: str = ""


@dataclass(frozen=True)
class LLMRepairResult:
    """Outcome of one Two-Stage repair call.

    ``suggestions`` is the parsed list, sorted by descending
    confidence (so the caller iterates highest-confidence first).
    Empty when Stage 2 produced no usable entries (after parse
    + filter + retry).

    ``stage1_reasoning`` is the verbatim Stage-1 CoT response.
    ``raw_response`` is the verbatim Stage-2 JSON string, kept for
    replay-debugging.
    """

    suggestions: tuple[LLMRepairSuggestion, ...] = field(default_factory=tuple)
    stage1_reasoning: str = ""
    raw_response: str = ""


class LLMRepairError(Exception):
    """Raised when Stage 2 cannot produce any usable suggestions."""


# ---------------------------------------------------------------------------
# Prompt schema (spec §4.7 Two-Stage adapted to repair)
# ---------------------------------------------------------------------------


_STAGE1_SYSTEM = (
    "You are a programmatic-synthesis repair advisor. Given an ARC-AGI "
    "grid task, a candidate DSL program that fails some demos, and the "
    "diff between actual and expected outputs, you reason briefly about "
    "what the program does wrong — which subtree is misbehaving, what "
    "transformation is missing, what is over-applied. Be concise: at "
    "most six sentences. Do not return a program. Do not return JSON. "
    "Return plain prose only."
)


_STAGE2_SYSTEM = (
    "You are a programmatic-synthesis repair advisor. Given the prior "
    "reasoning, the failing program, and the diff, return a JSON object "
    'of the form {"suggestions": [...]}. Each entry is an object with '
    "keys ``replacement_source`` (a DSL source string for the entire "
    "replacement program), ``confidence`` (a float in [0, 1]), and an "
    "optional ``reasoning`` (a short prose note). Return at most five "
    "entries, sorted by descending confidence. Use only DSL primitives "
    "from the provided whitelist. Do not include text outside the "
    "JSON object. Do not wrap the JSON in markdown code fences."
)


def _format_demo_block(demos: Iterable[tuple[Any, Any, Any]]) -> str:
    rendered: list[str] = []
    for i, (inp, expected, actual) in enumerate(demos):
        rendered.append(
            f"Demo {i + 1}:\n"
            f"  input    = {_compact_repr(inp)}\n"
            f"  expected = {_compact_repr(expected)}\n"
            f"  actual   = {_compact_repr(actual)}"
        )
    return "\n".join(rendered)


def _compact_repr(value: Any) -> str:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return repr(tolist())
    return repr(value)


def _format_diff_block(diff: DiffReport | None) -> str:
    if diff is None:
        return "No structured diff supplied."
    parts: list[str] = []
    if diff.identical:
        parts.append("identical=True")
    parts.append(
        f"shape_mismatch={diff.structure.shape_mismatch}, "
        f"actual_shape={diff.structure.actual_shape}, "
        f"expected_shape={diff.structure.expected_shape}"
    )
    parts.append(
        f"pixel_diff_count={diff.pixels.count}, "
        f"colors_introduced={sorted(diff.colors.introduced)}, "
        f"colors_missing={sorted(diff.colors.missing)}"
    )
    return " | ".join(parts)


def _build_stage1_user(
    program: ProgramNode,
    demos: Iterable[tuple[Any, Any, Any]],
    diff: DiffReport | None,
) -> str:
    return (
        f"Failing program (DSL source):\n  {program.to_source()}\n\n"
        f"Failing demos:\n{_format_demo_block(demos)}\n\n"
        f"Diff summary:\n  {_format_diff_block(diff)}"
    )


def _build_stage2_user(
    program: ProgramNode,
    demos: Iterable[tuple[Any, Any, Any]],
    diff: DiffReport | None,
    *,
    primitive_whitelist: list[str],
    stage1_reasoning: str,
) -> str:
    whitelist_block = ", ".join(sorted(primitive_whitelist))
    return (
        f"Failing program (DSL source):\n  {program.to_source()}\n\n"
        f"Failing demos:\n{_format_demo_block(demos)}\n\n"
        f"Diff summary:\n  {_format_diff_block(diff)}\n\n"
        f"Prior reasoning:\n{stage1_reasoning}\n\n"
        f"Allowed DSL primitives: {whitelist_block}\n\n"
        "Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMRepairTwoStageClient:
    """Two-Stage CoT→JSON LLM-repair client over an injected ``LLMBackend``.

    Mirrors :class:`LLMPriorClient` but for repair: the prompts focus
    on *why does this program fail* and *what would fix it*. The
    backend is the abstract :class:`LLMBackend`; production wires
    :class:`VLLMBackend`, tests wire a stub.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        primitive_whitelist: list[str] | None = None,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._backend = backend
        self._explicit_whitelist = primitive_whitelist
        self._config = config

    async def repair(
        self,
        program: ProgramNode,
        failing_demos: Iterable[tuple[Any, Any, Any]],
        diff: DiffReport | None = None,
    ) -> LLMRepairResult:
        """Run the Two-Stage repair prompt and return parsed suggestions.

        ``failing_demos`` is an iterable of ``(input, expected, actual)``
        triples. ``diff`` is an optional pre-computed
        :class:`DiffReport` from one of the demos — Sprint-1 hands it
        in for prompt richness; future sprints may compute one diff
        per demo and concatenate.
        """
        materialised = list(failing_demos)
        whitelist = self._resolve_whitelist()
        stage1_reasoning = await self._stage1(program, materialised, diff)
        raw_json, parsed = await self._stage2_with_retry(
            program, materialised, diff, stage1_reasoning, whitelist
        )
        suggestions = self._extract_suggestions(parsed)
        return LLMRepairResult(
            suggestions=suggestions,
            stage1_reasoning=stage1_reasoning,
            raw_response=raw_json,
        )

    # -- Internals ---------------------------------------------------

    def _resolve_whitelist(self) -> list[str]:
        if self._explicit_whitelist is not None:
            return list(self._explicit_whitelist)
        from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

        return list(REGISTRY.names())

    async def _stage1(
        self,
        program: ProgramNode,
        demos: list[tuple[Any, Any, Any]],
        diff: DiffReport | None,
    ) -> str:
        response = await self._backend.chat(
            model=self._config.llm_model_name,
            messages=[
                {"role": "system", "content": _STAGE1_SYSTEM},
                {
                    "role": "user",
                    "content": _build_stage1_user(program, demos, diff),
                },
            ],
            temperature=self._config.llm_temperature_stage1,
        )
        return response.content.strip()

    async def _stage2_with_retry(
        self,
        program: ProgramNode,
        demos: list[tuple[Any, Any, Any]],
        diff: DiffReport | None,
        stage1_reasoning: str,
        whitelist: list[str],
    ) -> tuple[str, dict[str, Any]]:
        attempts = 1 + max(0, self._config.llm_json_max_retries)
        last_error: Exception | None = None
        for _ in range(attempts):
            response = await self._backend.chat(
                model=self._config.llm_model_name,
                messages=[
                    {"role": "system", "content": _STAGE2_SYSTEM},
                    {
                        "role": "user",
                        "content": _build_stage2_user(
                            program,
                            demos,
                            diff,
                            primitive_whitelist=whitelist,
                            stage1_reasoning=stage1_reasoning,
                        ),
                    },
                ],
                temperature=self._config.llm_temperature_stage2,
                format_json=True,
            )
            try:
                parsed = json.loads(response.content)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(parsed, dict):
                last_error = LLMRepairError(
                    f"Stage-2 returned non-object JSON: {response.content[:120]!r}"
                )
                continue
            return response.content, parsed
        raise LLMRepairError(f"Stage-2 JSON parse failed after {attempts} attempts: {last_error}")

    def _extract_suggestions(
        self,
        parsed: dict[str, Any],
    ) -> tuple[LLMRepairSuggestion, ...]:
        raw_list = parsed.get("suggestions")
        if not isinstance(raw_list, list):
            raise LLMRepairError(
                "Stage-2 missing 'suggestions' list (or wrong type); "
                f"got: {type(raw_list).__name__}"
            )
        out: list[LLMRepairSuggestion] = []
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            source = entry.get("replacement_source")
            if not isinstance(source, str) or not source.strip():
                continue
            try:
                confidence = float(entry.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            if not math.isfinite(confidence):
                confidence = 0.5
            confidence = _clamp(confidence, 0.0, 1.0)
            reasoning_raw = entry.get("reasoning", "")
            reasoning = reasoning_raw if isinstance(reasoning_raw, str) else ""
            out.append(
                LLMRepairSuggestion(
                    replacement_source=source.strip(),
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
        # Stable sort by descending confidence; ties preserve LLM order.
        out.sort(key=lambda s: -s.confidence)
        return tuple(out)


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


__all__ = [
    "LLMRepairError",
    "LLMRepairResult",
    "LLMRepairSuggestion",
    "LLMRepairTwoStageClient",
]
