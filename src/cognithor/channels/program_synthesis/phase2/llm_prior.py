# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Module A — LLM-Prior over vLLM/Qwen 3.6 27B (spec v1.4 §4).

The LLM-Prior takes a Phase-1 :class:`TaskSpec` plus a depth context
and returns a per-primitive probability distribution that the
Dual-Prior mixer combines with the Symbolic-Prior. Spec §4 anchors the
contract; Sprint-1 ships:

* the **client** that talks to a vLLM-served Qwen 3.6 27B via the
  existing :class:`VLLMBackend` from ``cognithor.core.vllm_backend``;
* the **Two-Stage prompt schema** (CoT → JSON) per spec §4.7, with the
  retry-once-on-parse-failure rule;
* the **LLMPrior** dataclass (immutable), keyed by primitive name and
  carrying the alpha_entropy hint the mixer reads.

What it deliberately does not do yet:

* Constrained decoding (§4.6) — vLLM doesn't enforce a closed token
  set automatically. Sprint-1 filters the JSON keys against the live
  ``REGISTRY``; tightening this with vLLM ``guided_grammar`` lands
  later.
* Caching (§4.9) — α-aware mixing cache is a Module-A piece scheduled
  for the same Sprint-2 PR that lights up the symbolic-heuristic catalog.
* Top-K depth-dependent narrowing (§4.5) — Sprint-1 returns the full
  primitive distribution; the search engine slices it.

The client is fully :class:`Phase2Config`-driven. Tests inject a
fake ``LLMBackend`` so no vLLM has to be running for CI.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.phase2.alpha_mixer import alpha_bounds
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cognithor.core.llm_backend import LLMBackend


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMPrior:
    """One LLM-prior call's output.

    ``primitive_scores`` is keyed by primitive name and carries a
    softmax-like distribution over the registered primitives. Sums to
    approximately 1.0 (slack permitted within ``EPS`` for numerical
    drift). Names not present in the registered set are dropped before
    returning, so the consumer never has to whitelist again.

    ``alpha_entropy_hint`` is the LLM's self-reported confidence-from-
    entropy, clamped to the ``alpha_entropy`` band in :class:`Phase2Config`.
    The mixer multiplies it by ``alpha_performance`` to yield the final
    Search-α (spec §4.4.4).

    ``stage1_reasoning`` is the free-form CoT block from Stage 1; kept
    only for telemetry / debug, never read by the search engine.

    ``raw_response`` is the verbatim JSON string the LLM returned in
    Stage 2, useful for replay-debugging when the parser needs to be
    audited.
    """

    primitive_scores: dict[str, float]
    alpha_entropy_hint: float
    stage1_reasoning: str = ""
    raw_response: str = ""


class LLMPriorError(Exception):
    """Raised when the LLM-Prior call cannot produce a usable distribution."""


# ---------------------------------------------------------------------------
# Prompt schema (spec §4.7 Two-Stage)
# ---------------------------------------------------------------------------


_STAGE1_SYSTEM = (
    "You are a programmatic-synthesis advisor. Given an ARC-AGI grid task "
    "with paired example inputs and outputs, you reason briefly about what "
    "transformation is being applied — what changes (size, palette, "
    "structure, objects), what stays the same, and which DSL primitive "
    "categories most plausibly produce that change. Be concise: at most "
    "six sentences. Do not return a program. Do not return JSON. Return "
    "plain prose only."
)


_STAGE2_SYSTEM = (
    "You are a programmatic-synthesis advisor. Given the prior reasoning "
    "and the same task, return a JSON object whose keys are DSL primitive "
    "names from the provided whitelist and whose values are floats in "
    "[0, 1] approximating their relative usefulness for synthesizing this "
    "task. Include an additional key ``alpha_entropy_hint`` ∈ [0, 1] "
    "indicating how confident you are. Do not include any keys outside "
    "the whitelist. Do not include text outside the JSON object. Do not "
    "wrap the JSON in markdown code fences."
)


def _format_examples_block(examples: Iterable[tuple[Any, Any]]) -> str:
    rendered: list[str] = []
    for i, (inp, out) in enumerate(examples):
        # numpy / list-of-lists both support repr — keep it terse.
        rendered.append(
            f"Example {i + 1}:\n  input  = {_compact_repr(inp)}\n  output = {_compact_repr(out)}"
        )
    return "\n".join(rendered)


def _compact_repr(value: Any) -> str:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return repr(tolist())
    return repr(value)


def _build_stage1_user(examples: Iterable[tuple[Any, Any]]) -> str:
    return f"Task examples (input → output pairs):\n{_format_examples_block(examples)}"


def _build_stage2_user(
    examples: Iterable[tuple[Any, Any]],
    *,
    primitive_whitelist: list[str],
    stage1_reasoning: str,
) -> str:
    whitelist_block = ", ".join(sorted(primitive_whitelist))
    return (
        f"Task examples (input → output pairs):\n"
        f"{_format_examples_block(examples)}\n\n"
        f"Prior reasoning:\n{stage1_reasoning}\n\n"
        f"Allowed primitive keys (use only these): {whitelist_block}\n\n"
        "Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMPriorClient:
    """Two-Stage CoT→JSON LLM-prior client over an injected ``LLMBackend``.

    The backend is the abstract :class:`LLMBackend` from
    ``cognithor.core.llm_backend``; production wires
    :class:`VLLMBackend`, tests wire a stub. The client is otherwise
    pure: it formats the prompts, calls the backend twice (Stage-1 CoT,
    Stage-2 JSON with a single retry), parses and filters the JSON,
    and returns an :class:`LLMPrior`.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        primitive_whitelist: list[str] | None = None,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._backend = backend
        # Lazy-resolve the live REGISTRY if no explicit whitelist is
        # passed, but only at call time — keeps imports clean for tests
        # that don't need the registry.
        self._explicit_whitelist = primitive_whitelist
        self._config = config

    async def get_prior(
        self,
        examples: Iterable[tuple[Any, Any]],
    ) -> LLMPrior:
        """Run the two-stage prompt and return the parsed prior."""
        materialised = list(examples)
        whitelist = self._resolve_whitelist()
        stage1_reasoning = await self._stage1(materialised)
        raw_json, parsed = await self._stage2_with_retry(materialised, stage1_reasoning, whitelist)
        primitive_scores = self._extract_primitive_scores(parsed, whitelist)
        alpha_hint = self._extract_alpha_hint(parsed)
        return LLMPrior(
            primitive_scores=primitive_scores,
            alpha_entropy_hint=alpha_hint,
            stage1_reasoning=stage1_reasoning,
            raw_response=raw_json,
        )

    # -- Internals ---------------------------------------------------

    def _resolve_whitelist(self) -> list[str]:
        if self._explicit_whitelist is not None:
            return list(self._explicit_whitelist)
        # Lazy import — the prior module loads cleanly even without the
        # full DSL wired (e.g. for unit tests on the parser).
        from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

        return list(REGISTRY.names())

    async def _stage1(self, examples: list[tuple[Any, Any]]) -> str:
        response = await self._backend.chat(
            model=self._config.llm_model_name,
            messages=[
                {"role": "system", "content": _STAGE1_SYSTEM},
                {"role": "user", "content": _build_stage1_user(examples)},
            ],
            temperature=self._config.llm_temperature_stage1,
        )
        return response.content.strip()

    async def _stage2_with_retry(
        self,
        examples: list[tuple[Any, Any]],
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
                            examples,
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
                last_error = LLMPriorError(
                    f"Stage-2 returned non-object JSON: {response.content[:120]!r}"
                )
                continue
            return response.content, parsed
        raise LLMPriorError(f"Stage-2 JSON parse failed after {attempts} attempts: {last_error}")

    def _extract_primitive_scores(
        self,
        parsed: dict[str, Any],
        whitelist: list[str],
    ) -> dict[str, float]:
        allowed = set(whitelist)
        scores: dict[str, float] = {}
        for key, value in parsed.items():
            if key == "alpha_entropy_hint":
                continue
            if key not in allowed:
                # Quietly drop hallucinated primitives — spec §4.6's
                # "constrained decoding" intent.
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f) or f < 0.0:
                continue
            scores[key] = f
        if not scores:
            raise LLMPriorError("Stage-2 produced no usable primitive scores after filtering.")
        return _normalise(scores)

    def _extract_alpha_hint(self, parsed: dict[str, Any]) -> float:
        raw = parsed.get("alpha_entropy_hint")
        if raw is None:
            f = 0.5
        else:
            try:
                f = float(raw)
            except (TypeError, ValueError):
                f = 0.5  # neutral default — mid of the spec band
        if not math.isfinite(f):
            f = 0.5
        # Clamp to the configured α_entropy band so a misbehaving LLM
        # cannot push α outside the spec range.
        lo = self._config.alpha_entropy_lower
        hi = self._config.alpha_entropy_upper
        if f < lo:
            return lo
        if f > hi:
            return hi
        return f


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        # All zero — degrade to uniform over the keys we kept.
        n = len(scores) or 1
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


__all__ = [
    "LLMPrior",
    "LLMPriorClient",
    "LLMPriorError",
]


# Suppress unused-import lint — alpha_bounds is used by future Module
# A wiring (mixer integration). Keeping the import sets a clean import
# surface for the next sprint without an awkward re-export-only file.
_ = alpha_bounds
