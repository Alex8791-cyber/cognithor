# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-15 — LLM call telemetry for evidence-based tuning.

Phase-A blind-spot: every Qwen3.6 inference takes 30-40s and we have
no idea whether the time goes into prefill (input tokens), decode
(output tokens), or wasted ``<think>`` ramble that hits ``max_tokens``
and gets truncated. Without that data, every tuning decision (raise
``max_model_len``? raise ``max_tokens``? add KV-FP8?) is a guess.

This module ships :class:`LLMTelemetry`, a per-episode aggregator
that records for each LLM call:

* ``input_tokens`` — tokens fed to the model (prompt size).
* ``output_tokens`` — tokens emitted including any ``<think>`` block.
* ``think_tokens`` — tokens consumed by Qwen3.6's ``<think>...
  </think>`` block, if present (so we can split reasoning from final
  answer cost).
* ``finish_reason`` — ``"stop"`` (model done), ``"length"`` (hit
  ``max_tokens``), or ``"abort"`` (engine error).
* ``wall_clock_s`` — end-to-end choice-fn duration.

Two integration paths:

1. ``wrap_text_choice_fn(fn)`` — decorator for the single-step
   choice-fn signature ``(ctx) -> (action_name, reasoning)``.
2. ``wrap_planning_choice_fn(fn)`` — decorator for the planning
   signature ``(ctx) -> (list[PlanStep], reasoning)``.

Both decorators preserve the original return value and add a
side-effect: append a :class:`LLMCallRecord` to the wrapped
:class:`LLMTelemetry`. Use :meth:`summary` to print or persist the
aggregated stats at episode end.

The telemetry is **non-invasive** — without wrapping, the existing
choice-fns behave exactly as before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
        FrameContext,
    )

__all__ = [
    "LLMCallRecord",
    "LLMTelemetry",
    "estimate_token_count",
    "extract_think_tokens",
    "record_vllm_request_output",
    "wrap_planning_choice_fn",
    "wrap_text_choice_fn",
]


@dataclass(frozen=True)
class LLMCallRecord:
    """One LLM call's measurements."""

    call_index: int
    input_tokens: int
    output_tokens: int
    think_tokens: int
    finish_reason: str
    wall_clock_s: float

    @property
    def post_think_tokens(self) -> int:
        """Output tokens that were NOT inside the ``<think>`` block."""
        return max(0, self.output_tokens - self.think_tokens)


@dataclass
class LLMTelemetry:
    """Per-episode aggregator for LLM call records.

    Pass an instance to :func:`wrap_text_choice_fn` or
    :func:`wrap_planning_choice_fn` and the wrapped fn will append
    one :class:`LLMCallRecord` per call.

    The aggregator is intentionally trivial — no dependencies, no
    persistence; the caller decides when to dump it (typically at
    ``EpisodeRunner`` finalisation alongside the audit-trail JSONL).
    """

    records: list[LLMCallRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def summary(self) -> dict[str, Any]:
        """Aggregate stats useful for tuning decisions.

        Reports:
        * ``calls`` — total LLM calls
        * ``finish_reason_dist`` — ``{"stop": N1, "length": N2, ...}``
          (Claude-style truncation diagnostic)
        * ``input_tokens_{avg,max,total}``
        * ``output_tokens_{avg,max,total}``
        * ``think_tokens_{avg,max,total}``
        * ``wall_clock_s_{avg,max,total}``
        * ``length_truncation_rate`` — fraction of calls hitting
          ``max_tokens``; if > 0.10 you should raise the cap.
        """
        n = len(self.records)
        if n == 0:
            return {"calls": 0}

        from collections import Counter

        finish_dist = dict(Counter(r.finish_reason for r in self.records))
        in_t = [r.input_tokens for r in self.records]
        out_t = [r.output_tokens for r in self.records]
        think_t = [r.think_tokens for r in self.records]
        wall = [r.wall_clock_s for r in self.records]
        return {
            "calls": n,
            "finish_reason_dist": finish_dist,
            "length_truncation_rate": finish_dist.get("length", 0) / n,
            "input_tokens_avg": sum(in_t) / n,
            "input_tokens_max": max(in_t),
            "input_tokens_total": sum(in_t),
            "output_tokens_avg": sum(out_t) / n,
            "output_tokens_max": max(out_t),
            "output_tokens_total": sum(out_t),
            "think_tokens_avg": sum(think_t) / n,
            "think_tokens_max": max(think_t),
            "think_tokens_total": sum(think_t),
            "wall_clock_s_avg": sum(wall) / n,
            "wall_clock_s_max": max(wall),
            "wall_clock_s_total": sum(wall),
        }


# --------------------------------------------------------------------------
# Token-estimation helpers
# --------------------------------------------------------------------------


def estimate_token_count(text: str) -> int:
    """Rough token count from character length.

    Uses the ~4-chars-per-token English approximation (close enough
    for telemetry; for hard sizing use a real tokenizer). Returns 0
    for empty input.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_think_tokens(text: str) -> int:
    """Return the estimated token count inside ``<think>...</think>``.

    Qwen3.6 wraps its reasoning trace in ``<think>...</think>{json}``;
    this helper isolates the reasoning portion so the summary can
    distinguish "model thought a lot" from "model produced a long
    answer".

    Edge case (length-truncation): when the model hits ``max_tokens``
    mid-``<think>`` and never emits the closing ``</think>``, the
    *entire* output is reasoning. Treat that as "all output is think"
    rather than "no think" — those long-truncated calls are exactly
    the diagnostic-richest ones for the workload-MTP-mismatch
    hypothesis, and silently dropping them to ``think_tokens=0`` would
    invert the signal in the Reasoning-vs-Output split.
    """
    if "<think>" in text and "</think>" not in text:
        # Truncation case: open tag but no close — count from after
        # ``<think>`` to EOF.
        head = text.split("<think>", 1)[1]
        return estimate_token_count(head)
    if "</think>" not in text:
        return 0
    head, _, _ = text.partition("</think>")
    # Strip leading "<think>" if present.
    if "<think>" in head:
        head = head.split("<think>", 1)[1]
    return estimate_token_count(head)


# --------------------------------------------------------------------------
# Decorators
# --------------------------------------------------------------------------


def _record_call(
    telemetry: LLMTelemetry,
    *,
    input_text: str,
    output_text: str,
    finish_reason: str,
    wall_clock_s: float,
) -> None:
    record = LLMCallRecord(
        call_index=len(telemetry.records),
        input_tokens=estimate_token_count(input_text),
        output_tokens=estimate_token_count(output_text),
        think_tokens=extract_think_tokens(output_text),
        finish_reason=finish_reason,
        wall_clock_s=wall_clock_s,
    )
    telemetry.records.append(record)


def _ctx_to_input_text(ctx: FrameContext) -> str:
    """Approximate the prompt's character length from the context."""
    parts: list[str] = [
        ctx.history_summary,
        " ".join(ctx.available_action_names),
    ]
    parts.append(getattr(ctx, "action_effects_summary", ""))
    parts.append(getattr(ctx, "goal_summary", ""))
    parts.append(getattr(ctx, "game_id", ""))
    # The grid contributes the bulk of the prompt; approximate as
    # one token per cell (rough — vision input would actually be
    # much smaller, but this errs on the safe side for diagnostics).
    if hasattr(ctx, "grid") and ctx.grid is not None:
        parts.append("X" * (4 * ctx.grid.size))
    return "\n".join(parts)


def record_vllm_request_output(
    telemetry: LLMTelemetry,
    request_output: Any,
    *,
    wall_clock_s: float,
    input_text_for_estimate: str = "",
) -> None:
    """Record a real vLLM ``RequestOutput`` into the telemetry.

    Use this from inside an in-process choice-fn (where you have the
    actual ``llm.chat(...)`` return value) to capture vLLM's
    authoritative numbers:

    * ``request_output.prompt_token_ids`` length → exact ``input_tokens``
    * ``request_output.outputs[0].token_ids`` length → exact
      ``output_tokens``
    * ``request_output.outputs[0].finish_reason`` → exact reason
      (``stop`` / ``length`` / ``tool_calls`` / ``abort``)

    The ``input_text_for_estimate`` fallback is used when
    ``prompt_token_ids`` isn't accessible (some vLLM versions hide it).
    """
    out = None
    finish_reason = "stop"
    output_text = ""
    output_tokens = 0
    try:
        if request_output.outputs:
            out = request_output.outputs[0]
            output_text = getattr(out, "text", "") or ""
            output_tokens = len(getattr(out, "token_ids", []) or [])
            finish_reason = getattr(out, "finish_reason", "stop") or "stop"
    except Exception:
        finish_reason = "abort"

    input_tokens = 0
    try:
        prompt_ids = getattr(request_output, "prompt_token_ids", None)
        if prompt_ids:
            input_tokens = len(prompt_ids)
    except Exception:
        pass
    if input_tokens == 0:
        input_tokens = estimate_token_count(input_text_for_estimate)
    if output_tokens == 0:
        output_tokens = estimate_token_count(output_text)

    record = LLMCallRecord(
        call_index=len(telemetry.records),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        think_tokens=extract_think_tokens(output_text),
        finish_reason=str(finish_reason),
        wall_clock_s=wall_clock_s,
    )
    telemetry.records.append(record)


def wrap_text_choice_fn(
    fn: Callable[[FrameContext], tuple[str, str]],
    telemetry: LLMTelemetry,
) -> Callable[[FrameContext], tuple[str, str]]:
    """Decorator for single-step text choice-fns.

    The wrapped fn returns the same ``(action_name, reasoning)`` tuple
    but also appends a :class:`LLMCallRecord` to ``telemetry`` per
    call. The reasoning string is treated as the *output text* for
    token estimation (which gives a small underestimate — it doesn't
    include any stripped ``<think>`` block — but the decorator can't
    see the raw model output through the choice-fn boundary).
    """

    def _wrapped(ctx: FrameContext) -> tuple[str, str]:
        t0 = time.monotonic()
        finish_reason = "stop"
        try:
            action_name, reasoning = fn(ctx)
        except Exception:
            finish_reason = "abort"
            _record_call(
                telemetry,
                input_text=_ctx_to_input_text(ctx),
                output_text="",
                finish_reason=finish_reason,
                wall_clock_s=time.monotonic() - t0,
            )
            raise
        _record_call(
            telemetry,
            input_text=_ctx_to_input_text(ctx),
            output_text=reasoning,
            finish_reason=finish_reason,
            wall_clock_s=time.monotonic() - t0,
        )
        return action_name, reasoning

    return _wrapped


def wrap_planning_choice_fn(
    fn: Callable[[FrameContext], tuple[Any, str]],
    telemetry: LLMTelemetry,
) -> Callable[[FrameContext], tuple[Any, str]]:
    """Decorator for multi-step planning choice-fns.

    Same telemetry semantics as :func:`wrap_text_choice_fn`. The
    "output text" is the top-level reasoning; the plan-step list is
    not directly token-measured (its length is bounded and predictable
    given ``plan_horizon``).
    """

    def _wrapped(ctx: FrameContext) -> tuple[Any, str]:
        t0 = time.monotonic()
        finish_reason = "stop"
        try:
            plan, reasoning = fn(ctx)
        except Exception:
            finish_reason = "abort"
            _record_call(
                telemetry,
                input_text=_ctx_to_input_text(ctx),
                output_text="",
                finish_reason=finish_reason,
                wall_clock_s=time.monotonic() - t0,
            )
            raise
        _record_call(
            telemetry,
            input_text=_ctx_to_input_text(ctx),
            output_text=reasoning,
            finish_reason=finish_reason,
            wall_clock_s=time.monotonic() - t0,
        )
        return plan, reasoning

    return _wrapped
