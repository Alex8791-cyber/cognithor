# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-5 — LLMReasoningAgent over vLLM/qwen3.6:27b.

The Wave-5 :class:`LLMReasoningAgent` subclasses Wave-4's
:class:`Sprint10DSLAgent` and replaces the heuristic
:class:`DSLActionDecoder` with an LLM-driven
:class:`LLMActionDecoder`. The memory + stuck-detection plumbing is
inherited unchanged.

Construction is parameter-light: pass a ``choice_fn`` callable that
turns a :class:`FrameContext` into ``(action_name, reasoning)``.
Production wires :func:`build_vllm_choice_fn` (Sprint-10 Track B's
:class:`VLLMBackend`); tests pass deterministic stubs.

Without a running vLLM server the production factory falls back to
the Wave-4 :class:`DSLActionDecoder` policy at the first failed call.
This mirrors Sprint-10 Track B's hardware-gated wiring.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    FrameContext,
    LLMActionDecoder,
    render_grid,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
        StuckDetector,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
    from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import ChoiceFn
    from cognithor.core.llm_backend import LLMBackend


_LLM_SYSTEM_PROMPT = """You are an ARC-AGI-3 game-playing agent backed by Cognithor PSE.
You receive a frame from a small grid-based game and a list of available actions.
Pick the action that most likely makes progress toward winning the level.

Respond with strict JSON:
{
  "action": "ACTIONx",   // must be one of the available actions
  "reasoning": "<one short sentence>"
}

Do not output anything outside the JSON block."""


def _build_user_prompt(ctx: FrameContext) -> str:
    return (
        f"Current grid ({ctx.grid.shape[0]}x{ctx.grid.shape[1]}):\n"
        f"{render_grid(ctx.grid)}\n\n"
        f"Available actions: {', '.join(ctx.available_action_names)}\n"
        f"Recent history: {ctx.history_summary}\n"
        f"Progress: {ctx.levels_completed}/{ctx.win_levels} levels\n\n"
        f"Pick one action and respond as JSON."
    )


def build_vllm_choice_fn(
    *,
    backend: LLMBackend,
    model_name: str = "Qwen/Qwen3.6-27B-FP8",
    temperature: float = 0.3,
    timeout_seconds: float = 8.0,
) -> ChoiceFn:
    """Adapter: wrap a Sprint-10 :class:`VLLMBackend` as a synchronous
    :class:`ChoiceFn`.

    The wrapper does the async-in-sync via :func:`asyncio.run`. It
    parses the LLM's JSON response and returns ``(action, reasoning)``;
    on parse failure it raises, so the upstream
    :class:`LLMActionDecoder` catches it and falls back to the Wave-4
    DSL policy.

    Without a running vLLM server, the first call raises a connection
    error and the decoder falls back. The wiring is correct either
    way — it's hardware-gated, not code-gated.
    """

    async def _ask(ctx: FrameContext) -> tuple[str, str]:
        response = await asyncio.wait_for(
            backend.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(ctx)},
                ],
                temperature=temperature,
            ),
            timeout=timeout_seconds,
        )
        text = response.content.strip()
        # Find the JSON block — the model might wrap it in code fences.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"LLM response missing JSON block: {text[:200]!r}")
        parsed: dict[str, Any] = json.loads(text[start : end + 1])
        action_name = str(parsed.get("action", "")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()
        if not action_name:
            raise ValueError(f"LLM response missing 'action' field: {parsed!r}")
        return action_name, reasoning

    def _sync_choice(ctx: FrameContext) -> tuple[str, str]:
        return asyncio.run(_ask(ctx))

    return _sync_choice


class LLMReasoningAgent(Sprint10DSLAgent):
    """Wave-5: Sprint10DSLAgent with the decoder swapped for an LLM-driven one.

    The agent inherits the memory + stuck-detection + frame-bridging
    plumbing unchanged. Construction takes the same FrameBridge /
    EpisodeMemory / StuckDetector knobs as the parent, plus a
    mandatory ``choice_fn`` that drives the LLM call.
    """

    def __init__(
        self,
        *,
        choice_fn: ChoiceFn,
        bridge: FrameBridge | None = None,
        memory: EpisodeMemory | None = None,
        stuck_detector: StuckDetector | None = None,
        history_steps: int = 8,
    ) -> None:
        super().__init__(bridge=bridge, memory=memory, stuck_detector=stuck_detector)
        # Override the Wave-4 DSL decoder with the LLM-driven one.
        # Sprint10DSLAgent's choose_action delegates to ``self._decoder``,
        # so swapping it is sufficient — no additional override needed.
        self._decoder = LLMActionDecoder(
            bridge=self._bridge,
            memory=self._memory,
            choice_fn=choice_fn,
            history_steps=history_steps,
        )


__all__ = [
    "LLMReasoningAgent",
    "build_vllm_choice_fn",
]
