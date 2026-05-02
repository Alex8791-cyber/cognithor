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
    parts = [
        f"Current grid ({ctx.grid.shape[0]}x{ctx.grid.shape[1]}):",
        render_grid(ctx.grid),
        "",
        f"Available actions: {', '.join(ctx.available_action_names)}",
        f"Recent history: {ctx.history_summary}",
    ]
    if ctx.action_effects_summary:
        parts.append(f"Learned action effects: {ctx.action_effects_summary}")
    parts.extend(
        [
            f"Progress: {ctx.levels_completed}/{ctx.win_levels} levels",
            "",
            "Pick one action and respond as JSON.",
        ]
    )
    return "\n".join(parts)


def build_vllm_choice_fn(
    *,
    backend: LLMBackend,
    model_name: str = "sakamakismile/Qwen3.6-27B-NVFP4",
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


def build_inprocess_vllm_choice_fn(
    *,
    model_name: str = "sakamakismile/Qwen3.6-27B-NVFP4",
    max_model_len: int = 32768,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.92,
    enforce_eager: bool = False,
    cuda_home: str = "/usr/local/cuda-13.0",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> ChoiceFn:
    """Sprint-12 production factory: in-process vLLM (no HTTP).

    Validated 2026-05-02 on RTX 5090 + WSL2 + CUDA 13.0:
    50 tok/s decode, 240 tok/s prefill, 370-400 W sustained. The
    in-process path sidesteps the WSL2 mirror-mode networking quirk
    that breaks vLLM's uvicorn HTTP server (TCP listen but accept()
    deadlocks).

    Loads the engine on first call. Subsequent calls reuse the
    warmed-up model. Parses Qwen3.6's ``<think>...</think>{json}``
    output format — the thinking block is stripped before JSON parse.

    Use this when running Cognithor inside the same Python process as
    vLLM (typical: WSL2 with vllm + cognithor in one venv). Use the
    HTTP-based :func:`build_vllm_choice_fn` only if vLLM runs in a
    separate process on a host where TCP works.
    """
    import os as _os

    if cuda_home and _os.path.isdir(cuda_home):
        _os.environ.setdefault("CUDA_HOME", cuda_home)
        _os.environ["PATH"] = f"{cuda_home}/bin:{_os.environ.get('PATH', '')}"

    # Heavy imports at first call to keep the module import-clean on
    # hosts without vllm installed.
    _engine_state: dict[str, Any] = {}

    def _ensure_engine() -> tuple[Any, Any]:
        if "llm" in _engine_state:
            return _engine_state["llm"], _engine_state["sampling"]
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Run `pip install vllm` inside a "
                "Linux + CUDA-capable venv (WSL2 Ubuntu 24.04 verified)."
            ) from exc
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            dtype="auto",
        )
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        _engine_state["llm"] = llm
        _engine_state["sampling"] = sampling
        return llm, sampling

    def _sync_choice(ctx: FrameContext) -> tuple[str, str]:
        llm, sampling = _ensure_engine()
        outs = llm.chat(
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(ctx)},
            ],
            sampling_params=sampling,
        )
        text = outs[0].outputs[0].text
        # Qwen3.6 thinking-mode wraps reasoning in <think>…</think>{json}.
        # Strip the thinking section before JSON parse so the action
        # extraction is unambiguous.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
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
    "build_inprocess_vllm_choice_fn",
    "build_vllm_choice_fn",
]
