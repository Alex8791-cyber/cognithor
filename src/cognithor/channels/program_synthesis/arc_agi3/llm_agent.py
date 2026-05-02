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
from cognithor.channels.program_synthesis.arc_agi3.game_prompts import (
    build_system_prompt,
    game_prefix,
)
from cognithor.channels.program_synthesis.arc_agi3.goal_inferer import GoalInferer
from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    FrameContext,
    LLMActionDecoder,
    render_grid,
)
from cognithor.channels.program_synthesis.arc_agi3.llm_telemetry import (
    record_vllm_request_output,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
        StuckDetector,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
    from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import ChoiceFn
    from cognithor.channels.program_synthesis.arc_agi3.llm_telemetry import LLMTelemetry
    from cognithor.channels.program_synthesis.arc_agi3.mtp_stats import MTPStats
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


def _build_system_prompt(ctx: FrameContext) -> str:
    """System prompt = generic JSON contract + per-game rules.

    The Sprint-12 :mod:`game_prompts` module ships verbatim per-game
    rule sets (LS20 Locksmith mechanics, FT09 click hint, etc.). When
    ``ctx.game_id`` matches a known prefix, those rules are appended
    to the generic system prompt so the LLM gets game-specific
    grounding.
    """
    if ctx.game_id and game_prefix(ctx.game_id):
        return build_system_prompt(ctx.game_id, ", ".join(ctx.available_action_names))
    return _LLM_SYSTEM_PROMPT


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
    if ctx.goal_summary:
        parts.append(f"Goal hypothesis: {ctx.goal_summary}")
    # Sprint-16 anti-loop: surface per-state action counts + the
    # forbidden list so the LLM has a fighting chance of breaking out
    # of the deterministic-loop trap. The post-LLM override enforces
    # this even if the LLM ignores it, but the prompt-side hint is
    # what gives the LLM enough state to reason about *why* it's
    # being constrained.
    if ctx.state_action_summary:
        parts.append(f"At this state — action history: {ctx.state_action_summary}")
    if ctx.forbidden_action_names:
        forbidden_list = ", ".join(ctx.forbidden_action_names)
        parts.append(
            f"DO NOT pick: {forbidden_list} (already tried at this state without "
            "useful effect; pick a DIFFERENT action)."
        )
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
                    {"role": "system", "content": _build_system_prompt(ctx)},
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
    max_num_seqs: int = 4,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
    cuda_home: str = "/usr/local/cuda-13.0",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    kv_cache_dtype: str | None = None,
    speculative_config: dict[str, Any] | None = None,
    mtp_stats: MTPStats | None = None,
    telemetry: LLMTelemetry | None = None,
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
        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "dtype": "auto",
            # Sprint-16 perf: ARC-AGI-3 step prompts share ~8 KB of
            # system prompt + accumulated history across consecutive
            # calls; prefix caching reuses the prefill KV blocks for
            # the shared portion → 10-30 % wall-clock reduction on
            # iterative chat without changing acceptance/throughput
            # otherwise. Phase-A baseline showed
            # ``Prefix cache hit rate: 0.0 %`` (off by default).
            "enable_prefix_caching": True,
        }
        # Sprint-15 vLLM tuning: opt-in FP8 KV cache halves KV memory
        # footprint at <1% quality loss, enables much higher concurrency.
        if kv_cache_dtype is not None:
            llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
        # Sprint-15 vLLM tuning: opt-in MTP speculative decoding gives
        # ~1.9× decode throughput on Blackwell-class GPUs. Requires an
        # MTP-aware checkpoint (e.g. *-NVFP4-MTP family).
        if speculative_config is not None:
            llm_kwargs["speculative_config"] = speculative_config
        # Sprint-15 telemetry: vLLM's offline LLM class defaults
        # ``disable_log_stats=True`` which strips ``RequestOutput.metrics``
        # AND raises on ``LLM.get_metrics()``. Both are pre-conditions for
        # capturing TTFT and per-call MTP deltas. Re-enable when MTP or
        # telemetry is wired so the plumbing actually receives data; leave
        # it alone otherwise so callers that didn't ask for telemetry don't
        # pay the (small) logging overhead.
        if mtp_stats is not None or telemetry is not None:
            llm_kwargs["disable_log_stats"] = False
        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        _engine_state["llm"] = llm
        _engine_state["sampling"] = sampling
        return llm, sampling

    # vLLM 0.20-v1 doesn't carry ``spec_token_acceptance_counts`` per
    # ``RequestOutput`` — the spec-decode counts live on the engine's
    # cumulative Prometheus metrics. Track a running snapshot of the
    # cumulative counter and compute per-call deltas inside the
    # closure so ``mtp_stats.snapshots`` ends up with per-call entries
    # comparable to the per-request path on older vLLM.
    _last_engine_stats: dict[str, int] = {"drafts": 0, "accepted": 0, "emitted": 0}

    def _sync_choice(ctx: FrameContext) -> tuple[str, str]:
        import time as _time

        from cognithor.channels.program_synthesis.arc_agi3.mtp_stats import (
            MTPSnapshot,
            poll_engine_mtp_metrics,
        )

        llm, sampling = _ensure_engine()
        t0 = _time.monotonic()
        outs = llm.chat(
            messages=[
                {"role": "system", "content": _build_system_prompt(ctx)},
                {"role": "user", "content": _build_user_prompt(ctx)},
            ],
            sampling_params=sampling,
        )
        wall_clock_s = _time.monotonic() - t0
        # Sprint-15: capture per-call MTP + token telemetry into the
        # caller's aggregators if wired. Both side-effects only —
        # downstream behaviour is unchanged when the kwargs are None.
        req_out = outs[0]
        if mtp_stats is not None:
            # Prefer the per-request acceptance histogram if vLLM
            # exposes it (older vLLM); fall back to engine-cumulative
            # delta polling on vLLM 0.20-v1 which only ships
            # `SpecDecodingStats` engine-side.
            per_req = mtp_stats.add_request(req_out)
            if per_req is None:
                cumulative = poll_engine_mtp_metrics(llm)
                if cumulative is not None:
                    delta = MTPSnapshot(
                        drafts_proposed=max(
                            0, cumulative.drafts_proposed - _last_engine_stats["drafts"]
                        ),
                        drafts_accepted=max(
                            0,
                            cumulative.drafts_accepted - _last_engine_stats["accepted"],
                        ),
                        tokens_emitted=max(
                            0, cumulative.tokens_emitted - _last_engine_stats["emitted"]
                        ),
                        num_speculative_tokens=cumulative.num_speculative_tokens,
                    )
                    if delta.drafts_proposed > 0:
                        mtp_stats.snapshots.append(delta)
                    _last_engine_stats["drafts"] = cumulative.drafts_proposed
                    _last_engine_stats["accepted"] = cumulative.drafts_accepted
                    _last_engine_stats["emitted"] = cumulative.tokens_emitted
        if telemetry is not None:
            record_vllm_request_output(telemetry, req_out, wall_clock_s=wall_clock_s)
        text = req_out.outputs[0].text
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
        goal_inferer: GoalInferer | None = None,
        **parent_kwargs: Any,
    ) -> None:
        # Forward every Sprint10DSLAgent kwarg (audit_trail, game_profile,
        # strategy_name, frame_analyzer, fast_path_enabled,
        # click_target_sampler, state_counter, state_graph) to the parent
        # so the LLM agent automatically gets the same persistence +
        # analyzer + click-sampler plumbing as the heuristic agent.
        super().__init__(
            bridge=bridge,
            memory=memory,
            stuck_detector=stuck_detector,
            **parent_kwargs,
        )
        # Sprint-13: GoalInferer defaults to a stateless one when None
        # so the LLM always gets a goal-hypothesis line. Construction
        # cost is negligible (one int).
        if goal_inferer is None:
            goal_inferer = GoalInferer()
        # Sprint-16 Hebel 2: state-agnostic action-streak detector.
        # Hebel 1 (state_counter) didn't break the click-game loop
        # because each pick mutates the cursor pixel → new state hash
        # → counter resets. The streak detector instead looks at
        # recent action *names* (independent of state) and forbids any
        # action that dominates the last 4-of-5 picks while
        # ``levels_completed`` stays flat.
        from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
            ActionStreakDetector,
        )

        # Override the Wave-4 DSL decoder with the LLM-driven one. The
        # decoder also takes an optional frame_analyzer for prompt-side
        # action-effects rendering (Sprint-12 PR-12) and a goal_inferer
        # for goal-hypothesis injection (Sprint-13 PR-1). Sprint-16
        # forwards both the parent's state_counter (Hebel 1) and a
        # default ActionStreakDetector (Hebel 2) so the LLM agent
        # automatically gets the loop-breaking machinery without
        # opt-in.
        self._decoder = LLMActionDecoder(
            bridge=self._bridge,
            memory=self._memory,
            choice_fn=choice_fn,
            history_steps=history_steps,
            goal_inferer=goal_inferer,
            frame_analyzer=self._frame_analyzer,
            state_counter=self._state_counter,
            action_streak_detector=ActionStreakDetector(),
        )


__all__ = [
    "LLMReasoningAgent",
    "build_inprocess_vllm_choice_fn",
    "build_vllm_choice_fn",
]
