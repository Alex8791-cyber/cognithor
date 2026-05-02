# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-14 PR-1 — PlanningLLMReasoningAgent + vLLM planning choice-fns.

Wraps :class:`PlanningLLMActionDecoder` into a ready-to-use agent.
The system prompt asks the LLM for a JSON **array** of action steps;
the parser extracts up to ``plan_horizon`` :class:`PlanStep` entries
that the decoder then drives one step at a time.

Two factories ship for the production wiring:

* :func:`build_inprocess_vllm_planning_choice_fn` — in-process vLLM
  (validated path on RTX 5090 + WSL2 + NVFP4).
* :func:`build_vllm_planning_choice_fn` — HTTP-backed for the rare
  hosts where TCP works.

Both lazy-import their backends and reuse the :func:`build_*_vllm_*`
init machinery from :mod:`llm_agent`.
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
from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    FrameContext,
    render_grid,
)
from cognithor.channels.program_synthesis.arc_agi3.llm_telemetry import (
    record_vllm_request_output,
)
from cognithor.channels.program_synthesis.arc_agi3.planning_decoder import (
    PlanningLLMActionDecoder,
    PlanStep,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
        StuckDetector,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
    from cognithor.channels.program_synthesis.arc_agi3.goal_inferer import GoalInferer
    from cognithor.channels.program_synthesis.arc_agi3.llm_telemetry import LLMTelemetry
    from cognithor.channels.program_synthesis.arc_agi3.mtp_stats import MTPStats
    from cognithor.core.llm_backend import LLMBackend


__all__ = [
    "PlanningLLMReasoningAgent",
    "build_inprocess_vllm_planning_choice_fn",
    "build_inprocess_vllm_vision_planning_choice_fn",
    "build_vllm_planning_choice_fn",
    "parse_plan_response",
]


_PLANNING_SYSTEM_PROMPT = """You are an ARC-AGI-3 game-playing agent backed by Cognithor PSE.
You receive a frame from a small grid-based game and must produce a SHORT
PLAN of the next 3-5 actions you intend to take.

Respond with strict JSON — a single object with exactly two keys:
{
  "reasoning": "<one sentence describing the strategy>",
  "plan": [
    {"action": "ACTIONx", "data": null,                  "reasoning": "..."},
    {"action": "ACTION6", "data": {"x": 12, "y": 18},    "reasoning": "..."}
  ]
}

Rules:
- Each "action" must be one of the available actions named in the user message.
- "data" is null for simple actions (RESET, ACTION1..5) and {"x": ..., "y": ...}
  for click actions (ACTION6, ACTION7).
- Pick a plan length the situation justifies — 1 step if uncertain, up to 5 if
  you have a clear strategy. The agent will execute the plan one step at a
  time and re-plan if the level changes or it gets stuck.
- Do not output anything outside the JSON block."""


def _build_planning_system_prompt(ctx: FrameContext) -> str:
    if getattr(ctx, "game_id", "") and game_prefix(getattr(ctx, "game_id", "")):
        per_game = build_system_prompt(
            getattr(ctx, "game_id", ""), ", ".join(ctx.available_action_names)
        )
        return per_game + "\n\n" + _PLANNING_SYSTEM_PROMPT
    return _PLANNING_SYSTEM_PROMPT


def _build_planning_user_prompt(ctx: FrameContext) -> str:
    parts = [
        f"Current grid ({ctx.grid.shape[0]}x{ctx.grid.shape[1]}):",
        render_grid(ctx.grid),
        "",
        f"Available actions: {', '.join(ctx.available_action_names)}",
        f"Recent history: {ctx.history_summary}",
    ]
    if ctx.action_effects_summary:
        parts.append(f"Learned action effects: {ctx.action_effects_summary}")
    if getattr(ctx, "goal_summary", ""):
        parts.append(f"Goal hypothesis: {getattr(ctx, 'goal_summary', '')}")
    parts.extend(
        [
            f"Progress: {ctx.levels_completed}/{ctx.win_levels} levels",
            "",
            "Plan the next 3-5 actions and respond as JSON.",
        ]
    )
    return "\n".join(parts)


def parse_plan_response(text: str) -> tuple[list[PlanStep], str]:
    """Extract ``(plan_steps, top_level_reasoning)`` from raw LLM output.

    Tolerates Qwen3.6's ``<think>...</think>{json}`` wrapping and
    surrounding markdown fences. Raises :class:`ValueError` on unparseable
    or empty plans so the upstream decoder falls back cleanly.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"plan response missing JSON object: {text[:200]!r}")
    parsed: dict[str, Any] = json.loads(text[start : end + 1])
    plan_raw = parsed.get("plan") or []
    if not isinstance(plan_raw, list):
        raise ValueError(f"plan field must be a list, got {type(plan_raw).__name__}")
    steps: list[PlanStep] = []
    for entry in plan_raw:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action", "")).strip()
        if not action:
            continue
        data = entry.get("data")
        if data is not None and not isinstance(data, dict):
            data = None
        steps.append(
            PlanStep(
                action_name=action,
                data={"x": int(data["x"]), "y": int(data["y"])}
                if (data and "x" in data and "y" in data)
                else None,
                reasoning=str(entry.get("reasoning", "")).strip(),
            )
        )
    if not steps:
        raise ValueError(f"plan response had no usable steps: {parsed!r}")
    top_level_reasoning = str(parsed.get("reasoning", "")).strip()
    return steps, top_level_reasoning


def build_vllm_planning_choice_fn(
    *,
    backend: LLMBackend,
    model_name: str = "sakamakismile/Qwen3.6-27B-NVFP4",
    temperature: float = 0.3,
    timeout_seconds: float = 30.0,
) -> Any:
    """HTTP-backed planning choice-fn (mirror of :func:`build_vllm_choice_fn`)."""

    async def _ask(ctx: FrameContext) -> tuple[list[PlanStep], str]:
        response = await asyncio.wait_for(
            backend.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": _build_planning_system_prompt(ctx)},
                    {"role": "user", "content": _build_planning_user_prompt(ctx)},
                ],
                temperature=temperature,
            ),
            timeout=timeout_seconds,
        )
        return parse_plan_response(response.content.strip())

    def _sync_choice(ctx: FrameContext) -> tuple[list[PlanStep], str]:
        return asyncio.run(_ask(ctx))

    return _sync_choice


def build_inprocess_vllm_planning_choice_fn(
    *,
    model_name: str = "sakamakismile/Qwen3.6-27B-NVFP4",
    max_model_len: int = 32768,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.92,
    enforce_eager: bool = False,
    cuda_home: str = "/usr/local/cuda-13.0",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    kv_cache_dtype: str | None = None,
    speculative_config: dict[str, Any] | None = None,
    mtp_stats: MTPStats | None = None,
    telemetry: LLMTelemetry | None = None,
) -> Any:
    """In-process vLLM planning choice-fn — production path.

    Loads the engine on first call; reuses on subsequent. ``max_tokens``
    defaults higher than the single-step variant because a 5-step plan
    needs more room.

    Sprint-15 vLLM tuning knobs:

    * ``kv_cache_dtype="fp8"`` — halves KV memory for ~5-10% decode
      overhead. Big concurrency win on 32 GB GPUs.
    * ``speculative_config={"model": "...", "num_speculative_tokens": 3}``
      — MTP speculative decoding gives ~1.9× decode throughput on
      Blackwell. Requires an MTP-aware checkpoint (e.g. ``*-NVFP4-MTP``
      family). Note: those checkpoints are text-only — incompatible
      with the vision factory.
    """
    import os as _os

    if cuda_home and _os.path.isdir(cuda_home):
        _os.environ.setdefault("CUDA_HOME", cuda_home)
        _os.environ["PATH"] = f"{cuda_home}/bin:{_os.environ.get('PATH', '')}"

    _engine_state: dict[str, Any] = {}

    def _ensure_engine() -> tuple[Any, Any]:
        if "llm" in _engine_state:
            return _engine_state["llm"], _engine_state["sampling"]
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Run `pip install vllm` inside a Linux + CUDA-capable venv."
            ) from exc
        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "dtype": "auto",
        }
        if kv_cache_dtype is not None:
            llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if speculative_config is not None:
            llm_kwargs["speculative_config"] = speculative_config
        # Re-enable stats logging when telemetry/MTP-stats wired —
        # offline LLM defaults `disable_log_stats=True` which strips
        # RequestOutput.metrics and raises on get_metrics().
        if mtp_stats is not None or telemetry is not None:
            llm_kwargs["disable_log_stats"] = False
        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        _engine_state["llm"] = llm
        _engine_state["sampling"] = sampling
        return llm, sampling

    def _sync_choice(ctx: FrameContext) -> tuple[list[PlanStep], str]:
        import time as _time

        llm, sampling = _ensure_engine()
        t0 = _time.monotonic()
        outs = llm.chat(
            messages=[
                {"role": "system", "content": _build_planning_system_prompt(ctx)},
                {"role": "user", "content": _build_planning_user_prompt(ctx)},
            ],
            sampling_params=sampling,
        )
        wall_clock_s = _time.monotonic() - t0
        req_out = outs[0]
        if mtp_stats is not None:
            mtp_stats.add_request(req_out)
        if telemetry is not None:
            record_vllm_request_output(telemetry, req_out, wall_clock_s=wall_clock_s)
        return parse_plan_response(req_out.outputs[0].text)

    return _sync_choice


_VISION_PLANNING_USER_TEMPLATE = (
    "The image above is the current game grid (64x64 cells, ARC-AGI-3 16-colour"
    " palette). "
    "Available actions: {actions}. "
    "Recent history: {history}. "
    "{effects_line}"
    "{goal_line}"
    "Progress: {levels}/{win_levels} levels.\n\n"
    "Plan the next 3-5 actions and respond as JSON with the "
    '{{"reasoning": "...", "plan": [...]}} schema.'
)


def _build_vision_user_content(ctx: FrameContext) -> list[dict[str, Any]]:
    """Multimodal content list: image + text describing what to plan.

    The image is the rendered grid PNG; the text describes the
    available actions, history, effects, and goal hypothesis. This
    lets vision-capable LLMs (Qwen3.6) "see" the grid directly
    rather than parsing 4096 ASCII characters.
    """
    from cognithor.channels.program_synthesis.arc_agi3.vision_render import (
        render_grid_data_uri,
    )

    image_url = render_grid_data_uri(ctx.grid, scale=8)
    effects_line = (
        f"Learned action effects: {ctx.action_effects_summary}. "
        if ctx.action_effects_summary
        else ""
    )
    goal_line = (
        f"Goal hypothesis: {getattr(ctx, 'goal_summary', '')}. "
        if getattr(ctx, "goal_summary", "")
        else ""
    )
    text = _VISION_PLANNING_USER_TEMPLATE.format(
        actions=", ".join(ctx.available_action_names),
        history=ctx.history_summary,
        effects_line=effects_line,
        goal_line=goal_line,
        levels=ctx.levels_completed,
        win_levels=ctx.win_levels,
    )
    return [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": text},
    ]


def build_inprocess_vllm_vision_planning_choice_fn(
    *,
    model_name: str = "sakamakismile/Qwen3.6-27B-NVFP4",
    max_model_len: int = 32768,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.92,
    enforce_eager: bool = False,
    cuda_home: str = "/usr/local/cuda-13.0",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    grid_scale: int = 8,
    kv_cache_dtype: str | None = None,
    telemetry: LLMTelemetry | None = None,
) -> Any:
    """Vision-mode planning choice-fn: feed Qwen3.6 the grid as a PNG.

    Same engine init as the text-only variant; differs in the chat
    message construction — multimodal content list with the grid as a
    rendered image. The text part stays compact (actions / history /
    effects / goal) since the image carries the spatial info.

    ``grid_scale`` controls the upscale factor (default 8 → 64x64
    grid → 512x512 image).
    """
    import os as _os

    if cuda_home and _os.path.isdir(cuda_home):
        _os.environ.setdefault("CUDA_HOME", cuda_home)
        _os.environ["PATH"] = f"{cuda_home}/bin:{_os.environ.get('PATH', '')}"

    _engine_state: dict[str, Any] = {}

    def _ensure_engine() -> tuple[Any, Any]:
        if "llm" in _engine_state:
            return _engine_state["llm"], _engine_state["sampling"]
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Run `pip install vllm` inside a Linux + CUDA-capable venv."
            ) from exc
        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "dtype": "auto",
        }
        # Sprint-15 vLLM tuning: opt-in FP8 KV cache. MTP not supported
        # for the vision factory because vision-NVFP4-MTP checkpoints
        # don't exist as of 2026-05-02 (MTP variants are text-only).
        if kv_cache_dtype is not None:
            llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
        # Re-enable stats logging for the vision path too — without it
        # RequestOutput.metrics is stripped and TTFT capture silently
        # falls to zero.
        if telemetry is not None:
            llm_kwargs["disable_log_stats"] = False
        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        _engine_state["llm"] = llm
        _engine_state["sampling"] = sampling
        return llm, sampling

    def _sync_choice(ctx: FrameContext) -> tuple[list[PlanStep], str]:
        llm, sampling = _ensure_engine()
        # Override grid_scale per-call by building content here.
        from cognithor.channels.program_synthesis.arc_agi3.vision_render import (
            render_grid_data_uri,
        )

        image_url = render_grid_data_uri(ctx.grid, scale=grid_scale)
        text_after_image = _VISION_PLANNING_USER_TEMPLATE.format(
            actions=", ".join(ctx.available_action_names),
            history=ctx.history_summary,
            effects_line=(
                f"Learned action effects: {ctx.action_effects_summary}. "
                if ctx.action_effects_summary
                else ""
            ),
            goal_line=(
                f"Goal hypothesis: {getattr(ctx, 'goal_summary', '')}. "
                if getattr(ctx, "goal_summary", "")
                else ""
            ),
            levels=ctx.levels_completed,
            win_levels=ctx.win_levels,
        )
        import time as _time

        t0 = _time.monotonic()
        outs = llm.chat(
            messages=[
                {"role": "system", "content": _build_planning_system_prompt(ctx)},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": text_after_image},
                    ],
                },
            ],
            sampling_params=sampling,
        )
        wall_clock_s = _time.monotonic() - t0
        req_out = outs[0]
        # Vision factory: no MTP (no multimodal MTP-NVFP4 ckpt as of
        # 2026-05-02), but token + finish-reason telemetry is still
        # actionable for tuning the vision pass.
        if telemetry is not None:
            record_vllm_request_output(telemetry, req_out, wall_clock_s=wall_clock_s)
        return parse_plan_response(req_out.outputs[0].text)

    return _sync_choice


class PlanningLLMReasoningAgent(Sprint10DSLAgent):
    """Sprint-14 LLM agent using :class:`PlanningLLMActionDecoder`.

    Drop-in replacement for :class:`LLMReasoningAgent` whose decoder
    asks the LLM for *plans* instead of single actions. Same persistence
    + analyzer + click-fast-path plumbing as the heuristic agent (all
    Sprint-12/13 kwargs forward to ``Sprint10DSLAgent.__init__``).
    """

    def __init__(
        self,
        *,
        choice_fn: Any,
        bridge: FrameBridge | None = None,
        memory: EpisodeMemory | None = None,
        stuck_detector: StuckDetector | None = None,
        history_steps: int = 8,
        plan_horizon: int = 5,
        goal_inferer: GoalInferer | None = None,
        **parent_kwargs: Any,
    ) -> None:
        super().__init__(
            bridge=bridge,
            memory=memory,
            stuck_detector=stuck_detector,
            **parent_kwargs,
        )
        if goal_inferer is None:
            try:
                from cognithor.channels.program_synthesis.arc_agi3.goal_inferer import (
                    GoalInferer as _GI,
                )

                goal_inferer = _GI()
            except ImportError:
                goal_inferer = None
        self._decoder = PlanningLLMActionDecoder(
            bridge=self._bridge,
            memory=self._memory,
            choice_fn=choice_fn,
            history_steps=history_steps,
            plan_horizon=plan_horizon,
            frame_analyzer=self._frame_analyzer,
            goal_inferer=goal_inferer,
        )

    @property
    def plan_remaining(self) -> int:
        """Test/debug accessor for the active plan queue depth."""
        decoder = self._decoder
        if isinstance(decoder, PlanningLLMActionDecoder):
            return decoder.plan_remaining
        return 0
