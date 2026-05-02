# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-5 — LLM-driven action decoder.

The Wave-5 :class:`LLMActionDecoder` replaces Wave-4's heuristic
"least-tried action" policy with an LLM call: per frame, format a
prompt that describes the current grid + recent history + available
actions, ask the LLM which action to take, and return the
LLM-suggested action.

The actual LLM call is injected as a ``prompt_to_choice`` callable.
This keeps the decoder synchronous (mandatory — the upstream
``Agent.choose_action`` is sync) AND testable (tests pass a stub
callable that returns deterministic choices). A production factory
:func:`build_vllm_choice_fn` wraps a Sprint-10 :class:`VLLMBackend`
into the right shape, doing the async-in-sync via :func:`asyncio.run`.

Without a running vLLM server, the production factory fails fast on
the first call; the decoder then falls back to its Wave-4
:class:`DSLActionDecoder` policy if a fallback is configured. This
mirrors Sprint-10 Track B's hardware-gated wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.arc_agi3.action_decoder import ActionDecoder
from cognithor.channels.program_synthesis.arc_agi3.dsl_action_decoder import (
    DSLActionDecoder,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )

    _Grid = NDArray[np.int8]


@dataclass(frozen=True)
class FrameContext:
    """Bundle of inputs the LLM-prompt-builder reads.

    The decoder constructs this once per :meth:`pick_action` call,
    then hands it to the injected ``prompt_to_choice`` callable. The
    callable returns ``(action_name, reasoning)``; the decoder
    matches the name against ``available_actions`` and returns the
    matching action.
    """

    grid: _Grid
    available_action_names: list[str]
    history_summary: str
    levels_completed: int
    win_levels: int


# A callable that takes a :class:`FrameContext` and returns
# ``(chosen_action_name, reasoning_text)``. The action name MUST be
# one of ``ctx.available_action_names``; the decoder validates this.
ChoiceFn = Callable[[FrameContext], tuple[str, str]]


def render_grid(grid: _Grid) -> str:
    """Render an ``int8`` grid as a multi-line ASCII string for the LLM.

    Single-digit values mean each row is a contiguous ``012345``-style
    string. The renderer pads with spaces between columns for readability,
    which makes the column structure visible in the LLM's chat-window
    rendering.
    """
    if grid.ndim != 2:
        raise ValueError(f"render_grid: expected 2-D grid, got {grid.ndim}-D")
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def summarise_history(memory: EpisodeMemory, max_steps: int = 8) -> str:
    """Compact text description of the last ``max_steps`` actions.

    Format: ``"step -3: ACTION1 (no change), step -2: RESET (level up), ..."``
    Used in the Stage-1 prompt so the LLM knows what's been tried.
    Empty history yields ``"(no actions yet)"``.
    """
    if max_steps < 1:
        raise ValueError(f"summarise_history: max_steps must be >= 1, got {max_steps}")
    window = memory.window(max_steps)
    if not window:
        return "(no actions yet)"
    parts: list[str] = []
    for i, step in enumerate(window):
        # ``window`` is most-recent first; show step -1 for the
        # most recent, step -2 for the prior, etc.
        idx = -(i + 1)
        levels_marker = f", level={step.levels_completed}" if step.levels_completed > 0 else ""
        parts.append(f"step {idx}: {step.action_name}{levels_marker}")
    return "; ".join(parts)


class LLMActionDecoder(ActionDecoder):
    """Stateful decoder that delegates the choice to an LLM.

    Construct with:

    * ``bridge`` — :class:`FrameBridge` to convert the live frame to
      the Cognithor int8 grid that gets rendered for the prompt
    * ``memory`` — :class:`EpisodeMemory` for the history-summary
    * ``choice_fn`` — synchronous callable from
      :class:`FrameContext` to ``(action_name, reasoning)``. Production
      uses :func:`build_vllm_choice_fn`; tests pass deterministic stubs.
    * ``fallback`` (optional) — another :class:`ActionDecoder` used
      when ``choice_fn`` raises or returns an unknown action.
      Default: a fresh :class:`DSLActionDecoder` over the same memory.
    * ``history_steps`` — how many recent steps to summarise in the
      prompt. Default 8.
    """

    def __init__(
        self,
        *,
        bridge: FrameBridge,
        memory: EpisodeMemory,
        choice_fn: ChoiceFn,
        fallback: ActionDecoder | None = None,
        history_steps: int = 8,
    ) -> None:
        self._bridge = bridge
        self._memory = memory
        self._choice_fn = choice_fn
        self._fallback = fallback if fallback is not None else DSLActionDecoder(memory=memory)
        self._history_steps = history_steps

    def pick_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
        available_actions: list[GameActionProtocol],
    ) -> tuple[GameActionProtocol, str]:
        # Build the prompt context.
        try:
            grid = self._bridge.extract_grid(latest_frame)
        except Exception as exc:  # pragma: no cover — bridge errors propagate
            return self._fallback.pick_action(frames, latest_frame, available_actions) + (
                f" [LLM bridge failed: {type(exc).__name__}]",
            )[0:0] or self._fallback.pick_action(frames, latest_frame, available_actions)

        ctx = FrameContext(
            grid=grid,
            available_action_names=[a.name for a in available_actions],
            history_summary=summarise_history(self._memory, self._history_steps),
            levels_completed=latest_frame.levels_completed,
            win_levels=latest_frame.win_levels,
        )

        # Call the LLM (or stub). Failures fall back to the DSL decoder.
        try:
            chosen_name, reasoning = self._choice_fn(ctx)
        except Exception as exc:
            fallback_action, fb_reasoning = self._fallback.pick_action(
                frames, latest_frame, available_actions
            )
            return fallback_action, (
                f"LLMActionDecoder fallback ({type(exc).__name__}): {fb_reasoning}"
            )

        # Validate the LLM's name against the whitelist.
        for action in available_actions:
            if action.name == chosen_name:
                return action, reasoning or f"LLMActionDecoder chose {chosen_name}"

        # LLM picked an action not in the whitelist — fall back.
        fallback_action, fb_reasoning = self._fallback.pick_action(
            frames, latest_frame, available_actions
        )
        return fallback_action, (
            f"LLMActionDecoder fallback (LLM returned {chosen_name!r} which is "
            f"not available): {fb_reasoning}"
        )


__all__ = [
    "ChoiceFn",
    "FrameContext",
    "LLMActionDecoder",
    "render_grid",
    "summarise_history",
]
