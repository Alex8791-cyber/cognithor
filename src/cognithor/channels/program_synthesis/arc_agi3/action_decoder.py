# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-2 — Programm output → GameAction decoder.

The DSL Cognithor inherited from Sprint-10 transforms a *grid* into
another *grid*. ARC-AGI-3 expects the agent to emit a *GameAction*
(one of ``RESET``, ``ACTION1``..``ACTION7``) per frame. Bridging the
two requires a policy that picks the action most likely to drive the
current frame towards a desired target.

This module ships the abstraction (:class:`ActionDecoder`) and a
spec-default :class:`UniformActionDecoder` that simply picks the
first available action per frame. The DSL- and LLM-driven decoders
land in Wave-3 and Wave-4 respectively; both subclass
:class:`ActionDecoder` and override :meth:`pick_action`.

The decoder is responsible for three things:

1. **Filtering** — only choose from ``frame.available_actions``.
   Picking outside that whitelist is undefined upstream.
2. **Reasoning text** — set ``action.reasoning`` to a short
   natural-language string. The harness records this for replay.
3. **Complex-action data** — for actions where ``is_complex()`` is
   ``True``, the decoder must call ``set_data({"x": .., "y": ..})``
   before returning; ``ACTION6`` and ``ACTION7`` need x/y target
   coordinates. Wave-2's :class:`UniformActionDecoder` defaults to
   ``(0, 0)`` for complex actions; subclasses override.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )


class ActionDecoder(ABC):
    """Abstract: pick one of ``frame.available_actions`` per frame.

    Subclasses implement :meth:`pick_action`. The base class handles
    the surrounding contract (filter + reasoning text + complex-action
    data wiring) by calling the subclass and then post-processing the
    returned action.
    """

    DEFAULT_REASONING: str = "Cognithor PSE: action picked by decoder"

    def decode(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> GameActionProtocol:
        """Return the next :class:`GameActionProtocol` for the harness.

        Raises :class:`RuntimeError` if the frame has no available
        actions. The returned action has ``reasoning`` set to the
        decoder's per-call justification, and (for complex actions)
        ``set_data`` already invoked.
        """
        actions = list(latest_frame.available_actions)
        if not actions:
            raise RuntimeError(
                f"{type(self).__name__}: latest frame for "
                f"{latest_frame.game_id!r} has no available_actions"
            )
        action, reasoning = self.pick_action(frames, latest_frame, actions)
        if action not in actions:
            raise RuntimeError(
                f"{type(self).__name__}: pick_action returned an action "
                f"({action.name!r}) that's not in available_actions"
            )
        # Read-only stubs may reject the assignment; safe to skip.
        with contextlib.suppress(AttributeError):
            action.reasoning = reasoning or self.DEFAULT_REASONING
        if action.is_complex():
            self._wire_complex_action_data(action, frames, latest_frame)
        return action

    @abstractmethod
    def pick_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
        available_actions: list[GameActionProtocol],
    ) -> tuple[GameActionProtocol, str]:
        """Return ``(chosen_action, reasoning_text)``.

        ``available_actions`` is a copy of
        ``latest_frame.available_actions`` for convenience and to
        guarantee the subclass can iterate without exhausting an
        iterator. The chosen action MUST be ``in available_actions``.
        """

    def _wire_complex_action_data(
        self,
        action: GameActionProtocol,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> None:
        """Default: ``(0, 0)`` target coordinates for complex actions.

        Subclasses override to derive coordinates from the current
        frame (e.g. centre of largest object, position of marker).
        """
        del frames, latest_frame  # unused in the default policy
        action.set_data({"x": 0, "y": 0})


class UniformActionDecoder(ActionDecoder):
    """Wave-2 baseline: always pick the first available action.

    Useful as a deterministic smoke baseline for the Wave-3 DSL
    decoder and the Wave-4 LLM decoder — when those regress, the
    test harness compares against ``UniformActionDecoder`` to
    isolate the regression to the policy layer.
    """

    def pick_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
        available_actions: list[GameActionProtocol],
    ) -> tuple[GameActionProtocol, str]:
        return available_actions[0], "UniformActionDecoder: first available"


__all__ = [
    "ActionDecoder",
    "UniformActionDecoder",
]
