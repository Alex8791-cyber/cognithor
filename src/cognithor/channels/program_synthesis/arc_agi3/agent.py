# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 — Cognithor PSE agent base + smoke-baseline.

:class:`CognithorPSEAgent` is the abstract base every Cognithor
ARC-AGI-3 agent inherits from. It mirrors the upstream
``arcprize.ARC-AGI-3-Agents.agents.Agent`` ABC contract: two abstract
methods ``is_done`` and ``choose_action``.

When running inside the official harness the upstream wrapper calls
:meth:`is_done` after every frame and :meth:`choose_action` to get
the next move; Cognithor's only contract with the harness is that
these two methods return the right types.

This module deliberately has **no dependency on arc-agi or
arcengine** — Cognithor's :class:`CognithorPSEAgent` is a thin local
abstraction that the Wave-5 :mod:`arcengine_adapter` plugs into the
real harness. That keeps the import graph clean for the 90 % of
Cognithor that doesn't care about ARC-AGI-3.

:class:`RandomActionAgent` is a concrete smoke-baseline that picks a
uniformly random action per frame. It exists so that Wave-1 has at
least one runnable agent; Wave-3 ships the real :class:`Sprint10DSLAgent`.
"""

from __future__ import annotations

import contextlib
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )


class CognithorPSEAgent(ABC):
    """Abstract base for every Cognithor PSE agent that plays ARC-AGI-3.

    Subclass contract:

    * :meth:`is_done` — given the full episode history and the latest
      frame, decide whether to stop. The official harness calls this
      every iteration; returning True ends the game.
    * :meth:`choose_action` — return the next :class:`GameActionProtocol`
      to send to the game.

    The base class is otherwise empty by design — the official
    upstream :class:`Agent` ABC owns the game-loop, recording, and
    network glue. Cognithor's job is just the two pure decisions.
    """

    @abstractmethod
    def is_done(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> bool:
        """Return True to stop the episode.

        Cognithor's default policy is: stop on WIN. The harness has
        its own MAX_ACTIONS guard, so we don't need a separate timeout
        here. Subclasses may override to also stop on GAME_OVER (e.g.
        for one-shot evaluation runs).
        """

    @abstractmethod
    def choose_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> GameActionProtocol:
        """Return the next action to send to the game.

        Must respect ``latest_frame.available_actions`` — picking an
        action outside that whitelist is undefined behaviour
        upstream. Implementations should set ``action.reasoning`` to
        a short natural-language string (used by the recorder).
        """


class RandomActionAgent(CognithorPSEAgent):
    """Smoke-baseline: picks one of the available actions at random.

    Sprint-11 Wave-1's only concrete agent. Useful for:

    * proving the :class:`CognithorPSEAgent` ABC contract is sound;
    * smoke-testing the frame ↔ action plumbing once Wave-2 lands;
    * a lower-bound score on each ARC-AGI-3 game (anything below
      this is broken, anything above starts measuring real ability).

    The agent stops on WIN. It does *not* RESET on GAME_OVER —
    Cognithor's evaluation runs are one-shot per game, not retry
    loops. (Subclasses can override :meth:`is_done` to keep playing.)
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def is_done(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> bool:
        return latest_frame.state.name in {"WIN", "GAME_OVER"}

    def choose_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> GameActionProtocol:
        actions = list(latest_frame.available_actions)
        if not actions:
            raise RuntimeError(
                f"RandomActionAgent: latest frame for {latest_frame.game_id!r} "
                "has no available_actions; cannot pick a move."
            )
        choice = self._rng.choice(actions)
        # The harness reads `reasoning` from the action object before
        # transmitting it. Some stub implementations may make the field
        # read-only; the harness still works without it set.
        with contextlib.suppress(AttributeError):
            choice.reasoning = "RandomActionAgent: uniform random over available actions"
        return choice


__all__ = [
    "CognithorPSEAgent",
    "RandomActionAgent",
]
