# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — in-process EpisodeRunner for the new arc_agi3 stack.

The official ``arcprize/ARC-AGI-3-Agents`` harness drives the game
loop itself when running benchmarks. For ad-hoc episodes (MCP tools,
tests, debugging) we want an in-process runner that drives a
:class:`CognithorPSEAgent` against a connected ``arc_agi.Arcade()``
without the full harness.

This module ships :class:`EpisodeRunner` — a thin loop:

1. Lazy-import ``arc_agi`` (raises a clear error if not installed).
2. Reset the env, hand the first frame to the agent.
3. Repeat: call ``agent.choose_action(...)``, send the action back to
   the env, store the result. Stop on WIN, GAME_OVER, or
   ``max_steps``.
4. Return :class:`EpisodeResult` with the headline numbers.

``EpisodeRunner`` is intentionally narrow: it doesn't manage profiles
or audit trails itself — those are the agent's concern (the new
:class:`Sprint10DSLAgent` already wires both via PR-6/7). The runner
just provides the event loop the harness would otherwise provide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.agent import CognithorPSEAgent
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
    )

__all__ = ["EpisodeResult", "EpisodeRunner", "run_episode"]

log = get_logger(__name__)


_INSTALL_HINT = (
    "arc_agi (the ARC-AGI-3 SDK) is not installed. Install via:\n"
    "  uv pip install arc-agi\n"
    "or run inside a venv that already has it (e.g. the official\n"
    "ARC-AGI-3-Agents harness venv with `uv sync` already done)."
)


@dataclass(frozen=True)
class EpisodeResult:
    """Headline numbers from a completed (or aborted) episode."""

    game_id: str
    levels_completed: int
    win_levels: int
    total_steps: int
    final_state: str  # "WIN" / "GAME_OVER" / "TIMEOUT" / "ERROR"
    won: bool
    score: float  # levels_completed / win_levels (clamped to [0, 1])
    error: str | None = None


class EpisodeRunner:
    """Drives one Cognithor agent through one game session in-process.

    Usage::

        from cognithor.channels.program_synthesis.arc_agi3 import (
            Sprint10DSLAgent, EpisodeRunner
        )
        agent = Sprint10DSLAgent(fast_path_enabled=True)
        runner = EpisodeRunner(agent=agent, game_id="ls20", max_steps=80)
        result = runner.run()
        print(f"score={result.score} levels={result.levels_completed}")
    """

    def __init__(
        self,
        *,
        agent: CognithorPSEAgent,
        game_id: str,
        max_steps: int = 80,
    ) -> None:
        self._agent = agent
        self._game_id = game_id
        self._max_steps = max_steps

    def run(self) -> EpisodeResult:
        """Drive one episode end-to-end. Catches arcengine import +
        connection errors and returns a populated :class:`EpisodeResult`
        instead of raising.
        """
        try:
            arcade, env, first_frame = self._connect()
        except RuntimeError as exc:
            log.error("episode_runner.connect_failed", game_id=self._game_id, error=str(exc))
            return EpisodeResult(
                game_id=self._game_id,
                levels_completed=0,
                win_levels=0,
                total_steps=0,
                final_state="ERROR",
                won=False,
                score=0.0,
                error=str(exc),
            )

        return self._loop(env, first_frame)

    def _connect(self) -> tuple[Any, Any, FrameDataProtocol]:
        try:
            import arc_agi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        arcade = arc_agi.Arcade()
        env = arcade.make(self._game_id)
        if env is None:
            raise RuntimeError(f"arc_agi.Arcade.make({self._game_id!r}) returned None")
        first_frame = self._normalize_frame(env.reset())
        return arcade, env, first_frame

    @staticmethod
    def _normalize_frame(frame: Any) -> Any:
        """Ensure ``frame.available_actions`` are GameAction enum members.

        The pydantic-validated ``FrameData`` from arc_agi 0.9.x coerces
        the enum to bare ints. Cognithor agents read ``action.name`` /
        ``action.value`` / call ``action.set_data(...)`` — all attribute
        accesses that fail on raw ints. We re-wrap each available action
        as ``arcengine.GameAction.from_id(int)`` so the agent sees real
        enum members again. Same for ``frame.state`` if it's a bare str.
        """
        try:
            from arcengine import GameAction, GameState
        except ImportError:
            return frame  # arcengine missing — let downstream code fail naturally
        actions = getattr(frame, "available_actions", None)
        if actions is not None:
            normalized = []
            for a in actions:
                if isinstance(a, int):
                    normalized.append(GameAction.from_id(a))
                else:
                    normalized.append(a)
            try:
                frame.available_actions = normalized
            except Exception:
                # Pydantic frame may be frozen — try replace via model_copy.
                if hasattr(frame, "model_copy"):
                    frame = frame.model_copy(update={"available_actions": normalized})
        state = getattr(frame, "state", None)
        if isinstance(state, str):
            import contextlib

            try:
                frame.state = GameState(state)
            except Exception:
                if hasattr(frame, "model_copy"):
                    with contextlib.suppress(Exception):
                        frame = frame.model_copy(update={"state": GameState(state)})
        return frame

    def _loop(self, env: Any, first_frame: FrameDataProtocol) -> EpisodeResult:
        frames: list[FrameDataProtocol] = [first_frame]
        latest = first_frame
        steps_taken = 0
        try:
            while steps_taken < self._max_steps:
                if self._agent.is_done(frames, latest):
                    break
                action = self._agent.choose_action(frames, latest)
                # The arc_agi env.step accepts the GameAction value (or
                # ``(value, data)`` for click actions). Action6/Action7
                # carry click coords via set_data() inside the agent.
                payload = self._action_payload(action)
                latest = self._normalize_frame(env.step(*payload))
                frames.append(latest)
                steps_taken += 1
        except Exception as exc:
            log.error("episode_runner.loop_error", game_id=self._game_id, error=str(exc))
            return EpisodeResult(
                game_id=self._game_id,
                levels_completed=int(getattr(latest, "levels_completed", 0)),
                win_levels=int(getattr(latest, "win_levels", 1)),
                total_steps=steps_taken,
                final_state="ERROR",
                won=False,
                score=0.0,
                error=str(exc),
            )

        final_state = latest.state.name
        levels_completed = int(latest.levels_completed)
        win_levels = max(int(latest.win_levels), 1)
        won = final_state == "WIN"
        score = min(levels_completed / win_levels, 1.0)

        # Hand the agent a chance to roll up its persistence (audit +
        # profile) when the wiring exists. Sprint10DSLAgent has
        # finalize_episode; the ABC doesn't, so we duck-check.
        finalize = getattr(self._agent, "finalize_episode", None)
        if callable(finalize):
            try:
                finalize(
                    score=levels_completed,
                    won=won,
                    levels_solved=levels_completed,
                    budget_ratio=steps_taken / self._max_steps if self._max_steps else 0.0,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("episode_runner.finalize_failed", error=str(exc))

        return EpisodeResult(
            game_id=self._game_id,
            levels_completed=levels_completed,
            win_levels=win_levels,
            total_steps=steps_taken,
            final_state=final_state,
            won=won,
            score=score,
        )

    @staticmethod
    def _action_payload(action: Any) -> tuple[Any, ...]:
        """Convert a Cognithor action object into ``env.step()`` args.

        Simple actions (``RESET``, ``ACTION1``..``ACTION5``) take just
        the value. Complex actions (``ACTION6``, ``ACTION7``) carry
        ``(x, y)`` data the agent attached via ``set_data()``.
        """
        value = getattr(action, "value", action)
        data = getattr(action, "_data", None)
        if data:
            return (value, {"data": dict(data)})
        return (value,)


def run_episode(
    *,
    agent: CognithorPSEAgent,
    game_id: str,
    max_steps: int = 80,
) -> EpisodeResult:
    """Convenience wrapper: ``EpisodeRunner(...).run()``."""
    return EpisodeRunner(agent=agent, game_id=game_id, max_steps=max_steps).run()
