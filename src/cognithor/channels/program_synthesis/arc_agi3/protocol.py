# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 — Protocol classes that mirror the official ARC-AGI-3 API.

These :class:`Protocol` definitions describe the exact fields
Cognithor's PSE agents read from / write to when running inside the
official ``arcprize/ARC-AGI-3-Agents`` harness. They are intentionally
duck-typed: any object with the right attributes satisfies the
protocol, so ``arcengine.FrameData`` / ``arcengine.GameAction`` /
``arcengine.GameState`` (the real types) can be passed in without an
explicit adapter.

The protocols mirror the upstream ``arcengine`` package as of
ARC-AGI-3-Agents 0.9.3 (2026-01-29). Field changes:

* ``score`` → ``levels_completed``
* ``win_score`` → ``win_levels``

Cognithor sticks to the post-0.9.3 names. Older 0.9.1 / 0.9.2 frames
won't satisfy the protocol; the Wave-5 adapter can normalise if
needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class GameStateProtocol(Protocol):
    """Subset of ``arcengine.GameState`` that Cognithor agents read.

    GameState is an enum upstream; the protocol only needs the
    ``name`` accessor and equality semantics. Cognithor compares
    against string literals (``"WIN"``, ``"GAME_OVER"``,
    ``"NOT_PLAYED"``, ``"NOT_FINISHED"``) rather than importing the
    enum, so the agent code remains import-clean.
    """

    @property
    def name(self) -> str:  # pragma: no cover — Protocol body
        ...


@runtime_checkable
class GameActionProtocol(Protocol):
    """Subset of ``arcengine.GameAction`` that Cognithor agents emit.

    Upstream ``GameAction`` is an enum with members ``RESET``,
    ``ACTION1``..``ACTION7``. ``is_simple()`` returns True for
    parameter-less actions (RESET, ACTION1..ACTION5); ``is_complex()``
    returns True for actions that need ``set_data({"x": .., "y": ..})``
    (ACTION6, ACTION7). The ``reasoning`` field carries the agent's
    natural-language justification.

    Cognithor's :class:`ActionDecoder` (Wave-2) emits objects that
    satisfy this protocol — either real ``arcengine.GameAction``
    instances when running inside the official harness, or
    plain :class:`dataclass`-style stubs in unit tests.
    """

    name: str
    value: int
    reasoning: str

    def is_simple(self) -> bool: ...
    def is_complex(self) -> bool: ...
    def set_data(self, data: dict[str, Any]) -> None: ...


@runtime_checkable
class FrameDataProtocol(Protocol):
    """Subset of ``arcengine.FrameData`` that Cognithor agents read.

    ``frame`` is a list of 2-D grids (one per "layer" / channel,
    typically length 1 for the visible play-field but sometimes >1 for
    games with multiple semantic planes). Each grid is upstream a
    ``numpy.ndarray`` of shape ``(H, W)`` with int values 0..15
    (ARC-AGI-3 uses a 16-colour palette, wider than ARC-AGI-1's 0..9).

    Cognithor's :class:`FrameBridge` (Wave-2) selects the relevant
    layer, clamps to the Phase-1 ``int8 [0..9]`` range or extends
    Phase-1 to the wider palette (Wave-2 design decision).

    ``available_actions`` is the per-frame whitelist of allowed
    GameActions (Sprint-11 prereq for action enumeration).
    """

    game_id: str
    state: GameStateProtocol
    levels_completed: int
    win_levels: int
    guid: str
    full_reset: bool
    frame: Sequence[Any]  # list[NDArray] upstream, opaque to Cognithor
    available_actions: Sequence[GameActionProtocol]


__all__ = [
    "FrameDataProtocol",
    "GameActionProtocol",
    "GameStateProtocol",
]
