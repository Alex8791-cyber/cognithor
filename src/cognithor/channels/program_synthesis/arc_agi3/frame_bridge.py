# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-2 — FrameData → Cognithor Grid bridge.

ARC-AGI-3 frames are multi-layer 2-D arrays with values in the
16-colour ``0..15`` palette (wider than ARC-AGI-1's ``0..9``). The
Sprint-10 DSL primitives are typed against ``int8`` grids in the
``0..9`` range. The bridge handles two concerns:

1. **Layer selection** — most ARC-AGI-3 games expose a single visible
   play-field layer. ``layer_index=0`` is the spec default; games with
   semantic layers (e.g. backgrounds, foregrounds, HUD) can pick
   another index.

2. **Palette mapping** — values 10..15 don't exist in the DSL.
   Three policies are exposed:

   - ``ClampPolicy.SATURATE`` (default) — clamp 10..15 to 9. Loses
     information but every primitive runs without errors.
   - ``ClampPolicy.MODULO`` — wrap 10..15 to 0..5. Loses semantics
     entirely; useful only for games where the wider colours are
     decorative.
   - ``ClampPolicy.STRICT`` — raise on any value > 9. Useful for
     unit tests of games that *should* stay in range, and for the
     Wave-3 enumeration loop where surprising values flag a bug.

   The Wave-3+ DSL extension to a 16-colour palette is queued for a
   later sprint; until then, every Sprint-10 primitive that touches
   colour explicitly runs against the clamped grid.

The bridge is pure: same input → same output, no shared state.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
    )

_Grid = NDArray[np.int8]


class ClampPolicy(Enum):
    """How to handle ARC-AGI-3 colour values 10..15.

    The DSL is currently typed against ``int8 [0..9]``; values
    outside that range need explicit handling at the boundary.
    """

    SATURATE = "saturate"
    MODULO = "modulo"
    STRICT = "strict"


class FrameBridge:
    """Converts an ARC-AGI-3 :class:`FrameDataProtocol` into the
    Cognithor Grid format the Sprint-10 DSL operates on.

    Construction is parameter-light by design — runtime behaviour is
    controlled by ``layer_index`` and ``clamp_policy``. The bridge is
    stateless: each call to :meth:`extract_grid` is independent.
    """

    def __init__(
        self,
        *,
        layer_index: int = 0,
        clamp_policy: ClampPolicy = ClampPolicy.SATURATE,
    ) -> None:
        if layer_index < 0:
            raise ValueError(f"FrameBridge: layer_index must be >= 0, got {layer_index}")
        self._layer_index = layer_index
        self._clamp_policy = clamp_policy

    @property
    def layer_index(self) -> int:
        return self._layer_index

    @property
    def clamp_policy(self) -> ClampPolicy:
        return self._clamp_policy

    def extract_grid(self, frame: FrameDataProtocol) -> _Grid:
        """Return the chosen layer of *frame* as an ``int8`` grid in [0, 9].

        Raises :class:`IndexError` if ``layer_index`` is out of range
        for this frame, :class:`ValueError` if the layer can't be
        coerced to a 2-D array, and :class:`ValueError` for
        out-of-range values under :attr:`ClampPolicy.STRICT`.
        """
        layers = list(frame.frame)
        if self._layer_index >= len(layers):
            raise IndexError(
                f"FrameBridge: layer_index={self._layer_index} but frame "
                f"{frame.game_id!r} only has {len(layers)} layer(s)"
            )
        raw = self._coerce_to_2d(layers[self._layer_index])
        return self._apply_clamp_policy(raw)

    @staticmethod
    def _coerce_to_2d(layer: Any) -> NDArray[np.int_]:
        """Accept ``np.ndarray`` or ``list[list[int]]``; return int array."""
        if isinstance(layer, np.ndarray):
            arr = layer
        else:
            arr = np.asarray(layer)
        if arr.ndim != 2:
            raise ValueError(
                f"FrameBridge: layer must be 2-D (got {arr.ndim}-D, shape {arr.shape})"
            )
        return arr.astype(np.int_, copy=False)

    def _apply_clamp_policy(self, raw: NDArray[np.int_]) -> _Grid:
        if self._clamp_policy is ClampPolicy.STRICT:
            if int(raw.min()) < 0 or int(raw.max()) > 9:
                raise ValueError(
                    f"FrameBridge[STRICT]: values out of [0, 9] "
                    f"(min={int(raw.min())}, max={int(raw.max())})"
                )
            clamped = raw
        elif self._clamp_policy is ClampPolicy.MODULO:
            clamped = raw % 10
        else:  # SATURATE — default
            clamped = np.clip(raw, 0, 9)
        return clamped.astype(np.int8, copy=False)


__all__ = [
    "ClampPolicy",
    "FrameBridge",
]
