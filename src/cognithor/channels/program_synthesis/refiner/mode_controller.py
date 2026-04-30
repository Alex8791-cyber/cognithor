# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec v1.4 §6.5.2 — three-zone refiner mode-selection (F2).

Picks one of three refiner modes from the current ``α`` value:

* ``full_llm`` for ``α ≥ zone1_lower`` (default 0.45) —
  Two-Stage CoT→JSON LLM repair with retry.
* ``hybrid`` for ``zone3_upper ≤ α < zone1_lower`` (default 0.35–0.45)
  — symbolic and single-stage LLM in parallel; better wins.
* ``symbolic`` for ``α < zone3_upper`` (default 0.35) — rule-based
  heuristics only.

Plus a hysteresis gate: an once-chosen mode stays for at least
``hysteresis_window`` repair calls (default 3) even if α nudges over a
boundary. This neutralises the v1.3 cliff where α=0.39 vs α=0.41
flipped the entire repair pipeline.

The controller is intentionally a tiny stateful object — the actual
LLM/symbolic/hybrid implementations live elsewhere; this file only
decides *which* of the three to dispatch to.
"""

from __future__ import annotations

from typing import Literal

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

RefinerMode = Literal["full_llm", "hybrid", "symbolic"]


def _propose_mode(alpha: float, config: Phase2Config) -> RefinerMode:
    if alpha >= config.repair_alpha_zone1_lower:
        return "full_llm"
    if alpha >= config.repair_alpha_zone3_upper:
        return "hybrid"
    return "symbolic"


class RefinerModeController:
    """Picks a refiner mode for each repair call, with hysteresis.

    Stateful: instantiate once per :class:`Channel`, call
    :meth:`select_mode` per repair attempt. The state machine:

    * On the first call, it adopts whatever the proposed mode is.
    * On subsequent calls, the proposed mode is allowed only if the
      current mode has been held for at least ``hysteresis_window``
      calls. Otherwise the current mode is repeated (and the held-call
      counter increments).

    The window default is 3 — spec rationale is that 3 repair
    iterations is "long enough for α to actually have moved on
    purpose, short enough to keep the controller responsive".
    """

    def __init__(
        self,
        *,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._config = config
        self._current_mode: RefinerMode | None = None
        self._calls_in_current_mode: int = 0
        self._hysteresis_holds: int = 0  # cumulative count, exposed for telemetry

    @property
    def current_mode(self) -> RefinerMode | None:
        """Latest mode the controller is in. ``None`` before first call."""
        return self._current_mode

    @property
    def calls_in_current_mode(self) -> int:
        """How many times :meth:`select_mode` has returned the current mode."""
        return self._calls_in_current_mode

    @property
    def hysteresis_holds_total(self) -> int:
        """Number of mode-changes that hysteresis has held back so far.

        Telemetry: feeds the
        ``cognithor_synthesis_refiner_mode_hysteresis_held_total``
        counter the spec defines in §11.
        """
        return self._hysteresis_holds

    def select_mode(self, alpha: float) -> RefinerMode:
        """Return the mode for this repair call. See class docstring."""
        proposed = _propose_mode(alpha, self._config)

        # First call: adopt without hysteresis.
        if self._current_mode is None:
            self._current_mode = proposed
            self._calls_in_current_mode = 1
            return proposed

        # Same mode proposed: nothing to gate.
        if proposed == self._current_mode:
            self._calls_in_current_mode += 1
            return proposed

        # Mode change proposed but we're inside the hold window: stay.
        if self._calls_in_current_mode < self._config.refiner_hysteresis_window:
            self._calls_in_current_mode += 1
            self._hysteresis_holds += 1
            return self._current_mode

        # Hold window cleared — switch.
        self._current_mode = proposed
        self._calls_in_current_mode = 1
        return proposed

    def reset(self) -> None:
        """Forget all state. Used between independent synthesis runs."""
        self._current_mode = None
        self._calls_in_current_mode = 0
        self._hysteresis_holds = 0


__all__ = ["RefinerMode", "RefinerModeController"]
