# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""α-Controller + Prior-Performance-Tracker (spec §4.4.4, plan task 6).

The Search-α formula in :mod:`phase2.alpha_mixer` takes
``α_performance`` as one of its two factors. Sprint-1 has been
defaulting that to whatever the symbolic side reports as
``effective_confidence``; the spec §4.4.4 design is richer:

* A :class:`PriorPerformanceTracker` keeps a sliding window of the
  most recent prior outcomes (Did the LLM-favoured action survive the
  verifier? Did the symbolic-favoured one?). The window size lives in
  :class:`Phase2Config`.
* An :class:`AlphaController` reads the tracker and computes a fresh
  ``α_performance`` per call, with **hysteresis**: it only *lowers*
  α_performance once the tracker has reported low LLM-success for
  ``alpha_hysteresis_iterations`` consecutive calls. That makes the
  controller robust against single bad LLM responses.
* Cold-start value: ``alpha_cold_start`` (default 0.85, spec §4.4.4).
  The controller returns that until at least one observation lands.

Plan acceptance criteria (task 6):

* Multiplicative α formula already done (``mix_alpha``).
* Hysterese-window from config ✅.
* Sample-size dampening already done (``apply_sample_size_dampening``).
* A/B-Test: künstlich-verschlechterter LLM → α sinkt unter 0.4 nach
  Window-Iterationen — covered in the test suite below.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)


@dataclass(frozen=True)
class PriorObservation:
    """One per-call performance snapshot the tracker consumes.

    ``llm_success`` and ``symbolic_success`` are floats in ``[0, 1]``
    indicating how well each prior side did on this call. The tracker
    averages them over its window; the controller maps the averages
    to ``α_performance``.
    """

    llm_success: float
    symbolic_success: float

    def __post_init__(self) -> None:
        for name, value in (
            ("llm_success", self.llm_success),
            ("symbolic_success", self.symbolic_success),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"PriorObservation.{name} must be in [0, 1]; got {value}")


class PriorPerformanceTracker:
    """Sliding-window tracker of LLM and symbolic prior success.

    Window size is fixed at construction (read from
    :class:`Phase2Config.alpha_performance_window`). Adding an
    observation past the window evicts the oldest entry, so the
    tracker always reflects the *recent* prior quality, not the
    lifetime average.
    """

    def __init__(self, *, config: Phase2Config = DEFAULT_PHASE2_CONFIG) -> None:
        self._config = config
        self._observations: deque[PriorObservation] = deque(maxlen=config.alpha_performance_window)

    @property
    def window_size(self) -> int:
        return self._config.alpha_performance_window

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    def is_warm(self) -> bool:
        """At least one observation accumulated."""
        return self.n_observations > 0

    def record(self, observation: PriorObservation) -> None:
        self._observations.append(observation)

    def average_llm_success(self) -> float:
        if not self._observations:
            return 0.0
        return sum(o.llm_success for o in self._observations) / len(self._observations)

    def average_symbolic_success(self) -> float:
        if not self._observations:
            return 0.0
        return sum(o.symbolic_success for o in self._observations) / len(self._observations)

    def reset(self) -> None:
        self._observations.clear()


class AlphaController:
    """Resolves the Search-α_performance factor with hysteresis.

    The controller reads from a :class:`PriorPerformanceTracker` and
    returns a value in the configured ``α_performance`` band
    ([0.5, 1.0] by default). It tracks consecutive low-LLM-success
    observations and only allows ``α_performance`` to drop after the
    hysteresis window has been crossed — this makes it robust to a
    single bad LLM call.

    The controller is stateful (consecutive-low counter); call
    :meth:`reset` between independent synthesis runs.
    """

    def __init__(
        self,
        tracker: PriorPerformanceTracker | None = None,
        *,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
        low_llm_threshold: float = 0.4,
    ) -> None:
        self._config = config
        self._tracker = tracker or PriorPerformanceTracker(config=config)
        self._consecutive_low_count = 0
        # Threshold below which the LLM is "unreliable" — taken from
        # spec §4.4.4 narrative ("künstlich-verschlechterter LLM
        # → α sinkt korrekt unter 0.4"). Caller can override; not
        # exposed via Phase2Config because it's a controller-internal
        # detail rather than a YAML-anchored constant.
        self._low_llm_threshold = low_llm_threshold

    @property
    def tracker(self) -> PriorPerformanceTracker:
        return self._tracker

    @property
    def consecutive_low_count(self) -> int:
        return self._consecutive_low_count

    def observe(self, observation: PriorObservation) -> None:
        """Record an observation and update the hysteresis counter."""
        self._tracker.record(observation)
        if observation.llm_success < self._low_llm_threshold:
            self._consecutive_low_count += 1
        else:
            self._consecutive_low_count = 0

    def alpha_performance(self) -> float:
        """Current α_performance ∈ [α_performance_lower, α_performance_upper].

        * Cold-start (no observations): returns ``alpha_cold_start``,
          clamped into the α_performance band.
        * Warm + LLM has been below the low-threshold for fewer than
          ``alpha_hysteresis_iterations`` consecutive calls: returns
          the band ceiling (α_performance_upper).
        * Warm + LLM has been below the low-threshold for at least
          ``alpha_hysteresis_iterations`` consecutive calls: returns
          the band floor (α_performance_lower).

        The cold-start value is always emitted at the *first* observe
        until enough data accumulates to apply the hysteresis rule.
        """
        cfg = self._config
        if not self._tracker.is_warm():
            return _clamp(
                cfg.alpha_cold_start,
                cfg.alpha_performance_lower,
                cfg.alpha_performance_upper,
            )
        if self._consecutive_low_count >= cfg.alpha_hysteresis_iterations:
            return cfg.alpha_performance_lower
        return cfg.alpha_performance_upper

    def reset(self) -> None:
        self._tracker.reset()
        self._consecutive_low_count = 0


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


__all__ = [
    "AlphaController",
    "PriorObservation",
    "PriorPerformanceTracker",
]
