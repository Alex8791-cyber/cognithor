# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 telemetry counters (spec v1.4 §11).

Spec §11 lists four new Prometheus counters that the Phase-2 surface
must emit. They mirror the existing Phase-1 ``standard_counters`` shape
so the channel binds them once and references them by short key.

The four counters:

* ``cognithor_synthesis_refiner_mode_total`` (label: ``mode``)
  — every refiner-mode selection, partitioned by the chosen mode
  (``full_llm`` / ``hybrid`` / ``symbolic``).
* ``cognithor_synthesis_refiner_hybrid_winner_total`` (label: ``winner``)
  — every hybrid-mode repair, partitioned by who won
  (``symbolic`` / ``llm``).
* ``cognithor_synthesis_refiner_mode_hysteresis_held_total``
  — every time the hysteresis window blocked a mode change.
* ``cognithor_synthesis_structural_abstraction_token_total``
  — count of structural-abstraction tokens seen by the verifier
  per evaluated program.

Returned as a typed dict so the channel can pass the relevant
counter to the matching collaborator (e.g. give the
``RefinerModeController`` a callback bound to ``refiner_mode_total``).
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.observability.metrics import (
    DEFAULT_REGISTRY,
    Counter,
    Registry,
)


def phase2_counters(registry: Registry | None = None) -> dict[str, Counter]:
    """Bind the spec §11 Phase-2 counters once.

    Returned dict keys (kept short for call sites):

    * ``refiner_mode_total``
    * ``refiner_hybrid_winner_total``
    * ``refiner_mode_hysteresis_held_total``
    * ``structural_abstraction_token_total``
    """
    r = registry if registry is not None else DEFAULT_REGISTRY
    return {
        "refiner_mode_total": r.counter(
            "cognithor_synthesis_refiner_mode_total",
            "Refiner mode-selection events partitioned by the chosen mode.",
        ),
        "refiner_hybrid_winner_total": r.counter(
            "cognithor_synthesis_refiner_hybrid_winner_total",
            "Hybrid-mode repair winners (symbolic vs llm).",
        ),
        "refiner_mode_hysteresis_held_total": r.counter(
            "cognithor_synthesis_refiner_mode_hysteresis_held_total",
            "Number of mode-changes blocked by the hysteresis window.",
        ),
        "structural_abstraction_token_total": r.counter(
            "cognithor_synthesis_structural_abstraction_token_total",
            "Total structural-abstraction tokens seen by the verifier.",
        ),
    }


__all__ = ["phase2_counters"]
