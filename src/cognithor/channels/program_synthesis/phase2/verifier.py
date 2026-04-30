# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 Verifier extension — Triviality + Suspicion (spec v1.4 §7.3).

Spec v1.4 §7.3 splits the verifier extension into two pieces:

* **§7.3.1 Regelbasierte Triviality** — a list of structural rules
  that detect "this program is too simple to plausibly produce that
  partial score" (e.g. ``InputRef`` alone with score 0.85). Carried
  over from v1.3 unchanged.
* **§7.3.2 Suspicion-Score** — multiplies a syntactic-complexity
  measure (which now distinguishes High-Impact / Structural-
  Abstraction / Regular tokens, the F1 split) with the partial score
  to produce a "should-we-trust-it" number.

This Sprint-1 module ships the **Suspicion-Score** surface only. The
exact number-formula for §7.3.3 ("Effekt auf Score") lives in a
later sprint that wires this into the live verifier. Here we pin:

* the public dataclass shape (so callers can subscribe early);
* the qualitative F1 invariant (1-token ``objects()`` is *more*
  suspect than 1-token ``tile()``, which is more suspect than
  ``recolor(...)``);
* the config-overridability rule (Sprint-1 contract).

The numeric formula intentionally stays simple — ``partial_score *
(1 - syntactic_complexity)``. This satisfies the F1 ordering and is
safe to replace when later sprints land the spec's exact §7.3.3
formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)
from cognithor.channels.program_synthesis.phase2.suspicion import (
    compute_syntactic_complexity,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )


@dataclass(frozen=True)
class SuspicionScore:
    """The result of one suspicion evaluation.

    ``value`` is in ``[0, 1]``. Higher means *more* suspect — i.e.
    the partial score looks too good for how simple the program is.
    Consumers compare against a configurable threshold (typically 0.5)
    to decide whether to apply a Triviality penalty.

    ``syntactic_complexity`` and ``partial_score`` are echoed back so
    telemetry can store the inputs alongside the verdict without
    re-computing.
    """

    value: float
    syntactic_complexity: float
    partial_score: float


def compute_suspicion(
    program: ProgramNode,
    partial_score: float,
    *,
    config: Phase2Config = DEFAULT_PHASE2_CONFIG,
) -> SuspicionScore:
    """Score how suspect *program* is given its *partial_score*.

    Sprint-1 formula::

        syntactic_complexity = compute_syntactic_complexity(program, config)
        suspicion = partial_score * (1 - syntactic_complexity)

    F1 invariant follows from this formula:

    * ``objects(g)`` alone has a lower complexity (1.5× multiplier)
      than ``tile(g)`` alone (3× multiplier), so the same partial
      score yields a *higher* suspicion for ``objects``. That
      reproduces the spec §7.3.2 ordering "1-token ``objects()`` ohne
      nachgelagerte Verarbeitung ist tatsächlich verdächtig".
    * Pure ``InputRef`` (no primitives at all) has zero complexity →
      suspicion equals the full ``partial_score``.

    The exact §7.3.3 score-effect formula is reserved for a later
    sprint; this helper exists so verifier callers can adopt the
    public shape now without pinning to a number that will change.
    """
    if not 0.0 <= partial_score <= 1.0:
        raise ValueError(
            f"compute_suspicion: partial_score must be in [0, 1]; got {partial_score!r}"
        )
    sc = compute_syntactic_complexity(program, config=config)
    value = partial_score * (1.0 - sc)
    return SuspicionScore(
        value=value,
        syntactic_complexity=sc,
        partial_score=partial_score,
    )


__all__ = ["SuspicionScore", "compute_suspicion"]
