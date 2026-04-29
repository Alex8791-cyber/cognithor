# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Trace system for the PSE channel — K9 and K10 hard gates (spec §3 + §22).

This module turns a synthesised :class:`Program` into a deterministic,
human-readable, replay-able pseudo-code trace. The trace is the
differentiator vs. LLM-only ARC solvers — every solved task ships with
a step-by-step explanation that an end user can read.

K9 — Trace-Vollständigkeit: every solved program must have a complete,
human-readable trace.

K10 — Programm-Replay-Reproduzierbarkeit: re-executing the program on
the same input produces identical output, and the replay completes in
P95 ≤ 100 ms.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.trace.builder import (
    TraceLine,
    TraceResult,
    build_trace,
    format_trace,
)
from cognithor.channels.program_synthesis.trace.replay import (
    ReplayResult,
    replay_program,
)

__all__ = [
    "ReplayResult",
    "TraceLine",
    "TraceResult",
    "build_trace",
    "format_trace",
    "replay_program",
]
