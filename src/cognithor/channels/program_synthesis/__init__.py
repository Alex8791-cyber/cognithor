# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cognithor Program Synthesis Engine (PSE) — Phase 1 Channel.

Synthesizes deterministic, replay-able programs over the ARC-DSL instead of
producing free-form LLM answers. See
``docs/superpowers/specs/2026-04-29-pse-phase1-spec-v1.2.md`` for the full
specification.

Phase 1 (``pse-1.2.0``) is feature-complete: 61 base primitives + 5
higher-order, K9/K10 trace + replay, K4 typed-payload + subprocess
sandbox, D5 benchmark vs baseline, D7 cache hit-rate, D15 mypy --strict
on every PSE source.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.core.version import (
    DSL_VERSION,
    PSE_VERSION,
)

__all__ = ["DSL_VERSION", "PSE_VERSION"]
