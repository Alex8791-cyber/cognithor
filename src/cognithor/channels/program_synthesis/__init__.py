# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cognithor Program Synthesis Engine (PSE) — Phase 1 Channel.

Synthesizes deterministic, replay-able programs over the ARC-DSL instead of
producing free-form LLM answers. See
``docs/superpowers/specs/2026-04-29-pse-phase1-spec-v1.2.md`` for the full
specification.

This package is currently in scaffold state (Week 1 of the 7-week roadmap).
Public API will be exposed once the engine is operational.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.core.version import (
    DSL_VERSION,
    PSE_VERSION,
)

__all__ = ["DSL_VERSION", "PSE_VERSION"]
