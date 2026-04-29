# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verifier pipeline — five-stage program validation (spec §10).

Public API: :class:`Verifier`, :class:`VerificationResult`, and the
individual :class:`Stage` subclasses for tests that want to inject a
custom pipeline.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.verify.pipeline import Verifier
from cognithor.channels.program_synthesis.verify.properties import (
    DEFAULT_PROPERTIES,
)
from cognithor.channels.program_synthesis.verify.result import VerificationResult
from cognithor.channels.program_synthesis.verify.stages import (
    DemoStage,
    HeldOutStage,
    PropertyStage,
    Stage,
    SyntaxStage,
    TypeStage,
    default_pipeline,
)

__all__ = [
    "DEFAULT_PROPERTIES",
    "DemoStage",
    "HeldOutStage",
    "PropertyStage",
    "Stage",
    "SyntaxStage",
    "TypeStage",
    "VerificationResult",
    "Verifier",
    "default_pipeline",
]
