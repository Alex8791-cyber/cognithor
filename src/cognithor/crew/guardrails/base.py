"""Guardrail protocol + result dataclass.

A Guardrail is a callable that takes a TaskOutput and returns a
GuardrailResult. Concrete implementations live in ``function_guardrail.py``
(Python callable wrapper), ``string_guardrail.py`` (LLM-validated natural
language), and ``builtin.py`` (factory-produced presets).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from cognithor.crew.output import TaskOutput


class GuardrailResult(BaseModel):
    """Immutable verdict returned by every Guardrail."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    feedback: str | None = None  # Required when passed is False
    pii_detected: bool = False  # Set by no_pii and related guardrails


@runtime_checkable
class Guardrail(Protocol):
    def __call__(self, output: TaskOutput) -> GuardrailResult: ...
