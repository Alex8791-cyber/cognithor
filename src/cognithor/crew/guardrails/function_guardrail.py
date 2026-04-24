"""Function-based guardrail — wraps a user callable into the protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.crew.guardrails.base import GuardrailResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from cognithor.crew.output import TaskOutput


class FunctionGuardrail:
    """Adapter: user provides a callable with signature
        ``Callable[[TaskOutput], tuple[bool, str | TaskOutput]]``
    and gets a Guardrail that catches exceptions and normalizes return shape.
    """

    def __init__(self, fn: Callable[[TaskOutput], tuple[bool, Any]]) -> None:
        self._fn = fn

    def __call__(self, output: TaskOutput) -> GuardrailResult:
        try:
            raw = self._fn(output)
        except Exception as exc:
            return GuardrailResult(passed=False, feedback=f"Guardrail raised: {exc}")
        # Pass-through: callables may return a GuardrailResult directly
        # (Protocol-style). Skip tuple unpacking in that case.
        if isinstance(raw, GuardrailResult):
            return raw
        try:
            ok, payload = raw
        except (TypeError, ValueError) as exc:
            return GuardrailResult(passed=False, feedback=f"Guardrail raised: {exc}")
        if ok:
            return GuardrailResult(passed=True, feedback=None)
        feedback = payload if isinstance(payload, str) else "validation failed"
        return GuardrailResult(passed=False, feedback=feedback)
