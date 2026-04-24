"""Built-in Crew guardrail factories."""

from __future__ import annotations

import json as _json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from cognithor.crew.guardrails.base import GuardrailResult

if TYPE_CHECKING:
    from cognithor.crew.output import TaskOutput


# Regex patterns for common German PII
_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE),
    "iban": re.compile(r"\bDE\d{2}(?:\s?\d{4}){4}\s?\d{2}\b"),
    "phone": re.compile(r"(?:\+49|0049|0)[\s.-]?\d{2,4}[\s.-]?\d{3,6}[\s.-]?\d{0,6}"),
    "steuer_id": re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
}


def word_count(min_words: int | None = None, max_words: int | None = None):
    """Guardrail that checks output word count."""
    if min_words is None and max_words is None:
        raise ValueError("word_count requires at least min_words or max_words")

    def _guard(output: TaskOutput) -> GuardrailResult:
        count = len(output.raw.split())
        if min_words is not None and count < min_words:
            return GuardrailResult(
                passed=False,
                feedback=f"Output hat {count} Wörter, mindestens {min_words} erwartet.",
            )
        if max_words is not None and count > max_words:
            return GuardrailResult(
                passed=False,
                feedback=f"Output hat {count} Wörter, höchstens {max_words} erlaubt.",
            )
        return GuardrailResult(passed=True, feedback=None)

    return _guard


def no_pii():
    """Guardrail that blocks outputs containing German PII.

    Detects email addresses, German IBANs, German phone numbers, and 11-digit
    Steuer-IDs. Emits a combined feedback listing every category found.
    """

    def _guard(output: TaskOutput) -> GuardrailResult:
        hits: list[str] = []
        for name, pat in _PATTERNS.items():
            if pat.search(output.raw):
                hits.append(name)
        if not hits:
            return GuardrailResult(passed=True, feedback=None, pii_detected=False)
        categories = ", ".join(hits)
        return GuardrailResult(
            passed=False,
            feedback=f"PII erkannt: {categories}. Bitte anonymisieren.",
            pii_detected=True,
        )

    return _guard


def schema(model_cls: type[BaseModel]):
    """Guardrail that enforces a Pydantic schema on the output JSON."""

    def _guard(output: TaskOutput) -> GuardrailResult:
        try:
            data = _json.loads(output.raw)
        except _json.JSONDecodeError as exc:
            return GuardrailResult(passed=False, feedback=f"Output ist kein valides JSON: {exc}")
        try:
            model_cls.model_validate(data)
        except ValidationError as exc:
            errs = "; ".join(
                f"{'/'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return GuardrailResult(
                passed=False, feedback=f"Schema-Validierung fehlgeschlagen: {errs}"
            )
        return GuardrailResult(passed=True, feedback=None)

    return _guard
