"""Built-in Crew guardrail factories."""

from __future__ import annotations

import inspect
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


def hallucination_check(*, reference: str, min_overlap: float = 0.5):
    """Compare output tokens against a reference corpus. Fails when too few
    of the output's informative tokens appear in the reference (simple
    heuristic — not a substitute for retrieval grounding).
    """
    ref_tokens = {t.lower() for t in reference.split() if len(t) > 2}
    _number_re = re.compile(r"\b\d{3,}\b")  # 3+ digit numbers

    def _guard(output: TaskOutput) -> GuardrailResult:
        if min_overlap <= 0.0:
            return GuardrailResult(passed=True, feedback=None)

        out_tokens = [t.lower() for t in output.raw.split() if len(t) > 2]
        if not out_tokens:
            return GuardrailResult(passed=True, feedback=None)

        overlap = sum(1 for t in out_tokens if t in ref_tokens) / len(out_tokens)

        # Additionally fail when any 3+ digit number in output is not in reference
        invented = [n for n in _number_re.findall(output.raw) if n not in reference]
        if invented:
            return GuardrailResult(
                passed=False,
                feedback=(f"Output enthält Zahlen ohne Referenz-Nachweis: {', '.join(invented)}"),
            )
        if overlap < min_overlap:
            return GuardrailResult(
                passed=False,
                feedback=(
                    f"Output-Referenz-Überlappung {overlap:.0%} unter Schwelle {min_overlap:.0%}."
                ),
            )
        return GuardrailResult(passed=True, feedback=None)

    return _guard


def chain(*guards):
    """Run guardrails in order; first failure short-circuits.

    R4-C4: this combinator MUST be async so ``StringGuardrail`` (whose
    ``__call__`` is ``async def``) actually runs. The previous synchronous
    version invoked ``g(output)`` and got a coroutine back — which is always
    truthy — so ``if not r.passed`` was evaluated against an un-awaited
    coroutine, and the second guardrail never ran. The
    ``versicherungs-vergleich`` template's ``chain(no_pii(), StringGuardrail(...))``
    required this fix.

    Returned ``GuardrailResult`` preserves the ``pii_detected`` flag from
    whichever guard signaled it, so the audit-chain record is complete.
    """

    async def _combined(output: TaskOutput) -> GuardrailResult:
        for g in guards:
            r = g(output)
            if inspect.iscoroutine(r):
                r = await r
            if not r.passed:
                return r
        return GuardrailResult(passed=True, feedback=None)

    # Mark the closure as a Guardrail so the compiler's ``_normalize_guardrail``
    # doesn't re-wrap it in a ``FunctionGuardrail`` (which would tuple-unpack
    # our coroutine return and blow up with "cannot unpack non-iterable
    # coroutine object"). Any object with ``_is_guardrail`` set is considered
    # already-normalized by ``_is_already_guardrail``.
    _combined._is_guardrail = True  # type: ignore[attr-defined]
    return _combined
