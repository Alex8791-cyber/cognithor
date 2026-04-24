"""Built-in Crew guardrail factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.crew.guardrails.base import GuardrailResult

if TYPE_CHECKING:
    from cognithor.crew.output import TaskOutput


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
