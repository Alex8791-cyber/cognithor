"""Cognithor Crew-Layer Guardrails.

Two flavors:
  * function-based — Python callable, pass into ``CrewTask(guardrail=fn)``
  * string-based — natural language rule, evaluated by an LLM

Built-ins (factories):
  * ``word_count(min_words=..., max_words=...)``
  * ``no_pii()`` — detects German PII (emails, IBANs, phone numbers, Steuer-IDs)
  * ``hallucination_check(reference=..., min_overlap=...)``
  * ``schema(pydantic_model)``
  * ``chain(*guardrails)`` — composes async/sync guardrails with short-circuit

Example::

    from cognithor.crew import Crew, CrewAgent, CrewTask
    from cognithor.crew.guardrails import chain, no_pii, word_count

    task = CrewTask(
        description="Draft a customer email",
        expected_output="Polite, professional email under 200 words",
        agent=my_agent,
        guardrail=chain(no_pii(), word_count(max_words=200)),
        max_retries=2,
    )

Failed guardrails trigger retry-with-feedback up to ``max_retries``, then raise
``GuardrailFailure``. Every verdict (pass/fail/retry) emits a
``crew_guardrail_check`` event on the Hashline-Guard audit chain.
"""

from __future__ import annotations

from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails.base import Guardrail, GuardrailResult
from cognithor.crew.guardrails.builtin import (
    chain,
    hallucination_check,
    no_pii,
    schema,
    word_count,
)
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.guardrails.string_guardrail import StringGuardrail

__all__ = [
    "FunctionGuardrail",
    "Guardrail",
    "GuardrailFailure",
    "GuardrailResult",
    "StringGuardrail",
    "chain",
    "hallucination_check",
    "no_pii",
    "schema",
    "word_count",
]
