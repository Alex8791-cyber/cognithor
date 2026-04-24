"""String-based guardrail — LLM validates output against a natural-language rule.

The guardrail is **async** — it runs an LLM call via an OllamaClient-shaped
duck type (async ``.chat(model, messages)`` returning a dict with nested
``message.content``). The compiler awaits it inside ``execute_task_async``.

Function-based guardrails (FunctionGuardrail) remain sync. The compiler
awaits only if the guardrail's ``__call__`` returns a coroutine.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from cognithor.crew.guardrails.base import GuardrailResult

if TYPE_CHECKING:
    from cognithor.crew.output import TaskOutput

_VALIDATOR_SYSTEM_PROMPT = (
    "You are a strict output validator. You will receive a RULE and an OUTPUT. "
    "Respond with a single JSON object: "
    '{"passed": boolean, "feedback": string_or_null}. '
    "If the output satisfies the rule, passed=true and feedback=null. "
    "If not, passed=false and feedback is a short German explanation."
)


class StringGuardrail:
    """LLM-validated guardrail. Offline-safe fallback: if the LLM is
    unavailable the result is ``passed=False`` with a clear feedback, so
    production can't skip validation silently.

    ``llm_client`` must expose an async ``.chat(model, messages, ...)``
    method returning an Ollama-shaped dict
    (``{"message": {"content": "..."}}``).
    ``cognithor.core.model_router.OllamaClient`` satisfies this contract
    directly; the compiler passes ``planner._ollama`` into this guardrail.
    """

    def __init__(
        self,
        rule: str,
        *,
        llm_client: Any,
        model: str,
    ) -> None:
        self._rule = rule
        self._llm = llm_client
        self._model = model

    async def __call__(self, output: TaskOutput) -> GuardrailResult:
        user_prompt = f"RULE: {self._rule}\n\nOUTPUT:\n{output.raw}"
        messages = [
            {"role": "system", "content": _VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = ""
        try:
            resp = await self._llm.chat(
                model=self._model,
                messages=messages,
                format_json=True,
                temperature=0.0,
            )
            raw = (resp.get("message", {}) or {}).get("content", "") or ""
        except Exception as exc:
            return GuardrailResult(
                passed=False,
                feedback=f"Validator-LLM nicht verfuegbar: {exc}",
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return GuardrailResult(
                passed=False,
                feedback=f"Validator konnte LLM-Antwort nicht parsen: {raw[:100]}",
            )
        passed = bool(data.get("passed"))
        feedback = data.get("feedback") if not passed else None
        return GuardrailResult(passed=passed, feedback=feedback)
