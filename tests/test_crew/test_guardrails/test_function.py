"""Task 22 — FunctionGuardrail adapter tests."""

from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.output import TaskOutput


def test_function_guardrail_passes():
    def min_len(out: TaskOutput) -> tuple[bool, str | TaskOutput]:
        return (True, out) if len(out.raw) >= 3 else (False, "too short")

    g = FunctionGuardrail(min_len)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="hello"))
    assert isinstance(r, GuardrailResult)
    assert r.passed


def test_function_guardrail_fails_with_feedback():
    def min_len(out: TaskOutput) -> tuple[bool, str | TaskOutput]:
        return (False, "output ist kürzer als erwartet")

    g = FunctionGuardrail(min_len)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="hi"))
    assert not r.passed
    assert r.feedback == "output ist kürzer als erwartet"


def test_function_guardrail_wraps_unexpected_exception_as_fail():
    def buggy(out: TaskOutput) -> tuple[bool, str | TaskOutput]:
        raise RuntimeError("unexpected")

    g = FunctionGuardrail(buggy)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="x"))
    assert not r.passed
    assert "unexpected" in (r.feedback or "")
