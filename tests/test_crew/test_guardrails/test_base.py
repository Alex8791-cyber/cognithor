"""Task 21 - Guardrail Protocol + GuardrailResult contract tests."""

import pytest

from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.output import TaskOutput


class TestGuardrailResult:
    def test_pass_result(self):
        r = GuardrailResult(passed=True, feedback=None)
        assert r.passed
        assert r.feedback is None

    def test_fail_result(self):
        r = GuardrailResult(passed=False, feedback="too short")
        assert not r.passed
        assert r.feedback == "too short"

    def test_frozen(self):
        r = GuardrailResult(passed=True, feedback=None)
        with pytest.raises(Exception):  # noqa: B017 - frozen raises ValidationError
            r.passed = False  # type: ignore[misc]


class TestGuardrailProtocol:
    def test_callable_satisfies_protocol(self):
        """Duck-typing: any callable returning GuardrailResult satisfies the Protocol."""

        def my_guard(output: TaskOutput) -> GuardrailResult:
            return GuardrailResult(passed=True, feedback=None)

        assert callable(my_guard)
        result = my_guard(TaskOutput(task_id="t", agent_role="w", raw="x"))
        assert isinstance(result, GuardrailResult)
