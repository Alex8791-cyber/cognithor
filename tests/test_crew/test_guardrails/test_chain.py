"""Task 28 — chain() combinator tests.

R4-C4: chain() returns an ASYNC callable (needed so StringGuardrail, whose
__call__ is async, actually runs). All tests here await the chained result.
"""

from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.guardrails.builtin import chain, no_pii, word_count
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


async def test_chain_all_pass():
    g = chain(word_count(min_words=1), no_pii())
    result = await g(_out("Hallo Welt"))
    assert result.passed


async def test_chain_stops_on_first_failure():
    calls = []

    def tracker(label):
        def _g(out):
            calls.append(label)
            return GuardrailResult(passed=(label != "B"), feedback=f"from-{label}")

        return _g

    g = chain(tracker("A"), tracker("B"), tracker("C"))
    r = await g(_out("x"))
    assert not r.passed
    assert r.feedback == "from-B"
    assert calls == ["A", "B"]  # C never runs


async def test_chain_pii_in_first_fails_even_if_second_would_pass():
    g = chain(no_pii(), word_count(min_words=1))
    r = await g(_out("Kontakt: x@example.com"))
    assert not r.passed
    assert r.pii_detected is True


async def test_chain_awaits_async_guardrails():
    """R4-C4 regression: async guards inside chain() MUST actually run."""
    call_count = {"async_g": 0, "sync_g": 0}

    async def async_g(_out):
        call_count["async_g"] += 1
        return GuardrailResult(passed=True, feedback=None)

    def sync_g(_out):
        call_count["sync_g"] += 1
        return GuardrailResult(passed=True, feedback=None)

    g = chain(async_g, sync_g)
    r = await g(_out("anything"))
    assert r.passed
    assert call_count == {"async_g": 1, "sync_g": 1}


async def test_chain_short_circuits_on_first_async_failure():
    """First (async) guard fails → second guard never called."""
    second_calls = []

    async def failing_async(_out):
        return GuardrailResult(passed=False, feedback="blocked-async")

    def never_called(_out):
        second_calls.append(1)
        return GuardrailResult(passed=True, feedback=None)

    g = chain(failing_async, never_called)
    r = await g(_out("irrelevant"))
    assert not r.passed
    assert r.feedback == "blocked-async"
    assert second_calls == []
