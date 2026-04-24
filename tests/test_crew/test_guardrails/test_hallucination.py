"""Task 27 — hallucination_check built-in guardrail tests."""

from cognithor.crew.guardrails.builtin import hallucination_check
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_passes_when_output_is_subset_of_reference():
    ref = "Der Tarif PrivatPlus kostet 450 Euro pro Monat und deckt stationäre Leistungen ab."
    g = hallucination_check(reference=ref)
    r = g(_out("PrivatPlus kostet 450 Euro."))
    assert r.passed


def test_fails_when_output_invents_a_number():
    ref = "Der Tarif kostet 450 Euro."
    g = hallucination_check(reference=ref)
    r = g(_out("Der Tarif kostet 99999 Euro."))
    assert not r.passed
    assert "99999" in (r.feedback or "")


def test_passes_when_min_overlap_is_zero():
    # Edge case: min_overlap=0 disables the check (useful as a test-only mode)
    g = hallucination_check(reference="x", min_overlap=0.0)
    r = g(_out("completely unrelated"))
    assert r.passed
