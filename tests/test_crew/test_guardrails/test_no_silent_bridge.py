"""Task 21 regression: the PR 1 -> PR 2 bridge guard is gone; real apply
path (Task 29) replaces it.
"""


def test_no_bridge_guard_in_compiler():
    from cognithor.crew import compiler as m

    assert not hasattr(m, "_warn_if_guardrail_silently_ignored")
    assert not hasattr(m, "_guardrails_available")
