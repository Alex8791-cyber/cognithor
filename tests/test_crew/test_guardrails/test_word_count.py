"""Task 24 — word_count built-in guardrail tests."""

import pytest

from cognithor.crew.guardrails.builtin import word_count
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_word_count_min_pass():
    g = word_count(min_words=3)
    assert g(_out("one two three")).passed


def test_word_count_min_fail():
    g = word_count(min_words=5)
    r = g(_out("only three words"))
    assert not r.passed
    assert "5" in (r.feedback or "") or "mindestens" in (r.feedback or "").lower()


def test_word_count_max_pass():
    g = word_count(max_words=5)
    assert g(_out("one two")).passed


def test_word_count_max_fail():
    g = word_count(max_words=2)
    r = g(_out("one two three four"))
    assert not r.passed


def test_word_count_both_bounds():
    g = word_count(min_words=2, max_words=4)
    assert g(_out("a b c")).passed
    assert not g(_out("a")).passed
    assert not g(_out("a b c d e")).passed


def test_word_count_empty_string_fails_min():
    g = word_count(min_words=1)
    assert not g(_out("")).passed


def test_word_count_neither_bound_raises():
    with pytest.raises(ValueError):
        word_count()
