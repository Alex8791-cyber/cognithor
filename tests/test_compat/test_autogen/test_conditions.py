"""Termination conditions — count + text-match + composite."""

from __future__ import annotations

from cognithor.compat.autogen.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from cognithor.compat.autogen.messages import TextMessage


def _msg(content: str) -> TextMessage:
    return TextMessage(content=content, source="a")


def test_max_message_termination_below_threshold() -> None:
    cond = MaxMessageTermination(3)
    assert not cond.is_terminated([_msg("a"), _msg("b")])


def test_max_message_termination_at_threshold() -> None:
    cond = MaxMessageTermination(3)
    assert cond.is_terminated([_msg("a"), _msg("b"), _msg("c")])


def test_text_mention_termination_match() -> None:
    cond = TextMentionTermination("DONE")
    assert cond.is_terminated([_msg("step1"), _msg("we are DONE here")])


def test_text_mention_termination_no_match() -> None:
    cond = TextMentionTermination("DONE")
    assert not cond.is_terminated([_msg("step1"), _msg("step2")])


def test_text_mention_termination_only_inspects_last_message() -> None:
    """Spec says termination matches against last raw output, not history."""
    cond = TextMentionTermination("DONE")
    assert not cond.is_terminated([_msg("DONE earlier"), _msg("not now")])


def test_combined_and_both_must_match() -> None:
    cond = MaxMessageTermination(2) & TextMentionTermination("DONE")
    assert not cond.is_terminated([_msg("a"), _msg("b")])  # count ok, no DONE
    assert not cond.is_terminated([_msg("DONE")])  # DONE ok, count too low
    assert cond.is_terminated([_msg("a"), _msg("DONE")])  # both


def test_combined_or_either_can_match() -> None:
    cond = MaxMessageTermination(5) | TextMentionTermination("DONE")
    assert cond.is_terminated([_msg("DONE")])  # text matches
    assert cond.is_terminated([_msg("a")] * 5)  # count matches
    assert not cond.is_terminated([_msg("a")] * 4)  # neither
