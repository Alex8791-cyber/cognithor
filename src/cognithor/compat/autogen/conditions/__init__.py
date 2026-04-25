"""Termination conditions — AutoGen-shape, supporting __and__ / __or__ composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cognithor.compat.autogen.messages import TextMessage


class _TerminationCondition:
    """Base — supports `__and__` / `__or__` to compose conditions."""

    def is_terminated(self, messages: Sequence[TextMessage]) -> bool:  # pragma: no cover — abstract
        raise NotImplementedError

    def __and__(self, other: _TerminationCondition) -> _AndTermination:
        return _AndTermination(self, other)

    def __or__(self, other: _TerminationCondition) -> _OrTermination:
        return _OrTermination(self, other)


class _AndTermination(_TerminationCondition):
    def __init__(self, left: _TerminationCondition, right: _TerminationCondition) -> None:
        self.left = left
        self.right = right

    def is_terminated(self, messages: Sequence[TextMessage]) -> bool:
        return self.left.is_terminated(messages) and self.right.is_terminated(messages)


class _OrTermination(_TerminationCondition):
    def __init__(self, left: _TerminationCondition, right: _TerminationCondition) -> None:
        self.left = left
        self.right = right

    def is_terminated(self, messages: Sequence[TextMessage]) -> bool:
        return self.left.is_terminated(messages) or self.right.is_terminated(messages)


class MaxMessageTermination(_TerminationCondition):
    """Terminate when the message count reaches max_messages."""

    def __init__(self, max_messages: int) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages

    def is_terminated(self, messages: Sequence[TextMessage]) -> bool:
        return len(messages) >= self.max_messages


class TextMentionTermination(_TerminationCondition):
    """Terminate when the LAST message contains `mention` (case-insensitive substring)."""

    def __init__(self, mention: str) -> None:
        if not mention:
            raise ValueError("mention must be a non-empty string")
        self.mention = mention

    def is_terminated(self, messages: Sequence[TextMessage]) -> bool:
        if not messages:
            return False
        last = str(messages[-1].content).lower()
        return self.mention.lower() in last


__all__ = [
    "MaxMessageTermination",
    "TextMentionTermination",
]
