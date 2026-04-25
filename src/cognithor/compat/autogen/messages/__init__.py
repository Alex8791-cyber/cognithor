"""AutoGen-shaped message classes — used by the source-compat shim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _BaseMessage:
    content: Any
    source: str
    type: str = "Message"
    models_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return str(self.content)


@dataclass
class TextMessage(_BaseMessage):
    type: str = "TextMessage"


@dataclass
class ToolCallSummaryMessage(_BaseMessage):
    type: str = "ToolCallSummaryMessage"
    tool_name: str = ""


@dataclass
class HandoffMessage(_BaseMessage):
    type: str = "HandoffMessage"
    target: str = ""


@dataclass
class StructuredMessage(_BaseMessage):
    type: str = "StructuredMessage"


__all__ = ["HandoffMessage", "StructuredMessage", "TextMessage", "ToolCallSummaryMessage"]
