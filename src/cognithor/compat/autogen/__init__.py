"""AutoGen-AgentChat source-compatibility shim.

Translates a subset of `autogen_agentchat`'s public API onto Cognithor's
PGE-Trinity + cognithor.crew. Designed for search-and-replace migration:

    from autogen_agentchat.agents import AssistantAgent
        ↓
    from cognithor.compat.autogen import AssistantAgent

Supported:
- AssistantAgent (1-shot AssistantAgent.run / run_stream)
- RoundRobinGroupChat (multi-round via custom adapter)
- TextMessage, HandoffMessage, ToolCallSummaryMessage, StructuredMessage
- MaxMessageTermination, TextMentionTermination (with __and__/__or__)
- OpenAIChatCompletionClient (wrapper on cognithor.core.model_router)

Not supported by design (see ADR 0001):
- SelectorGroupChat (LLM as security boundary)
- Swarm (HandoffMessage freedom conflicts with PGE-Trinity)
- MagenticOneGroupChat (separate workstream)
- autogen-core classes (RoutedAgent, @message_handler, etc.)

References:
- Migration guide: cognithor/compat/autogen/README.md
- ADR 0001: docs/adr/0001-pge-trinity-vs-group-chat.md
- License: This shim is Apache 2.0; the API shape is concept-inspired
  from autogen-agentchat (MIT). NOTICE carries the attribution.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cognithor.compat.autogen is a source-compat shim. For new code, prefer "
    "cognithor.crew directly. See migration guide: "
    "https://github.com/Alex8791-cyber/cognithor/blob/main/src/cognithor/compat/autogen/README.md",
    DeprecationWarning,
    stacklevel=2,
)

from cognithor.compat.autogen.agents import AssistantAgent
from cognithor.compat.autogen.conditions import MaxMessageTermination, TextMentionTermination
from cognithor.compat.autogen.messages import (
    HandoffMessage,
    StructuredMessage,
    TextMessage,
    ToolCallSummaryMessage,
)
from cognithor.compat.autogen.models import OpenAIChatCompletionClient
from cognithor.compat.autogen.teams import RoundRobinGroupChat

__all__ = [
    "AssistantAgent",
    "HandoffMessage",
    "MaxMessageTermination",
    "OpenAIChatCompletionClient",
    "RoundRobinGroupChat",
    "StructuredMessage",
    "TextMentionTermination",
    "TextMessage",
    "ToolCallSummaryMessage",
]
