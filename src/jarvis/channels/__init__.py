"""Jarvis channels module.

Alle Kommunikationskanäle zwischen User und Gateway.
Bibel-Referenz: §9 (Gateway & Channels)
"""

from jarvis.channels.base import Channel, MessageHandler
from jarvis.channels.slack import SlackChannel
from jarvis.channels.discord import DiscordChannel
from jarvis.channels.interactive import (
    AdaptiveCard,
    DiscordMessageBuilder,
    FallbackRenderer,
    FormField,
    InteractionStateStore,
    ModalHandler,
    ProgressTracker,
    SignatureVerifier,
    SlackMessageBuilder,
    SlashCommandRegistry,
)
from jarvis.channels.commands import (  # noqa: F401
    CommandRegistry,
    FallbackRenderer as FallbackRendererV2,
    InteractionStore,
)
from jarvis.channels.connectors import (  # noqa: E402
    ConnectorRegistry,
    JiraConnector,
    ServiceNowConnector,
    TeamsConnector,
)

# v22: Neue Channels (lazy imports um optionale Dependencies zu vermeiden)
from jarvis.channels.google_chat import GoogleChatChannel  # noqa: E402
from jarvis.channels.mattermost import MattermostChannel  # noqa: E402
from jarvis.channels.feishu import FeishuChannel  # noqa: E402
from jarvis.channels.irc import IRCChannel  # noqa: E402
from jarvis.channels.twitch import TwitchChannel  # noqa: E402

# v22: Canvas
from jarvis.channels.canvas import CanvasManager  # noqa: E402

__all__ = [
    "Channel",
    "MessageHandler",
    "SlackChannel",
    "DiscordChannel",
    "SlackMessageBuilder",
    "DiscordMessageBuilder",
    "ProgressTracker",
    "AdaptiveCard",
    "FormField",
    "SlashCommandRegistry",
    "ModalHandler",
    "SignatureVerifier",
    "InteractionStateStore",
    "FallbackRenderer",
    "CommandRegistry",
    "InteractionStore",
    "ConnectorRegistry",
    "JiraConnector",
    "ServiceNowConnector",
    "TeamsConnector",
    # v22: Neue Channels
    "GoogleChatChannel",
    "MattermostChannel",
    "FeishuChannel",
    "IRCChannel",
    "TwitchChannel",
    "CanvasManager",
]
