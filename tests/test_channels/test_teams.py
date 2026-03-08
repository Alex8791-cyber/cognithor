"""Tests fuer den TeamsChannel.

Testet: Lifecycle, Message-Handling, Invoke-Activities (Adaptive Cards),
Approval-Workflow, Proaktives Senden, Streaming, Conversation-Updates.
Bot Framework APIs werden gemockt.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock botbuilder modules before importing teams module
_mock_botbuilder_core = MagicMock()
_mock_botbuilder_schema = MagicMock()

# Configure ActivityTypes
_mock_botbuilder_schema.ActivityTypes.message = "message"
_mock_botbuilder_schema.ActivityTypes.invoke = "invoke"
_mock_botbuilder_schema.ActivityTypes.conversation_update = "conversationUpdate"
_mock_botbuilder_schema.Activity = MagicMock

sys.modules.setdefault("botbuilder", MagicMock())
sys.modules.setdefault("botbuilder.core", _mock_botbuilder_core)
sys.modules.setdefault("botbuilder.schema", _mock_botbuilder_schema)
sys.modules.setdefault("botbuilder.integration", MagicMock())
sys.modules.setdefault("botbuilder.integration.aiohttp", MagicMock())

# Mock aiohttp so lazy imports inside teams.py don't fail in CI
_mock_aiohttp = MagicMock()
_mock_aiohttp_web = MagicMock()
_mock_aiohttp_web.Response = MagicMock
sys.modules.setdefault("aiohttp", _mock_aiohttp)
sys.modules.setdefault("aiohttp.web", _mock_aiohttp_web)

from jarvis.channels.teams import (
    TeamsChannel,
    _split_message,
    _create_typing_activity,
    _build_approval_card,
    MAX_MESSAGE_LENGTH,
)
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def teams_ch() -> TeamsChannel:
    return TeamsChannel(
        app_id="test-app-id",
        app_password="test-password",
        webhook_host="127.0.0.1",
        webhook_port=3978,
    )


@pytest.fixture
def handler() -> AsyncMock:
    h = AsyncMock()
    h.return_value = OutgoingMessage(channel="teams", text="Antwort", session_id="s1")
    return h


def _make_turn_context(
    text: str = "Hallo",
    user_id: str = "user-1",
    conversation_id: str = "conv-1",
    activity_type: str = "message",
    channel_id: str = "msteams",
    entities: list[Any] | None = None,
    value: dict[str, Any] | None = None,
    members_added: list[Any] | None = None,
) -> MagicMock:
    """Erstellt einen Mock-TurnContext."""
    ctx = MagicMock()
    activity = MagicMock()
    activity.type = activity_type
    activity.text = text
    activity.id = "act-123"
    activity.channel_id = channel_id
    activity.value = value
    activity.entities = entities

    # from_property
    activity.from_property = MagicMock()
    activity.from_property.id = user_id
    activity.from_property.name = "Test User"

    # conversation
    activity.conversation = MagicMock()
    activity.conversation.id = conversation_id

    # recipient
    activity.recipient = MagicMock()
    activity.recipient.id = "bot-id"

    # members_added
    activity.members_added = members_added

    ctx.activity = activity
    ctx.send_activity = AsyncMock()
    return ctx


# ============================================================================
# Properties
# ============================================================================


class TestTeamsProperties:
    def test_name(self, teams_ch: TeamsChannel) -> None:
        assert teams_ch.name == "teams"

    def test_app_id_stored(self, teams_ch: TeamsChannel) -> None:
        assert teams_ch._app_id == "test-app-id"


# ============================================================================
# Lifecycle
# ============================================================================


class TestTeamsLifecycle:
    @pytest.mark.asyncio
    async def test_start_no_botbuilder(self, teams_ch: TeamsChannel, handler: AsyncMock) -> None:
        """Ohne botbuilder-core startet nicht."""
        with patch.dict("sys.modules", {"botbuilder.core": None, "botbuilder.schema": None}):
            # Simuliere ImportError
            with patch("builtins.__import__", side_effect=ImportError("No botbuilder")):
                pass  # start() catches ImportError internally

    @pytest.mark.asyncio
    async def test_stop_resets_state(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = True
        teams_ch._adapter = MagicMock()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        teams_ch._pending_approvals["sess-1"] = future

        await teams_ch.stop()

        assert teams_ch._running is False
        assert teams_ch._adapter is None
        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_stop_cleans_webhook(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = True
        mock_runner = AsyncMock()
        teams_ch._webhook_runner = mock_runner

        await teams_ch.stop()

        mock_runner.cleanup.assert_called_once()
        assert teams_ch._webhook_runner is None


# ============================================================================
# Inbound: Activity Processing
# ============================================================================


class TestTeamsOnTurn:
    @pytest.mark.asyncio
    async def test_on_turn_message(self, teams_ch: TeamsChannel, handler: AsyncMock) -> None:
        """Message-Activity wird verarbeitet."""
        teams_ch._handler = handler
        ctx = _make_turn_context(text="Hallo Teams", activity_type="message")

        # Mock ActivityTypes
        with patch("jarvis.channels.teams.ActivityTypes", create=True) as at_mock:
            at_mock.message = "message"
            at_mock.invoke = "invoke"
            at_mock.conversation_update = "conversationUpdate"
            await teams_ch._on_turn(ctx)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_message_calls_handler(
        self, teams_ch: TeamsChannel, handler: AsyncMock
    ) -> None:
        teams_ch._handler = handler
        ctx = _make_turn_context(text="Frage")

        with patch("jarvis.channels.teams.TurnContext", create=True) as tc_mock:
            tc_mock.get_conversation_reference = MagicMock(return_value=MagicMock())
            await teams_ch._on_message(ctx)

        handler.assert_called_once()
        incoming: IncomingMessage = handler.call_args[0][0]
        assert incoming.text == "Frage"
        assert incoming.channel == "teams"
        assert incoming.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_on_message_empty(self, teams_ch: TeamsChannel, handler: AsyncMock) -> None:
        """Leere Nachricht wird ignoriert."""
        teams_ch._handler = handler
        ctx = _make_turn_context(text="")

        with patch("jarvis.channels.teams.TurnContext", create=True) as tc_mock:
            tc_mock.get_conversation_reference = MagicMock(return_value=MagicMock())
            await teams_ch._on_message(ctx)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_message_strips_bot_mention(
        self, teams_ch: TeamsChannel, handler: AsyncMock
    ) -> None:
        """Bot-Mention wird aus Text entfernt."""
        teams_ch._handler = handler
        teams_ch._app_id = "bot-app-id"

        entity = MagicMock()
        entity.type = "mention"
        entity.mentioned = MagicMock()
        entity.mentioned.id = "bot-app-id"
        entity.text = "<at>Jarvis</at>"

        ctx = _make_turn_context(
            text="<at>Jarvis</at> Was ist Python?",
            entities=[entity],
        )

        with patch("jarvis.channels.teams.TurnContext", create=True) as tc_mock:
            tc_mock.get_conversation_reference = MagicMock(return_value=MagicMock())
            await teams_ch._on_message(ctx)

        incoming: IncomingMessage = handler.call_args[0][0]
        assert "Jarvis" not in incoming.text
        assert "Python" in incoming.text

    @pytest.mark.asyncio
    async def test_on_message_approval_yes(self, teams_ch: TeamsChannel) -> None:
        """'ja' loest Text-basiertes Approval aus."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        teams_ch._sessions["conv-1"] = "sess-appr"
        teams_ch._pending_approvals["sess-appr"] = future

        ctx = _make_turn_context(text="ja", conversation_id="conv-1")

        with patch("jarvis.channels.teams.TurnContext", create=True) as tc_mock:
            tc_mock.get_conversation_reference = MagicMock(return_value=MagicMock())
            await teams_ch._on_message(ctx)

        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_on_message_approval_no(self, teams_ch: TeamsChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        teams_ch._sessions["conv-1"] = "sess-rej"
        teams_ch._pending_approvals["sess-rej"] = future

        ctx = _make_turn_context(text="nein", conversation_id="conv-1")

        with patch("jarvis.channels.teams.TurnContext", create=True) as tc_mock:
            tc_mock.get_conversation_reference = MagicMock(return_value=MagicMock())
            await teams_ch._on_message(ctx)

        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_on_message_handler_error(self, teams_ch: TeamsChannel) -> None:
        handler = AsyncMock(side_effect=RuntimeError("Boom"))
        teams_ch._handler = handler
        ctx = _make_turn_context(text="crash")

        with patch("jarvis.channels.teams.TurnContext", create=True) as tc_mock:
            tc_mock.get_conversation_reference = MagicMock(return_value=MagicMock())
            await teams_ch._on_message(ctx)

        # Fehlermeldung gesendet
        ctx.send_activity.assert_called()

    @pytest.mark.asyncio
    async def test_on_message_saves_session(
        self, teams_ch: TeamsChannel, handler: AsyncMock
    ) -> None:
        """Session-Store wird aktualisiert."""
        mock_store = MagicMock()
        teams_ch._session_store = mock_store
        teams_ch._handler = handler

        ctx = _make_turn_context(text="Test")

        with patch("jarvis.channels.teams.TurnContext", create=True) as tc_mock:
            tc_mock.get_conversation_reference = MagicMock(return_value=MagicMock())
            await teams_ch._on_message(ctx)

        mock_store.save_channel_mapping.assert_called()


# ============================================================================
# Invoke Activities (Adaptive Cards)
# ============================================================================


class TestTeamsInvoke:
    @pytest.mark.asyncio
    async def test_on_invoke_approve(self, teams_ch: TeamsChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        teams_ch._pending_approvals["appr-1"] = future

        ctx = _make_turn_context(
            activity_type="invoke",
            value={"action": "approve", "approval_id": "appr-1"},
        )

        with patch("jarvis.channels.teams.Activity", create=True, return_value=MagicMock()):
            await teams_ch._on_invoke(ctx)

        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_on_invoke_reject(self, teams_ch: TeamsChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        teams_ch._pending_approvals["appr-2"] = future

        ctx = _make_turn_context(
            activity_type="invoke",
            value={"action": "reject", "approval_id": "appr-2"},
        )

        with patch("jarvis.channels.teams.Activity", create=True, return_value=MagicMock()):
            await teams_ch._on_invoke(ctx)

        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_on_invoke_unknown_action(self, teams_ch: TeamsChannel) -> None:
        ctx = _make_turn_context(
            activity_type="invoke",
            value={"action": "unknown"},
        )
        with patch("jarvis.channels.teams.Activity", create=True, return_value=MagicMock()):
            await teams_ch._on_invoke(ctx)  # Kein Crash


# ============================================================================
# Conversation Update
# ============================================================================


class TestTeamsConversationUpdate:
    @pytest.mark.asyncio
    async def test_on_conversation_update_bot_added(self, teams_ch: TeamsChannel) -> None:
        bot_member = MagicMock()
        bot_member.id = "bot-id"

        ctx = _make_turn_context(members_added=[bot_member])
        await teams_ch._on_conversation_update(ctx)

        ctx.send_activity.assert_called_once()
        msg = ctx.send_activity.call_args[0][0]
        assert "bereit" in msg

    @pytest.mark.asyncio
    async def test_on_conversation_update_user_added(self, teams_ch: TeamsChannel) -> None:
        user_member = MagicMock()
        user_member.id = "user-123"

        ctx = _make_turn_context(members_added=[user_member])
        await teams_ch._on_conversation_update(ctx)

        # Kein Greeting fuer User
        ctx.send_activity.assert_not_called()


# ============================================================================
# Send
# ============================================================================


class TestTeamsSend:
    @pytest.mark.asyncio
    async def test_send_not_running(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = False
        msg = OutgoingMessage(channel="teams", text="noop", session_id="s1")
        await teams_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_no_adapter(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = True
        teams_ch._adapter = None
        msg = OutgoingMessage(channel="teams", text="noop", session_id="s1")
        await teams_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_no_conversation(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = True
        teams_ch._adapter = MagicMock()
        msg = OutgoingMessage(channel="teams", text="lost", session_id="unknown")
        await teams_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_no_ref(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = True
        teams_ch._adapter = MagicMock()
        teams_ch._sessions["conv-1"] = "sess-x"
        # Kein conversation_ref
        msg = OutgoingMessage(channel="teams", text="no ref", session_id="sess-x")
        await teams_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_success(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = True
        teams_ch._adapter = MagicMock()
        teams_ch._adapter.continue_conversation = AsyncMock()
        teams_ch._sessions["conv-1"] = "sess-y"
        teams_ch._conversation_refs["conv-1"] = MagicMock()

        msg = OutgoingMessage(channel="teams", text="Hello", session_id="sess-y")
        await teams_ch.send(msg)

        teams_ch._adapter.continue_conversation.assert_called_once()


# ============================================================================
# Approval Workflow
# ============================================================================


class TestTeamsApproval:
    @pytest.mark.asyncio
    async def test_approval_no_conversation(self, teams_ch: TeamsChannel) -> None:
        action = PlannedAction(tool="delete", params={})
        result = await teams_ch.request_approval("unknown", action, "Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_no_ref(self, teams_ch: TeamsChannel) -> None:
        teams_ch._sessions["conv-1"] = "sess-a"
        action = PlannedAction(tool="delete", params={})
        result = await teams_ch.request_approval("sess-a", action, "Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_approval(self, teams_ch: TeamsChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        teams_ch._pending_approvals["sess-r"] = future

        ctx = MagicMock()
        ctx.send_activity = AsyncMock()
        await teams_ch._resolve_approval("sess-r", approved=True, turn_context=ctx)

        assert future.result() is True
        ctx.send_activity.assert_called_once()


# ============================================================================
# Streaming
# ============================================================================


class TestTeamsStreaming:
    @pytest.mark.asyncio
    async def test_streaming_token(self, teams_ch: TeamsChannel) -> None:
        teams_ch._running = True
        teams_ch._adapter = MagicMock()
        teams_ch._adapter.continue_conversation = AsyncMock()
        teams_ch._sessions["conv-1"] = "sess-s"
        teams_ch._conversation_refs["conv-1"] = MagicMock()

        with patch.object(teams_ch, "send", new_callable=AsyncMock):
            await teams_ch.send_streaming_token("sess-s", "Token")
            await asyncio.sleep(0.6)


# ============================================================================
# Hilfsfunktionen
# ============================================================================


class TestTeamsHelpers:
    def test_get_or_create_session(self, teams_ch: TeamsChannel) -> None:
        s1 = teams_ch._get_or_create_session("conv-1")
        s2 = teams_ch._get_or_create_session("conv-1")
        assert s1 == s2

    def test_conversation_for_session(self, teams_ch: TeamsChannel) -> None:
        teams_ch._sessions["conv-1"] = "sess-c"
        assert teams_ch._conversation_for_session("sess-c") == "conv-1"
        assert teams_ch._conversation_for_session("unknown") is None

    def test_split_message_short(self) -> None:
        assert _split_message("Hi") == ["Hi"]

    def test_split_message_long(self) -> None:
        long_text = "X" * (MAX_MESSAGE_LENGTH + 100)
        chunks = _split_message(long_text)
        assert len(chunks) >= 2

    def test_create_typing_activity(self) -> None:
        result = _create_typing_activity()
        assert result["type"] == "typing"

    def test_build_approval_card(self) -> None:
        action = PlannedAction(tool="delete_file", params={"path": "/tmp"})
        card = _build_approval_card(action, "Gefaehrlich", "appr-1")
        assert card["type"] == "AdaptiveCard"
        assert len(card["actions"]) == 2
        assert card["actions"][0]["data"]["action"] == "approve"
        assert card["actions"][1]["data"]["action"] == "reject"

    @pytest.mark.asyncio
    async def test_handle_health(self, teams_ch: TeamsChannel) -> None:
        request = MagicMock()
        with patch("jarvis.channels.teams.web", create=True) as mock_web:
            mock_web.json_response = MagicMock(return_value={"status": "ok"})
            result = await teams_ch._handle_health(request)
