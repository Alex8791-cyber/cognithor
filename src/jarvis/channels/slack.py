"""Slack-Channel: Bidirektionale Kommunikation über Slack.

Nutzt Slack Socket Mode für eingehende Nachrichten (kein HTTP-Server nötig)
und Web-API für ausgehende Nachrichten. Unterstützt:
  - Eingehende Nachrichten (message Event via Socket Mode)
  - Ausgehende Nachrichten (chat.postMessage)
  - Interaktive Approvals (Block Kit Buttons)
  - Streaming (Buffer → einzelne Nachricht)
  - App-Mentions (@Jarvis)

Konfiguration:
  - JARVIS_SLACK_BOT_TOKEN: Bot-Token (xoxb-...)
  - JARVIS_SLACK_APP_TOKEN: App-Token für Socket Mode (xapp-...)
  - JARVIS_SLACK_CHANNEL: Standard-Kanal für Benachrichtigungen

Abhängigkeiten:
  pip install slack_sdk slack_bolt

Fallback: Ohne App-Token arbeitet der Channel wie bisher (nur senden).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.channels.interactive import (
    AdaptiveCard,
    ProgressTracker,
    SlackMessageBuilder,
)
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)


class SlackChannel(Channel):
    """Bidirektionale Slack-Integration für Jarvis.

    Empfängt Nachrichten via Socket Mode, sendet via Web-API,
    und unterstützt interaktive Approvals über Block Kit Buttons.
    Ohne App-Token: reiner Send-Only-Modus (backward-compatible).
    """

    def __init__(
        self,
        token: str,
        app_token: str = "",
        default_channel: str | None = None,
    ) -> None:
        self.token = token
        self.app_token = app_token
        self.default_channel = default_channel
        self._client: Any | None = None
        self._socket_handler: Any | None = None
        self._handler: MessageHandler | None = None
        self._running = False
        self._bidirectional = False
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()
        self._stream_buffers: dict[str, list[str]] = {}
        self._bot_user_id: str = ""

    @property
    def name(self) -> str:
        return "slack"

    @property
    def is_bidirectional(self) -> bool:
        """True wenn Socket Mode aktiv ist (Empfang + Senden)."""
        return self._bidirectional

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Slack-Client mit optionalem Socket Mode."""
        self._handler = handler

        try:
            from slack_sdk.web.async_client import AsyncWebClient  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "slack_sdk nicht installiert. pip install slack_sdk slack_bolt"
            )
            return

        self._client = AsyncWebClient(token=self.token)

        # Bot-User-ID ermitteln (um eigene Nachrichten zu ignorieren)
        try:
            auth = await self._client.auth_test()
            self._bot_user_id = auth.get("user_id", "")
            logger.info("Slack-Bot authentifiziert als %s", auth.get("user", "?"))
        except Exception as exc:
            logger.warning("Slack auth_test fehlgeschlagen: %s", exc)

        # Socket Mode starten wenn App-Token vorhanden
        if self.app_token:
            await self._start_socket_mode()
        else:
            logger.info(
                "Slack: Kein App-Token → Send-Only-Modus. "
                "Setze JARVIS_SLACK_APP_TOKEN für bidirektionalen Betrieb."
            )

        self._running = True
        logger.info(
            "SlackChannel gestartet (bidirektional=%s)", self._bidirectional,
        )

    async def _start_socket_mode(self) -> None:
        """Startet Socket Mode für eingehende Events + interaktive Buttons."""
        try:
            from slack_bolt.adapter.socket_mode.async_handler import (  # type: ignore[import-untyped]
                AsyncSocketModeHandler,
            )
            from slack_bolt.async_app import AsyncApp  # type: ignore[import-untyped]
        except ImportError:
            logger.error("slack_bolt nicht installiert. pip install slack_bolt")
            return

        app = AsyncApp(token=self.token)

        @app.event("message")
        async def _on_msg(event: dict[str, Any], say: Any) -> None:  # noqa: ARG001
            await self._on_message(event)

        @app.event("app_mention")
        async def _on_mention(event: dict[str, Any], say: Any) -> None:  # noqa: ARG001
            await self._on_message(event)

        @app.action("jarvis_approve")
        async def _approve(ack: Any, body: dict[str, Any]) -> None:
            await ack()
            await self._on_approval(body, approved=True)

        @app.action("jarvis_reject")
        async def _reject(ack: Any, body: dict[str, Any]) -> None:
            await ack()
            await self._on_approval(body, approved=False)

        self._socket_handler = AsyncSocketModeHandler(app, self.app_token)
        asyncio.get_running_loop().create_task(self._socket_handler.start_async())
        self._bidirectional = True
        logger.info("Slack Socket Mode gestartet")

    # ------------------------------------------------------------------
    # Eingehende Nachrichten
    # ------------------------------------------------------------------

    async def _on_message(self, event: dict[str, Any]) -> None:
        """Verarbeitet eingehende Slack-Nachrichten."""
        user_id = event.get("user", "")
        if user_id == self._bot_user_id or event.get("bot_id"):
            return  # Eigene/Bot-Nachrichten ignorieren

        text = event.get("text", "").strip()
        if not text:
            return

        # Bot-Mention entfernen (<@U12345>)
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts", "")

        incoming = IncomingMessage(
            channel="slack",
            user_id=user_id,
            text=text,
            metadata={
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "ts": event.get("ts", ""),
            },
        )

        if self._handler:
            response = await self._handler(incoming)
            try:
                await self._client.chat_postMessage(
                    channel=channel_id,
                    text=response.text,
                    thread_ts=thread_ts,
                )
            except Exception as exc:
                logger.error("Slack Antwort fehlgeschlagen: %s", exc)

    # ------------------------------------------------------------------
    # Approvals via Block Kit Buttons
    # ------------------------------------------------------------------

    async def _on_approval(self, body: dict[str, Any], *, approved: bool) -> None:
        """Verarbeitet Approval-Button-Klicks."""
        actions = body.get("actions", [{}])
        approval_id = actions[0].get("value", "") if actions else ""

        async with self._approval_lock:
            future = self._approval_futures.pop(approval_id, None)
        if future and not future.done():
            future.set_result(approved)

        # Original-Nachricht aktualisieren (Buttons entfernen)
        status = "✅ Genehmigt" if approved else "❌ Abgelehnt"
        user_name = body.get("user", {}).get("name", "?")
        try:
            ch = body.get("channel", {}).get("id", "")
            ts = body.get("message", {}).get("ts", "")
            if ch and ts:
                await self._client.chat_update(
                    channel=ch, ts=ts,
                    text=f"{status} von {user_name}",
                    blocks=[],
                )
        except Exception as exc:
            logger.warning("Slack Approval-Update fehlgeschlagen: %s", exc)

    # ------------------------------------------------------------------
    # Senden
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        self._running = False
        if self._socket_handler:
            try:
                await self._socket_handler.close_async()
            except Exception:
                logger.debug("Slack socket handler close skipped", exc_info=True)
            self._socket_handler = None
        self._client = None
        self._bidirectional = False
        logger.info("SlackChannel gestoppt")

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an Slack (optional in Thread)."""
        if not self._client:
            logger.warning("SlackChannel ist nicht einsatzbereit")
            return

        channel = message.metadata.get("channel_id") or self.default_channel
        thread_ts = message.metadata.get("thread_ts")
        if channel is None:
            logger.warning("Keine Slack-Channel-ID; Nachricht verworfen")
            return

        try:
            kwargs: dict[str, Any] = {"channel": channel, "text": message.text}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            await self._client.chat_postMessage(**kwargs)
        except Exception:
            logger.exception("Fehler beim Senden über Slack")

    async def send_rich(
        self,
        builder: SlackMessageBuilder,
        channel: str = "",
        thread_ts: str = "",
    ) -> None:
        """Sendet eine Rich Message (Block Kit) an Slack."""
        if not self._client:
            logger.warning("SlackChannel ist nicht einsatzbereit")
            return

        target = channel or self.default_channel
        if not target:
            return

        try:
            msg = builder.build()
            kwargs: dict[str, Any] = {"channel": target, **msg}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            await self._client.chat_postMessage(**kwargs)
        except Exception:
            logger.exception("Fehler beim Rich-Senden über Slack")

    async def send_card(
        self,
        card: AdaptiveCard,
        channel: str = "",
    ) -> None:
        """Sendet eine plattform-übergreifende AdaptiveCard als Slack-Nachricht."""
        if not self._client:
            return

        target = channel or self.default_channel
        if not target:
            return

        try:
            msg = card.to_slack()
            await self._client.chat_postMessage(channel=target, **msg)
        except Exception:
            logger.exception("Fehler beim Card-Senden über Slack")

    async def send_progress(
        self,
        tracker: ProgressTracker,
        channel: str = "",
    ) -> None:
        """Sendet eine Fortschritts-Anzeige an Slack."""
        if not self._client:
            return

        target = channel or self.default_channel
        if not target:
            return

        try:
            blocks = tracker.to_slack_blocks()
            await self._client.chat_postMessage(
                channel=target,
                text=f"Fortschritt: {tracker.percent_complete}%",
                blocks=blocks,
            )
        except Exception:
            logger.exception("Fehler beim Progress-Senden über Slack")

    async def request_approval(
        self, session_id: str, action: PlannedAction, reason: str,
    ) -> bool:
        """Fragt den User per Block Kit Buttons um Erlaubnis.

        Falls nicht bidirektional: gibt False zurück (wie bisher).
        """
        if not self._bidirectional or not self._client:
            logger.warning("Slack: Approval nicht möglich (kein Socket Mode)")
            return False

        target_channel = self.default_channel
        if not target_channel:
            logger.warning("Slack: Approval ohne default_channel unmöglich")
            return False

        approval_id = f"appr_{session_id}_{action.tool}_{id(action)}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._approval_futures[approval_id] = future

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"🔶 *Genehmigung erforderlich*\n"
                        f"*Tool:* `{action.tool}`\n"
                        f"*Grund:* {reason}\n"
                        f"*Parameter:* ```{str(action.params)[:200]}```"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Genehmigen"},
                        "style": "primary",
                        "action_id": "jarvis_approve",
                        "value": approval_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Ablehnen"},
                        "style": "danger",
                        "action_id": "jarvis_reject",
                        "value": approval_id,
                    },
                ],
            },
        ]

        try:
            await self._client.chat_postMessage(
                channel=target_channel,
                text=f"Genehmigung erforderlich: {action.tool}",
                blocks=blocks,
            )
        except Exception as exc:
            logger.error("Slack Approval fehlgeschlagen: %s", exc)
            async with self._approval_lock:
                self._approval_futures.pop(approval_id, None)
            return False

        try:
            return await asyncio.wait_for(future, timeout=300.0)
        except asyncio.TimeoutError:
            logger.warning("Slack Approval Timeout: %s", action.tool)
            async with self._approval_lock:
                self._approval_futures.pop(approval_id, None)
            return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Buffert Streaming-Tokens und sendet sie als eine Nachricht."""
        buf = self._stream_buffers.setdefault(session_id, [])
        buf.append(token)
        # Beim ersten Token einen verzögerten Flush starten
        if len(buf) == 1:
            await asyncio.sleep(0.5)
            text = "".join(self._stream_buffers.pop(session_id, []))
            if text.strip():
                await self.send(
                    OutgoingMessage(
                        channel=self.name, text=text, session_id=session_id,
                    )
                )
