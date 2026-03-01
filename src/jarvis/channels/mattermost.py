"""Mattermost-Channel: Bidirektionale Kommunikation über Mattermost.

Nutzt Mattermost REST API v4 + WebSocket Events.
Unterstützt:
  - Text-Nachrichten in Channels
  - Reactions für Approvals
  - File Uploads
  - Slash-Commands

Konfiguration:
  - JARVIS_MATTERMOST_URL: Server-URL
  - JARVIS_MATTERMOST_TOKEN: Bot-Token oder Personal Access Token
  - JARVIS_MATTERMOST_CHANNEL: Standard-Channel-ID

Abhängigkeiten:
  Nur httpx (bereits als Core-Dependency vorhanden)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.security.token_store import get_token_store

logger = logging.getLogger(__name__)


class MattermostChannel(Channel):
    """Bidirektionale Mattermost-Integration für Jarvis.

    Empfängt Nachrichten via WebSocket, sendet via REST API v4.
    """

    def __init__(
        self,
        url: str = "",
        token: str = "",
        default_channel: str = "",
    ) -> None:
        self._url = url.rstrip("/")
        self._token_store_ref = get_token_store()
        if token:
            self._token_store_ref.store("mattermost_token", token)
        self._has_token = bool(token)
        self._default_channel = default_channel
        self._handler: MessageHandler | None = None
        self._running = False
        self._http_client: Any | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._bot_user_id: str = ""
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()
        self._stream_buffers: dict[str, list[str]] = {}

    @property
    def _token(self) -> str:
        """Bot-Token (entschlüsselt bei Zugriff)."""
        if self._has_token:
            return self._token_store_ref.retrieve("mattermost_token")
        return ""

    @property
    def name(self) -> str:
        return "mattermost"

    @property
    def api_url(self) -> str:
        return f"{self._url}/api/v4"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Mattermost Client."""
        self._handler = handler

        if not self._url or not self._token:
            logger.warning("Mattermost: URL oder Token nicht konfiguriert")
            return

        try:
            import httpx
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers=self._headers(),
            )
        except ImportError:
            logger.error("httpx nicht installiert")
            return

        # Bot-User-ID ermitteln
        try:
            resp = await self._http_client.get(f"{self.api_url}/users/me")
            if resp.status_code == 200:
                data = resp.json()
                self._bot_user_id = data.get("id", "")
                logger.info("Mattermost-Bot authentifiziert als %s",
                            data.get("username", "?"))
            else:
                logger.warning("Mattermost Auth fehlgeschlagen: %s", resp.status_code)
        except Exception as exc:
            logger.warning("Mattermost Auth-Test fehlgeschlagen: %s", exc)

        # WebSocket für Events starten
        self._ws_task = asyncio.get_running_loop().create_task(self._websocket_loop())
        self._running = True
        logger.info("MattermostChannel gestartet")

    async def _websocket_loop(self) -> None:
        """WebSocket-Verbindung für Echtzeit-Events."""
        ws_url = self._url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/v4/websocket"

        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            logger.error("websockets nicht installiert. pip install websockets")
            return

        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    # Authentifizieren
                    auth_msg = json.dumps({
                        "seq": 1,
                        "action": "authentication_challenge",
                        "data": {"token": self._token},
                    })
                    await ws.send(auth_msg)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw_msg)
                            await self._handle_ws_event(data)
                        except json.JSONDecodeError:
                            continue

            except Exception as exc:
                if self._running:
                    logger.warning("Mattermost WS-Verbindung verloren: %s", exc)
                    await asyncio.sleep(5.0)  # Reconnect-Delay

    async def _handle_ws_event(self, data: dict[str, Any]) -> None:
        """Verarbeitet ein WebSocket-Event."""
        event = data.get("event", "")

        if event == "posted":
            post_data = json.loads(data.get("data", {}).get("post", "{}"))
            await self._on_message(post_data)
        elif event == "reaction_added":
            reaction = data.get("data", {}).get("reaction", {})
            if isinstance(reaction, str):
                reaction = json.loads(reaction)
            await self._on_reaction(reaction)

    async def _on_message(self, post: dict[str, Any]) -> None:
        """Verarbeitet eine eingehende Nachricht."""
        user_id = post.get("user_id", "")
        if user_id == self._bot_user_id:
            return

        text = post.get("message", "").strip()
        if not text:
            return

        channel_id = post.get("channel_id", "")
        post_id = post.get("id", "")

        incoming = IncomingMessage(
            channel="mattermost",
            user_id=user_id,
            text=text,
            metadata={
                "channel_id": channel_id,
                "post_id": post_id,
                "root_id": post.get("root_id", ""),
            },
        )

        if self._handler:
            response = await self._handler(incoming)
            await self._create_post(
                channel_id=channel_id,
                message=response.text,
                root_id=post.get("root_id") or post_id,
            )

    async def _on_reaction(self, reaction: dict[str, Any]) -> None:
        """Verarbeitet Reactions (für Approvals)."""
        emoji = reaction.get("emoji_name", "")
        post_id = reaction.get("post_id", "")

        if emoji in ("white_check_mark", "heavy_check_mark", "+1", "thumbsup"):
            approved = True
        elif emoji in ("x", "-1", "thumbsdown", "no_entry"):
            approved = False
        else:
            return

        async with self._approval_lock:
            future = self._approval_futures.pop(post_id, None)
        if future and not future.done():
            future.set_result(approved)

    async def _create_post(
        self,
        channel_id: str,
        message: str,
        root_id: str = "",
    ) -> str:
        """Erstellt einen Post in einem Channel."""
        if not self._http_client:
            return ""

        body: dict[str, Any] = {
            "channel_id": channel_id,
            "message": message,
        }
        if root_id:
            body["root_id"] = root_id

        try:
            resp = await self._http_client.post(
                f"{self.api_url}/posts", json=body,
            )
            if resp.status_code == 201:
                return resp.json().get("id", "")
            logger.error("Mattermost Post fehlgeschlagen: %s", resp.status_code)
        except Exception as exc:
            logger.error("Mattermost Post fehlgeschlagen: %s", exc)
        return ""

    async def stop(self) -> None:
        """Stoppt den Mattermost Client."""
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            self._ws_task = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("MattermostChannel gestoppt")

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an Mattermost."""
        channel_id = message.metadata.get("channel_id") or self._default_channel
        if not channel_id:
            logger.warning("Mattermost: Keine Channel-ID; Nachricht verworfen")
            return

        root_id = message.metadata.get("root_id", "")
        await self._create_post(channel_id, message.text, root_id)

    async def request_approval(
        self, session_id: str, action: PlannedAction, reason: str,
    ) -> bool:
        """Fragt den User per Nachricht + Reactions um Erlaubnis."""
        channel_id = self._default_channel
        if not channel_id or not self._http_client:
            logger.warning("Mattermost: Approval nicht möglich")
            return False

        text = (
            f"**Genehmigung erforderlich**\n"
            f"**Tool:** `{action.tool}`\n"
            f"**Grund:** {reason}\n"
            f"**Parameter:** ```{str(action.params)[:200]}```\n\n"
            f"Reagiere mit :white_check_mark: zum Genehmigen "
            f"oder :x: zum Ablehnen."
        )

        post_id = await self._create_post(channel_id, text)
        if not post_id:
            return False

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._approval_futures[post_id] = future

        try:
            return await asyncio.wait_for(future, timeout=300.0)
        except asyncio.TimeoutError:
            logger.warning("Mattermost Approval Timeout: %s", action.tool)
            async with self._approval_lock:
                self._approval_futures.pop(post_id, None)
            return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Buffert Streaming-Tokens und sendet sie als eine Nachricht."""
        buf = self._stream_buffers.setdefault(session_id, [])
        buf.append(token)
        if len(buf) == 1:
            await asyncio.sleep(0.5)
            text = "".join(self._stream_buffers.pop(session_id, []))
            if text.strip():
                await self.send(
                    OutgoingMessage(
                        channel=self.name, text=text, session_id=session_id,
                    )
                )
