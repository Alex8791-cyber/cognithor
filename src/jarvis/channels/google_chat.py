"""Google Chat Channel: Bidirektionale Kommunikation über Google Chat.

Nutzt Google Chat API (REST via httpx) mit Pub/Sub oder Webhook für Events.
Unterstützt:
  - Text-Nachrichten in Spaces und DMs
  - Cards für strukturierte Antworten
  - Approval-Buttons über interaktive Cards
  - Streaming (Buffer → einzelne Nachricht)

Konfiguration:
  - JARVIS_GOOGLE_CHAT_CREDENTIALS: Pfad zur Service Account JSON
  - JARVIS_GOOGLE_CHAT_ALLOWED_SPACES: Erlaubte Space-IDs

Abhängigkeiten:
  pip install google-auth google-api-core
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

_CHAT_API_BASE = "https://chat.googleapis.com/v1"


class GoogleChatChannel(Channel):
    """Google Chat Integration für Jarvis.

    Empfängt Nachrichten via Webhook/Pub/Sub, sendet via REST API.
    Unterstützt Spaces, DMs und interaktive Cards.
    """

    def __init__(
        self,
        credentials_path: str = "",
        allowed_spaces: list[str] | None = None,
    ) -> None:
        self._credentials_path = credentials_path
        self._allowed_spaces: set[str] = set(allowed_spaces or [])
        self._handler: MessageHandler | None = None
        self._running = False
        self._http_client: Any | None = None
        self._credentials: Any | None = None
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()
        self._stream_buffers: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "google_chat"

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Google Chat Client."""
        self._handler = handler

        if not self._credentials_path:
            logger.warning("Google Chat: Kein Credentials-Pfad konfiguriert")
            return

        try:
            from google.oauth2 import service_account  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "google-auth nicht installiert. pip install google-auth google-api-core"
            )
            return

        creds_path = Path(self._credentials_path)
        if not creds_path.exists():
            logger.error("Google Chat Credentials nicht gefunden: %s", creds_path)
            return

        try:
            self._credentials = service_account.Credentials.from_service_account_file(
                str(creds_path),
                scopes=["https://www.googleapis.com/auth/chat.bot"],
            )
        except Exception as exc:
            logger.error("Google Chat Auth fehlgeschlagen: %s", exc)
            return

        try:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30.0)
        except ImportError:
            logger.error("httpx nicht installiert")
            return

        self._running = True
        logger.info("GoogleChatChannel gestartet")

    async def stop(self) -> None:
        """Stoppt den Google Chat Client."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._credentials = None
        logger.info("GoogleChatChannel gestoppt")

    def _is_space_allowed(self, space_name: str) -> bool:
        """Prüft ob der Space erlaubt ist."""
        if not self._allowed_spaces:
            return True  # Keine Whitelist = alles erlaubt
        return space_name in self._allowed_spaces

    async def _get_auth_headers(self) -> dict[str, str]:
        """Erstellt Auth-Header mit aktuellem Token."""
        if not self._credentials:
            return {}
        try:
            from google.auth.transport.requests import Request  # type: ignore[import-untyped]
            self._credentials.refresh(Request())
            return {"Authorization": f"Bearer {self._credentials.token}"}
        except Exception as exc:
            logger.error("Google Chat Token-Refresh fehlgeschlagen: %s", exc)
            return {}

    async def handle_webhook(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Verarbeitet eingehende Webhook-Events von Google Chat.

        Args:
            payload: Das JSON-Payload vom Webhook.

        Returns:
            Optional: Synchrone Antwort für den Webhook.
        """
        event_type = payload.get("type", "")

        if event_type == "MESSAGE":
            return await self._handle_message_event(payload)
        elif event_type == "CARD_CLICKED":
            await self._handle_card_click(payload)
        elif event_type == "ADDED_TO_SPACE":
            logger.info("Google Chat: Zu Space hinzugefügt: %s",
                        payload.get("space", {}).get("name", "?"))

        return None

    async def _handle_message_event(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Verarbeitet eine eingehende Nachricht."""
        space = payload.get("space", {})
        space_name = space.get("name", "")

        if not self._is_space_allowed(space_name):
            logger.warning("Google Chat: Space nicht erlaubt: %s", space_name)
            return None

        message = payload.get("message", {})
        text = message.get("argumentText", "") or message.get("text", "")
        sender = message.get("sender", {})
        user_name = sender.get("displayName", "unknown")

        if not text.strip():
            return None

        incoming = IncomingMessage(
            channel="google_chat",
            user_id=sender.get("name", ""),
            text=text.strip(),
            metadata={
                "space_name": space_name,
                "message_name": message.get("name", ""),
                "thread_name": message.get("thread", {}).get("name", ""),
                "user_display_name": user_name,
            },
        )

        if self._handler:
            response = await self._handler(incoming)
            return {"text": response.text}

        return None

    async def _handle_card_click(self, payload: dict[str, Any]) -> None:
        """Verarbeitet Card-Button-Klicks (Approvals)."""
        action = payload.get("action", {})
        action_method = action.get("actionMethodName", "")
        parameters = {p["key"]: p["value"] for p in action.get("parameters", [])}

        approval_id = parameters.get("approval_id", "")
        if not approval_id:
            return

        approved = action_method == "jarvis_approve"

        async with self._approval_lock:
            future = self._approval_futures.pop(approval_id, None)
        if future and not future.done():
            future.set_result(approved)

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an einen Google Chat Space."""
        if not self._http_client or not self._credentials:
            logger.warning("GoogleChatChannel ist nicht einsatzbereit")
            return

        space_name = message.metadata.get("space_name", "")
        if not space_name:
            logger.warning("Google Chat: Kein Space angegeben")
            return

        headers = await self._get_auth_headers()
        if not headers:
            return

        url = f"{_CHAT_API_BASE}/{space_name}/messages"
        body: dict[str, Any] = {"text": message.text}

        thread_name = message.metadata.get("thread_name")
        if thread_name:
            body["thread"] = {"name": thread_name}

        try:
            resp = await self._http_client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                logger.error("Google Chat Senden fehlgeschlagen: %s %s",
                             resp.status_code, resp.text)
        except Exception as exc:
            logger.error("Google Chat Senden fehlgeschlagen: %s", exc)

    async def request_approval(
        self, session_id: str, action: PlannedAction, reason: str,
    ) -> bool:
        """Fragt den User per Card-Buttons um Erlaubnis."""
        if not self._http_client or not self._credentials:
            logger.warning("Google Chat: Approval nicht möglich")
            return False

        approval_id = f"appr_{session_id}_{action.tool}_{id(action)}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._approval_futures[approval_id] = future

        card = {
            "cardsV2": [{
                "cardId": approval_id,
                "card": {
                    "header": {"title": "Genehmigung erforderlich"},
                    "sections": [{
                        "widgets": [
                            {"textParagraph": {"text": f"<b>Tool:</b> {action.tool}"}},
                            {"textParagraph": {"text": f"<b>Grund:</b> {reason}"}},
                            {"textParagraph": {
                                "text": f"<b>Parameter:</b> {json.dumps(action.params)[:200]}"
                            }},
                            {"buttonList": {"buttons": [
                                {
                                    "text": "Genehmigen",
                                    "onClick": {"action": {
                                        "actionMethodName": "jarvis_approve",
                                        "parameters": [
                                            {"key": "approval_id", "value": approval_id}
                                        ],
                                    }},
                                    "color": {"red": 0.2, "green": 0.7, "blue": 0.3, "alpha": 1},
                                },
                                {
                                    "text": "Ablehnen",
                                    "onClick": {"action": {
                                        "actionMethodName": "jarvis_reject",
                                        "parameters": [
                                            {"key": "approval_id", "value": approval_id}
                                        ],
                                    }},
                                    "color": {"red": 0.8, "green": 0.2, "blue": 0.2, "alpha": 1},
                                },
                            ]}},
                        ],
                    }],
                },
            }],
        }

        # Card wird über Webhook-Antwort gesendet, hier nur Future warten
        try:
            return await asyncio.wait_for(future, timeout=300.0)
        except asyncio.TimeoutError:
            logger.warning("Google Chat Approval Timeout: %s", action.tool)
            async with self._approval_lock:
                self._approval_futures.pop(approval_id, None)
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
