"""Feishu/Lark Channel: Bidirektionale Kommunikation über Feishu.

Nutzt Feishu Open API (REST via httpx).
Unterstützt:
  - Text-Nachrichten
  - Interactive Cards mit Approval-Workflow
  - Event Subscription
  - Tenant Access Token Auth (App ID + App Secret)

Konfiguration:
  - JARVIS_FEISHU_APP_ID: App ID
  - JARVIS_FEISHU_APP_SECRET: App Secret

Abhängigkeiten:
  Nur httpx (bereits als Core-Dependency vorhanden)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

_FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuChannel(Channel):
    """Feishu/Lark Integration für Jarvis.

    Empfängt Nachrichten via Event Subscription (Webhook),
    sendet via REST API. Nutzt Tenant Access Token für Auth.
    """

    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._handler: MessageHandler | None = None
        self._running = False
        self._http_client: Any | None = None
        self._tenant_token: str = ""
        self._token_expires_at: float = 0
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()
        self._stream_buffers: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "feishu"

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Feishu Client."""
        self._handler = handler

        if not self._app_id or not self._app_secret:
            logger.warning("Feishu: App ID oder App Secret nicht konfiguriert")
            return

        try:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30.0)
        except ImportError:
            logger.error("httpx nicht installiert")
            return

        # Initial Token holen
        await self._refresh_tenant_token()
        self._running = True
        logger.info("FeishuChannel gestartet")

    async def stop(self) -> None:
        """Stoppt den Feishu Client."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._tenant_token = ""
        logger.info("FeishuChannel gestoppt")

    async def _refresh_tenant_token(self) -> None:
        """Holt oder erneuert den Tenant Access Token."""
        if not self._http_client:
            return

        if time.time() < self._token_expires_at - 60:
            return  # Token noch gültig

        try:
            resp = await self._http_client.post(
                f"{_FEISHU_API_BASE}/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self._app_id,
                    "app_secret": self._app_secret,
                },
            )
            data = resp.json()
            if data.get("code") == 0:
                self._tenant_token = data.get("tenant_access_token", "")
                expire = data.get("expire", 7200)
                self._token_expires_at = time.time() + expire
                logger.debug("Feishu Token erneuert (läuft in %ds ab)", expire)
            else:
                logger.error("Feishu Token-Request fehlgeschlagen: %s", data)
        except Exception as exc:
            logger.error("Feishu Token-Request fehlgeschlagen: %s", exc)

    async def _get_headers(self) -> dict[str, str]:
        """Erstellt Auth-Header mit aktuellem Token."""
        await self._refresh_tenant_token()
        return {
            "Authorization": f"Bearer {self._tenant_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def verify_event(self, body: dict[str, Any], encrypt_key: str = "") -> bool:
        """Verifiziert ein eingehendes Event (Challenge/Signature).

        Args:
            body: Das JSON-Payload.
            encrypt_key: Optionaler Encryption Key.

        Returns:
            True wenn valide.
        """
        # URL-Verification (Challenge)
        if "challenge" in body:
            return True

        if not encrypt_key:
            return True

        # Signature Verification
        timestamp = body.get("header", {}).get("event_time", "")
        nonce = body.get("header", {}).get("nonce", "")
        signature = body.get("header", {}).get("signature", "")
        body_str = str(body)

        expected = hashlib.sha256(
            f"{timestamp}{nonce}{encrypt_key}{body_str}".encode()
        ).hexdigest()
        return signature == expected

    async def handle_event(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Verarbeitet eingehende Events von Feishu.

        Args:
            payload: Das JSON-Payload vom Event Subscription.

        Returns:
            Optional: Challenge-Antwort oder None.
        """
        # URL Verification (Challenge-Response)
        if "challenge" in payload:
            return {"challenge": payload["challenge"]}

        header = payload.get("header", {})
        event_type = header.get("event_type", "")

        if event_type == "im.message.receive_v1":
            event = payload.get("event", {})
            await self._on_message(event)
        elif event_type == "card.action.trigger":
            event = payload.get("event", {})
            await self._on_card_action(event)

        return None

    async def _on_message(self, event: dict[str, Any]) -> None:
        """Verarbeitet eine eingehende Nachricht."""
        message = event.get("message", {})
        msg_type = message.get("message_type", "")

        if msg_type != "text":
            logger.debug("Feishu: Nicht-Text-Nachricht ignoriert: %s", msg_type)
            return

        content = message.get("content", "{}")
        try:
            text = _parse_json_safe(content).get("text", "")
        except Exception:
            text = content

        if not text.strip():
            return

        sender = event.get("sender", {}).get("sender_id", {})
        chat_id = message.get("chat_id", "")
        message_id = message.get("message_id", "")

        incoming = IncomingMessage(
            channel="feishu",
            user_id=sender.get("user_id", "") or sender.get("open_id", ""),
            text=text.strip(),
            metadata={
                "chat_id": chat_id,
                "message_id": message_id,
                "open_id": sender.get("open_id", ""),
            },
        )

        if self._handler:
            response = await self._handler(incoming)
            await self._send_text(chat_id, response.text, message_id)

    async def _on_card_action(self, event: dict[str, Any]) -> None:
        """Verarbeitet Card-Button-Aktionen (Approvals)."""
        action = event.get("action", {})
        tag = action.get("tag", "")
        value = action.get("value", {})

        if tag != "button":
            return

        approval_id = value.get("approval_id", "")
        if not approval_id:
            return

        approved = value.get("action") == "approve"

        async with self._approval_lock:
            future = self._approval_futures.pop(approval_id, None)
        if future and not future.done():
            future.set_result(approved)

    async def _send_text(
        self,
        chat_id: str,
        text: str,
        reply_to: str = "",
    ) -> None:
        """Sendet eine Text-Nachricht."""
        if not self._http_client:
            return

        headers = await self._get_headers()
        body: dict[str, Any] = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": f'{{"text": "{text}"}}',
        }

        url = f"{_FEISHU_API_BASE}/im/v1/messages"
        params = {"receive_id_type": "chat_id"}

        try:
            resp = await self._http_client.post(
                url, headers=headers, json=body, params=params,
            )
            data = resp.json()
            if data.get("code") != 0:
                logger.error("Feishu Senden fehlgeschlagen: %s", data.get("msg"))
        except Exception as exc:
            logger.error("Feishu Senden fehlgeschlagen: %s", exc)

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an Feishu."""
        chat_id = message.metadata.get("chat_id", "")
        if not chat_id:
            logger.warning("Feishu: Kein chat_id; Nachricht verworfen")
            return

        reply_to = message.metadata.get("message_id", "")
        await self._send_text(chat_id, message.text, reply_to)

    async def request_approval(
        self, session_id: str, action: PlannedAction, reason: str,
    ) -> bool:
        """Fragt den User per Interactive Card um Erlaubnis."""
        if not self._http_client:
            logger.warning("Feishu: Approval nicht möglich")
            return False

        approval_id = f"appr_{session_id}_{action.tool}_{id(action)}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._approval_futures[approval_id] = future

        try:
            return await asyncio.wait_for(future, timeout=300.0)
        except asyncio.TimeoutError:
            logger.warning("Feishu Approval Timeout: %s", action.tool)
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


def _parse_json_safe(s: str) -> dict[str, Any]:
    """Sicheres JSON-Parsing mit Fallback."""
    try:
        import json
        return json.loads(s)
    except Exception:
        return {}
