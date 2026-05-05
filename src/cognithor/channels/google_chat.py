"""Google Chat Channel: Bidirektionale Kommunikation ueber Google Chat.

Nutzt Google Chat API (REST via httpx) mit Pub/Sub oder Webhook fuer Events.
Unterstuetzt:
  - Text-Nachrichten in Spaces und DMs
  - Cards fuer strukturierte Antworten
  - Approval-Buttons ueber interaktive Cards
  - Streaming (Buffer → einzelne Nachricht)

Konfiguration:
  - COGNITHOR_GOOGLE_CHAT_CREDENTIALS: Pfad zur Service Account JSON
  - COGNITHOR_GOOGLE_CHAT_ALLOWED_SPACES: Erlaubte Space-IDs

Abhaengigkeiten:
  pip install google-auth google-api-core
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from cognithor.channels.base import Channel, MessageHandler
from cognithor.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

_CHAT_API_BASE = "https://chat.googleapis.com/v1"


class GoogleChatChannel(Channel):
    """Google Chat Integration fuer Cognithor.

    Empfaengt Nachrichten via Webhook/Pub/Sub, sendet via REST API.
    Unterstuetzt Spaces, DMs und interaktive Cards.
    """

    def __init__(
        self,
        credentials_path: str = "",
        allowed_spaces: list[str] | None = None,
        expected_audience: str = "",
    ) -> None:
        self._credentials_path = credentials_path
        self._allowed_spaces: set[str] = set(allowed_spaces or [])
        # PASS-4 SEC-CRIT: Google Chat signs every webhook payload with a
        # JWT keyed against the configured project / bot service account.
        # When ``expected_audience`` is non-empty, ``handle_webhook``
        # refuses any payload whose Authorization header does not carry
        # a JWT signed by the Chat-system service account and addressed
        # to this audience. Empty string keeps the unauthenticated path
        # for tests / local dev — operators must set it in production.
        self._expected_audience = expected_audience
        self._handler: MessageHandler | None = None
        self._running = False
        self._http_client: Any | None = None
        self._credentials: Any | None = None
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._approval_users: dict[str, str] = {}  # approval_id → user name
        self._approval_lock = asyncio.Lock()
        self._session_users: dict[str, str] = {}  # session_id → user name
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
            from google.oauth2 import service_account
        except ImportError:
            logger.error("google-auth nicht installiert. pip install google-auth google-api-core")
            return

        creds_path = Path(self._credentials_path)
        if not creds_path.exists():
            logger.error("Google Chat Credentials nicht gefunden: %s", creds_path)
            return

        try:
            self._credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call, unused-ignore]
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
        """Prueft ob der Space erlaubt ist."""
        if not self._allowed_spaces:
            return True  # Keine Whitelist = alles erlaubt
        return space_name in self._allowed_spaces

    async def _get_auth_headers(self) -> dict[str, str]:
        """Erstellt Auth-Header mit aktuellem Token."""
        if not self._credentials:
            return {}
        try:
            from google.auth.transport.requests import Request

            self._credentials.refresh(Request())  # type: ignore[no-untyped-call, unused-ignore]
            return {"Authorization": f"Bearer {self._credentials.token}"}
        except Exception as exc:
            logger.error("Google Chat Token-Refresh fehlgeschlagen: %s", exc)
            return {}

    def _verify_chat_jwt(self, auth_header: str | None) -> bool:
        """Verify a Google-Chat-issued JWT against the configured audience.

        Google Chat sends ``Authorization: Bearer <jwt>`` on every
        webhook call. The JWT is signed by the Chat system service
        account ``chat@system.gserviceaccount.com`` and the ``aud``
        claim equals the bot's project number / endpoint URL (whatever
        the operator registered in the Chat publishing console).
        Returns ``True`` if the token is valid for the configured
        audience, ``False`` otherwise.
        """
        if not self._expected_audience:
            return True  # Audience unconfigured -- caller policy.
        if not auth_header or not auth_header.lower().startswith("bearer "):
            return False
        token = auth_header[7:].strip()
        if not token:
            return False
        try:
            from google.auth.transport import requests as ga_requests
            from google.oauth2 import id_token as ga_id_token
        except ImportError:
            logger.error(
                "Google Chat JWT verification requires google-auth + "
                "google-api-core. Install via: pip install google-auth"
            )
            return False
        try:
            claims = ga_id_token.verify_token(  # type: ignore[no-untyped-call, unused-ignore]
                token,
                ga_requests.Request(),  # type: ignore[no-untyped-call, unused-ignore]
                audience=self._expected_audience,
            )
        except Exception as exc:
            logger.warning("Google Chat JWT signature verification failed: %s", exc)
            return False
        issuer = claims.get("iss", "")
        # The Chat service account is the only legitimate issuer.
        if issuer not in (
            "chat@system.gserviceaccount.com",
            "https://chat.google.com",
        ):
            logger.warning("Google Chat JWT issuer not allowlisted: %s", issuer)
            return False
        return True

    async def handle_webhook(
        self,
        payload: dict[str, Any],
        auth_header: str | None = None,
    ) -> dict[str, Any] | None:
        """Verarbeitet eingehende Webhook-Events von Google Chat.

        Args:
            payload: Das JSON-Payload vom Webhook.
            auth_header: Der ``Authorization``-Header aus dem
                FastAPI-Request. Wird gegen das configured
                ``expected_audience`` per JWT verifiziert wenn gesetzt.

        Returns:
            Optional: Synchrone Antwort fuer den Webhook.
        """
        if not self._verify_chat_jwt(auth_header):
            logger.warning("Google Chat: Webhook rejected - JWT verification failed")
            return None
        event_type = payload.get("type", "")

        if event_type == "MESSAGE":
            return await self._handle_message_event(payload)
        elif event_type == "CARD_CLICKED":
            await self._handle_card_click(payload)
        elif event_type == "ADDED_TO_SPACE":
            logger.info(
                "Google Chat: Zu Space hinzugefügt: %s", payload.get("space", {}).get("name", "?")
            )

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

        sender_name = sender.get("name", "")
        session_id = f"gchat_{sender_name}_{space_name}"
        self._session_users[session_id] = sender_name

        incoming = IncomingMessage(
            channel="google_chat",
            user_id=sender_name,
            text=text.strip(),
            session_id=session_id,
            metadata={
                "space_name": space_name,
                "message_name": message.get("name", ""),
                "thread_name": message.get("thread", {}).get("name", ""),
                "user_display_name": user_name,
            },
        )

        if self._handler:
            try:
                response = await self._handler(incoming)
                return {"text": response.text}
            except Exception as exc:
                logger.error("Google Chat: Handler-Fehler: %s", exc)
                try:
                    from cognithor.utils.error_messages import classify_error_for_user

                    friendly = classify_error_for_user(exc)
                except Exception:
                    friendly = "Ein Fehler ist bei der Verarbeitung aufgetreten."
                return {"text": friendly}

        return None

    async def _handle_card_click(self, payload: dict[str, Any]) -> None:
        """Verarbeitet Card-Button-Klicks (Approvals).

        Nur der User, der die Aktion urspruenglich ausgeloest hat,
        darf genehmigen oder ablehnen.
        """
        action = payload.get("action", {})
        action_method = action.get("actionMethodName", "")
        parameters = {p["key"]: p["value"] for p in action.get("parameters", [])}

        approval_id = parameters.get("approval_id", "")
        if not approval_id:
            return

        clicker = payload.get("user", {}).get("name", "")
        approved = action_method == "jarvis_approve"

        async with self._approval_lock:
            expected_user = self._approval_users.get(approval_id, "")
            if expected_user and clicker != expected_user:
                logger.warning(
                    "Google Chat Approval von fremdem User ignoriert: %s (erwartet: %s)",
                    clicker,
                    expected_user,
                )
                return
            future = self._approval_futures.pop(approval_id, None)
            self._approval_users.pop(approval_id, None)
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
                logger.error(
                    "Google Chat Senden fehlgeschlagen: %s %s", resp.status_code, resp.text
                )
        except Exception as exc:
            logger.error("Google Chat Senden fehlgeschlagen: %s", exc)

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Fragt den User per Card-Buttons um Erlaubnis."""
        if not self._http_client or not self._credentials:
            logger.warning("Google Chat: Approval nicht möglich")
            return False

        approval_id = f"appr_{session_id}_{action.tool}_{id(action)}"
        requester_user = self._session_users.get(session_id, "")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._approval_futures[approval_id] = future
            if requester_user:
                self._approval_users[approval_id] = requester_user

        _card = {
            "cardsV2": [
                {
                    "cardId": approval_id,
                    "card": {
                        "header": {"title": "Genehmigung erforderlich"},
                        "sections": [
                            {
                                "widgets": [
                                    {"textParagraph": {"text": f"<b>Tool:</b> {action.tool}"}},
                                    {"textParagraph": {"text": f"<b>Grund:</b> {reason}"}},
                                    {
                                        "textParagraph": {
                                            "text": (
                                                f"<b>Parameter:</b> "
                                                f"{json.dumps(action.params)[:200]}"
                                            )
                                        }
                                    },
                                    {
                                        "buttonList": {
                                            "buttons": [
                                                {
                                                    "text": "Genehmigen",
                                                    "onClick": {
                                                        "action": {
                                                            "actionMethodName": "jarvis_approve",
                                                            "parameters": [
                                                                {
                                                                    "key": "approval_id",
                                                                    "value": approval_id,
                                                                }
                                                            ],
                                                        }
                                                    },
                                                    "color": {
                                                        "red": 0.2,
                                                        "green": 0.7,
                                                        "blue": 0.3,
                                                        "alpha": 1,
                                                    },
                                                },
                                                {
                                                    "text": "Ablehnen",
                                                    "onClick": {
                                                        "action": {
                                                            "actionMethodName": "jarvis_reject",
                                                            "parameters": [
                                                                {
                                                                    "key": "approval_id",
                                                                    "value": approval_id,
                                                                }
                                                            ],
                                                        }
                                                    },
                                                    "color": {
                                                        "red": 0.8,
                                                        "green": 0.2,
                                                        "blue": 0.2,
                                                        "alpha": 1,
                                                    },
                                                },
                                            ]
                                        }
                                    },
                                ],
                            }
                        ],
                    },
                }
            ],
        }

        # Card wird ueber Webhook-Antwort gesendet, hier nur Future warten
        try:
            return await asyncio.wait_for(future, timeout=300.0)
        except TimeoutError:
            logger.warning("Google Chat Approval Timeout: %s", action.tool)
            async with self._approval_lock:
                self._approval_futures.pop(approval_id, None)
                self._approval_users.pop(approval_id, None)
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
                        channel=self.name,
                        text=text,
                        session_id=session_id,
                    )
                )
