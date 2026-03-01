"""Microsoft Teams Channel: Bidirektionale Kommunikation ueber Bot Framework.

Nutzt botbuilder-core/botbuilder-integration-aiohttp fuer den
Microsoft Bot Framework v4:
  https://github.com/microsoft/botbuilder-python

Features:
  - Text senden/empfangen via Bot Framework Adapter
  - Adaptive Cards fuer Approval-Workflows
  - Proaktive Nachrichten (via ConversationReference)
  - Activity-based Messaging (Typing-Indicator)
  - Channel- und 1:1-Chat Support
  - Mentions-Handling
  - Message-Splitting bei 4000 Zeichen
  - Session->Conversation Mapping
  - Graceful Shutdown

Konfiguration:
  - JARVIS_TEAMS_APP_ID: Bot-App-ID (Azure AD)
  - JARVIS_TEAMS_APP_PASSWORD: Bot-App-Passwort
  - JARVIS_TEAMS_WEBHOOK_HOST: Webhook-Host (Default: 127.0.0.1)
  - JARVIS_TEAMS_WEBHOOK_PORT: Webhook-Port (Default: 3978)

Abhaengigkeiten:
  pip install botbuilder-core botbuilder-integration-aiohttp
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.security.token_store import get_token_store

if TYPE_CHECKING:
    from jarvis.gateway.session_store import SessionStore

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
APPROVAL_TIMEOUT = 300


class TeamsChannel(Channel):
    """Microsoft Teams als bidirektionaler Kommunikationskanal.

    Nutzt das Microsoft Bot Framework v4 (botbuilder-python)
    fuer die Kommunikation mit Teams.
    """

    def __init__(
        self,
        *,
        app_id: str = "",
        app_password: str = "",
        webhook_host: str = "127.0.0.1",
        webhook_port: int = 3978,
        ssl_certfile: str = "",
        ssl_keyfile: str = "",
        session_store: SessionStore | None = None,
    ) -> None:
        self._app_id = app_id
        self._token_store = get_token_store()
        if app_password:
            self._token_store.store("teams_app_password", app_password)
        self._has_app_password = bool(app_password)
        self._webhook_host = webhook_host
        self._webhook_port = webhook_port
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._session_store = session_store

        self._handler: MessageHandler | None = None
        self._running = False

        # Bot Framework Komponenten
        self._adapter: Any | None = None
        self._bot: Any | None = None
        self._webhook_runner: Any | None = None

        # Conversation-References fuer proaktive Nachrichten
        self._conversation_refs: dict[str, Any] = {}

        # Session-Mapping: conversation_id -> session_id
        self._sessions: dict[str, str] = {}

        # Approval-Workflow
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()

        # Streaming-Buffer
        self._stream_buffers: dict[str, list[str]] = {}

    @property
    def _app_password(self) -> str:
        """App-Passwort (entschlüsselt bei Zugriff)."""
        if self._has_app_password:
            return self._token_store.retrieve("teams_app_password")
        return ""

    @property
    def name(self) -> str:
        return "teams"

    # -- Lifecycle ---------------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Teams Bot mit Webhook-Server."""
        self._handler = handler

        # Persistierte Mappings laden
        if self._session_store:
            for key, val in self._session_store.load_all_channel_mappings("teams_sessions").items():
                self._sessions[key] = val

        try:
            from botbuilder.core import (  # type: ignore[import-untyped]
                BotFrameworkAdapter,
                BotFrameworkAdapterSettings,
                TurnContext,
            )
            from botbuilder.schema import Activity, ActivityTypes  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "botbuilder-core nicht installiert. "
                "Installiere mit: pip install botbuilder-core botbuilder-integration-aiohttp"
            )
            return

        # Adapter erstellen
        settings = BotFrameworkAdapterSettings(self._app_id, self._app_password)
        self._adapter = BotFrameworkAdapter(settings)

        # Error-Handler
        async def on_error(context: Any, error: Exception) -> None:
            logger.error("Teams Bot-Fehler: %s", error)
            try:
                await context.send_activity("Ein interner Fehler ist aufgetreten.")
            except Exception:
                pass

        self._adapter.on_turn_error = on_error

        # Webhook-Server starten
        await self._setup_webhook()

        self._running = True
        logger.info(
            "TeamsChannel gestartet (App-ID=%s, Port=%d)",
            self._app_id[:8] + "..." if self._app_id else "?",
            self._webhook_port,
        )

    async def _setup_webhook(self) -> None:
        """Startet den aiohttp Webhook-Server fuer Bot Framework Messages."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("Teams: aiohttp nicht installiert")
            return

        app = web.Application()
        app.router.add_post("/api/messages", self._handle_messages)
        app.router.add_get("/api/health", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()

        # TLS-Support
        ssl_ctx = None
        if self._ssl_certfile and self._ssl_keyfile:
            from jarvis.security.token_store import create_ssl_context
            ssl_ctx = create_ssl_context(self._ssl_certfile, self._ssl_keyfile)

        if not ssl_ctx and self._webhook_host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning("WARNUNG: Teams-Webhook auf %s ohne TLS gestartet!", self._webhook_host)

        site = web.TCPSite(runner, self._webhook_host, self._webhook_port, ssl_context=ssl_ctx)
        await site.start()

        self._webhook_runner = runner
        logger.info(
            "Teams: Webhook-Server gestartet auf %s:%d (TLS=%s)",
            self._webhook_host,
            self._webhook_port,
            ssl_ctx is not None,
        )

    async def _handle_health(self, request: Any) -> Any:
        """GET /api/health -- Healthcheck."""
        from aiohttp import web
        return web.json_response({"status": "ok", "channel": "teams"})

    async def _handle_messages(self, request: Any) -> Any:
        """POST /api/messages -- Eingehende Bot Framework Activities."""
        from aiohttp import web
        from botbuilder.core import BotFrameworkAdapter, TurnContext  # type: ignore[import-untyped]
        from botbuilder.schema import Activity  # type: ignore[import-untyped]

        if not self._adapter:
            return web.Response(status=503, text="Bot not ready")

        try:
            body = await request.json()
            activity = Activity().deserialize(body)
            auth_header = request.headers.get("Authorization", "")

            async def _turn_handler(turn_context: TurnContext) -> None:
                await self._on_turn(turn_context)

            await self._adapter.process_activity(
                activity, auth_header, _turn_handler,
            )
            return web.Response(status=200)

        except Exception as exc:
            logger.error("Teams: Activity-Verarbeitung fehlgeschlagen: %s", exc)
            return web.Response(status=500, text=str(exc))

    async def stop(self) -> None:
        """Stoppt den Teams-Channel sauber."""
        self._running = False

        # Pending approvals abbrechen
        async with self._approval_lock:
            for future in self._pending_approvals.values():
                if not future.done():
                    future.set_result(False)
            self._pending_approvals.clear()

        # Webhook stoppen
        if self._webhook_runner:
            await self._webhook_runner.cleanup()
            self._webhook_runner = None

        self._adapter = None
        logger.info("TeamsChannel gestoppt")

    # -- Inbound: Activity-Verarbeitung ------------------------------------------

    async def _on_turn(self, turn_context: Any) -> None:
        """Verarbeitet eingehende Bot Framework Activities."""
        from botbuilder.schema import ActivityTypes  # type: ignore[import-untyped]

        activity = turn_context.activity
        activity_type = activity.type

        if activity_type == ActivityTypes.message:
            await self._on_message(turn_context)
        elif activity_type == ActivityTypes.invoke:
            await self._on_invoke(turn_context)
        elif activity_type == ActivityTypes.conversation_update:
            await self._on_conversation_update(turn_context)

    async def _on_message(self, turn_context: Any) -> None:
        """Verarbeitet eingehende Textnachrichten."""
        from botbuilder.core import TurnContext  # type: ignore[import-untyped]

        activity = turn_context.activity
        text = (activity.text or "").strip()
        user_id = activity.from_property.id if activity.from_property else ""
        conversation_id = activity.conversation.id if activity.conversation else ""

        # Conversation-Reference speichern (fuer proaktive Nachrichten)
        ref = TurnContext.get_conversation_reference(activity)
        self._conversation_refs[conversation_id] = ref

        if not text:
            return

        # Bot-Mention entfernen
        if activity.entities:
            for entity in activity.entities:
                if getattr(entity, "type", "") == "mention":
                    mentioned = getattr(entity, "mentioned", None)
                    if mentioned and getattr(mentioned, "id", "") == self._app_id:
                        mention_text = getattr(entity, "text", "")
                        text = text.replace(mention_text, "").strip()

        # Approval-Antwort pruefen
        session_for_conv = self._sessions.get(conversation_id, "")
        if session_for_conv in self._pending_approvals:
            normalized = text.lower()
            if normalized in ("ja", "yes", "ok", "genehmigen", "approve"):
                await self._resolve_approval(session_for_conv, approved=True, turn_context=turn_context)
                return
            elif normalized in ("nein", "no", "ablehnen", "reject"):
                await self._resolve_approval(session_for_conv, approved=False, turn_context=turn_context)
                return

        # Session-Mapping
        session_id = self._get_or_create_session(conversation_id)

        incoming = IncomingMessage(
            channel="teams",
            user_id=user_id,
            text=text,
            session_id=session_id,
            metadata={
                "conversation_id": conversation_id,
                "activity_id": activity.id or "",
                "user_name": activity.from_property.name if activity.from_property else "",
                "channel_id": activity.channel_id or "",
            },
        )

        if self._handler:
            # Typing-Indicator senden
            try:
                await turn_context.send_activity(
                    _create_typing_activity()
                )
            except Exception:
                pass

            try:
                response = await self._handler(incoming)
                if response.session_id:
                    self._sessions[conversation_id] = response.session_id
                    if self._session_store:
                        self._session_store.save_channel_mapping(
                            "teams_sessions", conversation_id, response.session_id,
                        )

                # Antwort senden (mit Splitting)
                chunks = _split_message(response.text)
                for chunk in chunks:
                    await turn_context.send_activity(chunk)

            except Exception as exc:
                logger.error("Teams: Handler-Fehler: %s", exc)
                await turn_context.send_activity(
                    "Ein Fehler ist bei der Verarbeitung aufgetreten."
                )

    async def _on_invoke(self, turn_context: Any) -> None:
        """Verarbeitet Invoke-Activities (Adaptive Card Actions)."""
        from botbuilder.schema import Activity  # type: ignore[import-untyped]

        activity = turn_context.activity
        value = activity.value or {}

        if isinstance(value, dict):
            action_type = value.get("action", "")
            approval_id = value.get("approval_id", "")

            if action_type in ("approve", "reject") and approval_id:
                approved = action_type == "approve"
                async with self._approval_lock:
                    future = self._pending_approvals.get(approval_id)
                if future and not future.done():
                    future.set_result(approved)
                    status = "genehmigt" if approved else "abgelehnt"
                    await turn_context.send_activity(f"Aktion {status}.")

        # Invoke braucht eine Response
        invoke_response = Activity(
            type="invokeResponse",
            value={"status": 200, "body": {}},
        )
        await turn_context.send_activity(invoke_response)

    async def _on_conversation_update(self, turn_context: Any) -> None:
        """Begruessungsnachricht wenn Bot zum Chat hinzugefuegt wird."""
        activity = turn_context.activity
        if activity.members_added:
            for member in activity.members_added:
                if member.id != activity.recipient.id:
                    continue
                # Bot wurde hinzugefuegt
                await turn_context.send_activity(
                    "Jarvis ist bereit. Schreibe eine Nachricht um zu beginnen."
                )

    # -- Outbound: Nachrichten senden --------------------------------------------

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht proaktiv an eine Conversation."""
        if not self._running or not self._adapter:
            return

        conversation_id = self._conversation_for_session(message.session_id)
        if not conversation_id:
            conversation_id = message.metadata.get("conversation_id", "")
        if not conversation_id:
            logger.warning("Teams: Keine Conversation fuer Session %s", message.session_id[:8])
            return

        ref = self._conversation_refs.get(conversation_id)
        if not ref:
            logger.warning("Teams: Keine ConversationReference fuer %s", conversation_id[:16])
            return

        try:
            chunks = _split_message(message.text)

            async def _send_callback(turn_context: Any) -> None:
                for chunk in chunks:
                    await turn_context.send_activity(chunk)

            await self._adapter.continue_conversation(
                ref, _send_callback, self._app_id,
            )
        except Exception as exc:
            logger.error("Teams: Proaktives Senden fehlgeschlagen: %s", exc)

    # -- Approval-Workflow -------------------------------------------------------

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Sendet Approval-Anfrage als Adaptive Card."""
        conversation_id = self._conversation_for_session(session_id)
        if not conversation_id:
            return False

        ref = self._conversation_refs.get(conversation_id)
        if not ref or not self._adapter:
            return False

        approval_id = session_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._pending_approvals[approval_id] = future

        # Adaptive Card senden
        card = _build_approval_card(action, reason, approval_id)

        try:
            async def _send_card(turn_context: Any) -> None:
                from botbuilder.schema import (  # type: ignore[import-untyped]
                    Activity,
                    Attachment,
                )
                card_activity = Activity(
                    type="message",
                    attachments=[
                        Attachment(
                            content_type="application/vnd.microsoft.card.adaptive",
                            content=card,
                        )
                    ],
                )
                await turn_context.send_activity(card_activity)

            await self._adapter.continue_conversation(
                ref, _send_card, self._app_id,
            )
        except Exception as exc:
            logger.error("Teams: Approval-Card senden fehlgeschlagen: %s", exc)
            async with self._approval_lock:
                self._pending_approvals.pop(approval_id, None)
            return False

        try:
            return await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Teams: Approval-Timeout fuer Session %s", session_id[:8])
            return False
        finally:
            async with self._approval_lock:
                self._pending_approvals.pop(approval_id, None)

    async def _resolve_approval(
        self, session_id: str, *, approved: bool, turn_context: Any = None,
    ) -> None:
        """Loest ein Approval-Future auf."""
        async with self._approval_lock:
            future = self._pending_approvals.get(session_id)
        if future and not future.done():
            future.set_result(approved)
            if turn_context:
                status = "genehmigt" if approved else "abgelehnt"
                await turn_context.send_activity(f"Aktion {status}.")

    # -- Streaming ---------------------------------------------------------------

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Buffert Streaming-Tokens und sendet gebuendelt."""
        buf = self._stream_buffers.setdefault(session_id, [])
        buf.append(token)
        if len(buf) == 1:
            await asyncio.sleep(0.5)
            text = "".join(self._stream_buffers.pop(session_id, []))
            if text.strip():
                await self.send(
                    OutgoingMessage(channel=self.name, text=text, session_id=session_id)
                )

    # -- Hilfsmethoden -----------------------------------------------------------

    def _get_or_create_session(self, conversation_id: str) -> str:
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = uuid.uuid4().hex
        return self._sessions[conversation_id]

    def _conversation_for_session(self, session_id: str) -> str | None:
        """Findet die Conversation-ID fuer eine Session."""
        for conv_id, sid in self._sessions.items():
            if sid == session_id:
                return conv_id
        return None


def _split_message(text: str) -> list[str]:
    """Teilt Nachrichten bei MAX_MESSAGE_LENGTH Zeichen."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1 or split_pos < MAX_MESSAGE_LENGTH // 2:
            split_pos = text.rfind(" ", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1:
            split_pos = MAX_MESSAGE_LENGTH
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    return chunks


def _create_typing_activity() -> dict[str, Any]:
    """Erstellt eine Typing-Activity fuer Teams."""
    return {"type": "typing"}


def _build_approval_card(
    action: PlannedAction,
    reason: str,
    approval_id: str,
) -> dict[str, Any]:
    """Erstellt eine Adaptive Card fuer den Approval-Workflow."""
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "Genehmigung erforderlich",
                "weight": "Bolder",
                "size": "Medium",
                "color": "Warning",
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Tool:", "value": action.tool},
                    {"title": "Grund:", "value": reason},
                    {"title": "Parameter:", "value": str(action.params)[:300]},
                ],
            },
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "Genehmigen",
                "style": "positive",
                "data": {
                    "action": "approve",
                    "approval_id": approval_id,
                },
            },
            {
                "type": "Action.Submit",
                "title": "Ablehnen",
                "style": "destructive",
                "data": {
                    "action": "reject",
                    "approval_id": approval_id,
                },
            },
        ],
    }
