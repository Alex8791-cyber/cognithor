"""WhatsApp Cloud API Channel -- Meta Graph API v21.0.

Vollstaendige WhatsApp Business Integration:
  - Text senden/empfangen
  - Voice -> faster-whisper Transkription
  - Foto/Dokument-Download (2-Step)
  - Interactive Buttons fuer Approval-Workflows
  - Session->Phone-Number Mapping
  - Allowed-Numbers Whitelist
  - Message-Splitting bei 4096 Zeichen

Dependencies: aiohttp (im Core enthalten), httpx
Architektur-Bibel: SS9 (Channels)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import secrets
import uuid
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import httpx

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.security.token_store import get_token_store

if TYPE_CHECKING:
    from jarvis.gateway.session_store import SessionStore

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"
MAX_TEXT_LENGTH = 4096


class WhatsAppChannel(Channel):
    """WhatsApp Business Cloud API Channel.

    Nutzt die offizielle Meta Graph API v21.0 fuer bidirektionale
    Kommunikation. Eingehende Nachrichten werden ueber einen
    Webhook-Server empfangen, ausgehende ueber HTTP POST gesendet.
    """

    def __init__(
        self,
        *,
        api_token: str,
        phone_number_id: str,
        verify_token: str = "",
        app_secret: str = "",
        webhook_host: str = "127.0.0.1",
        webhook_port: int = 8443,
        allowed_numbers: list[str] | None = None,
        ssl_certfile: str = "",
        ssl_keyfile: str = "",
        session_store: SessionStore | None = None,
    ) -> None:
        self._token_store = get_token_store()
        self._token_store.store("whatsapp_api_token", api_token)
        if app_secret:
            self._token_store.store("whatsapp_app_secret", app_secret)
        self._has_app_secret = bool(app_secret)
        self._phone_number_id = phone_number_id
        self._verify_token = verify_token or secrets.token_urlsafe(16)
        self._webhook_host = webhook_host
        self._webhook_port = webhook_port
        self._allowed_numbers = set(allowed_numbers or [])
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._session_store = session_store

        self._handler: MessageHandler | None = None
        self._running = False
        self._http: httpx.AsyncClient | None = None
        self._webhook_runner: Any | None = None
        self._webhook_site: Any | None = None

        # Session mapping: phone_number -> session_id
        self._sessions: dict[str, str] = {}
        # Pending approvals: session_id -> asyncio.Future
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}

        # Optional voice transcription
        self._whisper = None
        try:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel("base", compute_type="int8")
            logger.info("WhatsApp: faster-whisper geladen fuer Voice-Transkription")
        except ImportError:
            logger.debug("WhatsApp: faster-whisper nicht verfuegbar, Voice wird als Text-Hinweis weitergeleitet")

    @property
    def _api_token(self) -> str:
        """API-Token (entschlüsselt bei Zugriff)."""
        return self._token_store.retrieve("whatsapp_api_token")

    @property
    def _app_secret(self) -> str:
        """App-Secret (entschlüsselt bei Zugriff)."""
        if self._has_app_secret:
            return self._token_store.retrieve("whatsapp_app_secret")
        return ""

    @property
    def name(self) -> str:
        return "whatsapp"

    # -- Outbound: Nachrichten senden ------------------------------------------

    async def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=GRAPH_API_BASE,
                headers={
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._http

    async def _send_text(self, to: str, text: str) -> None:
        """Sendet eine Textnachricht (mit Splitting bei >4096 Zeichen)."""
        client = await self._ensure_http()
        chunks = self._split_message(text)

        for chunk in chunks:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"body": chunk},
            }
            try:
                resp = await client.post(
                    f"/{self._phone_number_id}/messages",
                    json=payload,
                )
                resp.raise_for_status()
                logger.debug("WhatsApp: Nachricht an %s gesendet", to)
            except httpx.HTTPError as e:
                logger.error("WhatsApp: Senden fehlgeschlagen: %s", e)

    async def _send_interactive_buttons(
        self,
        to: str,
        body_text: str,
        buttons: list[dict[str, str]],
    ) -> None:
        """Sendet Interactive Buttons (max 3)."""
        client = await self._ensure_http()
        button_list = []
        for i, btn in enumerate(buttons[:3]):
            button_list.append({
                "type": "reply",
                "reply": {
                    "id": btn.get("id", f"btn_{i}"),
                    "title": btn.get("title", f"Option {i+1}")[:20],
                },
            })

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body_text[:1024]},
                "action": {"buttons": button_list},
            },
        }
        try:
            resp = await client.post(
                f"/{self._phone_number_id}/messages",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("WhatsApp: Button-Senden fehlgeschlagen: %s", e)

    # -- Media Download (2-Step) -----------------------------------------------

    async def _download_media(self, media_id: str) -> bytes | None:
        """Laedt eine Mediendatei herunter (2-Step: URL holen, dann Download)."""
        client = await self._ensure_http()
        try:
            # Step 1: Media-URL holen
            resp = await client.get(f"/{media_id}")
            resp.raise_for_status()
            media_url = resp.json().get("url")
            if not media_url:
                return None

            # Step 2: Datei herunterladen
            dl_resp = await client.get(
                media_url,
                headers={"Authorization": f"Bearer {self._api_token}"},
            )
            dl_resp.raise_for_status()
            return dl_resp.content
        except httpx.HTTPError as e:
            logger.error("WhatsApp: Media-Download fehlgeschlagen: %s", e)
            return None

    async def _transcribe_voice(self, audio_data: bytes) -> str:
        """Transkribiert Voice-Nachricht mit faster-whisper."""
        if self._whisper is None:
            return "[Voice-Nachricht empfangen -- Transkription nicht verfuegbar]"

        try:
            segments, _ = self._whisper.transcribe(io.BytesIO(audio_data), language="de")
            text = " ".join(seg.text for seg in segments)
            return text.strip() or "[Leere Voice-Nachricht]"
        except Exception as e:
            logger.error("WhatsApp: Transkription fehlgeschlagen: %s", e)
            return f"[Voice-Transkription fehlgeschlagen: {e}]"

    # -- Inbound: Webhook-Server -----------------------------------------------

    async def _setup_webhook(self) -> None:
        """Startet den aiohttp Webhook-Server."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("WhatsApp: aiohttp nicht installiert -- Webhook nicht verfuegbar")
            return

        app = web.Application()
        app.router.add_get("/webhook", self._handle_verification)
        app.router.add_post("/webhook", self._handle_incoming)

        runner = web.AppRunner(app)
        await runner.setup()

        # TLS-Support
        ssl_ctx = None
        if self._ssl_certfile and self._ssl_keyfile:
            from jarvis.security.token_store import create_ssl_context
            ssl_ctx = create_ssl_context(self._ssl_certfile, self._ssl_keyfile)

        if not ssl_ctx and self._webhook_host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning("WARNUNG: WhatsApp-Webhook auf %s ohne TLS gestartet!", self._webhook_host)

        site = web.TCPSite(runner, self._webhook_host, self._webhook_port, ssl_context=ssl_ctx)
        await site.start()

        self._webhook_runner = runner
        self._webhook_site = site
        logger.info(
            "WhatsApp: Webhook-Server gestartet auf Port %d (TLS=%s)",
            self._webhook_port,
            ssl_ctx is not None,
        )

    async def _handle_verification(self, request: Any) -> Any:
        """GET /webhook -- Meta Verification Challenge."""
        from aiohttp import web

        mode = request.query.get("hub.mode")
        token = request.query.get("hub.verify_token")
        challenge = request.query.get("hub.challenge")

        if mode == "subscribe" and token == self._verify_token:
            logger.info("WhatsApp: Webhook-Verifizierung erfolgreich")
            return web.Response(text=challenge or "", content_type="text/plain")

        logger.warning("WhatsApp: Webhook-Verifizierung fehlgeschlagen")
        return web.Response(status=403, text="Forbidden")

    async def _handle_incoming(self, request: Any) -> Any:
        """POST /webhook -- Eingehende Nachrichten verarbeiten.

        Verifies the X-Hub-Signature-256 HMAC header before processing.
        Requests with missing or invalid signatures are rejected with 403.
        """
        from aiohttp import web

        # -- HMAC-SHA256 signature verification --------------------------------
        raw_body = await request.read()
        signature_header = request.headers.get("X-Hub-Signature-256", "")

        if not self._verify_signature(raw_body, signature_header):
            logger.warning("WhatsApp: Ungueltige Webhook-Signatur abgelehnt")
            return web.Response(status=403, text="Invalid signature")

        try:
            body = json.loads(raw_body)
        except Exception:
            return web.Response(status=400)

        # Meta sendet immer 200 zurueck, sonst Retry-Storm
        asyncio.create_task(self._process_webhook_payload(body))
        return web.Response(status=200, text="OK")

    def _verify_signature(self, payload: bytes, signature_header: str) -> bool:
        """Verify the X-Hub-Signature-256 header using HMAC-SHA256.

        The header value has the form ``sha256=<hex-digest>``.
        Uses ``hmac.compare_digest`` for constant-time comparison.
        """
        if not signature_header:
            return False

        if not signature_header.startswith("sha256="):
            return False

        expected_sig = signature_header[len("sha256="):]
        # Meta signs webhooks with the App Secret, not the API token
        hmac_key = self._app_secret or self._api_token
        computed = hmac.new(
            hmac_key.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, expected_sig)

    async def _process_webhook_payload(self, body: dict[str, Any]) -> None:
        """Verarbeitet den Webhook-Payload asynchron."""
        try:
            entries = body.get("entry", [])
            for entry in entries:
                changes = entry.get("changes", [])
                for change in changes:
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    for msg in messages:
                        await self._process_message(msg, value)
        except Exception as e:
            logger.error("WhatsApp: Payload-Verarbeitung fehlgeschlagen: %s", e)

    async def _process_message(self, msg: dict[str, Any], value: dict[str, Any]) -> None:
        """Verarbeitet eine einzelne eingehende Nachricht."""
        from_number = msg.get("from", "")
        msg_type = msg.get("type", "")
        msg_id = msg.get("id", uuid.uuid4().hex)

        # Whitelist pruefen
        if self._allowed_numbers and from_number not in self._allowed_numbers:
            logger.warning("WhatsApp: Nachricht von nicht-erlaubter Nummer %s", from_number)
            return

        # Text extrahieren
        text = ""
        attachments: list[str] = []

        if msg_type == "text":
            text = msg.get("text", {}).get("body", "")

        elif msg_type == "interactive":
            # Button-Antwort
            interactive = msg.get("interactive", {})
            if interactive.get("type") == "button_reply":
                button_id = interactive.get("button_reply", {}).get("id", "")
                button_title = interactive.get("button_reply", {}).get("title", "")
                text = button_title
                # Approval-Workflow
                await self._handle_button_response(from_number, button_id)
                return

        elif msg_type == "audio":
            # Voice-Nachricht
            media_id = msg.get("audio", {}).get("id", "")
            if media_id:
                audio_data = await self._download_media(media_id)
                if audio_data:
                    text = await self._transcribe_voice(audio_data)
                else:
                    text = "[Voice-Nachricht konnte nicht heruntergeladen werden]"

        elif msg_type == "image":
            media_id = msg.get("image", {}).get("id", "")
            caption = msg.get("image", {}).get("caption", "")
            text = caption or "[Bild empfangen]"
            if media_id:
                attachments.append(f"whatsapp://media/{media_id}")

        elif msg_type == "document":
            media_id = msg.get("document", {}).get("id", "")
            filename = msg.get("document", {}).get("filename", "document")
            caption = msg.get("document", {}).get("caption", "")
            text = caption or f"[Dokument empfangen: {filename}]"
            if media_id:
                attachments.append(f"whatsapp://media/{media_id}")

        else:
            text = f"[Unterstuetzter Nachrichtentyp: {msg_type}]"

        if not text.strip():
            return

        # Session-Mapping
        session_id = self._get_or_create_session(from_number)

        # IncomingMessage erstellen
        incoming = IncomingMessage(
            channel="whatsapp",
            user_id=from_number,
            text=text,
            session_id=session_id,
            attachments=attachments,
            metadata={
                "whatsapp_msg_id": msg_id,
                "phone_number": from_number,
                "msg_type": msg_type,
            },
        )

        # An Handler weiterleiten
        if self._handler:
            try:
                response = await self._handler(incoming)
                await self._send_text(from_number, response.text)
            except Exception as e:
                logger.error("WhatsApp: Handler-Fehler: %s", e)
                await self._send_text(from_number, "Ein Fehler ist bei der Verarbeitung aufgetreten.")

    async def _handle_button_response(self, from_number: str, button_id: str) -> None:
        """Verarbeitet Button-Antworten fuer Approval-Workflows."""
        session_id = self._sessions.get(from_number, "")
        future = self._pending_approvals.pop(session_id, None)
        if future and not future.done():
            approved = button_id.startswith("approve")
            future.set_result(approved)
            logger.info(
                "WhatsApp: Approval %s von %s",
                "genehmigt" if approved else "abgelehnt",
                from_number,
            )

    def _get_or_create_session(self, phone_number: str) -> str:
        if phone_number not in self._sessions:
            self._sessions[phone_number] = uuid.uuid4().hex
            if self._session_store:
                self._session_store.save_channel_mapping(
                    "whatsapp_sessions", phone_number, self._sessions[phone_number],
                )
        return self._sessions[phone_number]

    # -- Channel Interface -----------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._running = True

        # Persistierte Mappings laden
        if self._session_store:
            for key, val in self._session_store.load_all_channel_mappings("whatsapp_sessions").items():
                self._sessions[key] = val

        await self._setup_webhook()
        logger.info("WhatsAppChannel gestartet (Cloud API v21.0)")

    async def stop(self) -> None:
        self._running = False
        if self._webhook_runner:
            await self._webhook_runner.cleanup()
        if self._http:
            await self._http.aclose()
        # Cancel pending approvals
        for future in self._pending_approvals.values():
            if not future.done():
                future.set_result(False)
        self._pending_approvals.clear()
        logger.info("WhatsAppChannel gestoppt")

    async def send(self, message: OutgoingMessage) -> None:
        if not self._running:
            return
        # Find phone number from session
        phone = self._phone_for_session(message.session_id)
        if phone:
            await self._send_text(phone, message.text)
        else:
            logger.warning("WhatsApp: Keine Telefonnummer fuer Session %s", message.session_id[:8])

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Sendet Approval-Anfrage mit Interactive Buttons."""
        phone = self._phone_for_session(session_id)
        if not phone:
            return False

        body_text = (
            f"Aktion benoetigt Genehmigung:\n\n"
            f"Tool: {action.tool}\n"
            f"Grund: {reason}\n\n"
            f"Genehmigen oder Ablehnen?"
        )

        buttons = [
            {"id": "approve_yes", "title": "Genehmigen"},
            {"id": "reject_no", "title": "Ablehnen"},
        ]

        # Future fuer Antwort erstellen
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending_approvals[session_id] = future

        await self._send_interactive_buttons(phone, body_text, buttons)

        try:
            return await asyncio.wait_for(future, timeout=300.0)
        except TimeoutError:
            self._pending_approvals.pop(session_id, None)
            await self._send_text(phone, "Genehmigung abgelaufen (Timeout).")
            return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        # WhatsApp unterstuetzt kein Token-Streaming
        pass

    # -- Hilfsmethoden ---------------------------------------------------------

    def _phone_for_session(self, session_id: str) -> str | None:
        """Findet die Telefonnummer fuer eine Session-ID."""
        for phone, sid in self._sessions.items():
            if sid == session_id:
                return phone
        return None

    @staticmethod
    def _split_message(text: str) -> list[str]:
        """Teilt Nachrichten bei MAX_TEXT_LENGTH Zeichen."""
        if len(text) <= MAX_TEXT_LENGTH:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= MAX_TEXT_LENGTH:
                chunks.append(text)
                break
            # Am letzten Newline oder Leerzeichen vor dem Limit splitten
            split_pos = text.rfind("\n", 0, MAX_TEXT_LENGTH)
            if split_pos == -1:
                split_pos = text.rfind(" ", 0, MAX_TEXT_LENGTH)
            if split_pos == -1:
                split_pos = MAX_TEXT_LENGTH
            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip()
        return chunks
