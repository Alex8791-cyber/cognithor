"""Signal-Channel: Bidirektionale Kommunikation ueber Signal Messenger.

Nutzt signal-cli-rest-api als HTTP-Bridge:
  https://github.com/bbernhard/signal-cli-rest-api

Features:
  - Text senden/empfangen via REST + Webhook
  - Gruppen-Support (optional)
  - Attachment-Download und -Versand
  - Voice-Transkription via faster-whisper
  - Approval-Workflow via Reply-Nachrichten
  - Allowed-Numbers Whitelist
  - Message-Splitting bei 2000 Zeichen
  - Session->Number Mapping
  - Graceful Shutdown

Konfiguration:
  - JARVIS_SIGNAL_API_URL: signal-cli-rest-api Basis-URL (z.B. http://localhost:8080)
  - JARVIS_SIGNAL_PHONE: Registrierte Telefonnummer (+49...)
  - JARVIS_SIGNAL_ALLOWED_NUMBERS: Komma-getrennte Whitelist

Abhaengigkeiten:
  - httpx (im Core enthalten)
  - aiohttp (fuer Webhook-Server)
  - Optional: faster-whisper (Voice-Transkription)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2000
APPROVAL_TIMEOUT = 300  # 5 Minuten


class SignalChannel(Channel):
    """Signal Messenger als Kommunikationskanal.

    Kommuniziert ueber signal-cli-rest-api (HTTP-Bridge).
    Eingehende Nachrichten werden entweder per Webhook oder Polling empfangen,
    ausgehende per REST-API gesendet.
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8080",
        phone_number: str = "",
        allowed_numbers: list[str] | None = None,
        webhook_host: str = "127.0.0.1",
        webhook_port: int = 8090,
        use_polling: bool = False,
        polling_interval: float = 1.0,
        workspace_dir: Path | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._phone_number = phone_number
        self._allowed_numbers = set(allowed_numbers or [])
        self._webhook_host = webhook_host
        self._webhook_port = webhook_port
        self._use_polling = use_polling
        self._polling_interval = polling_interval
        self._workspace_dir = workspace_dir or Path.home() / ".jarvis" / "workspace" / "signal"

        self._handler: MessageHandler | None = None
        self._running = False
        self._http: httpx.AsyncClient | None = None

        # Webhook-Server (aiohttp)
        self._webhook_runner: Any | None = None
        self._webhook_site: Any | None = None

        # Polling-Task
        self._poll_task: asyncio.Task[None] | None = None

        # Session-Mapping: phone_number -> session_id
        self._sessions: dict[str, str] = {}

        # Approval-Workflow
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()

        # Voice-Transkription
        self._whisper: Any | None = None

        # Streaming-Buffer
        self._stream_buffers: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "signal"

    # -- Lifecycle ---------------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Signal-Channel mit Webhook oder Polling."""
        self._handler = handler
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        self._http = httpx.AsyncClient(
            base_url=self._api_url,
            timeout=30.0,
        )

        # Whisper fuer Voice-Transkription laden
        try:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel("base", device="auto", compute_type="int8")
            logger.info("Signal: faster-whisper geladen fuer Voice-Transkription")
        except ImportError:
            logger.debug("Signal: faster-whisper nicht verfuegbar")

        # Verbindung testen
        try:
            resp = await self._http.get("/v1/about")
            if resp.status_code == 200:
                info = resp.json()
                logger.info(
                    "Signal: Verbunden mit signal-cli-rest-api v%s",
                    info.get("versions", {}).get("signal-cli", "?"),
                )
        except Exception as exc:
            logger.warning("Signal: API nicht erreichbar (%s) -- starte trotzdem", exc)

        # Eingehende Nachrichten: Webhook oder Polling
        if self._use_polling:
            self._poll_task = asyncio.create_task(self._polling_loop())
            logger.info("Signal: Polling-Modus gestartet (Intervall %.1fs)", self._polling_interval)
        else:
            await self._setup_webhook()

        self._running = True
        phone_status = "konfiguriert" if self._phone_number else "nicht-konfiguriert"
        logger.info(
            "SignalChannel gestartet (Nummer=%s, Modus=%s)",
            phone_status,
            "polling" if self._use_polling else "webhook",
        )

    async def stop(self) -> None:
        """Stoppt den Signal-Channel sauber."""
        self._running = False

        # Polling stoppen
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        # Webhook stoppen
        if self._webhook_runner:
            await self._webhook_runner.cleanup()
            self._webhook_runner = None
            self._webhook_site = None

        # Pending approvals abbrechen
        async with self._approval_lock:
            for future in self._pending_approvals.values():
                if not future.done():
                    future.set_result(False)
            self._pending_approvals.clear()

        # HTTP-Client schliessen
        if self._http:
            await self._http.aclose()
            self._http = None

        logger.info("SignalChannel gestoppt")

    # -- Outbound: Nachrichten senden --------------------------------------------

    async def _send_text(self, recipient: str, text: str) -> None:
        """Sendet eine Textnachricht an eine Nummer (mit Splitting)."""
        if not self._http:
            return

        chunks = _split_message(text)
        for chunk in chunks:
            payload = {
                "message": chunk,
                "number": self._phone_number,
                "recipients": [recipient],
            }
            try:
                resp = await self._http.post("/v2/send", json=payload)
                if resp.status_code not in (200, 201):
                    logger.error(
                        "Signal: Senden fehlgeschlagen (HTTP %d): %s",
                        resp.status_code,
                        resp.text[:200],
                    )
            except httpx.HTTPError as exc:
                logger.error("Signal: Senden fehlgeschlagen: %s", exc)

    async def _send_attachment(
        self, recipient: str, file_path: Path, message: str = "",
    ) -> None:
        """Sendet eine Datei als Attachment."""
        if not self._http:
            return

        try:
            with open(file_path, "rb") as fh:
                files = {"attachment": (file_path.name, fh)}
                data = {
                    "number": self._phone_number,
                    "recipients": json.dumps([recipient]),
                    "message": message,
                }
                resp = await self._http.post("/v2/send", data=data, files=files)
                if resp.status_code not in (200, 201):
                    logger.error("Signal: Attachment-Senden fehlgeschlagen: %s", resp.text[:200])
        except Exception as exc:
            logger.error("Signal: Attachment-Senden fehlgeschlagen: %s", exc)

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an den User."""
        if not self._running:
            return

        phone = self._phone_for_session(message.session_id)
        if not phone:
            # Fallback: Metadata pruefen
            phone = message.metadata.get("phone_number", "")
        if not phone:
            logger.warning("Signal: Keine Telefonnummer fuer Session %s", message.session_id[:8])
            return

        await self._send_text(phone, message.text)

    # -- Inbound: Webhook --------------------------------------------------------

    async def _setup_webhook(self) -> None:
        """Startet den aiohttp Webhook-Server fuer eingehende Signal-Nachrichten."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error(
                "Signal: aiohttp nicht installiert -- Webhook nicht verfuegbar. "
                "Nutze use_polling=True oder pip install aiohttp"
            )
            self._use_polling = True
            self._poll_task = asyncio.create_task(self._polling_loop())
            return

        app = web.Application()
        app.router.add_post("/signal/webhook", self._handle_webhook)
        app.router.add_get("/signal/health", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._webhook_host, self._webhook_port)
        await site.start()

        self._webhook_runner = runner
        self._webhook_site = site

        # Webhook bei signal-cli-rest-api registrieren
        if self._http:
            try:
                webhook_url = f"http://{self._webhook_host}:{self._webhook_port}/signal/webhook"
                await self._http.post(
                    f"/v1/receive/{self._phone_number}",
                    json={"webhook": webhook_url},
                )
            except Exception:
                logger.debug("Signal: Webhook-Registrierung uebersprungen (nicht fatal)")

        logger.info(
            "Signal: Webhook-Server gestartet auf %s:%d",
            self._webhook_host,
            self._webhook_port,
        )

    async def _handle_health(self, request: Any) -> Any:
        """GET /signal/health -- Healthcheck."""
        from aiohttp import web
        return web.json_response({"status": "ok", "channel": "signal"})

    async def _handle_webhook(self, request: Any) -> Any:
        """POST /signal/webhook -- Eingehende Signal-Nachrichten."""
        from aiohttp import web

        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON")

        asyncio.create_task(self._process_webhook_payload(body))
        return web.Response(status=200, text="OK")

    # -- Inbound: Polling --------------------------------------------------------

    async def _polling_loop(self) -> None:
        """Empfaengt Nachrichten per Polling (Fallback wenn kein Webhook)."""
        while self._running:
            try:
                if self._http:
                    resp = await self._http.get(
                        f"/v1/receive/{self._phone_number}",
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        messages = resp.json()
                        if isinstance(messages, list):
                            for msg in messages:
                                await self._process_webhook_payload(msg)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Signal: Polling-Fehler: %s", exc)

            try:
                await asyncio.sleep(self._polling_interval)
            except asyncio.CancelledError:
                break

    # -- Nachrichtenverarbeitung -------------------------------------------------

    async def _process_webhook_payload(self, payload: dict[str, Any]) -> None:
        """Verarbeitet einen eingehenden Signal-Payload."""
        try:
            envelope = payload.get("envelope", payload)
            source = envelope.get("sourceNumber") or envelope.get("source", "")
            source_name = envelope.get("sourceName", "")

            if not source:
                return

            # Whitelist pruefen
            if self._allowed_numbers and source not in self._allowed_numbers:
                logger.warning("Signal: Nachricht von nicht-erlaubter Nummer %s", source)
                return

            # Verschiedene Nachrichtentypen
            data_msg = envelope.get("dataMessage", {})
            if not data_msg:
                return

            msg_text = data_msg.get("message", "")
            timestamp = data_msg.get("timestamp", 0)
            attachments = data_msg.get("attachments", [])
            group_info = data_msg.get("groupInfo", {})
            quote = data_msg.get("quote", {})

            # Attachment-Handling
            attachment_refs: list[str] = []
            transcribed_voice = ""
            for att in attachments:
                content_type = att.get("contentType", "")
                att_id = att.get("id", "")
                filename = att.get("filename", "")

                if content_type.startswith("audio/") and self._whisper:
                    # Voice-Transkription
                    audio_data = await self._download_attachment(att_id)
                    if audio_data:
                        transcribed_voice = await self._transcribe_audio(audio_data)
                elif att_id:
                    attachment_refs.append(f"signal://attachment/{att_id}/{filename}")

            # Text zusammenbauen
            text = msg_text
            if transcribed_voice:
                text = transcribed_voice if not text else f"{text}\n[Voice: {transcribed_voice}]"
            if not text.strip():
                if attachment_refs:
                    text = f"[{len(attachment_refs)} Attachment(s) empfangen]"
                else:
                    return

            # Quote/Reply -> evtl. Approval-Antwort
            if quote:
                quote_text = quote.get("text", "")
                if "Genehmigung erforderlich" in quote_text:
                    await self._handle_approval_reply(source, text)
                    return

            # Session-Mapping
            session_id = self._get_or_create_session(source)

            incoming = IncomingMessage(
                channel="signal",
                user_id=source,
                text=text,
                session_id=session_id,
                attachments=attachment_refs,
                metadata={
                    "phone_number": source,
                    "source_name": source_name,
                    "timestamp": str(timestamp),
                    "group_id": group_info.get("groupId", ""),
                },
            )

            if self._handler:
                try:
                    response = await self._handler(incoming)
                    if response.session_id:
                        self._sessions[source] = response.session_id
                    await self._send_text(source, response.text)
                except Exception as exc:
                    logger.error("Signal: Handler-Fehler: %s", exc)
                    await self._send_text(source, "Ein Fehler ist bei der Verarbeitung aufgetreten.")

        except Exception as exc:
            logger.error("Signal: Payload-Verarbeitung fehlgeschlagen: %s", exc)

    async def _download_attachment(self, attachment_id: str) -> bytes | None:
        """Laedt ein Attachment von der signal-cli-rest-api herunter."""
        if not self._http:
            return None
        try:
            resp = await self._http.get(f"/v1/attachments/{attachment_id}")
            if resp.status_code == 200:
                return resp.content
        except Exception as exc:
            logger.error("Signal: Attachment-Download fehlgeschlagen: %s", exc)
        return None

    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transkribiert Audio-Daten mit faster-whisper."""
        if not self._whisper:
            return "[Voice-Nachricht -- Transkription nicht verfuegbar]"
        try:
            segments, _ = self._whisper.transcribe(io.BytesIO(audio_data), language="de")
            text = " ".join(seg.text.strip() for seg in segments)
            return text.strip() or "[Leere Voice-Nachricht]"
        except Exception as exc:
            logger.error("Signal: Transkription fehlgeschlagen: %s", exc)
            return f"[Voice-Transkription fehlgeschlagen: {exc}]"

    # -- Approval-Workflow -------------------------------------------------------

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Sendet Approval-Anfrage und wartet auf Antwort per Reply."""
        phone = self._phone_for_session(session_id)
        if not phone:
            return False

        text = (
            f"Genehmigung erforderlich\n\n"
            f"Tool: {action.tool}\n"
            f"Grund: {reason}\n"
            f"Parameter: {str(action.params)[:300]}\n\n"
            f"Antworte mit 'ja' oder 'nein' (Reply auf diese Nachricht)"
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._pending_approvals[session_id] = future

        await self._send_text(phone, text)

        try:
            return await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Signal: Approval-Timeout fuer Session %s", session_id[:8])
            await self._send_text(phone, "Genehmigung abgelaufen (Timeout).")
            return False
        finally:
            async with self._approval_lock:
                self._pending_approvals.pop(session_id, None)

    async def _handle_approval_reply(self, source: str, reply_text: str) -> None:
        """Verarbeitet eine Approval-Antwort."""
        session_id = self._sessions.get(source, "")
        async with self._approval_lock:
            future = self._pending_approvals.get(session_id)

        if not future or future.done():
            return

        normalized = reply_text.strip().lower()
        approved = normalized in ("ja", "yes", "ok", "genehmigen", "approve", "1")
        future.set_result(approved)

        status = "genehmigt" if approved else "abgelehnt"
        logger.info("Signal: Approval %s von %s", status, source)
        await self._send_text(source, f"Aktion {status}.")

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

    def _get_or_create_session(self, phone_number: str) -> str:
        if phone_number not in self._sessions:
            self._sessions[phone_number] = uuid.uuid4().hex
        return self._sessions[phone_number]

    def _phone_for_session(self, session_id: str) -> str | None:
        """Findet die Telefonnummer fuer eine Session-ID."""
        for phone, sid in self._sessions.items():
            if sid == session_id:
                return phone
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
