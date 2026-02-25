"""iMessage-Channel: Bidirektionale Kommunikation ueber Apple iMessage.

Nutzt zwei Ansaetze je nach Plattform:
  1. macOS nativ: AppleScript-Bridge fuer Senden, Messages.app SQLite-DB fuer Empfang
  2. Cross-platform: BlueBubbles-Server HTTP-API (selbst-gehostet auf Mac)
     https://github.com/BlueBubblesApp/bluebubbles-server

Features:
  - Text senden/empfangen
  - Attachment-Versand (Bilder, Dateien)
  - Polling-basierter Empfang (macOS: SQLite, BlueBubbles: REST)
  - Session->Handle Mapping
  - Reply-basierte Approvals
  - Message-Splitting bei 2000 Zeichen
  - Graceful Shutdown

Konfiguration (macOS nativ):
  - JARVIS_IMESSAGE_ALLOWED_HANDLES: Komma-getrennte Handle-Whitelist (+49..., email@...)

Konfiguration (BlueBubbles):
  - JARVIS_IMESSAGE_BB_URL: BlueBubbles Server URL (z.B. http://localhost:1234)
  - JARVIS_IMESSAGE_BB_PASSWORD: BlueBubbles Server-Passwort
  - JARVIS_IMESSAGE_ALLOWED_HANDLES: Komma-getrennte Handle-Whitelist

Abhaengigkeiten:
  - macOS: Keine (nutzt osascript + sqlite3)
  - BlueBubbles: httpx
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2000
APPROVAL_TIMEOUT = 300
POLL_INTERVAL = 2.0

# macOS Messages SQLite-DB Pfad
MESSAGES_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"


class IMessageChannel(Channel):
    """Apple iMessage als bidirektionaler Kommunikationskanal.

    Unterstuetzt zwei Modi:
    - 'native': Nutzt macOS AppleScript + SQLite (nur macOS)
    - 'bluebubbles': Nutzt BlueBubbles HTTP-API (macOS-Server noetig)
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        allowed_handles: list[str] | None = None,
        bb_url: str = "",
        bb_password: str = "",
        polling_interval: float = POLL_INTERVAL,
    ) -> None:
        # Auto-detect: Native wenn macOS, sonst BlueBubbles
        if mode == "auto":
            self._mode = "native" if sys.platform == "darwin" else "bluebubbles"
        else:
            self._mode = mode

        self._allowed_handles = set(allowed_handles or [])
        self._bb_url = bb_url.rstrip("/") if bb_url else ""
        self._bb_password = bb_password
        self._polling_interval = polling_interval

        self._handler: MessageHandler | None = None
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._http: Any | None = None  # httpx.AsyncClient fuer BlueBubbles

        # Session-Mapping: handle (phone/email) -> session_id
        self._sessions: dict[str, str] = {}

        # Approval-Workflow
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()

        # Streaming-Buffer
        self._stream_buffers: dict[str, list[str]] = {}

        # Letzte bekannte Message-ID (fuer Polling)
        self._last_message_rowid: int = 0
        self._last_bb_timestamp: int = 0

    @property
    def name(self) -> str:
        return "imessage"

    # -- Lifecycle ---------------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Startet den iMessage-Channel."""
        self._handler = handler

        if self._mode == "native":
            if sys.platform != "darwin":
                logger.error(
                    "iMessage: Native-Modus nur auf macOS verfuegbar. "
                    "Nutze mode='bluebubbles' mit einem BlueBubbles-Server."
                )
                return

            # Messages.app DB pruefen
            if not MESSAGES_DB_PATH.exists():
                logger.error(
                    "iMessage: Messages-Datenbank nicht gefunden: %s",
                    MESSAGES_DB_PATH,
                )
                return

            # Letzte Message-ID ermitteln
            self._last_message_rowid = await self._get_latest_rowid()
            logger.info("iMessage: Native-Modus (letzte Message-ID: %d)", self._last_message_rowid)

        elif self._mode == "bluebubbles":
            if not self._bb_url:
                logger.error("iMessage: BlueBubbles-URL nicht konfiguriert")
                return

            try:
                import httpx
                self._http = httpx.AsyncClient(
                    base_url=self._bb_url,
                    params={"password": self._bb_password},
                    timeout=30.0,
                )

                # Verbindung testen
                resp = await self._http.get("/api/v1/server/info")
                if resp.status_code == 200:
                    info = resp.json()
                    logger.info(
                        "iMessage: BlueBubbles verbunden (v%s)",
                        info.get("data", {}).get("server_version", "?"),
                    )
                else:
                    logger.warning("iMessage: BlueBubbles nicht erreichbar (HTTP %d)", resp.status_code)
            except ImportError:
                logger.error("iMessage: httpx nicht installiert fuer BlueBubbles-Modus")
                return
            except Exception as exc:
                logger.warning("iMessage: BlueBubbles-Verbindung fehlgeschlagen: %s", exc)

        # Polling starten
        self._running = True
        self._poll_task = asyncio.create_task(self._polling_loop())
        logger.info(
            "IMessageChannel gestartet (Modus=%s, Intervall=%.1fs)",
            self._mode,
            self._polling_interval,
        )

    async def stop(self) -> None:
        """Stoppt den iMessage-Channel sauber."""
        self._running = False

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        # Pending approvals abbrechen
        async with self._approval_lock:
            for future in self._pending_approvals.values():
                if not future.done():
                    future.set_result(False)
            self._pending_approvals.clear()

        if self._http:
            await self._http.aclose()
            self._http = None

        logger.info("IMessageChannel gestoppt")

    # -- Polling: Neue Nachrichten empfangen ------------------------------------

    async def _polling_loop(self) -> None:
        """Pollt nach neuen Nachrichten (Native oder BlueBubbles)."""
        while self._running:
            try:
                if self._mode == "native":
                    await self._poll_native()
                elif self._mode == "bluebubbles":
                    await self._poll_bluebubbles()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("iMessage: Polling-Fehler: %s", exc)

            try:
                await asyncio.sleep(self._polling_interval)
            except asyncio.CancelledError:
                break

    async def _poll_native(self) -> None:
        """Pollt die Messages.app SQLite-DB nach neuen Nachrichten."""
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(None, self._query_new_messages)
        for msg in messages:
            await self._process_native_message(msg)

    def _query_new_messages(self) -> list[dict[str, Any]]:
        """Liest neue Nachrichten aus der Messages.app SQLite-DB (synchron)."""
        try:
            conn = sqlite3.connect(f"file:{MESSAGES_DB_PATH}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    m.ROWID,
                    m.text,
                    m.date,
                    m.is_from_me,
                    m.cache_has_attachments,
                    h.id AS handle_id
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.ROWID > ?
                  AND m.is_from_me = 0
                ORDER BY m.ROWID ASC
                LIMIT 50
                """,
                (self._last_message_rowid,),
            )

            results = []
            for row in cursor.fetchall():
                rowid = row["ROWID"]
                if rowid > self._last_message_rowid:
                    self._last_message_rowid = rowid
                results.append({
                    "rowid": rowid,
                    "text": row["text"] or "",
                    "date": row["date"],
                    "handle": row["handle_id"] or "",
                    "has_attachments": bool(row["cache_has_attachments"]),
                })

            conn.close()
            return results

        except Exception as exc:
            logger.debug("iMessage: SQLite-Abfrage fehlgeschlagen: %s", exc)
            return []

    async def _get_latest_rowid(self) -> int:
        """Ermittelt die letzte Message-ROWID."""
        loop = asyncio.get_running_loop()

        def _query() -> int:
            try:
                conn = sqlite3.connect(f"file:{MESSAGES_DB_PATH}?mode=ro", uri=True)
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(ROWID) FROM message")
                result = cursor.fetchone()
                conn.close()
                return result[0] or 0
            except Exception:
                return 0

        return await loop.run_in_executor(None, _query)

    async def _process_native_message(self, msg: dict[str, Any]) -> None:
        """Verarbeitet eine native iMessage-Nachricht."""
        handle = msg["handle"]
        text = msg["text"]

        if not handle or not text:
            return

        # Whitelist pruefen
        if self._allowed_handles and handle not in self._allowed_handles:
            logger.warning("iMessage: Nachricht von nicht-erlaubtem Handle %s", handle)
            return

        # Approval-Antwort pruefen
        session_for_handle = self._sessions.get(handle, "")
        if session_for_handle in self._pending_approvals:
            normalized = text.strip().lower()
            if normalized in ("ja", "yes", "ok", "genehmigen", "approve"):
                await self._resolve_approval(session_for_handle, approved=True, handle=handle)
                return
            elif normalized in ("nein", "no", "ablehnen", "reject"):
                await self._resolve_approval(session_for_handle, approved=False, handle=handle)
                return

        session_id = self._get_or_create_session(handle)

        incoming = IncomingMessage(
            channel="imessage",
            user_id=handle,
            text=text,
            session_id=session_id,
            metadata={
                "handle": handle,
                "rowid": str(msg["rowid"]),
                "has_attachments": str(msg["has_attachments"]),
            },
        )

        if self._handler:
            try:
                response = await self._handler(incoming)
                if response.session_id:
                    self._sessions[handle] = response.session_id
                await self._send_native(handle, response.text)
            except Exception as exc:
                logger.error("iMessage: Handler-Fehler: %s", exc)
                await self._send_native(handle, "Ein Fehler ist bei der Verarbeitung aufgetreten.")

    async def _poll_bluebubbles(self) -> None:
        """Pollt BlueBubbles-API nach neuen Nachrichten."""
        if not self._http:
            return

        try:
            resp = await self._http.post(
                "/api/v1/message/query",
                json={
                    "limit": 50,
                    "sort": "DESC",
                    "after": self._last_bb_timestamp,
                    "with": ["handle"],
                },
            )
            if resp.status_code != 200:
                return

            data = resp.json().get("data", [])
            # Aelteste zuerst verarbeiten
            for msg in reversed(data):
                await self._process_bb_message(msg)

        except Exception as exc:
            logger.debug("iMessage: BlueBubbles-Polling fehlgeschlagen: %s", exc)

    async def _process_bb_message(self, msg: dict[str, Any]) -> None:
        """Verarbeitet eine BlueBubbles-Nachricht."""
        is_from_me = msg.get("isFromMe", False)
        if is_from_me:
            return

        text = msg.get("text", "") or ""
        date_created = msg.get("dateCreated", 0)
        handle_data = msg.get("handle", {})
        handle = handle_data.get("address", "") if isinstance(handle_data, dict) else ""

        # Timestamp aktualisieren
        if date_created > self._last_bb_timestamp:
            self._last_bb_timestamp = date_created

        if not handle or not text:
            return

        # Whitelist pruefen
        if self._allowed_handles and handle not in self._allowed_handles:
            return

        # Approval-Antwort pruefen
        session_for_handle = self._sessions.get(handle, "")
        if session_for_handle in self._pending_approvals:
            normalized = text.strip().lower()
            if normalized in ("ja", "yes", "ok", "genehmigen", "approve"):
                await self._resolve_approval(session_for_handle, approved=True, handle=handle)
                return
            elif normalized in ("nein", "no", "ablehnen", "reject"):
                await self._resolve_approval(session_for_handle, approved=False, handle=handle)
                return

        session_id = self._get_or_create_session(handle)

        incoming = IncomingMessage(
            channel="imessage",
            user_id=handle,
            text=text,
            session_id=session_id,
            metadata={
                "handle": handle,
                "guid": msg.get("guid", ""),
                "date_created": str(date_created),
            },
        )

        if self._handler:
            try:
                response = await self._handler(incoming)
                if response.session_id:
                    self._sessions[handle] = response.session_id
                await self._send_bb(handle, response.text)
            except Exception as exc:
                logger.error("iMessage: Handler-Fehler: %s", exc)
                await self._send_bb(handle, "Ein Fehler ist bei der Verarbeitung aufgetreten.")

    # -- Outbound: Nachrichten senden --------------------------------------------

    async def _send_native(self, handle: str, text: str) -> None:
        """Sendet eine iMessage via AppleScript (macOS)."""
        chunks = _split_message(text)
        loop = asyncio.get_running_loop()

        for chunk in chunks:
            # AppleScript fuer iMessage
            script = (
                f'tell application "Messages"\n'
                f'  set targetService to 1st account whose service type = iMessage\n'
                f'  set targetBuddy to participant "{_escape_applescript(handle)}" of targetService\n'
                f'  send "{_escape_applescript(chunk)}" to targetBuddy\n'
                f'end tell'
            )
            try:
                await loop.run_in_executor(
                    None,
                    lambda s=script: subprocess.run(
                        ["osascript", "-e", s],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    ),
                )
            except Exception as exc:
                logger.error("iMessage: AppleScript-Senden fehlgeschlagen: %s", exc)

    async def _send_bb(self, handle: str, text: str) -> None:
        """Sendet eine Nachricht via BlueBubbles-API."""
        if not self._http:
            return

        chunks = _split_message(text)
        for chunk in chunks:
            try:
                resp = await self._http.post(
                    "/api/v1/message/text",
                    json={
                        "chatGuid": f"iMessage;-;{handle}",
                        "tempGuid": uuid.uuid4().hex,
                        "message": chunk,
                        "method": "apple-script",
                    },
                )
                if resp.status_code not in (200, 201):
                    logger.error(
                        "iMessage: BlueBubbles-Senden fehlgeschlagen (HTTP %d)",
                        resp.status_code,
                    )
            except Exception as exc:
                logger.error("iMessage: BlueBubbles-Senden fehlgeschlagen: %s", exc)

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an den User."""
        if not self._running:
            return

        handle = self._handle_for_session(message.session_id)
        if not handle:
            handle = message.metadata.get("handle", "")
        if not handle:
            logger.warning("iMessage: Kein Handle fuer Session %s", message.session_id[:8])
            return

        if self._mode == "native":
            await self._send_native(handle, message.text)
        else:
            await self._send_bb(handle, message.text)

    # -- Approval-Workflow -------------------------------------------------------

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Sendet Approval-Anfrage per iMessage."""
        handle = self._handle_for_session(session_id)
        if not handle:
            return False

        text = (
            f"Genehmigung erforderlich\n\n"
            f"Tool: {action.tool}\n"
            f"Grund: {reason}\n"
            f"Parameter: {str(action.params)[:300]}\n\n"
            f"Antworte mit 'ja' oder 'nein'"
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._pending_approvals[session_id] = future

        if self._mode == "native":
            await self._send_native(handle, text)
        else:
            await self._send_bb(handle, text)

        try:
            return await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("iMessage: Approval-Timeout fuer Session %s", session_id[:8])
            send_fn = self._send_native if self._mode == "native" else self._send_bb
            await send_fn(handle, "Genehmigung abgelaufen (Timeout).")
            return False
        finally:
            async with self._approval_lock:
                self._pending_approvals.pop(session_id, None)

    async def _resolve_approval(
        self, session_id: str, *, approved: bool, handle: str,
    ) -> None:
        """Loest ein Approval-Future auf."""
        async with self._approval_lock:
            future = self._pending_approvals.get(session_id)
        if future and not future.done():
            future.set_result(approved)
            status = "genehmigt" if approved else "abgelehnt"
            if self._mode == "native":
                await self._send_native(handle, f"Aktion {status}.")
            else:
                await self._send_bb(handle, f"Aktion {status}.")

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

    def _get_or_create_session(self, handle: str) -> str:
        if handle not in self._sessions:
            self._sessions[handle] = uuid.uuid4().hex
        return self._sessions[handle]

    def _handle_for_session(self, session_id: str) -> str | None:
        """Findet das Handle fuer eine Session-ID."""
        for handle, sid in self._sessions.items():
            if sid == session_id:
                return handle
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


def _escape_applescript(text: str) -> str:
    """Escaped einen String fuer die Verwendung in AppleScript."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
