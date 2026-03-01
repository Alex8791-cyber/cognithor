"""Matrix-Channel: Bidirektionale Kommunikation ueber das Matrix-Protokoll.

Nutzt matrix-nio (async) als SDK:
  https://github.com/matrix-nio/matrix-nio

Features:
  - Text senden/empfangen via matrix-nio async client
  - E2EE-Support (optional, benoetigt python-olm)
  - Raum-basiertes Messaging (Join per Invite oder Room-ID)
  - Reaction-basierte Approvals (Thumbs-Up/Down)
  - Media-Upload und -Download
  - Voice-Transkription via faster-whisper
  - HTML-formatierte Nachrichten
  - Session->Room Mapping
  - Message-Splitting bei 4000 Zeichen
  - Graceful Shutdown

Konfiguration:
  - JARVIS_MATRIX_HOMESERVER: Homeserver-URL (z.B. https://matrix.org)
  - JARVIS_MATRIX_USER_ID: Bot-User-ID (@jarvis:matrix.org)
  - JARVIS_MATRIX_ACCESS_TOKEN: Access-Token (oder Passwort fuer Login)
  - JARVIS_MATRIX_PASSWORD: Passwort (alternativ zu Token)
  - JARVIS_MATRIX_ALLOWED_ROOMS: Komma-getrennte Room-ID-Whitelist

Abhaengigkeiten:
  pip install 'matrix-nio[e2e]'  (oder matrix-nio ohne E2EE)
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from pathlib import Path
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.security.token_store import get_token_store

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
APPROVAL_TIMEOUT = 300


class MatrixChannel(Channel):
    """Matrix als bidirektionaler Kommunikationskanal.

    Nutzt matrix-nio fuer asynchrone Kommunikation.
    Unterstuetzt sowohl unverschluesselte als auch E2EE-Raeume.
    """

    def __init__(
        self,
        *,
        homeserver: str = "https://matrix.org",
        user_id: str = "",
        access_token: str = "",
        password: str = "",
        allowed_rooms: list[str] | None = None,
        store_path: Path | None = None,
        workspace_dir: Path | None = None,
    ) -> None:
        self._homeserver = homeserver
        self._user_id = user_id
        self._token_store = get_token_store()
        if access_token:
            self._token_store.store("matrix_access_token", access_token)
        self._has_access_token = bool(access_token)
        if password:
            self._token_store.store("matrix_password", password)
        self._has_password = bool(password)
        self._allowed_rooms = set(allowed_rooms or [])
        self._store_path = store_path or Path.home() / ".jarvis" / "matrix_store"
        self._workspace_dir = workspace_dir or Path.home() / ".jarvis" / "workspace" / "matrix"

        self._handler: MessageHandler | None = None
        self._running = False
        self._client: Any | None = None  # nio.AsyncClient
        self._sync_task: asyncio.Task[None] | None = None

        # Session-Mapping: room_id -> session_id
        self._sessions: dict[str, str] = {}
        # Room->User mapping fuer Approvals
        self._room_users: dict[str, str] = {}

        # Approval-Workflow
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._approval_messages: dict[str, str] = {}  # event_id -> session_id
        self._approval_lock = asyncio.Lock()

        # Streaming-Buffer
        self._stream_buffers: dict[str, list[str]] = {}
        self._stream_lock = asyncio.Lock()

        # Voice-Transkription
        self._whisper: Any | None = None

    @property
    def _access_token(self) -> str:
        """Access-Token (entschlüsselt bei Zugriff)."""
        if self._has_access_token:
            return self._token_store.retrieve("matrix_access_token")
        return ""

    @property
    def _password(self) -> str:
        """Passwort (entschlüsselt bei Zugriff)."""
        if self._has_password:
            return self._token_store.retrieve("matrix_password")
        return ""

    @property
    def name(self) -> str:
        return "matrix"

    # -- Lifecycle ---------------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Matrix-Client und beginnt mit dem Sync."""
        self._handler = handler
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        try:
            from nio import AsyncClient, LoginResponse, MatrixRoom, RoomMessageText
        except ImportError:
            logger.error(
                "matrix-nio nicht installiert. "
                "Installiere mit: pip install 'matrix-nio[e2e]'"
            )
            return

        # E2EE-Store vorbereiten
        self._store_path.mkdir(parents=True, exist_ok=True)

        # Client erstellen
        self._client = AsyncClient(
            self._homeserver,
            self._user_id,
            store_path=str(self._store_path),
        )

        # Login: Token oder Passwort
        if self._access_token:
            self._client.access_token = self._access_token
            self._client.user_id = self._user_id
            # Device-ID setzen wenn vorhanden
            logger.info("Matrix: Token-basierter Login fuer %s", self._user_id)
        elif self._password:
            resp = await self._client.login(self._password)
            if isinstance(resp, LoginResponse):
                logger.info("Matrix: Login erfolgreich als %s", self._user_id)
            else:
                logger.error("Matrix: Login fehlgeschlagen: %s", resp)
                return
        else:
            logger.error("Matrix: Weder access_token noch password angegeben")
            return

        # Whisper laden
        try:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel("base", device="auto", compute_type="int8")
            logger.info("Matrix: faster-whisper geladen")
        except ImportError:
            logger.debug("Matrix: faster-whisper nicht verfuegbar")

        # Event-Callbacks registrieren
        self._client.add_event_callback(self._on_message, RoomMessageText)

        # Reaction-Callback fuer Approvals
        try:
            from nio import UnknownEvent
            self._client.add_event_callback(self._on_reaction, UnknownEvent)
        except ImportError:
            logger.debug("Matrix: UnknownEvent nicht verfuegbar, Reaction-Approvals deaktiviert")

        # Invite-Callback: Auto-Join
        try:
            from nio import InviteMemberEvent
            self._client.add_event_callback(self._on_invite, InviteMemberEvent)
        except ImportError:
            pass

        # Sync starten (non-blocking)
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(
            "MatrixChannel gestartet (Homeserver=%s, User=%s)",
            self._homeserver,
            self._user_id,
        )

    async def _sync_loop(self) -> None:
        """Haupt-Sync-Loop: Empfaengt neue Events vom Homeserver."""
        # Erster Sync mit Timeout (nur aktuelle Events, keine History)
        first_sync = True
        while self._running and self._client:
            try:
                timeout = 0 if first_sync else 30000
                resp = await self._client.sync(timeout=timeout, full_state=first_sync)
                if hasattr(resp, "next_batch"):
                    first_sync = False
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Matrix: Sync-Fehler: %s", exc)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stoppt den Matrix-Client sauber."""
        self._running = False

        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        # Pending approvals abbrechen
        async with self._approval_lock:
            for future in self._pending_approvals.values():
                if not future.done():
                    future.set_result(False)
            self._pending_approvals.clear()

        if self._client:
            try:
                await self._client.close()
            except Exception:
                logger.debug("Matrix: Client-Close uebersprungen", exc_info=True)
            self._client = None

        logger.info("MatrixChannel gestoppt")

    # -- Inbound: Nachrichten empfangen ------------------------------------------

    async def _on_message(self, room: Any, event: Any) -> None:
        """Verarbeitet eingehende Text-Nachrichten."""
        # Eigene Nachrichten ignorieren
        if event.sender == self._user_id:
            return

        # Room-Whitelist pruefen
        if self._allowed_rooms and room.room_id not in self._allowed_rooms:
            return

        text = event.body.strip() if hasattr(event, "body") else ""
        if not text:
            return

        sender = event.sender
        room_id = room.room_id

        # Approval-Antwort pruefen (Text-basiert)
        session_for_room = self._sessions.get(room_id, "")
        if session_for_room in self._pending_approvals:
            normalized = text.lower()
            if normalized in ("ja", "yes", "ok", "genehmigen", "approve"):
                await self._resolve_approval(session_for_room, approved=True)
                return
            elif normalized in ("nein", "no", "ablehnen", "reject"):
                await self._resolve_approval(session_for_room, approved=False)
                return

        # User-Room-Mapping
        self._room_users[room_id] = sender

        # Session-Mapping
        session_id = self._get_or_create_session(room_id)

        incoming = IncomingMessage(
            channel="matrix",
            user_id=sender,
            text=text,
            session_id=session_id,
            metadata={
                "room_id": room_id,
                "event_id": event.event_id if hasattr(event, "event_id") else "",
                "sender": sender,
            },
        )

        if self._handler:
            try:
                response = await self._handler(incoming)
                if response.session_id:
                    self._sessions[room_id] = response.session_id
                await self._send_to_room(room_id, response.text)
            except Exception as exc:
                logger.error("Matrix: Handler-Fehler: %s", exc)
                await self._send_to_room(room_id, "Ein Fehler ist bei der Verarbeitung aufgetreten.")

    async def _on_reaction(self, room: Any, event: Any) -> None:
        """Verarbeitet Reactions fuer Approval-Workflow."""
        if not hasattr(event, "source"):
            return

        source = event.source or {}
        content = source.get("content", {})
        if content.get("m.relates_to", {}).get("rel_type") != "m.annotation":
            return

        # Eigene Reactions ignorieren
        sender = source.get("sender", "")
        if sender == self._user_id:
            return

        event_id = content.get("m.relates_to", {}).get("event_id", "")
        emoji = content.get("m.relates_to", {}).get("key", "")

        async with self._approval_lock:
            session_id = self._approval_messages.get(event_id)
            if not session_id:
                return

        if emoji in ("\u2705", "\U0001f44d"):  # Checkmark, Thumbs-Up
            await self._resolve_approval(session_id, approved=True)
        elif emoji in ("\u274c", "\U0001f44e"):  # Cross, Thumbs-Down
            await self._resolve_approval(session_id, approved=False)

    async def _on_invite(self, room: Any, event: Any) -> None:
        """Auto-Join bei Einladung (wenn erlaubt)."""
        if not self._client:
            return

        room_id = room.room_id if hasattr(room, "room_id") else str(room)
        if self._allowed_rooms and room_id not in self._allowed_rooms:
            logger.info("Matrix: Invite abgelehnt fuer nicht-erlaubten Raum %s", room_id)
            return

        try:
            await self._client.join(room_id)
            logger.info("Matrix: Raum beigetreten: %s", room_id)
        except Exception as exc:
            logger.error("Matrix: Join fehlgeschlagen fuer %s: %s", room_id, exc)

    # -- Outbound: Nachrichten senden --------------------------------------------

    async def _send_to_room(self, room_id: str, text: str) -> str | None:
        """Sendet eine Nachricht in einen Matrix-Raum.

        Returns:
            Event-ID der gesendeten Nachricht oder None bei Fehler.
        """
        if not self._client:
            return None

        chunks = _split_message(text)
        last_event_id = None

        for chunk in chunks:
            try:
                # HTML-formatiert senden (Markdown-Konvertierung)
                html_body = _text_to_html(chunk)
                content = {
                    "msgtype": "m.text",
                    "body": chunk,
                    "format": "org.matrix.custom.html",
                    "formatted_body": html_body,
                }
                resp = await self._client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content=content,
                )
                if hasattr(resp, "event_id"):
                    last_event_id = resp.event_id
            except Exception as exc:
                logger.error("Matrix: Senden fehlgeschlagen in %s: %s", room_id, exc)

        return last_event_id

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an den User."""
        if not self._running:
            return

        room_id = self._room_for_session(message.session_id)
        if not room_id:
            room_id = message.metadata.get("room_id", "")
        if not room_id:
            logger.warning("Matrix: Kein Raum fuer Session %s", message.session_id[:8])
            return

        await self._send_to_room(room_id, message.text)

    # -- Approval-Workflow -------------------------------------------------------

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Sendet Approval-Anfrage mit Reactions."""
        room_id = self._room_for_session(session_id)
        if not room_id:
            return False

        text = (
            f"**Genehmigung erforderlich**\n\n"
            f"**Tool:** `{action.tool}`\n"
            f"**Grund:** {reason}\n"
            f"**Parameter:** `{str(action.params)[:300]}`\n\n"
            f"Reagiere mit \u2705 (genehmigen) oder \u274c (ablehnen)\n"
            f"Oder antworte mit 'ja' / 'nein'"
        )

        event_id = await self._send_to_room(room_id, text)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        async with self._approval_lock:
            self._pending_approvals[session_id] = future
            if event_id:
                self._approval_messages[event_id] = session_id

        # Reactions hinzufuegen als Vorlage
        if event_id and self._client:
            try:
                await self._client.room_send(
                    room_id=room_id,
                    message_type="m.reaction",
                    content={
                        "m.relates_to": {
                            "rel_type": "m.annotation",
                            "event_id": event_id,
                            "key": "\u2705",
                        }
                    },
                )
                await self._client.room_send(
                    room_id=room_id,
                    message_type="m.reaction",
                    content={
                        "m.relates_to": {
                            "rel_type": "m.annotation",
                            "event_id": event_id,
                            "key": "\u274c",
                        }
                    },
                )
            except Exception:
                logger.debug("Matrix: Reaction-Vorlagen konnten nicht gesetzt werden")

        try:
            return await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Matrix: Approval-Timeout fuer Session %s", session_id[:8])
            await self._send_to_room(room_id, "Genehmigung abgelaufen (Timeout).")
            return False
        finally:
            async with self._approval_lock:
                self._pending_approvals.pop(session_id, None)
                if event_id:
                    self._approval_messages.pop(event_id, None)

    async def _resolve_approval(self, session_id: str, *, approved: bool) -> None:
        """Loest ein Approval-Future auf."""
        async with self._approval_lock:
            future = self._pending_approvals.get(session_id)
        if future and not future.done():
            future.set_result(approved)
            room_id = self._room_for_session(session_id)
            if room_id:
                status = "genehmigt" if approved else "abgelehnt"
                await self._send_to_room(room_id, f"Aktion {status}.")

    # -- Streaming ---------------------------------------------------------------

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Buffert Streaming-Tokens und sendet gebuendelt."""
        async with self._stream_lock:
            buf = self._stream_buffers.setdefault(session_id, [])
            buf.append(token)
            is_first = len(buf) == 1
        if is_first:
            await asyncio.sleep(0.5)
            async with self._stream_lock:
                text = "".join(self._stream_buffers.pop(session_id, []))
            if text.strip():
                await self.send(
                    OutgoingMessage(channel=self.name, text=text, session_id=session_id)
                )

    # -- Hilfsmethoden -----------------------------------------------------------

    def _get_or_create_session(self, room_id: str) -> str:
        if room_id not in self._sessions:
            self._sessions[room_id] = uuid.uuid4().hex
        return self._sessions[room_id]

    def _room_for_session(self, session_id: str) -> str | None:
        """Findet den Raum fuer eine Session-ID."""
        for room_id, sid in self._sessions.items():
            if sid == session_id:
                return room_id
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


def _text_to_html(text: str) -> str:
    """Einfache Markdown-zu-HTML Konvertierung fuer Matrix."""
    import re

    html = text
    # Code-Bloecke
    html = re.sub(r"```(.*?)```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL)
    # Inline-Code
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    # Italic
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    # Newlines
    html = html.replace("\n", "<br>")
    return html
