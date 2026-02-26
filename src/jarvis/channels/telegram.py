"""Telegram-Channel: Kommunikation über Telegram-Bot.

Features:
  - User-ID-Whitelist (Sicherheit)
  - Inline-Keyboards für Approval-Workflow
  - Voice-Messages: Automatische Transkription via Whisper
  - Foto-/Dokument-Empfang mit Beschreibung
  - Typing-Indicator während der Verarbeitung
  - Datei-Versand (Bilder, PDFs, etc.)
  - Reconnect bei Verbindungsabbruch
  - Streaming-Simulation (lange Nachrichten in Teilen)
  - Graceful Shutdown

Bibel-Referenz: §9.3 (Telegram Channel)

Benötigt: pip install 'python-telegram-bot>=21.0,<22'
Konfiguration: JARVIS_TELEGRAM_TOKEN und JARVIS_TELEGRAM_ALLOWED_USERS
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

# Maximale Telegram-Nachrichtenlänge
MAX_MESSAGE_LENGTH = 4096

# Timeout für Approval-Anfragen (Sekunden)
APPROVAL_TIMEOUT = 300  # 5 Minuten


class TelegramChannel(Channel):
    """Telegram-Bot als Kommunikationskanal.

    Nutzt python-telegram-bot 21.x mit async/await.
    Filtert Nachrichten nach erlaubten User-IDs.

    Attributes:
        token: Bot-API-Token.
        allowed_users: Set erlaubter Telegram-User-IDs.
    """

    def __init__(
        self,
        token: str,
        allowed_users: set[int] | list[int] | None = None,
        workspace_dir: Path | None = None,
        max_reconnect_attempts: int = 5,
    ) -> None:
        """Initialisiert den Telegram-Channel.

        Args:
            token: Telegram Bot API Token.
            allowed_users: Erlaubte Telegram-User-IDs. None = alle erlaubt.
            workspace_dir: Verzeichnis für heruntergeladene Medien.
            max_reconnect_attempts: Maximale Reconnect-Versuche.
        """
        self.token = token
        self.allowed_users: set[int] = set(allowed_users or [])
        self._workspace_dir = workspace_dir or Path.home() / ".jarvis" / "workspace" / "telegram"
        self._max_reconnect = max_reconnect_attempts
        self._handler: MessageHandler | None = None
        self._app: Any | None = None  # telegram.ext.Application
        self._approval_events: dict[str, asyncio.Event] = {}
        self._approval_results: dict[str, bool] = {}
        self._approval_lock = asyncio.Lock()
        self._session_chat_map: dict[str, int] = {}
        self._running = False
        self._typing_tasks: dict[int, asyncio.Task[None]] = {}
        self._whisper_model: Any | None = None

    @property
    def name(self) -> str:
        """Eindeutiger Channel-Name."""
        return "telegram"

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Telegram-Bot.

        Args:
            handler: Async-Callback für eingehende Nachrichten.
        """
        self._handler = handler

        try:
            from telegram.ext import (
                Application,
                CallbackQueryHandler,
                filters,
            )
            from telegram.ext import (
                MessageHandler as TGMessageHandler,
            )
        except ImportError:
            logger.error(
                "python-telegram-bot nicht installiert. "
                "Installiere mit: pip install 'python-telegram-bot>=21.0,<22'"
            )
            return

        self._app = Application.builder().token(self.token).concurrent_updates(True).build()

        # Handler registrieren
        self._app.add_handler(
            TGMessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._on_telegram_message,
            )
        )
        # Voice-Messages (Sprachnachrichten)
        self._app.add_handler(
            TGMessageHandler(
                filters.VOICE | filters.AUDIO,
                self._on_voice_message,
            )
        )
        # Fotos
        self._app.add_handler(
            TGMessageHandler(
                filters.PHOTO,
                self._on_photo_message,
            )
        )
        # Dokumente (PDFs, etc.)
        self._app.add_handler(
            TGMessageHandler(
                filters.Document.ALL,
                self._on_document_message,
            )
        )
        self._app.add_handler(CallbackQueryHandler(self._on_approval_callback))

        # Bot starten (non-blocking)
        await self._app.initialize()
        await self._app.start()
        if self._app.updater is not None:
            await self._app.updater.start_polling(drop_pending_updates=True)

        self._running = True
        logger.info("Telegram-Bot gestartet")

    async def stop(self) -> None:
        """Stoppt den Telegram-Bot sauber."""
        if not self._running or self._app is None:
            return

        try:
            if self._app.updater is not None:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            logger.exception("Fehler beim Stoppen des Telegram-Bots")

        self._running = False
        self._app = None
        logger.info("Telegram-Bot gestoppt")

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an den User.

        Teilt lange Nachrichten automatisch in mehrere Teile.

        Args:
            message: Die zu sendende Nachricht.
        """
        if self._app is None:
            logger.warning("Telegram-Bot nicht gestartet")
            return

        chat_id = message.metadata.get("chat_id")
        if chat_id is None:
            logger.warning("Keine chat_id in message.metadata")
            return

        text = message.text
        chunks = _split_message(text)

        for chunk in chunks:
            try:
                await self._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=chunk,
                    parse_mode="Markdown",
                )
            except Exception:
                logger.debug("Markdown-Parsing fehlgeschlagen, Fallback auf Plain-Text", exc_info=True)
                # Fallback ohne Markdown falls Parsing fehlschlägt
                try:
                    await self._app.bot.send_message(
                        chat_id=int(chat_id),
                        text=chunk,
                    )
                except Exception:
                    logger.exception("Fehler beim Senden an chat_id=%s", chat_id)

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Fragt den User via Inline-Keyboard um Bestätigung.

        Args:
            session_id: Aktive Session-ID.
            action: Die zu bestätigende Aktion.
            reason: Begründung für die Bestätigung.

        Returns:
            True wenn User bestätigt, False bei Ablehnung oder Timeout.
        """
        if self._app is None:
            return False

        chat_id = self._extract_chat_id_from_session(session_id)
        if chat_id is None:
            logger.warning("Keine chat_id für Session %s", session_id)
            return False

        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        except ImportError:
            return False

        approval_id = f"approval-{session_id}-{action.tool}"

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✅ Erlauben",
                        callback_data=f"approve:{approval_id}",
                    ),
                    InlineKeyboardButton(
                        "❌ Ablehnen",
                        callback_data=f"deny:{approval_id}",
                    ),
                ]
            ]
        )

        text = (
            f"🔶 *Bestätigung erforderlich*\n\n"
            f"**Aktion:** `{action.tool}`\n"
            f"**Grund:** {reason}\n"
            f"**Parameter:** `{action.params}`"
        )

        event = asyncio.Event()
        async with self._approval_lock:
            self._approval_events[approval_id] = event
            self._approval_results[approval_id] = False

        try:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("Fehler beim Senden der Approval-Anfrage")
            async with self._approval_lock:
                self._approval_events.pop(approval_id, None)
                self._approval_results.pop(approval_id, None)
            return False

        # Warte auf User-Antwort (mit Timeout)
        try:
            await asyncio.wait_for(event.wait(), timeout=APPROVAL_TIMEOUT)
            async with self._approval_lock:
                return self._approval_results.get(approval_id, False)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Approval-Timeout für %s", approval_id)
            return False
        finally:
            async with self._approval_lock:
                self._approval_events.pop(approval_id, None)
                self._approval_results.pop(approval_id, None)

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Streaming ist bei Telegram nicht sinnvoll unterstützt.

        Telegram hat kein echtes Token-Streaming. Nachrichten werden
        als Ganzes gesendet (via send()).

        Args:
            session_id: Aktive Session-ID.
            token: Einzelnes Token (wird ignoriert).
        """
        # Telegram unterstützt kein echtes Streaming.
        # Nachrichten werden als Ganzes über send() gesendet.
        pass

    # === Interne Handler ===

    async def _on_telegram_message(self, update: Any, context: Any) -> None:
        """Verarbeitet eingehende Telegram-Textnachrichten.

        Prüft User-Whitelist und leitet an den Gateway-Handler weiter.
        """
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        text = update.effective_message.text or ""

        # Whitelist-Prüfung
        if self.allowed_users and user_id not in self.allowed_users:
            logger.warning("Unerlaubter Zugriff von User %d (Chat %d)", user_id, chat_id)
            await update.effective_message.reply_text(
                "⛔ Zugriff verweigert. Deine User-ID ist nicht autorisiert."
            )
            return

        await self._process_incoming(chat_id, user_id, text, update)

    async def _on_voice_message(self, update: Any, context: Any) -> None:
        """Verarbeitet Sprachnachrichten: Download → Transkription → Gateway.

        Nutzt faster-whisper oder whisper.cpp für lokale Transkription.
        Fallback: Nachricht an den User, dass Voice nicht verfügbar ist.
        """
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if self.allowed_users and user_id not in self.allowed_users:
            return

        voice = update.effective_message.voice or update.effective_message.audio
        if voice is None:
            return

        # Typing-Indicator starten
        typing_task = self._start_typing(chat_id)

        try:
            # Audio herunterladen
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            file = await voice.get_file()
            audio_path = self._workspace_dir / f"voice-{voice.file_unique_id}.ogg"
            await file.download_to_drive(str(audio_path))
            logger.info("Voice heruntergeladen: %s (%d bytes)", audio_path, voice.file_size or 0)

            # Transkription versuchen
            text = await self._transcribe_audio(audio_path)

            if text:
                # Transkription dem User zeigen
                await update.effective_message.reply_text(f"🎤 _{text}_", parse_mode="Markdown")
                await self._process_incoming(chat_id, user_id, text, update)
            else:
                await update.effective_message.reply_text(
                    "⚠️ Spracherkennung nicht verfügbar.\n"
                    "Installiere `faster-whisper` für lokale Transkription:\n"
                    "`pip install faster-whisper`"
                )
        except Exception:
            logger.exception("Fehler bei Voice-Verarbeitung")
            await update.effective_message.reply_text("❌ Fehler bei der Sprachverarbeitung.")
        finally:
            self._stop_typing(chat_id, typing_task)

    async def _on_photo_message(self, update: Any, context: Any) -> None:
        """Verarbeitet Fotos: Download + Caption als Nachricht weiterleiten."""
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if self.allowed_users and user_id not in self.allowed_users:
            return

        photos = update.effective_message.photo
        if not photos:
            return

        # Größtes Foto wählen
        photo = photos[-1]
        caption = update.effective_message.caption or ""

        try:
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            file = await photo.get_file()
            photo_path = self._workspace_dir / f"photo-{photo.file_unique_id}.jpg"
            await file.download_to_drive(str(photo_path))

            text = f"[Foto empfangen: {photo_path.name}, {photo.width}x{photo.height}]"
            if caption:
                text += f"\nBeschreibung: {caption}"

            await self._process_incoming(chat_id, user_id, text, update)

        except Exception:
            logger.exception("Fehler beim Foto-Download")
            await update.effective_message.reply_text("❌ Fehler beim Empfangen des Fotos.")

    async def _on_document_message(self, update: Any, context: Any) -> None:
        """Verarbeitet Dokumente: Download + Metadaten als Nachricht weiterleiten."""
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if self.allowed_users and user_id not in self.allowed_users:
            return

        doc = update.effective_message.document
        if doc is None:
            return

        caption = update.effective_message.caption or ""

        try:
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            file = await doc.get_file()
            raw_name = doc.file_name or f"doc-{doc.file_unique_id}"
            # Sanitize filename to prevent path traversal
            filename = Path(raw_name).name.lstrip(".")
            if not filename:
                filename = f"doc-{doc.file_unique_id}"
            doc_path = (self._workspace_dir / filename).resolve()
            if not str(doc_path).startswith(str(self._workspace_dir.resolve())):
                logger.warning("Path traversal attempt blocked: %s", raw_name)
                return
            await file.download_to_drive(str(doc_path))

            size_kb = (doc.file_size or 0) // 1024
            text = f"[Dokument empfangen: {filename}, {size_kb} KB, MIME: {doc.mime_type or 'unbekannt'}]"
            if caption:
                text += f"\nBeschreibung: {caption}"
            text += f"\nGespeichert unter: {doc_path}"

            await self._process_incoming(chat_id, user_id, text, update)

        except Exception:
            logger.exception("Fehler beim Dokument-Download")
            await update.effective_message.reply_text("❌ Fehler beim Empfangen des Dokuments.")

    # === Hilfsmethoden ===

    async def _process_incoming(
        self,
        chat_id: int,
        user_id: int,
        text: str,
        update: Any,
    ) -> None:
        """Zentrale Verarbeitung eingehender Nachrichten aller Typen."""
        if self._handler is None:
            await update.effective_message.reply_text("⚠️ Jarvis ist noch nicht bereit.")
            return

        # Typing-Indicator starten
        typing_task = self._start_typing(chat_id)

        msg = IncomingMessage(
            channel="telegram",
            user_id=str(user_id),
            text=text,
        )

        try:
            response = await self._handler(msg)

            # Session → chat_id Mapping speichern
            if response.session_id:
                self._session_chat_map[response.session_id] = chat_id
            enriched = OutgoingMessage(
                channel="telegram",
                text=response.text,
                session_id=response.session_id,
                is_final=response.is_final,
                reply_to=response.reply_to,
                metadata={**response.metadata, "chat_id": str(chat_id)},
            )
            await self.send(enriched)

        except Exception:
            logger.exception("Fehler bei Telegram-Nachricht von User %d", user_id)
            await update.effective_message.reply_text(
                "❌ Ein Fehler ist aufgetreten. Bitte versuche es erneut."
            )
        finally:
            self._stop_typing(chat_id, typing_task)

    async def _transcribe_audio(self, audio_path: Path) -> str | None:
        """Transkribiert eine Audiodatei mit faster-whisper (lokal).

        Returns:
            Transkribierter Text oder None wenn nicht verfügbar.
        """
        try:
            import os
            # CUDA deaktivieren falls cuDNN nicht verfügbar (verhindert DLL-Crash)
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            from faster_whisper import WhisperModel

            if self._whisper_model is None:
                self._whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            model = self._whisper_model
            segments, _info = model.transcribe(str(audio_path), language="de")
            text = " ".join(seg.text.strip() for seg in segments)
            logger.info("Voice transkribiert: %d Zeichen", len(text))
            return text if text.strip() else None

        except ImportError:
            logger.warning("faster-whisper nicht installiert -- Voice-Transkription deaktiviert")
            return None
        except Exception:
            logger.exception("Transkriptionsfehler")
            return None

    def _start_typing(self, chat_id: int) -> asyncio.Task[None] | None:
        """Startet den Typing-Indicator für einen Chat.

        Telegram setzt den Indicator nach ~5 Sekunden zurück,
        daher wird er periodisch erneuert bis stop_typing aufgerufen wird.
        """
        if self._app is None:
            return None

        async def _typing_loop() -> None:
            while True:
                try:
                    await self._app.bot.send_chat_action(
                        chat_id=chat_id,
                        action="typing",
                    )
                    await asyncio.sleep(4.5)
                except asyncio.CancelledError:
                    break
                except Exception:
                    break

        task = asyncio.create_task(_typing_loop())
        self._typing_tasks[chat_id] = task
        return task

    def _stop_typing(self, chat_id: int, task: asyncio.Task[None] | None = None) -> None:
        """Stoppt den Typing-Indicator."""
        if task:
            task.cancel()
        existing = self._typing_tasks.pop(chat_id, None)
        if existing and existing is not task:
            existing.cancel()

    async def send_file(self, chat_id: int, file_path: Path, caption: str = "") -> bool:
        """Sendet eine Datei an einen Telegram-Chat.

        Args:
            chat_id: Ziel-Chat-ID.
            file_path: Pfad zur Datei.
            caption: Optionale Beschreibung.

        Returns:
            True bei Erfolg.
        """
        if self._app is None:
            return False

        try:
            suffix = file_path.suffix.lower()
            if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                with open(file_path, "rb") as fh:
                    await self._app.bot.send_photo(
                        chat_id=chat_id,
                        photo=fh,
                        caption=caption or None,
                    )
            else:
                with open(file_path, "rb") as fh:
                    await self._app.bot.send_document(
                        chat_id=chat_id,
                        document=fh,
                        caption=caption or None,
                        filename=file_path.name,
                    )
            logger.info("Datei gesendet: %s an Chat %d", file_path.name, chat_id)
            return True
        except Exception:
            logger.exception("Fehler beim Senden der Datei %s", file_path)
            return False

    async def _on_approval_callback(self, update: Any, context: Any) -> None:
        """Verarbeitet Approval-Inline-Keyboard-Klicks."""
        query = update.callback_query
        if query is None:
            return

        await query.answer()
        data = query.data or ""

        if ":" not in data:
            return

        action, approval_id = data.split(":", 1)

        async with self._approval_lock:
            has_event = approval_id in self._approval_events
            if has_event:
                approved = action == "approve"
                self._approval_results[approval_id] = approved
                self._approval_events[approval_id].set()

        if has_event:
            status = "✅ Erlaubt" if approved else "❌ Abgelehnt"
            with contextlib.suppress(Exception):
                await query.edit_message_text(f"{query.message.text}\n\n→ {status}")
        else:
            with contextlib.suppress(Exception):
                await query.edit_message_text(f"{query.message.text}\n\n→ ⏰ Abgelaufen")

    def _extract_chat_id_from_session(self, session_id: str) -> int | None:
        """Extrahiert die chat_id aus einer Session-ID.

        Nutzt das interne Session→Chat-ID Mapping, das beim
        Empfang von Gateway-Antworten befüllt wird.

        Args:
            session_id: Session-ID.

        Returns:
            Chat-ID oder None.
        """
        return self._session_chat_map.get(session_id)


def _split_message(text: str) -> list[str]:
    """Teilt eine Nachricht in Telegram-kompatible Teile.

    Versucht an Zeilenumbrüchen zu splitten, nicht mitten in Wörtern.

    Args:
        text: Der vollständige Nachrichtentext.

    Returns:
        Liste von Nachrichtenteilen (max. MAX_MESSAGE_LENGTH Zeichen).
    """
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= MAX_MESSAGE_LENGTH:
            chunks.append(remaining)
            break

        # Versuche an einem Zeilenumbruch zu splitten
        split_pos = remaining.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1 or split_pos < MAX_MESSAGE_LENGTH // 2:
            # Kein guter Zeilenumbruch → an Leerzeichen splitten
            split_pos = remaining.rfind(" ", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1:
            # Kein Leerzeichen → harter Split
            split_pos = MAX_MESSAGE_LENGTH

        chunks.append(remaining[:split_pos])
        remaining = remaining[split_pos:].lstrip()

    return chunks
