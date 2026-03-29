"""Telegram-Channel: Kommunikation ueber Telegram-Bot.

Features:
  - User-ID-Whitelist (Sicherheit)
  - Inline-Keyboards fuer Approval-Workflow
  - Voice-Messages: Automatische Transkription via Whisper
  - Foto-/Dokument-Empfang mit Beschreibung
  - Typing-Indicator waehrend der Verarbeitung
  - Datei-Versand (Bilder, PDFs, etc.)
  - Reconnect bei Verbindungsabbruch
  - Streaming-Simulation (lange Nachrichten in Teilen)
  - Graceful Shutdown
  - Webhook-Modus (<100ms Latenz) mit Fallback auf Polling

Bibel-Referenz: §9.3 (Telegram Channel)

Benoetigt: pip install 'python-telegram-bot>=21.0,<22'
Konfiguration: JARVIS_TELEGRAM_TOKEN und JARVIS_TELEGRAM_ALLOWED_USERS
Optional fuer Webhook: JARVIS_TELEGRAM_USE_WEBHOOK, JARVIS_TELEGRAM_WEBHOOK_URL
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.channels.base import Channel, MessageHandler, StatusType
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.security.token_store import get_token_store
from jarvis.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from jarvis.utils.ttl_dict import TTLDict

if TYPE_CHECKING:
    from jarvis.gateway.session_store import SessionStore

logger = logging.getLogger(__name__)

# Maximum Telegram message length
MAX_MESSAGE_LENGTH = 4096

# Vision postprocessing: system prompt for text correction
_VISION_POLISH_PROMPT = (
    "Überarbeite den folgenden Text. Korrigiere Rechtschreibung, Grammatik, "
    "Satzbau und Satzzeichen. Behalte den Inhalt und die Sprache exakt bei. "
    "Gib NUR den korrigierten Text aus, ohne Erklärungen."
)

# Timeout for approval requests (seconds)
APPROVAL_TIMEOUT = 300  # 5 Minuten

# Maximum document size (50 MB)
MAX_DOCUMENT_SIZE = 52_428_800


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
        session_store: SessionStore | None = None,
        *,
        use_webhook: bool = False,
        webhook_url: str = "",
        webhook_port: int = 8443,
        webhook_host: str = "0.0.0.0",
        ssl_certfile: str = "",
        ssl_keyfile: str = "",
    ) -> None:
        """Initialisiert den Telegram-Channel.

        Args:
            token: Telegram Bot API Token.
            allowed_users: Erlaubte Telegram-User-IDs. None = alle erlaubt.
            workspace_dir: Verzeichnis fuer heruntergeladene Medien.
            max_reconnect_attempts: Maximale Reconnect-Versuche.
            session_store: Optionaler SessionStore fuer persistente Mappings.
            use_webhook: Webhook statt Polling verwenden.
            webhook_url: Externe URL, die Telegram fuer Updates nutzt.
            webhook_port: Lokaler Port fuer den Webhook-Server.
            webhook_host: Lokaler Bind-Host fuer den Webhook-Server.
            ssl_certfile: Pfad zum SSL-Zertifikat (PEM) fuer TLS.
            ssl_keyfile: Pfad zum SSL-Privat-Key (PEM) fuer TLS.
        """
        self._token_store = get_token_store()
        self._token_store.store("telegram_bot_token", token)
        self.allowed_users: set[int] = set(allowed_users or [])
        self._workspace_dir = workspace_dir or Path.home() / ".jarvis" / "workspace" / "telegram"
        self._max_reconnect = max_reconnect_attempts
        self._session_store = session_store
        self._handler: MessageHandler | None = None
        self._app: Any | None = None  # telegram.ext.Application
        self._approval_events: dict[str, asyncio.Event] = {}
        self._approval_results: dict[str, bool] = {}
        self._approval_lock = asyncio.Lock()
        self._session_chat_map: TTLDict[str, int] = TTLDict(max_size=10000, ttl_seconds=86400)
        self._user_chat_map: TTLDict[str, int] = TTLDict(max_size=10000, ttl_seconds=86400)
        self._running = False
        self._typing_tasks: TTLDict[int, asyncio.Task[None]] = TTLDict(
            max_size=1000, ttl_seconds=300
        )
        self._circuit_breaker = CircuitBreaker(
            name="telegram_api", failure_threshold=5, recovery_timeout=60.0
        )
        self._whisper_model: Any | None = None

        # Webhook configuration
        self._use_webhook = use_webhook
        self._webhook_url = webhook_url
        self._webhook_port = webhook_port
        self._webhook_host = webhook_host
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._webhook_runner: Any | None = None  # aiohttp.web.AppRunner
        # Secret token for webhook verification (per Telegram Bot API)
        import secrets as _secrets

        self._webhook_secret_token: str = _secrets.token_hex(32)

    @property
    def token(self) -> str:
        """Bot-API-Token (entschluesselt bei Zugriff)."""
        return self._token_store.retrieve("telegram_bot_token")

    @property
    def name(self) -> str:
        """Eindeutiger Channel-Name."""
        return "telegram"

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Telegram-Bot.

        Args:
            handler: Async-Callback fuer eingehende Nachrichten.
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

        # Load persisted mappings (if store available)
        if self._session_store:
            for key, val in self._session_store.load_all_channel_mappings(
                "telegram_session"
            ).items():
                self._session_chat_map[key] = int(val)
            for key, val in self._session_store.load_all_channel_mappings("telegram_user").items():
                self._user_chat_map[key] = int(val)
            if self._session_chat_map or self._user_chat_map:
                logger.info(
                    "Telegram-Mappings geladen: %d Sessions, %d Users",
                    len(self._session_chat_map),
                    len(self._user_chat_map),
                )

        self._app = Application.builder().token(self.token).concurrent_updates(True).build()

        # Register handlers
        self._app.add_handler(
            TGMessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._on_telegram_message,
            )
        )
        # Voice messages
        self._app.add_handler(
            TGMessageHandler(
                filters.VOICE | filters.AUDIO,
                self._on_voice_message,
            )
        )
        # Photos
        self._app.add_handler(
            TGMessageHandler(
                filters.PHOTO,
                self._on_photo_message,
            )
        )
        # Documents (PDFs, etc.)
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

        if self._use_webhook and self._webhook_url:
            try:
                await self._start_webhook()
                logger.info("Telegram-Bot gestartet (Webhook-Modus)")
            except Exception as wh_exc:
                logger.warning(
                    "Webhook fehlgeschlagen (%s), Fallback auf Polling-Modus",
                    wh_exc,
                )
                if self._app.updater is not None:
                    await self._app.updater.start_polling(drop_pending_updates=True)
                logger.info("Telegram-Bot gestartet (Polling-Fallback)")
        else:
            if self._app.updater is not None:
                await self._app.updater.start_polling(drop_pending_updates=True)
            logger.info("Telegram-Bot gestartet (Polling-Modus)")

        self._running = True

        # Periodic TTLDict cleanup (#47 optimization)
        self._cleanup_task = asyncio.create_task(self._periodic_ttl_cleanup())

    async def _periodic_ttl_cleanup(self) -> None:
        """Periodischer Sweep abgelaufener TTLDict-Eintraege (alle 5 Minuten)."""
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 Minuten
                n1 = self._session_chat_map.purge_expired()
                n2 = self._user_chat_map.purge_expired()
                n3 = self._typing_tasks.purge_expired()
                total = n1 + n2 + n3
                if total > 0:
                    logger.debug("TTLDict cleanup: %d abgelaufene Einträge entfernt", total)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("ttl_cleanup_error", exc_info=True)

    async def stop(self) -> None:
        """Stoppt den Telegram-Bot sauber."""
        if not self._running or self._app is None:
            return

        # Stop cleanup task
        if hasattr(self, "_cleanup_task") and self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        try:
            if self._use_webhook:
                # Unregister webhook at Telegram and stop local server
                with contextlib.suppress(Exception):
                    await self._app.bot.delete_webhook(drop_pending_updates=False)
                if self._webhook_runner:
                    await self._webhook_runner.cleanup()
                    self._webhook_runner = None
            else:
                if self._app.updater is not None:
                    await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            logger.exception("Fehler beim Stoppen des Telegram-Bots")

        self._running = False
        self._app = None
        logger.info("Telegram-Bot gestoppt")

    # === Webhook methods ===

    async def _start_webhook(self) -> None:
        """Startet Webhook-Server und registriert bei Telegram."""
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/telegram/webhook", self._handle_webhook)
        app.router.add_get("/telegram/health", self._handle_health)

        self._webhook_runner = web.AppRunner(app)
        await self._webhook_runner.setup()

        # TLS-Support (optional)
        ssl_ctx = None
        if self._ssl_certfile and self._ssl_keyfile:
            from jarvis.security.token_store import create_ssl_context

            ssl_ctx = create_ssl_context(self._ssl_certfile, self._ssl_keyfile)

        site = web.TCPSite(
            self._webhook_runner,
            self._webhook_host,
            self._webhook_port,
            ssl_context=ssl_ctx,
        )
        await site.start()

        # Register webhook at Telegram
        await self._app.bot.set_webhook(
            url=self._webhook_url,
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"],
            secret_token=self._webhook_secret_token,
        )

        logger.info(
            "Telegram-Webhook gestartet: url=%s port=%d",
            self._webhook_url,
            self._webhook_port,
        )

    async def _handle_webhook(self, request: Any) -> Any:
        """Verarbeitet eingehende Telegram-Updates via Webhook."""
        import hmac as _hmac

        from aiohttp import web
        from telegram import Update

        # Secret token verification (Telegram sends the header automatically)
        received_token = request.headers.get(
            "X-Telegram-Bot-Api-Secret-Token",
            "",
        )
        if not _hmac.compare_digest(received_token, self._webhook_secret_token):
            logger.warning("Telegram-Webhook: Ungueltige oder fehlende Secret-Token-Verifizierung")
            return web.Response(status=403, text="Forbidden")

        try:
            data = await request.json()
            update = Update.de_json(data, self._app.bot)
            await self._app.process_update(update)
            return web.Response(status=200)
        except Exception as exc:
            logger.error("Telegram-Webhook-Fehler: %s", exc)
            return web.Response(status=500)

    async def _handle_health(self, request: Any) -> Any:
        """Health-Check-Endpoint fuer den Webhook-Server."""
        from aiohttp import web

        return web.json_response(
            {
                "status": "ok",
                "channel": "telegram",
                "mode": "webhook",
                "running": self._running,
            }
        )

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
                await self._circuit_breaker.call(
                    self._app.bot.send_message(
                        chat_id=int(chat_id),
                        text=chunk,
                        parse_mode="Markdown",
                    )
                )
            except CircuitBreakerOpen:
                logger.warning("telegram_circuit_open", extra={"chat_id": chat_id})
                return
            except Exception:
                logger.debug(
                    "Markdown-Parsing fehlgeschlagen, Fallback auf Plain-Text", exc_info=True
                )
                # Fallback without Markdown if parsing fails
                try:
                    await self._circuit_breaker.call(
                        self._app.bot.send_message(
                            chat_id=int(chat_id),
                            text=chunk,
                        )
                    )
                except CircuitBreakerOpen:
                    logger.warning("telegram_circuit_open", extra={"chat_id": chat_id})
                    return
                except Exception:
                    logger.exception("Fehler beim Senden an chat_id=%s", chat_id)

        # Send attachments (e.g. generated PDF/DOCX documents)
        for att_path in message.attachments:
            try:
                await self.send_file(int(chat_id), Path(att_path))
            except Exception:
                logger.exception("Fehler beim Senden von Attachment %s", att_path)

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Fragt den User via Inline-Keyboard um Bestaetigung.

        Args:
            session_id: Aktive Session-ID.
            action: Die zu bestaetigende Aktion.
            reason: Begruendung fuer die Bestaetigung.

        Returns:
            True wenn User bestaetigt, False bei Ablehnung oder Timeout.
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

        # Wait for user response (with timeout)
        try:
            await asyncio.wait_for(event.wait(), timeout=APPROVAL_TIMEOUT)
            async with self._approval_lock:
                return self._approval_results.get(approval_id, False)
        except TimeoutError:
            logger.warning("Approval-Timeout für %s", approval_id)
            return False
        finally:
            async with self._approval_lock:
                self._approval_events.pop(approval_id, None)
                self._approval_results.pop(approval_id, None)

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Streaming ist bei Telegram nicht sinnvoll unterstuetzt.

        Telegram hat kein echtes Token-Streaming. Nachrichten werden
        als Ganzes gesendet (via send()).

        Args:
            session_id: Aktive Session-ID.
            token: Einzelnes Token (wird ignoriert).
        """
        # Telegram does not support real streaming.
        # Messages are sent as a whole via send().
        pass

    async def send_status(self, session_id: str, status: StatusType, text: str) -> None:
        """Sendet Typing-Indicator als Status-Feedback."""
        if self._app is None:
            return
        chat_id = self._session_chat_map.get(session_id) or self._user_chat_map.get(session_id)
        if chat_id is None:
            return
        try:
            bot = self._app.bot
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass  # Cleanup — typing indicator failure is non-critical

    # === Internal handlers ===

    async def _on_telegram_message(self, update: Any, context: Any) -> None:
        """Verarbeitet eingehende Telegram-Textnachrichten.

        Prueft User-Whitelist und leitet an den Gateway-Handler weiter.
        """
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        text = update.effective_message.text or ""

        # Whitelist check
        if self.allowed_users and user_id not in self.allowed_users:
            logger.warning("Unerlaubter Zugriff von User %d (Chat %d)", user_id, chat_id)
            await update.effective_message.reply_text(
                "⛔ Zugriff verweigert. Deine User-ID ist nicht autorisiert."
            )
            return

        await self._process_incoming(chat_id, user_id, text, update)

    async def _on_voice_message(self, update: Any, context: Any) -> None:
        """Verarbeitet Sprachnachrichten: Download → Transkription → Gateway.

        Nutzt faster-whisper oder whisper.cpp fuer lokale Transkription.
        Fallback: Nachricht an den User, dass Voice nicht verfuegbar ist.
        """
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if self.allowed_users and user_id not in self.allowed_users:
            return

        # GDPR consent check for voice messages
        consent_mgr = None
        try:
            if hasattr(self, "_handler") and hasattr(self._handler, "__self__"):
                consent_mgr = getattr(self._handler.__self__, "consent_manager", None)
        except Exception:
            pass
        if consent_mgr and consent_mgr.requires_consent(str(user_id), "telegram"):
            await context.bot.send_message(
                chat_id=chat_id,
                text="Bitte sende zuerst eine Textnachricht und akzeptiere den Datenschutzhinweis.",
            )
            return

        voice = update.effective_message.voice or update.effective_message.audio
        if voice is None:
            return

        # Start typing indicator
        typing_task = self._start_typing(chat_id)

        try:
            # Download audio
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            file = await voice.get_file()
            audio_path = self._workspace_dir / f"voice-{voice.file_unique_id}.ogg"
            await file.download_to_drive(str(audio_path))
            logger.info("Voice heruntergeladen: %s (%d bytes)", audio_path, voice.file_size or 0)

            # Attempt transcription
            text = await self._transcribe_audio(audio_path)

            if text:
                # Show transcription to user
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
        """Verarbeitet Fotos: Direkte Vision-Analyse (umgeht Planner).

        Fotos werden direkt mit dem Vision-LLM analysiert, statt den Planner
        entscheiden zu lassen -- das ist zuverlaessiger und schneller.
        Bei komplexen Captions (Follow-Up-Fragen) wird das Analyseergebnis
        als Kontext an den Planner weitergereicht.
        """
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if self.allowed_users and user_id not in self.allowed_users:
            return

        # GDPR consent check for photo messages
        consent_mgr = None
        try:
            if hasattr(self, "_handler") and hasattr(self._handler, "__self__"):
                consent_mgr = getattr(self._handler.__self__, "consent_manager", None)
        except Exception:
            pass
        if consent_mgr and consent_mgr.requires_consent(str(user_id), "telegram"):
            await context.bot.send_message(
                chat_id=chat_id,
                text="Datenschutzhinweis: Bitte sende zuerst eine Textnachricht "
                "und akzeptiere den Datenschutzhinweis bevor du Bilder sendest.",
            )
            return

        photos = update.effective_message.photo
        if not photos:
            return

        # Choose largest photo
        photo = photos[-1]
        caption = update.effective_message.caption or ""

        try:
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            file = await photo.get_file()
            photo_path = self._workspace_dir / f"photo-{photo.file_unique_id}.jpg"
            await file.download_to_drive(str(photo_path))
        except Exception:
            logger.exception("Fehler beim Foto-Download")
            await update.effective_message.reply_text("❌ Fehler beim Empfangen des Fotos.")
            return

        user_question = caption if caption else "Beschreibe dieses Bild detailliert auf Deutsch."

        # Direct vision analysis (no planner needed)
        typing_task = self._start_typing(chat_id)
        try:
            from jarvis.mcp.media import MediaPipeline

            # Vision model from config (auto-adapts based on backend)
            try:
                from jarvis.config import load_config

                _cfg = load_config()
                _vision_model = _cfg.vision_model
                _ollama_url = _cfg.ollama.base_url
                _openai_key = _cfg.openai_api_key or ""
                _openai_base = _cfg.openai_base_url or "https://api.openai.com/v1"
            except Exception:
                _vision_model = "openbmb/minicpm-v4.5"
                _ollama_url = "http://localhost:11434"
                _openai_key = ""
                _openai_base = "https://api.openai.com/v1"

            pipeline = MediaPipeline()
            result = await pipeline.analyze_image(
                str(photo_path),
                prompt=user_question,
                model=_vision_model,
                ollama_url=_ollama_url,
                openai_api_key=_openai_key,
                openai_base_url=_openai_base,
            )

            if result.success and result.text:
                response_text = await self._polish_vision_text(result.text)
            else:
                response_text = f"❌ Bildanalyse fehlgeschlagen: {result.error}"

            # Send response (respect Telegram limit)
            if len(response_text) <= MAX_MESSAGE_LENGTH:
                await update.effective_message.reply_text(response_text)
            else:
                # Split long responses
                for i in range(0, len(response_text), MAX_MESSAGE_LENGTH):
                    chunk = response_text[i : i + MAX_MESSAGE_LENGTH]
                    await update.effective_message.reply_text(chunk)

            logger.info(
                "Foto analysiert für User %d: %s (%s)",
                user_id,
                photo_path,
                _vision_model,
            )
        except Exception:
            logger.exception("Fehler bei Vision-Analyse für User %d", user_id)
            await update.effective_message.reply_text(
                "❌ Bildanalyse fehlgeschlagen. Bitte versuche es erneut."
            )
        finally:
            self._stop_typing(chat_id, typing_task)

    async def _on_document_message(self, update: Any, context: Any) -> None:
        """Verarbeitet Dokumente: Download + Metadaten als Nachricht weiterleiten."""
        if update.effective_message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if self.allowed_users and user_id not in self.allowed_users:
            return

        # GDPR consent check for document messages
        consent_mgr = None
        try:
            if hasattr(self, "_handler") and hasattr(self._handler, "__self__"):
                consent_mgr = getattr(self._handler.__self__, "consent_manager", None)
        except Exception:
            pass
        if consent_mgr and consent_mgr.requires_consent(str(user_id), "telegram"):
            await context.bot.send_message(
                chat_id=chat_id,
                text="Bitte sende zuerst eine Textnachricht und akzeptiere den Datenschutzhinweis.",
            )
            return

        doc = update.effective_message.document
        if doc is None:
            return

        # Check document size limit
        if doc.file_size and doc.file_size > MAX_DOCUMENT_SIZE:
            await update.effective_message.reply_text(
                f"Dokument zu gross "
                f"({doc.file_size // 1_048_576} MB, "
                f"max {MAX_DOCUMENT_SIZE // 1_048_576} MB)"
            )
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
            text = (
                f"[Dokument empfangen: {filename}, "
                f"{size_kb} KB, MIME: {doc.mime_type or 'unbekannt'}]"
            )
            if caption:
                text += f"\nBeschreibung: {caption}"
            text += f"\nGespeichert unter: {doc_path}"

            await self._process_incoming(chat_id, user_id, text, update)

        except Exception:
            logger.exception("Fehler beim Dokument-Download")
            await update.effective_message.reply_text("❌ Fehler beim Empfangen des Dokuments.")

    # === Helper methods ===

    async def _polish_vision_text(
        self,
        raw_text: str,
        *,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
    ) -> str:
        """Nachbearbeitung des Vision-Outputs durch ein Text-LLM.

        Korrigiert Rechtschreibung, Grammatik, Satzbau und Satzzeichen.
        Bei Fehler wird der Rohtext unveraendert zurueckgegeben.
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
                resp = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _VISION_POLISH_PROMPT},
                            {"role": "user", "content": raw_text},
                        ],
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    polished = resp.json().get("message", {}).get("content", "")
                    if polished.strip():
                        logger.info(
                            "Vision-Text nachbearbeitet (%d → %d Zeichen)",
                            len(raw_text),
                            len(polished),
                        )
                        return polished.strip()
        except Exception:
            logger.debug("Vision-Postprocessing fehlgeschlagen, nutze Rohtext", exc_info=True)

        return raw_text

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

        # Save user_id → chat_id mapping BEFORE handler call
        # (so approval requests can find the chat_id, even on the first request)
        self._user_chat_map[str(user_id)] = chat_id
        if self._session_store:
            self._session_store.save_channel_mapping("telegram_user", str(user_id), str(chat_id))

        # GDPR consent check
        consent_mgr = None
        if hasattr(self, "_handler") and self._handler and hasattr(self._handler, "__self__"):
            consent_mgr = getattr(self._handler.__self__, "consent_manager", None)

        if consent_mgr and consent_mgr.requires_consent(str(user_id), "telegram"):
            text_lower = (text or "").strip().lower()
            if text_lower in ("akzeptieren", "accept", "ja", "yes"):
                consent_mgr.grant_consent(
                    str(user_id),
                    "telegram",
                    "data_processing",
                    context=str(chat_id),
                )
                await update.effective_message.reply_text(
                    "Datenschutz-Einwilligung erteilt. Ich bin jetzt bereit!"
                )
                return
            elif text_lower in ("ablehnen", "decline", "nein", "no"):
                await update.effective_message.reply_text(
                    "Ich kann deine Nachrichten ohne Datenschutz-Einwilligung "
                    "nicht verarbeiten. Sende 'akzeptieren' wenn du es dir "
                    "anders ueberlegst."
                )
                return
            else:
                await update.effective_message.reply_text(
                    "Datenschutzhinweis: Ich speichere Nachrichten, Erinnerungen "
                    "und Verarbeitungsprotokolle. Details: cognithor.dev/privacy\n\n"
                    "Antworte mit 'akzeptieren' oder 'ablehnen'."
                )
                return

        # Start typing indicator
        typing_task = self._start_typing(chat_id)

        msg = IncomingMessage(
            channel="telegram",
            user_id=str(user_id),
            text=text,
        )

        try:
            response = await self._handler(msg)

            # Save session → chat_id mapping (for future lookups)
            if response.session_id:
                self._session_chat_map[response.session_id] = chat_id
                if self._session_store:
                    self._session_store.save_channel_mapping(
                        "telegram_session",
                        response.session_id,
                        str(chat_id),
                    )
            enriched = OutgoingMessage(
                channel="telegram",
                text=response.text,
                session_id=response.session_id,
                is_final=response.is_final,
                reply_to=response.reply_to,
                attachments=response.attachments,
                metadata={**response.metadata, "chat_id": str(chat_id)},
            )
            await self.send(enriched)

        except Exception as exc:
            logger.exception("Fehler bei Telegram-Nachricht von User %d", user_id)
            try:
                from jarvis.utils.error_messages import classify_error_for_user

                friendly = classify_error_for_user(exc)
            except Exception:
                friendly = "Ein Fehler ist aufgetreten. Bitte versuche es erneut."
            await update.effective_message.reply_text(f"❌ {friendly}")
        finally:
            self._stop_typing(chat_id, typing_task)

    async def _transcribe_audio(self, audio_path: Path) -> str | None:
        """Transkribiert eine Audiodatei mit faster-whisper (lokal).

        Returns:
            Transkribierter Text oder None wenn nicht verfuegbar.
        """
        try:
            import os

            # Disable CUDA if cuDNN is unavailable (prevents DLL crash)
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
        """Startet den Typing-Indicator fuer einen Chat.

        Telegram setzt den Indicator nach ~5 Sekunden zurueck,
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

        Versucht zuerst das Session→Chat-ID Mapping, dann als Fallback
        das User→Chat-ID Mapping (wichtig nach Neustart, wenn die
        Session-Map noch leer ist).

        Args:
            session_id: Session-ID.

        Returns:
            Chat-ID oder None.
        """
        # Primary: direct session → chat ID mapping
        chat_id = self._session_chat_map.get(session_id)
        if chat_id is not None:
            return chat_id

        # Fallback: if only one user is active (typical 1:1 bot chat),
        # use their chat_id. With multiple active users: no fallback.
        if len(self._user_chat_map) == 1:
            chat_id = next(iter(self._user_chat_map.values()))
            # Backfill mapping for future lookups
            self._session_chat_map[session_id] = chat_id
            logger.debug("chat_id via user_chat_map Fallback gefunden: %d", chat_id)
            return chat_id

        return None


def _split_message(text: str) -> list[str]:
    """Teilt eine Nachricht in Telegram-kompatible Teile.

    Versucht an Zeilenumbruechen zu splitten, nicht mitten in Woertern.

    Args:
        text: Der vollstaendige Nachrichtentext.

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

        # Try to split at a line break
        split_pos = remaining.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1 or split_pos < MAX_MESSAGE_LENGTH // 2:
            # No good line break → split at space
            split_pos = remaining.rfind(" ", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1:
            # No space → hard split
            split_pos = MAX_MESSAGE_LENGTH

        chunks.append(remaining[:split_pos])
        remaining = remaining[split_pos:].lstrip()

    return chunks
