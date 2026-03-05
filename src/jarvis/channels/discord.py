"""Discord-Channel: Bidirektionale Kommunikation über Discord.

Nutzt discord.py Gateway (WebSocket) für eingehende Nachrichten und
die REST-API für ausgehende. Unterstützt:
  - Eingehende Nachrichten (on_message Event)
  - App-Mentions (@Jarvis)
  - Ausgehende Nachrichten
  - Interaktive Approvals (Reaction-basiert: ✅/❌)
  - Streaming (Buffer → einzelne Nachricht)

Konfiguration:
  - JARVIS_DISCORD_TOKEN: Bot-Token
  - JARVIS_DISCORD_CHANNEL_ID: Standard-Kanal-ID

Abhängigkeiten:
  pip install discord.py

Bibliothek-Referenz: §9.2 (Channel-Interface)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from jarvis.channels.base import Channel, MessageHandler, StatusType
from jarvis.channels.interactive import (
    AdaptiveCard,
    DiscordMessageBuilder,
    ProgressTracker,
)
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.security.token_store import get_token_store
from jarvis.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from jarvis.utils.ttl_dict import TTLDict

if TYPE_CHECKING:
    from jarvis.gateway.session_store import SessionStore

logger = logging.getLogger(__name__)

_DISCORD_MAX_LENGTH = 2000


def _split_discord_message(text: str, limit: int = _DISCORD_MAX_LENGTH) -> list[str]:
    """Splits a message into chunks that fit Discord's character limit.

    Tries to split on newlines first, then on spaces, then hard-cuts.
    """
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            # Try to split at a space
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


class DiscordChannel(Channel):
    """Bidirektionale Discord-Integration für Jarvis.

    Empfängt Nachrichten via Gateway WebSocket, sendet via REST-API,
    und unterstützt interaktive Approvals über Reactions (✅/❌).
    """

    def __init__(
        self,
        token: str,
        channel_id: int,
        session_store: SessionStore | None = None,
    ) -> None:
        self._token_store = get_token_store()
        self._token_store.store("discord_bot_token", token)
        self.channel_id = channel_id
        self._session_store = session_store
        self._client: Any | None = None
        self._handler: MessageHandler | None = None
        self._running = False
        self._bidirectional = False
        self._approval_messages: dict[int, tuple[asyncio.Future[bool], int]] = {}
        self._approval_lock = asyncio.Lock()
        self._session_users: TTLDict[str, int] = TTLDict(max_size=10000, ttl_seconds=86400)
        self._stream_buffers: TTLDict[str, list[str]] = TTLDict(max_size=1000, ttl_seconds=60)
        self._stream_lock = asyncio.Lock()
        self._circuit_breaker = CircuitBreaker(name="discord_api", failure_threshold=5, recovery_timeout=60.0)

    @property
    def token(self) -> str:
        """Bot-Token (entschlüsselt bei Zugriff)."""
        return self._token_store.retrieve("discord_bot_token")

    @property
    def name(self) -> str:
        return "discord"

    @property
    def is_bidirectional(self) -> bool:
        """True wenn Gateway-Verbindung aktiv ist."""
        return self._bidirectional

    async def start(self, handler: MessageHandler) -> None:
        """Startet den Discord-Client mit Event-Handling."""
        self._handler = handler

        # Persistierte Mappings laden
        if self._session_store:
            for key, val in self._session_store.load_all_channel_mappings("discord_session_users").items():
                self._session_users[key] = int(val)

        try:
            import discord  # type: ignore[import-untyped]
        except ImportError:
            logger.error("discord.py nicht installiert. pip install discord.py")
            return

        intents = discord.Intents.default()
        intents.message_content = True  # Privileged Intent für Nachrichteninhalt
        intents.reactions = True  # Für Approval-Reactions

        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready() -> None:
            self._running = True
            self._bidirectional = True
            logger.info(
                "Discord-Client verbunden als %s (bidirektional)",
                client.user,
            )

        @client.event
        async def on_message(message: Any) -> None:
            await self._on_message(message)

        @client.event
        async def on_reaction_add(reaction: Any, user: Any) -> None:
            await self._on_reaction(reaction, user)

        loop = asyncio.get_running_loop()
        loop.create_task(client.start(self.token))

    async def _on_message(self, message: Any) -> None:
        """Verarbeitet eingehende Discord-Nachrichten."""
        # Eigene Nachrichten ignorieren
        if message.author == self._client.user:
            return
        # Bot-Nachrichten ignorieren
        if message.author.bot:
            return

        text = message.content.strip()
        if not text:
            return

        # Bot-Mention entfernen
        if self._client.user:
            mention = f"<@{self._client.user.id}>"
            mention_nick = f"<@!{self._client.user.id}>"
            text = text.replace(mention, "").replace(mention_nick, "").strip()

        # Nur reagieren wenn: DM, im konfigurierten Channel, oder Mention
        is_dm = message.guild is None
        is_target_channel = message.channel.id == self.channel_id
        was_mentioned = self._client.user in message.mentions if self._client.user else False

        if not (is_dm or is_target_channel or was_mentioned):
            return

        session_id = f"discord_{message.author.id}_{message.channel.id}"
        incoming = IncomingMessage(
            channel="discord",
            user_id=str(message.author.id),
            session_id=session_id,
            text=text,
            metadata={
                "channel_id": str(message.channel.id),
                "message_id": str(message.id),
                "guild_id": str(message.guild.id) if message.guild else "",
                "author_name": str(message.author),
            },
        )
        # Session → Discord User-ID Mapping für Approval-Validierung
        self._session_users[session_id] = message.author.id
        if self._session_store:
            self._session_store.save_channel_mapping(
                "discord_session_users", session_id, str(message.author.id),
            )

        if self._handler:
            try:
                async with message.channel.typing():
                    response = await self._handler(incoming)
                for chunk in _split_discord_message(response.text):
                    await message.channel.send(chunk)
            except Exception as exc:
                logger.error("Discord: Handler-Fehler: %s", exc)
                try:
                    from jarvis.utils.error_messages import classify_error_for_user
                    friendly = classify_error_for_user(exc)
                    await message.channel.send(f"❌ {friendly}")
                except Exception:
                    pass  # Can't send error message either — silently log

    # ------------------------------------------------------------------
    # Approvals via Reactions
    # ------------------------------------------------------------------

    async def _on_reaction(self, reaction: Any, user: Any) -> None:
        """Verarbeitet Reaction-basierte Approvals."""
        if user == self._client.user:
            return  # Eigene Reactions ignorieren

        msg_id = reaction.message.id
        async with self._approval_lock:
            entry = self._approval_messages.get(msg_id)
            if not entry:
                return
            future, requester_id = entry
            if future.done():
                # Stale entry — clean up
                del self._approval_messages[msg_id]
                return
            # Nur der ursprüngliche Anfragesteller darf genehmigen/ablehnen
            if requester_id and user.id != requester_id:
                logger.warning(
                    "Discord Approval von fremdem User ignoriert: %s (erwartet: %s)",
                    user.id, requester_id,
                )
                return

        emoji = str(reaction.emoji)
        if emoji == "✅":
            future.set_result(True)
        elif emoji == "❌":
            future.set_result(False)

    # ------------------------------------------------------------------
    # Senden
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        self._running = False
        self._bidirectional = False
        # Cancel all pending approval futures to prevent memory leaks
        async with self._approval_lock:
            for msg_id, (future, _) in list(self._approval_messages.items()):
                if not future.done():
                    future.cancel()
            self._approval_messages.clear()
        if self._client:
            try:
                await self._client.close()
            except Exception:
                logger.exception("Fehler beim Stoppen des Discord-Clients")
            self._client = None
        logger.info("DiscordChannel gestoppt")

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an Discord (mit automatischem Splitting)."""
        client = self._client
        if not client or not self._running:
            logger.warning("DiscordChannel ist nicht einsatzbereit")
            return
        try:
            while not client.is_ready():
                await asyncio.sleep(0.1)

            # Ziel-Channel aus Metadata oder Default
            target_id = int(message.metadata.get("channel_id", self.channel_id))
            channel = client.get_channel(target_id)
            if channel is None:
                logger.error("Unbekannter Discord-Channel: %s", target_id)
                return
            for chunk in _split_discord_message(message.text):
                await self._circuit_breaker.call(channel.send(chunk))
        except CircuitBreakerOpen:
            logger.warning("discord_circuit_open", extra={"channel_id": target_id})
        except Exception:
            logger.exception("Fehler beim Senden über Discord")

    async def send_rich(
        self,
        builder: DiscordMessageBuilder,
        channel_id: int = 0,
    ) -> None:
        """Sendet eine Rich Message (Embed + Components) an Discord."""
        client = self._client
        if not client or not self._running:
            logger.warning("DiscordChannel ist nicht einsatzbereit")
            return

        try:
            while not client.is_ready():
                await asyncio.sleep(0.1)

            target = client.get_channel(channel_id or self.channel_id)
            if target is None:
                return

            msg = builder.build()
            # discord.py erwartet Embed-Objekte, hier senden wir als raw dict
            # In Produktion: Konvertierung zu discord.Embed Objekten
            content = msg.get("content", "")
            await target.send(content or "📊")  # Fallback-Text
        except Exception:
            logger.exception("Fehler beim Rich-Senden über Discord")

    async def send_card(
        self,
        card: AdaptiveCard,
        channel_id: int = 0,
    ) -> None:
        """Sendet eine plattform-übergreifende AdaptiveCard als Discord Embed."""
        client = self._client
        if not client or not self._running:
            return

        try:
            while not client.is_ready():
                await asyncio.sleep(0.1)

            target = client.get_channel(channel_id or self.channel_id)
            if target is None:
                return

            msg = card.to_discord()
            await target.send(msg.get("content", "📋"))
        except Exception:
            logger.exception("Fehler beim Card-Senden über Discord")

    async def send_progress(
        self,
        tracker: ProgressTracker,
        channel_id: int = 0,
    ) -> None:
        """Sendet eine Fortschritts-Anzeige an Discord."""
        client = self._client
        if not client or not self._running:
            return

        try:
            while not client.is_ready():
                await asyncio.sleep(0.1)

            target = client.get_channel(channel_id or self.channel_id)
            if target is None:
                return

            await target.send(f"⏳ Fortschritt: {tracker.percent_complete}%")
        except Exception:
            logger.exception("Fehler beim Progress-Senden über Discord")

    async def request_approval(
        self, session_id: str, action: PlannedAction, reason: str,
    ) -> bool:
        """Fragt den User per Reaction-Buttons um Erlaubnis.

        Sendet eine Nachricht mit ✅ und ❌ Reactions und wartet auf
        die Antwort des Users (Timeout: 5 Minuten).
        """
        if not self._bidirectional or not self._client:
            logger.warning("Discord: Approval nicht möglich (nicht verbunden)")
            return False

        try:
            while not self._client.is_ready():
                await asyncio.sleep(0.1)

            channel = self._client.get_channel(self.channel_id)
            if channel is None:
                logger.error("Discord Approval-Channel nicht gefunden")
                return False

            text = (
                f"🔶 **Genehmigung erforderlich**\n"
                f"**Tool:** `{action.tool}`\n"
                f"**Grund:** {reason}\n"
                f"**Parameter:** ```{str(action.params)[:200]}```\n"
                f"Reagiere mit ✅ oder ❌"
            )

            msg = await channel.send(text)
            await msg.add_reaction("✅")
            await msg.add_reaction("❌")

            loop = asyncio.get_running_loop()
            future: asyncio.Future[bool] = loop.create_future()
            requester_id = self._session_users.get(session_id, 0)
            async with self._approval_lock:
                self._approval_messages[msg.id] = (future, requester_id)

            try:
                return await asyncio.wait_for(future, timeout=300.0)
            except asyncio.TimeoutError:
                logger.warning("Discord Approval Timeout: %s", action.tool)
                return False
            finally:
                async with self._approval_lock:
                    self._approval_messages.pop(msg.id, None)

        except Exception as exc:
            logger.error("Discord Approval fehlgeschlagen: %s", exc)
            return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Buffert Streaming-Tokens und sendet gebündelt."""
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
                    OutgoingMessage(
                        channel=self.name, text=text, session_id=session_id,
                    )
                )

    async def send_status(self, session_id: str, status: StatusType, text: str) -> None:
        """Sendet Typing-Indicator als Status-Feedback in Discord."""
        client = self._client
        if not client or not self._running:
            return
        try:
            target_id = self.channel_id
            channel = client.get_channel(target_id)
            if channel is not None:
                await channel.typing()
        except Exception:
            pass
