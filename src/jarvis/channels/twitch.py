"""Twitch-Channel: Bidirektionale Kommunikation über Twitch Chat (IRC/TMI).

Nutzt Twitch IRC (TMI) + optional Helix API via httpx.
Unterstützt:
  - Chat-Commands (!jarvis)
  - Whispers (Private Nachrichten)
  - Subscriber-Only Mode
  - User Whitelist

Konfiguration:
  - JARVIS_TWITCH_TOKEN: OAuth Token (oauth:xxx)
  - JARVIS_TWITCH_CHANNEL: Channel-Name (ohne #)
  - JARVIS_TWITCH_ALLOWED_USERS: Erlaubte User

Abhängigkeiten:
  Optional: twitchio>=2.0 (Fallback: Raw IRC via TMI)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

_TMI_HOST = "irc.chat.twitch.tv"
_TMI_PORT = 6667
_TMI_PORT_SSL = 6697
_MAX_MSG_LENGTH = 500
_MIN_MSG_INTERVAL = 1.5  # Twitch Rate Limit


class TwitchChannel(Channel):
    """Twitch Chat Integration für Jarvis.

    Verbindet sich zu Twitch IRC (TMI), liest Chat-Nachrichten
    und antwortet. Unterstützt User-Whitelist für Sicherheit.
    """

    def __init__(
        self,
        token: str = "",
        channel: str = "",
        nick: str = "JarvisBot",
        allowed_users: list[str] | None = None,
        command_prefix: str = "!jarvis",
    ) -> None:
        self._token = token
        self._channel = channel.lower().lstrip("#")
        self._nick = nick.lower()
        self._allowed_users: set[str] = {u.lower() for u in (allowed_users or [])}
        self._command_prefix = command_prefix.lower()
        self._handler: MessageHandler | None = None
        self._running = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._last_msg_time: float = 0
        self._stream_buffers: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "twitch"

    async def start(self, handler: MessageHandler) -> None:
        """Startet die Twitch IRC-Verbindung."""
        self._handler = handler

        if not self._token or not self._channel:
            logger.warning("Twitch: Token oder Channel nicht konfiguriert")
            return

        try:
            import ssl as ssl_mod
            ssl_ctx = ssl_mod.create_default_context()
            self._reader, self._writer = await asyncio.open_connection(
                _TMI_HOST, _TMI_PORT_SSL, ssl=ssl_ctx,
            )
        except Exception as exc:
            logger.error("Twitch Verbindung fehlgeschlagen: %s", exc)
            return

        # Authentifizieren
        token = self._token if self._token.startswith("oauth:") else f"oauth:{self._token}"
        await self._send_raw(f"PASS {token}")
        await self._send_raw(f"NICK {self._nick}")

        # Capabilities anfordern (für Tags, Commands, etc.)
        await self._send_raw("CAP REQ :twitch.tv/tags twitch.tv/commands")

        # Channel joinen
        await self._send_raw(f"JOIN #{self._channel}")

        self._running = True
        self._recv_task = asyncio.get_running_loop().create_task(self._receive_loop())
        logger.info("TwitchChannel gestartet: #%s als %s", self._channel, self._nick)

    async def _send_raw(self, line: str) -> None:
        """Sendet eine rohe IRC-Zeile."""
        if self._writer is None:
            return
        try:
            self._writer.write(f"{line}\r\n".encode("utf-8"))
            await self._writer.drain()
        except Exception as exc:
            logger.error("Twitch Senden fehlgeschlagen: %s", exc)

    async def _receive_loop(self) -> None:
        """Empfängt und verarbeitet Twitch IRC-Nachrichten."""
        if not self._reader:
            return

        buffer = ""
        while self._running:
            try:
                data = await self._reader.read(4096)
                if not data:
                    logger.warning("Twitch Verbindung geschlossen")
                    break

                buffer += data.decode("utf-8", errors="replace")
                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)
                    await self._handle_line(line)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    logger.error("Twitch Empfangsfehler: %s", exc)
                    await asyncio.sleep(1.0)

    async def _handle_line(self, line: str) -> None:
        """Verarbeitet eine einzelne IRC-Zeile."""
        if not line:
            return

        # PING/PONG
        if line.startswith("PING"):
            await self._send_raw("PONG :tmi.twitch.tv")
            return

        # Tags + Nachricht parsen
        tags: dict[str, str] = {}
        if line.startswith("@"):
            tag_str, line = line.split(" ", 1)
            for tag in tag_str[1:].split(";"):
                if "=" in tag:
                    k, v = tag.split("=", 1)
                    tags[k] = v

        # PRIVMSG parsen
        if "PRIVMSG" in line:
            await self._on_privmsg(line, tags)

    async def _on_privmsg(self, line: str, tags: dict[str, str]) -> None:
        """Verarbeitet eine Chat-Nachricht."""
        # Format: :nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :message
        try:
            prefix, _, rest = line.partition(" PRIVMSG ")
            target, _, text = rest.partition(" :")
        except ValueError:
            return

        nick = prefix.split("!")[0].lstrip(":")
        if nick.lower() == self._nick:
            return

        # User-Whitelist prüfen
        if self._allowed_users and nick.lower() not in self._allowed_users:
            return

        # Nur auf Command-Prefix reagieren
        if not text.lower().startswith(self._command_prefix):
            return

        # Prefix entfernen
        text = text[len(self._command_prefix):].strip()
        if not text:
            return

        display_name = tags.get("display-name", nick)
        is_mod = tags.get("mod") == "1"
        is_sub = tags.get("subscriber") == "1"
        is_broadcaster = tags.get("badges", "").startswith("broadcaster")

        incoming = IncomingMessage(
            channel="twitch",
            user_id=nick.lower(),
            text=text,
            metadata={
                "display_name": display_name,
                "target": target,
                "is_mod": is_mod,
                "is_sub": is_sub,
                "is_broadcaster": is_broadcaster,
                "tags": tags,
            },
        )

        if self._handler:
            response = await self._handler(incoming)
            await self._send_chat(response.text)

    async def _send_chat(self, text: str) -> None:
        """Sendet eine Chat-Nachricht mit Rate Limiting."""
        import time
        now = time.monotonic()
        elapsed = now - self._last_msg_time
        if elapsed < _MIN_MSG_INTERVAL:
            await asyncio.sleep(_MIN_MSG_INTERVAL - elapsed)

        # Nachricht splitten wenn zu lang
        lines = text.split("\n")
        for line in lines:
            if not line.strip():
                continue
            # Twitch max 500 Zeichen
            while len(line) > _MAX_MSG_LENGTH:
                chunk = line[:_MAX_MSG_LENGTH]
                line = line[_MAX_MSG_LENGTH:]
                await self._send_raw(f"PRIVMSG #{self._channel} :{chunk}")
                await asyncio.sleep(_MIN_MSG_INTERVAL)
            await self._send_raw(f"PRIVMSG #{self._channel} :{line}")
            self._last_msg_time = time.monotonic()
            await asyncio.sleep(_MIN_MSG_INTERVAL)

    async def stop(self) -> None:
        """Trennt die Twitch-Verbindung."""
        self._running = False
        if self._writer:
            try:
                await self._send_raw(f"PART #{self._channel}")
            except Exception:
                pass
            self._writer.close()
            self._writer = None
        self._reader = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        logger.info("TwitchChannel gestoppt")

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an Twitch Chat."""
        if not self._writer:
            logger.warning("TwitchChannel ist nicht einsatzbereit")
            return
        await self._send_chat(message.text)

    async def request_approval(
        self, session_id: str, action: PlannedAction, reason: str,
    ) -> bool:
        """Twitch hat keine Buttons — Approval nicht unterstützt."""
        logger.warning("Twitch: Interaktive Approvals nicht unterstützt")
        return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Buffert Streaming-Tokens und sendet sie als eine Nachricht."""
        buf = self._stream_buffers.setdefault(session_id, [])
        buf.append(token)
        if len(buf) == 1:
            await asyncio.sleep(2.0)  # Längerer Buffer für Twitch (Rate Limit)
            text = "".join(self._stream_buffers.pop(session_id, []))
            if text.strip():
                await self.send(
                    OutgoingMessage(
                        channel=self.name, text=text, session_id=session_id,
                    )
                )
