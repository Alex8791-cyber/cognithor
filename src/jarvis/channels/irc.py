"""IRC-Channel: Bidirektionale Kommunikation über IRC.

Nutzt asyncio-basierte IRC-Verbindung (raw sockets).
Unterstützt:
  - Text-Nachrichten in Channels und Private Messages
  - Channel Join
  - Flood Protection
  - NickServ Authentifizierung

Konfiguration:
  - JARVIS_IRC_SERVER: IRC-Server Hostname
  - JARVIS_IRC_PORT: Port (default 6667)
  - JARVIS_IRC_NICK: Bot-Nick
  - JARVIS_IRC_CHANNELS: Komma-separierte Channel-Liste

Abhängigkeiten:
  Optional: irc>=20.0 (Fallback: Raw asyncio sockets)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

logger = logging.getLogger(__name__)

# Flood Protection: Minimaler Abstand zwischen Nachrichten
_MIN_MSG_INTERVAL = 0.5  # Sekunden
_MAX_MSG_LENGTH = 450  # IRC max per line


class IRCChannel(Channel):
    """IRC Integration für Jarvis.

    Verbindet sich zu einem IRC-Server, joint Channels und
    verarbeitet eingehende Nachrichten. Sendet Antworten mit
    Flood Protection.
    """

    def __init__(
        self,
        server: str = "",
        port: int = 6667,
        nick: str = "JarvisBot",
        channels: list[str] | None = None,
        password: str = "",
        use_ssl: bool = False,
    ) -> None:
        self._server = server
        self._port = port
        self._nick = nick
        self._channels = channels or []
        self._password = password
        self._use_ssl = use_ssl
        self._handler: MessageHandler | None = None
        self._running = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._last_msg_time: float = 0
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._approval_lock = asyncio.Lock()
        self._stream_buffers: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "irc"

    async def start(self, handler: MessageHandler) -> None:
        """Startet die IRC-Verbindung."""
        self._handler = handler

        if not self._server:
            logger.warning("IRC: Server nicht konfiguriert")
            return

        try:
            if self._use_ssl:
                import ssl as ssl_mod
                ssl_ctx = ssl_mod.create_default_context()
                self._reader, self._writer = await asyncio.open_connection(
                    self._server, self._port, ssl=ssl_ctx,
                )
            else:
                self._reader, self._writer = await asyncio.open_connection(
                    self._server, self._port,
                )
        except Exception as exc:
            logger.error("IRC Verbindung fehlgeschlagen: %s", exc)
            return

        # Nick und User setzen
        if self._password:
            await self._send_raw(f"PASS {self._password}")
        await self._send_raw(f"NICK {self._nick}")
        await self._send_raw(f"USER {self._nick} 0 * :Jarvis Bot")

        # Receive-Loop starten
        self._running = True
        self._recv_task = asyncio.get_running_loop().create_task(self._receive_loop())
        logger.info("IRCChannel gestartet: %s:%d als %s",
                     self._server, self._port, self._nick)

    async def _send_raw(self, line: str) -> None:
        """Sendet eine rohe IRC-Zeile."""
        if self._writer is None:
            return
        try:
            self._writer.write(f"{line}\r\n".encode("utf-8"))
            await self._writer.drain()
        except Exception as exc:
            logger.error("IRC Senden fehlgeschlagen: %s", exc)

    async def _receive_loop(self) -> None:
        """Empfängt und verarbeitet IRC-Nachrichten."""
        if not self._reader:
            return

        buffer = ""
        while self._running:
            try:
                data = await self._reader.read(4096)
                if not data:
                    logger.warning("IRC Verbindung geschlossen")
                    break

                buffer += data.decode("utf-8", errors="replace")
                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)
                    await self._handle_line(line)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    logger.error("IRC Empfangsfehler: %s", exc)
                    await asyncio.sleep(1.0)

    async def _handle_line(self, line: str) -> None:
        """Verarbeitet eine einzelne IRC-Zeile."""
        if not line:
            return

        # PING/PONG
        if line.startswith("PING"):
            pong_arg = line[5:] if len(line) > 5 else ""
            await self._send_raw(f"PONG {pong_arg}")
            return

        # Numerics parsen
        parts = line.split(" ", 3)
        if len(parts) < 2:
            return

        prefix = parts[0] if parts[0].startswith(":") else ""
        command = parts[1] if prefix else parts[0]

        # 001 = Welcome → Channels joinen
        if command == "001":
            for channel in self._channels:
                await self._send_raw(f"JOIN {channel}")
            # NickServ Auth
            if self._password:
                await self._send_raw(f"PRIVMSG NickServ :IDENTIFY {self._password}")
            return

        # PRIVMSG = Nachricht
        if command == "PRIVMSG":
            await self._on_privmsg(prefix, parts)
            return

    async def _on_privmsg(self, prefix: str, parts: list[str]) -> None:
        """Verarbeitet eine PRIVMSG."""
        # Prefix: :nick!user@host
        nick = prefix[1:].split("!")[0] if prefix else ""
        if nick == self._nick:
            return

        target = parts[2] if len(parts) > 2 else ""
        text = parts[3][1:] if len(parts) > 3 and parts[3].startswith(":") else ""

        if not text.strip():
            return

        # In Channel: Nur reagieren wenn direkt angesprochen
        is_private = not target.startswith("#")
        if not is_private and not text.lower().startswith(self._nick.lower()):
            return

        # Nick-Prefix entfernen
        if not is_private:
            text = text[len(self._nick):].lstrip(":,").strip()

        reply_target = nick if is_private else target

        incoming = IncomingMessage(
            channel="irc",
            user_id=nick,
            text=text.strip(),
            metadata={
                "target": target,
                "reply_target": reply_target,
                "is_private": is_private,
            },
        )

        if self._handler:
            response = await self._handler(incoming)
            await self._send_message(reply_target, response.text)

    async def _send_message(self, target: str, text: str) -> None:
        """Sendet eine Nachricht mit Flood Protection und Message-Splitting."""
        import time
        now = time.monotonic()
        elapsed = now - self._last_msg_time
        if elapsed < _MIN_MSG_INTERVAL:
            await asyncio.sleep(_MIN_MSG_INTERVAL - elapsed)

        # Lange Nachrichten splitten
        lines = text.split("\n")
        for line in lines:
            while len(line) > _MAX_MSG_LENGTH:
                chunk = line[:_MAX_MSG_LENGTH]
                line = line[_MAX_MSG_LENGTH:]
                await self._send_raw(f"PRIVMSG {target} :{chunk}")
                await asyncio.sleep(_MIN_MSG_INTERVAL)
            if line.strip():
                await self._send_raw(f"PRIVMSG {target} :{line}")
                self._last_msg_time = time.monotonic()
                await asyncio.sleep(_MIN_MSG_INTERVAL)

    async def stop(self) -> None:
        """Trennt die IRC-Verbindung."""
        self._running = False
        if self._writer:
            try:
                await self._send_raw("QUIT :Jarvis shutting down")
            except Exception:
                pass
            self._writer.close()
            self._writer = None
        self._reader = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        logger.info("IRCChannel gestoppt")

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an IRC."""
        target = message.metadata.get("reply_target", "")
        if not target and self._channels:
            target = self._channels[0]
        if not target:
            logger.warning("IRC: Kein Ziel; Nachricht verworfen")
            return
        await self._send_message(target, message.text)

    async def request_approval(
        self, session_id: str, action: PlannedAction, reason: str,
    ) -> bool:
        """IRC hat keine Buttons — Approval via Textnachricht (ja/nein)."""
        logger.warning("IRC: Interaktive Approvals nicht unterstützt, verwende Fallback")
        return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Buffert Streaming-Tokens und sendet sie als eine Nachricht."""
        buf = self._stream_buffers.setdefault(session_id, [])
        buf.append(token)
        if len(buf) == 1:
            await asyncio.sleep(1.0)  # Längerer Buffer für IRC (Flood Protection)
            text = "".join(self._stream_buffers.pop(session_id, []))
            if text.strip():
                await self.send(
                    OutgoingMessage(
                        channel=self.name, text=text, session_id=session_id,
                    )
                )
