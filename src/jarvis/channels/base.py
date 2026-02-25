"""Abstract base class for all Jarvis channels.

A channel is a communication link between the user and the gateway.
Every channel must implement this interface.

Bible reference: §9.2 (Channel Interface)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

# Callback-Typ: Empfängt eine IncomingMessage, gibt OutgoingMessage zurück
MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, OutgoingMessage]]


class Channel(ABC):
    """Abstract base for all communication channels. [B§9.2]"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Name des Channels (z.B. 'cli', 'telegram')."""
        ...

    @abstractmethod
    async def start(self, handler: MessageHandler) -> None:
        """Startet den Channel und registriert den Message-Handler.

        Args:
            handler: Async-Funktion die eingehende Nachrichten verarbeitet.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stoppt den Channel sauber."""
        ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> None:
        """Sendet eine Nachricht an den User.

        Args:
            message: Die zu sendende Nachricht.
        """
        ...

    @abstractmethod
    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Fragt den User um Erlaubnis für eine ORANGE-Aktion.

        Args:
            session_id: Aktive Session-ID
            action: Die Aktion die bestätigt werden soll
            reason: Warum Bestätigung nötig ist

        Returns:
            True wenn User bestätigt, False wenn abgelehnt.
        """
        ...

    @abstractmethod
    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Sendet ein einzelnes Token (für Streaming-Ausgabe).

        Args:
            session_id: Aktive Session-ID
            token: Einzelnes Token (Text-Fragment)
        """
        ...
