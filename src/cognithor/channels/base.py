"""Abstract base class for all Cognithor channels.

A channel is a communication link between the user and the gateway.
Every channel must implement this interface.

Bible reference: §9.2 (Channel Interface)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from enum import StrEnum
from typing import Any

from cognithor.models import IncomingMessage, OutgoingMessage, PlannedAction

# Callback-Typ: Empfaengt eine IncomingMessage, gibt OutgoingMessage zurueck
MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, OutgoingMessage]]


class StatusType(StrEnum):
    """Status types for progress feedback during processing."""

    THINKING = "thinking"
    SEARCHING = "searching"
    EXECUTING = "executing"
    RETRYING = "retrying"
    PROCESSING = "processing"
    FINISHING = "finishing"


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
        """Fragt den User um Erlaubnis fuer eine ORANGE-Aktion.

        Args:
            session_id: Aktive Session-ID
            action: Die Aktion die bestaetigt werden soll
            reason: Warum Bestaetigung noetig ist

        Returns:
            True wenn User bestaetigt, False wenn abgelehnt.
        """
        ...

    @abstractmethod
    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Sendet ein einzelnes Token (fuer Streaming-Ausgabe).

        Args:
            session_id: Aktive Session-ID
            token: Einzelnes Token (Text-Fragment)
        """
        ...

    async def send_status(self, session_id: str, status: StatusType, text: str) -> None:
        """Sendet eine Status-Meldung an den User (z.B. 'Denke nach...').

        Default: no-op. Channels koennen dies ueberschreiben.

        Args:
            session_id: Aktive Session-ID
            status: Art des Status (THINKING, SEARCHING, etc.)
            text: Menschenlesbarer Status-Text
        """

    async def send_pipeline_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Sendet ein Pipeline-Event fuer die PGE-Visualisierung.

        Default: no-op. Nur WebUI implementiert dies.
        """

    async def send_plan_detail(self, session_id: str, plan_data: dict[str, Any]) -> None:
        """Plan detail for UI Plan Review panel.

        Default: no-op. Only WebUI implements this.
        """

    async def send_identity_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Identity state update — default no-op."""
