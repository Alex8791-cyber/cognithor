"""Live Canvas — Agent-Driven Visual Workspace.

Der Agent kann dynamisch HTML/CSS/JS in ein Canvas-Panel des Clients pushen.
Visualisierungen, Dashboards, interaktive Formulare — alles was der Agent braucht.

WebSocket-Protocol:
  {"type": "canvas_push", "html": "<div>...</div>", "title": "Dashboard"}
  {"type": "canvas_reset"}
  {"type": "canvas_eval", "js": "document.querySelector('#chart').update(data)"}
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Callback-Typ für Canvas-Broadcasts
CanvasBroadcaster = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class CanvasSnapshot:
    """Einzelner Canvas-Zustand."""
    html: str
    title: str = ""
    timestamp: float = 0.0


@dataclass
class CanvasState:
    """Canvas-Zustand pro Session mit Undo/Redo-History."""
    current_html: str = ""
    current_title: str = ""
    history: list[CanvasSnapshot] = field(default_factory=list)
    redo_stack: list[CanvasSnapshot] = field(default_factory=list)
    max_history: int = 50


class CanvasManager:
    """Verwaltet Canvas-Inhalte pro Session.

    Jede Session hat einen eigenen Canvas-Zustand mit History.
    Änderungen werden über einen Broadcaster an verbundene Clients gesendet.
    """

    def __init__(self, broadcaster: CanvasBroadcaster | None = None) -> None:
        self._sessions: dict[str, CanvasState] = {}
        self._broadcaster = broadcaster
        self._lock = asyncio.Lock()

    def _get_state(self, session_id: str) -> CanvasState:
        """Gibt den Canvas-Zustand für eine Session zurück (erstellt bei Bedarf)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = CanvasState()
        return self._sessions[session_id]

    async def push(self, session_id: str, html: str, title: str = "") -> None:
        """Pusht neuen HTML-Inhalt ans Canvas.

        Args:
            session_id: Die Session-ID.
            html: HTML/CSS/JS-Inhalt für das Canvas.
            title: Optionaler Titel für das Canvas-Panel.
        """
        import time

        async with self._lock:
            state = self._get_state(session_id)

            # Aktuellen Zustand in History sichern (für Undo)
            if state.current_html:
                state.history.append(CanvasSnapshot(
                    html=state.current_html,
                    title=state.current_title,
                    timestamp=time.time(),
                ))
                # History begrenzen
                if len(state.history) > state.max_history:
                    state.history = state.history[-state.max_history:]

            # Redo-Stack leeren bei neuem Push
            state.redo_stack.clear()

            # Neuen Zustand setzen
            state.current_html = html
            state.current_title = title

        # An Clients broadcasten
        if self._broadcaster:
            await self._broadcaster(session_id, {
                "type": "canvas_push",
                "html": html,
                "title": title,
            })

        logger.debug("Canvas push: session=%s title=%s len=%d",
                      session_id, title, len(html))

    async def reset(self, session_id: str) -> None:
        """Leert das Canvas einer Session.

        Args:
            session_id: Die Session-ID.
        """
        async with self._lock:
            state = self._get_state(session_id)

            if state.current_html:
                import time
                state.history.append(CanvasSnapshot(
                    html=state.current_html,
                    title=state.current_title,
                    timestamp=time.time(),
                ))

            state.current_html = ""
            state.current_title = ""
            state.redo_stack.clear()

        if self._broadcaster:
            await self._broadcaster(session_id, {"type": "canvas_reset"})

        logger.debug("Canvas reset: session=%s", session_id)

    async def snapshot(self, session_id: str) -> str:
        """Gibt den aktuellen Canvas-Inhalt zurück.

        Args:
            session_id: Die Session-ID.

        Returns:
            Aktueller HTML-Inhalt oder leerer String.
        """
        async with self._lock:
            state = self._get_state(session_id)
            return state.current_html

    async def eval_js(self, session_id: str, js: str) -> None:
        """Führt JavaScript im Canvas-Client aus.

        Args:
            session_id: Die Session-ID.
            js: JavaScript-Code zur Ausführung im Canvas-iframe.
        """
        if self._broadcaster:
            await self._broadcaster(session_id, {
                "type": "canvas_eval",
                "js": js,
            })

        logger.debug("Canvas eval: session=%s len=%d", session_id, len(js))

    async def undo(self, session_id: str) -> str | None:
        """Macht die letzte Canvas-Änderung rückgängig.

        Args:
            session_id: Die Session-ID.

        Returns:
            Der wiederhergestellte HTML-Inhalt oder None wenn keine History.
        """
        async with self._lock:
            state = self._get_state(session_id)
            if not state.history:
                return None

            # Aktuellen Zustand auf Redo-Stack
            import time
            state.redo_stack.append(CanvasSnapshot(
                html=state.current_html,
                title=state.current_title,
                timestamp=time.time(),
            ))

            # Letzten History-Eintrag wiederherstellen
            prev = state.history.pop()
            state.current_html = prev.html
            state.current_title = prev.title

        if self._broadcaster:
            await self._broadcaster(session_id, {
                "type": "canvas_push",
                "html": state.current_html,
                "title": state.current_title,
            })

        return state.current_html

    async def redo(self, session_id: str) -> str | None:
        """Stellt die letzte rückgängig gemachte Änderung wieder her.

        Args:
            session_id: Die Session-ID.

        Returns:
            Der wiederhergestellte HTML-Inhalt oder None wenn kein Redo verfügbar.
        """
        async with self._lock:
            state = self._get_state(session_id)
            if not state.redo_stack:
                return None

            import time
            state.history.append(CanvasSnapshot(
                html=state.current_html,
                title=state.current_title,
                timestamp=time.time(),
            ))

            next_state = state.redo_stack.pop()
            state.current_html = next_state.html
            state.current_title = next_state.title

        if self._broadcaster:
            await self._broadcaster(session_id, {
                "type": "canvas_push",
                "html": state.current_html,
                "title": state.current_title,
            })

        return state.current_html

    def get_title(self, session_id: str) -> str:
        """Gibt den aktuellen Canvas-Titel zurück."""
        state = self._sessions.get(session_id)
        return state.current_title if state else ""

    def has_content(self, session_id: str) -> bool:
        """Prüft ob das Canvas Inhalt hat."""
        state = self._sessions.get(session_id)
        return bool(state and state.current_html)

    def history_count(self, session_id: str) -> int:
        """Gibt die Anzahl der History-Einträge zurück."""
        state = self._sessions.get(session_id)
        return len(state.history) if state else 0

    def cleanup_session(self, session_id: str) -> None:
        """Räumt den Canvas-Zustand einer Session auf."""
        self._sessions.pop(session_id, None)
