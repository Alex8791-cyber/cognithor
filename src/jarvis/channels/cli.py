"""CLI-Channel: Interaktives Terminal-REPL.

Features:
  - Farbige Ausgabe (via Rich)
  - Token-für-Token Streaming
  - Approval-Workflow ([j/n] im Terminal)
  - Graceful Exit (Ctrl+C, /quit)

Bibel-Referenz: §9.3 (CLI Channel)
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# Farben
COLOR_USER = "bold cyan"
COLOR_JARVIS = "white"
COLOR_TOOL = "dim yellow"
COLOR_ERROR = "bold red"
COLOR_APPROVAL = "bold yellow"
COLOR_INFO = "dim"

BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
  Agent OS v{version} · Lokal · DSGVO-konform
"""


class CliChannel(Channel):
    """Terminal-REPL-Channel. [B§9.3]"""

    def __init__(self, version: str = "0.1.0") -> None:
        """Initialisiert den CLI-Channel mit Prompt-Toolkit."""
        self._handler: MessageHandler | None = None
        self._console = Console()
        self._running = False
        self._version = version
        self._session_id = "cli-session"

    @property
    def name(self) -> str:
        """Gibt den Channel-Namen zurück."""
        return "cli"

    async def start(self, handler: MessageHandler) -> None:
        """Startet die CLI-REPL."""
        self._handler = handler
        self._running = True

        # Banner anzeigen
        self._console.print(
            Panel(
                BANNER.format(version=self._version),
                border_style="cyan",
                expand=False,
            )
        )
        self._console.print("[dim]Tippe eine Nachricht oder /quit zum Beenden.[/dim]\n")

        # REPL-Loop
        while self._running:
            try:
                user_input = await self._read_input()
            except (EOFError, KeyboardInterrupt):
                self._console.print("\n[dim]Auf Wiedersehen![/dim]")
                break

            if user_input is None:
                break

            text = user_input.strip()
            if not text:
                continue

            # Slash-Commands
            if text.startswith("/"):
                should_continue = await self._handle_command(text)
                if not should_continue:
                    break
                continue

            # Nachricht an Gateway senden
            msg = IncomingMessage(
                channel="cli",
                user_id="local",
                text=text,
            )

            try:
                response = await self._handler(msg)
                await self.send(response)
            except Exception as exc:
                self._console.print(f"[{COLOR_ERROR}]Fehler: {exc}[/{COLOR_ERROR}]")
                log.error("cli_handler_error", error=str(exc))

    async def stop(self) -> None:
        """Stoppt die CLI."""
        self._running = False

    async def send(self, message: OutgoingMessage) -> None:
        """Gibt eine Jarvis-Antwort farbig im Terminal aus."""
        if not message.text:
            return

        self._console.print()
        self._console.print(
            "[bold green]Jarvis:[/bold green] ",
            end="",
        )
        self._console.print(message.text)
        self._console.print()

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Fragt den User im Terminal um Erlaubnis.

        Zeigt die geplante Aktion an und wartet auf [j/n].
        """
        self._console.print()
        self._console.print(
            Panel(
                f"[{COLOR_APPROVAL}]🔐 Bestätigung erforderlich[/{COLOR_APPROVAL}]\n\n"
                f"Tool: [bold]{action.tool}[/bold]\n"
                f"Parameter: {action.params}\n"
                f"Grund: {reason}\n"
                f"Begründung: {action.rationale}",
                border_style="yellow",
                title="Gatekeeper",
            )
        )

        while True:
            try:
                answer = await self._read_input(prompt="Erlauben? [j/n]: ")
                if answer is None:
                    return False
                answer = answer.strip().lower()
                if answer in ("j", "ja", "y", "yes"):
                    self._console.print("[green]✓ Erlaubt[/green]")
                    return True
                if answer in ("n", "nein", "no"):
                    self._console.print("[red]✗ Abgelehnt[/red]")
                    return False
                self._console.print("[dim]Bitte 'j' oder 'n' eingeben.[/dim]")
            except (EOFError, KeyboardInterrupt):
                return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Gibt ein einzelnes Token aus (für Streaming)."""
        self._console.print(token, end="", highlight=False)

    async def _read_input(self, prompt: str | None = None) -> str | None:
        """Liest User-Input nicht-blockierend.

        Nutzt asyncio.to_thread um den blockierenden input()-Call
        in den Thread-Pool auszulagern.
        """
        if prompt is None:
            prompt = "Du: "

        try:
            return await asyncio.to_thread(input, prompt)
        except EOFError:
            return None
        except KeyboardInterrupt:
            return None

    async def _handle_command(self, command: str) -> bool:
        """Verarbeitet Slash-Commands.

        Returns:
            True wenn die REPL weiterlaufen soll, False zum Beenden.
        """
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            self._console.print("[dim]Auf Wiedersehen![/dim]")
            return False

        if cmd == "/help":
            self._console.print(
                Panel(
                    "[bold]Verfügbare Befehle:[/bold]\n\n"
                    "/quit     – Jarvis beenden\n"
                    "/help     – Diese Hilfe anzeigen\n"
                    "/status   – Systemstatus anzeigen\n"
                    "/clear    – Bildschirm leeren\n"
                    "/version  – Versionsinformation",
                    border_style="dim",
                    title="Hilfe",
                )
            )
            return True

        if cmd == "/clear":
            self._console.clear()
            return True

        if cmd == "/version":
            self._console.print(f"[dim]Jarvis Agent OS v{self._version}[/dim]")
            return True

        if cmd == "/status":
            self._console.print(
                f"[dim]Status: Aktiv · Channel: CLI · Session: {self._session_id}[/dim]"
            )
            return True

        self._console.print(f"[dim]Unbekannter Befehl: {command}. Tippe /help für Hilfe.[/dim]")
        return True
