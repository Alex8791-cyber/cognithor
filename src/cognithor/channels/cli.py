"""CLI-Channel: Interaktives Terminal-REPL.

Features:
  - Farbige Ausgabe (via Rich)
  - Token-fuer-Token Streaming
  - Approval-Workflow ([j/n] im Terminal)
  - Graceful Exit (Ctrl+C, /quit)

Bibel-Referenz: §9.3 (CLI Channel)
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel

from cognithor.channels.base import Channel, MessageHandler, StatusType
from cognithor.models import IncomingMessage, OutgoingMessage, PlannedAction
from cognithor.utils.logging import get_logger

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
  Agent OS v{version} · Local-first · Privacy-focused
"""


class CliChannel(Channel):
    """Terminal-REPL-Channel. [B§9.3]"""

    def __init__(self, version: str = "0.1.0", config=None, api_port: int = 8741) -> None:
        """Initialisiert den CLI-Channel mit Prompt-Toolkit."""
        self._handler: MessageHandler | None = None
        self._console = Console()
        self._running = False
        self._version = version
        self._session_id = "cli-session"
        self._config = config
        self._api_port = api_port

    @property
    def name(self) -> str:
        """Gibt den Channel-Namen zurueck."""
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
        self._console.print("[dim]Type a message or /quit to exit.[/dim]\n")

        # REPL-Loop
        while self._running:
            try:
                user_input = await self._read_input()
            except (EOFError, KeyboardInterrupt):
                self._console.print("\n[dim]Goodbye![/dim]")
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
                with self._console.status(
                    "[bold cyan]Jarvis is thinking...[/bold cyan]",
                    spinner="dots",
                ):
                    response = await self._handler(msg)
                await self.send(response)
            except Exception as exc:
                try:
                    from cognithor.utils.error_messages import classify_error_for_user

                    friendly = classify_error_for_user(exc)
                except Exception:
                    friendly = f"An error occurred: {exc}"
                self._console.print(f"[{COLOR_ERROR}]{friendly}[/{COLOR_ERROR}]")
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
                f"[{COLOR_APPROVAL}]🔐 Approval required[/{COLOR_APPROVAL}]\n\n"
                f"Tool: [bold]{action.tool}[/bold]\n"
                f"Parameters: {action.params}\n"
                f"Reason: {reason}\n"
                f"Rationale: {action.rationale}",
                border_style="yellow",
                title="Gatekeeper",
            )
        )

        while True:
            try:
                answer = await self._read_input(prompt="Allow? [y/n]: ")
                if answer is None:
                    return False
                answer = answer.strip().lower()
                if answer in ("j", "ja", "y", "yes"):
                    self._console.print("[green]✓ Allowed[/green]")
                    return True
                if answer in ("n", "nein", "no"):
                    self._console.print("[red]✗ Denied[/red]")
                    return False
                self._console.print("[dim]Please enter 'y' or 'n'.[/dim]")
            except (EOFError, KeyboardInterrupt):
                return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Gibt ein einzelnes Token aus (fuer Streaming)."""
        self._console.print(token, end="", highlight=False)

    async def send_status(self, session_id: str, status: StatusType, text: str) -> None:
        """Zeigt einen Status-Text im Terminal an (dim italic)."""
        self._console.print(f"  [{COLOR_INFO} italic]{text}[/{COLOR_INFO} italic]", highlight=False)

    async def _read_input(self, prompt: str | None = None) -> str | None:
        """Liest User-Input nicht-blockierend.

        Nutzt asyncio.to_thread um den blockierenden input()-Call
        in den Thread-Pool auszulagern.
        """
        if prompt is None:
            prompt = "You: "

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
            self._console.print("[dim]Goodbye![/dim]")
            return False

        if cmd == "/help":
            self._console.print(
                Panel(
                    "[bold]Available commands:[/bold]\n\n"
                    "/quit     -- Exit Jarvis / Cognithor beenden\n"
                    "/help     -- Show this help / Diese Hilfe anzeigen\n"
                    "/status   -- Show system status / System-Status\n"
                    "/clear    -- Clear screen / Bildschirm loeschen\n"
                    "/version  -- Version info / Versionsinfo\n"
                    "/config   -- Open config editor / Config-Editor oeffnen\n"
                    "/ui       -- Open web UI in browser / Web-UI im Browser oeffnen",
                    border_style="dim",
                    title="Help",
                )
            )
            return True

        if cmd == "/config":
            try:
                from cognithor.cli.config_tui import launch as _launch_config_tui

                config_path = None
                if self._config is not None:
                    config_path = getattr(self._config, "cognithor_home", None)
                    if config_path is not None:
                        config_path = config_path / "config.yaml"
                # config_tui uses prompt_toolkit's sync API — running it in a
                # separate thread avoids the "event loop already running" error
                # from calling pt_prompt() inside this async handler.
                await asyncio.to_thread(_launch_config_tui, config_path)
                self._console.print(
                    f"[{COLOR_INFO}]Config geschlossen. Fortfahren mit Chat.[/{COLOR_INFO}]"
                )
            except Exception as exc:
                self._console.print(f"[{COLOR_ERROR}]Config TUI error: {exc}[/{COLOR_ERROR}]")
            return True

        if cmd == "/ui":
            import webbrowser

            try:
                port = self._api_port
                url = f"http://localhost:{port}"
                webbrowser.open(url)
                self._console.print(f"[{COLOR_INFO}]Browser geoeffnet: {url}[/{COLOR_INFO}]")
            except Exception as exc:
                self._console.print(f"[{COLOR_ERROR}]Failed to open browser: {exc}[/{COLOR_ERROR}]")
            return True

        if cmd == "/clear":
            self._console.clear()
            return True

        if cmd == "/version":
            self._console.print(f"[dim]Jarvis Agent OS v{self._version}[/dim]")
            return True

        if cmd == "/status":
            self._console.print(
                f"[dim]Status: Active · Channel: CLI · Session: {self._session_id}[/dim]"
            )
            return True

        self._console.print(f"[dim]Unknown command: {command}. Type /help for help.[/dim]")
        return True
