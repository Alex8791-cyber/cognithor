"""Interaktive Channel-Erweiterungen: Slash-Commands, State, Fallback.

Stellt bereit:
  - SlashCommand: Definition und Registry von Slash-Commands
  - InteractionState: Zustandsverwaltung für Button/Modal-Interaktionen
  - FallbackRenderer: Text-Fallback für Clients ohne Interaktivität
  - CommandRegistry: Zentrale Verwaltung aller Slash-Commands

Bibel-Referenz: §7 (Channels), §3 (Gateway)
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Slash-Commands
# ============================================================================


class CommandScope(Enum):
    """Wo der Command verfügbar ist."""

    SLACK = "slack"
    DISCORD = "discord"
    ALL = "all"


@dataclass
class SlashCommand:
    """Definition eines Slash-Commands."""

    name: str                  # z.B. "schedule", "briefing", "approve"
    description: str
    usage: str = ""            # z.B. "/jarvis schedule [task] [time]"
    scope: CommandScope = CommandScope.ALL
    agent_id: str = ""         # Leer = alle Agents
    requires_auth: bool = False
    admin_only: bool = False
    cooldown_seconds: int = 0  # Anti-Spam

    # Handler-Info (wird zur Laufzeit gesetzt)
    handler_name: str = ""

    def to_slack_definition(self) -> dict[str, Any]:
        """Generiert Slack Slash-Command Definition."""
        return {
            "command": f"/jarvis_{self.name}" if self.name != "jarvis" else "/jarvis",
            "description": self.description,
            "usage_hint": self.usage,
        }

    def to_discord_definition(self) -> dict[str, Any]:
        """Generiert Discord Application Command Definition."""
        return {
            "name": self.name,
            "description": self.description[:100],  # Discord: max 100 chars
            "type": 1,  # CHAT_INPUT
            "options": [],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "usage": self.usage,
            "scope": self.scope.value,
            "agent_id": self.agent_id,
            "requires_auth": self.requires_auth,
            "admin_only": self.admin_only,
            "cooldown_seconds": self.cooldown_seconds,
        }


class CommandRegistry:
    """Zentrale Verwaltung aller Slash-Commands.

    Registriert Commands, prüft Cooldowns und
    routet Aufrufe an den richtigen Handler.
    """

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._cooldowns: dict[str, float] = {}  # "user:command" → last_used
        self._usage_counts: dict[str, int] = defaultdict(int)
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Registriert Standard-Commands."""
        defaults = [
            SlashCommand(
                name="schedule",
                description="Erstellt oder zeigt Heartbeat-Aufgaben",
                usage="/jarvis schedule [daily|weekly] [task]",
            ),
            SlashCommand(
                name="briefing",
                description="Fordert ein sofortiges Briefing an",
                usage="/jarvis briefing",
            ),
            SlashCommand(
                name="approve",
                description="Genehmigt eine ausstehende Aktion",
                usage="/jarvis approve [action_id]",
                requires_auth=True,
            ),
            SlashCommand(
                name="delegate",
                description="Delegiert eine Aufgabe an einen anderen Agenten",
                usage="/jarvis delegate [agent] [task]",
                requires_auth=True,
            ),
            SlashCommand(
                name="status",
                description="Zeigt den aktuellen System-Status",
                usage="/jarvis status",
            ),
            SlashCommand(
                name="skills",
                description="Zeigt installierte Skills und Updates",
                usage="/jarvis skills [list|update|search]",
            ),
            SlashCommand(
                name="config",
                description="Zeigt oder ändert Konfiguration",
                usage="/jarvis config [show|set key value]",
                admin_only=True,
            ),
        ]
        for cmd in defaults:
            self.register(cmd)

    def register(self, command: SlashCommand) -> None:
        self._commands[command.name] = command

    def unregister(self, name: str) -> bool:
        if name in self._commands:
            del self._commands[name]
            return True
        return False

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def list_commands(self, scope: CommandScope | None = None) -> list[SlashCommand]:
        cmds = list(self._commands.values())
        if scope and scope != CommandScope.ALL:
            cmds = [c for c in cmds if c.scope in (scope, CommandScope.ALL)]
        return cmds

    def check_cooldown(self, user_id: str, command_name: str) -> bool:
        """Prüft ob der Cooldown abgelaufen ist."""
        cmd = self._commands.get(command_name)
        if not cmd or cmd.cooldown_seconds == 0:
            return True

        key = f"{user_id}:{command_name}"
        last = self._cooldowns.get(key, 0)
        return (time.time() - last) >= cmd.cooldown_seconds

    def record_usage(self, user_id: str, command_name: str) -> None:
        """Zeichnet eine Command-Nutzung auf."""
        key = f"{user_id}:{command_name}"
        self._cooldowns[key] = time.time()
        self._usage_counts[command_name] += 1

    def slack_definitions(self) -> list[dict[str, Any]]:
        """Alle Commands als Slack-Definitionen."""
        return [
            c.to_slack_definition()
            for c in self._commands.values()
            if c.scope in (CommandScope.SLACK, CommandScope.ALL)
        ]

    def discord_definitions(self) -> list[dict[str, Any]]:
        """Alle Commands als Discord Application Commands."""
        return [
            c.to_discord_definition()
            for c in self._commands.values()
            if c.scope in (CommandScope.DISCORD, CommandScope.ALL)
        ]

    @property
    def command_count(self) -> int:
        return len(self._commands)

    def usage_stats(self) -> dict[str, int]:
        return dict(self._usage_counts)


# ============================================================================
# Interaction-State: Zustandsverwaltung für Buttons/Modals
# ============================================================================


class InteractionType(Enum):
    BUTTON_CLICK = "button_click"
    MODAL_SUBMIT = "modal_submit"
    SELECT_MENU = "select_menu"
    APPROVAL = "approval"


@dataclass
class InteractionState:
    """Zustand einer laufenden Interaktion.

    Nach einem Button-Klick oder Modal-Submit muss der Agent
    wissen, welche Aktion gemeint ist. Diese Klasse speichert
    den Kontext zwischen Senden und Empfangen.
    """

    interaction_id: str
    interaction_type: InteractionType
    user_id: str
    agent_id: str = ""
    channel: str = ""          # slack, discord, etc.
    message_id: str = ""       # Original-Nachricht
    response_url: str = ""     # URL für verzögerte Antwort
    action_id: str = ""        # Welcher Button/welches Feld
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    resolved: bool = False
    result: Any = None

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def is_pending(self) -> bool:
        return not self.resolved and not self.is_expired

    def resolve(self, result: Any = None) -> None:
        """Löst die Interaktion auf."""
        self.resolved = True
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type.value,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "channel": self.channel,
            "action_id": self.action_id,
            "is_pending": self.is_pending,
            "resolved": self.resolved,
            "created_at": self.created_at.isoformat(),
        }


class InteractionStore:
    """Persistiert temporäre Interaktions-Zustände.

    Jede Button-/Modal-Interaktion bekommt eine eindeutige ID.
    Der Store verwaltet den Lifecycle:
      1. create() beim Senden der interaktiven Nachricht
      2. get() beim Empfangen der Antwort
      3. resolve() beim Verarbeiten der Antwort
      4. cleanup() für abgelaufene States
    """

    DEFAULT_TTL = 3600  # 1 Stunde

    def __init__(self, ttl_seconds: int = DEFAULT_TTL) -> None:
        self._states: dict[str, InteractionState] = {}
        self._by_message: dict[str, list[str]] = defaultdict(list)
        self._ttl = ttl_seconds

    def create(
        self,
        interaction_type: InteractionType,
        user_id: str,
        *,
        agent_id: str = "",
        channel: str = "",
        message_id: str = "",
        response_url: str = "",
        action_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> InteractionState:
        """Erstellt einen neuen Interaktions-Zustand."""
        interaction_id = secrets.token_hex(12)
        from datetime import timedelta
        expires = datetime.now(UTC) + timedelta(seconds=self._ttl)

        state = InteractionState(
            interaction_id=interaction_id,
            interaction_type=interaction_type,
            user_id=user_id,
            agent_id=agent_id,
            channel=channel,
            message_id=message_id,
            response_url=response_url,
            action_id=action_id,
            payload=payload or {},
            expires_at=expires,
        )

        self._states[interaction_id] = state
        if message_id:
            self._by_message[message_id].append(interaction_id)

        return state

    def get(self, interaction_id: str) -> InteractionState | None:
        state = self._states.get(interaction_id)
        if state and state.is_expired:
            return None
        return state

    def get_by_message(self, message_id: str) -> list[InteractionState]:
        """Alle Interaktionen für eine Nachricht."""
        ids = self._by_message.get(message_id, [])
        return [
            self._states[iid]
            for iid in ids
            if iid in self._states and not self._states[iid].is_expired
        ]

    def resolve(self, interaction_id: str, result: Any = None) -> bool:
        """Löst eine Interaktion auf."""
        state = self._states.get(interaction_id)
        if not state or state.is_expired:
            return False
        state.resolve(result)
        return True

    def pending_count(self) -> int:
        return sum(1 for s in self._states.values() if s.is_pending)

    def cleanup(self) -> int:
        """Entfernt abgelaufene Interaktionen."""
        expired = [
            iid for iid, s in self._states.items()
            if s.is_expired or s.resolved
        ]
        for iid in expired:
            state = self._states.pop(iid, None)
            if state and state.message_id:
                msg_ids = self._by_message.get(state.message_id, [])
                if iid in msg_ids:
                    msg_ids.remove(iid)
        return len(expired)

    def stats(self) -> dict[str, Any]:
        states = list(self._states.values())
        return {
            "total": len(states),
            "pending": sum(1 for s in states if s.is_pending),
            "resolved": sum(1 for s in states if s.resolved),
            "expired": sum(1 for s in states if s.is_expired),
        }


# ============================================================================
# Fallback-Renderer: Text-Fallback für Clients ohne Interaktivität
# ============================================================================


class FallbackRenderer:
    """Rendert interaktive Nachrichten als Plain-Text.

    Für Clients ohne Interaktivitäts-Support (E-Mail, SMS,
    einfache Chat-Bots) werden Buttons zu nummerierten Optionen,
    Formulare zu Text-Prompts und Progress-Bars zu ASCII.
    """

    @staticmethod
    def render_buttons(
        text: str,
        buttons: list[dict[str, str]],
    ) -> str:
        """Rendert Buttons als nummerierte Optionen.

        Input: [{"label": "Erlauben", "value": "allow"}, ...]
        Output: "Antworten Sie mit der Nummer:\n1) Erlauben\n2) Ablehnen"
        """
        lines = [text, "", "Antworten Sie mit der Nummer:"]
        for i, btn in enumerate(buttons, 1):
            label = btn.get("label", btn.get("text", f"Option {i}"))
            lines.append(f"  {i}) {label}")
        return "\n".join(lines)

    @staticmethod
    def render_approval(
        action_description: str,
        *,
        approve_word: str = "JA",
        reject_word: str = "NEIN",
    ) -> str:
        """Rendert eine Approval-Anfrage als Text."""
        return (
            f"Genehmigung erforderlich:\n"
            f"{action_description}\n\n"
            f"Antworten Sie mit '{approve_word}' zum Genehmigen "
            f"oder '{reject_word}' zum Ablehnen."
        )

    @staticmethod
    def render_select(
        prompt: str,
        options: list[dict[str, str]],
    ) -> str:
        """Rendert ein Select-Menü als nummerierte Liste."""
        lines = [prompt, ""]
        for i, opt in enumerate(options, 1):
            label = opt.get("label", opt.get("text", f"Option {i}"))
            desc = opt.get("description", "")
            line = f"  {i}) {label}"
            if desc:
                line += f" -- {desc}"
            lines.append(line)
        lines.append("")
        lines.append("Antworten Sie mit der Nummer Ihrer Wahl.")
        return "\n".join(lines)

    @staticmethod
    def render_progress(
        title: str,
        steps: list[dict[str, str]],
    ) -> str:
        """Rendert eine Progress-Anzeige als ASCII.

        steps: [{"name": "Download", "status": "completed"}, ...]
        """
        status_icons = {
            "completed": "✓",
            "running": "►",
            "pending": "○",
            "failed": "✗",
            "skipped": "--",
        }
        lines = [f"=== {title} ==="]
        for step in steps:
            icon = status_icons.get(step.get("status", "pending"), "?")
            lines.append(f"  [{icon}] {step.get('name', '?')}")
        return "\n".join(lines)

    @staticmethod
    def render_form(
        title: str,
        fields: list[dict[str, str]],
    ) -> str:
        """Rendert ein Formular als Text-Prompts."""
        lines = [f"=== {title} ===", "Bitte antworten Sie im Format:"]
        for f in fields:
            label = f.get("label", "")
            required = " (Pflicht)" if f.get("required") else ""
            default = f" [Standard: {f['default']}]" if f.get("default") else ""
            lines.append(f"  {label}{required}{default}: <Ihre Eingabe>")
        return "\n".join(lines)

    @staticmethod
    def parse_numbered_response(response: str, option_count: int) -> int | None:
        """Parst eine nummerierte Antwort.

        Returns:
            0-basierter Index oder None bei ungültiger Eingabe.
        """
        response = response.strip()
        try:
            num = int(response)
            if 1 <= num <= option_count:
                return num - 1
        except ValueError:
            pass
        return None

    @staticmethod
    def parse_approval_response(
        response: str,
        *,
        approve_word: str = "JA",
        reject_word: str = "NEIN",
    ) -> bool | None:
        """Parst eine Approval-Antwort.

        Returns:
            True = genehmigt, False = abgelehnt, None = ungültig.
        """
        normalized = response.strip().upper()
        if normalized == approve_word.upper():
            return True
        if normalized == reject_word.upper():
            return False
        return None
