"""Interaktive UI-Komponenten für Slack und Discord.

Erweitert die Sende-Only-Channels um:
  - SlackMessageBuilder: Block Kit Rich Messages, Modals, Forms
  - DiscordMessageBuilder: Embeds, Buttons, Select-Menus
  - ProgressTracker: Fortschritts-Updates in Echtzeit
  - AdaptiveCards: Plattform-übergreifende strukturierte Nachrichten

Bibel-Referenz: §9 (Gateway & Channels)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Gemeinsame Typen
# ============================================================================


class ButtonStyle(Enum):
    PRIMARY = "primary"
    DANGER = "danger"
    DEFAULT = "default"


class FieldType(Enum):
    TEXT = "text"
    EMAIL = "email"
    NUMBER = "number"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    DATE = "date"
    CHECKBOX = "checkbox"


@dataclass
class FormField:
    """Ein Eingabefeld in einem interaktiven Formular."""

    name: str
    label: str
    field_type: FieldType = FieldType.TEXT
    placeholder: str = ""
    required: bool = False
    options: list[dict[str, str]] = field(default_factory=list)  # [{text, value}]
    default_value: str = ""
    max_length: int = 500

    def to_slack_block(self) -> dict[str, Any]:
        """Konvertiert zu Slack Block Kit Input-Block."""
        element: dict[str, Any]
        if self.field_type == FieldType.SELECT:
            element = {
                "type": "static_select",
                "action_id": f"field_{self.name}",
                "placeholder": {"type": "plain_text", "text": self.placeholder or self.label},
                "options": [
                    {"text": {"type": "plain_text", "text": o["text"]}, "value": o["value"]}
                    for o in self.options
                ],
            }
        elif self.field_type == FieldType.MULTI_SELECT:
            element = {
                "type": "multi_static_select",
                "action_id": f"field_{self.name}",
                "placeholder": {"type": "plain_text", "text": self.placeholder or self.label},
                "options": [
                    {"text": {"type": "plain_text", "text": o["text"]}, "value": o["value"]}
                    for o in self.options
                ],
            }
        elif self.field_type == FieldType.DATE:
            element = {
                "type": "datepicker",
                "action_id": f"field_{self.name}",
            }
        elif self.field_type == FieldType.CHECKBOX:
            element = {
                "type": "checkboxes",
                "action_id": f"field_{self.name}",
                "options": [
                    {"text": {"type": "plain_text", "text": o["text"]}, "value": o["value"]}
                    for o in self.options
                ],
            }
        else:
            element = {
                "type": "plain_text_input",
                "action_id": f"field_{self.name}",
                "placeholder": {"type": "plain_text", "text": self.placeholder},
                "multiline": self.max_length > 200,
            }

        return {
            "type": "input",
            "block_id": f"block_{self.name}",
            "label": {"type": "plain_text", "text": self.label},
            "element": element,
            "optional": not self.required,
        }

    def to_discord_component(self) -> dict[str, Any]:
        """Konvertiert zu Discord-Komponente."""
        if self.field_type in (FieldType.SELECT, FieldType.MULTI_SELECT):
            return {
                "type": 3,  # SELECT_MENU
                "custom_id": f"field_{self.name}",
                "placeholder": self.placeholder or self.label,
                "options": [
                    {"label": o["text"], "value": o["value"]}
                    for o in self.options[:25]
                ],
                "min_values": 1 if self.required else 0,
                "max_values": len(self.options) if self.field_type == FieldType.MULTI_SELECT else 1,
            }
        return {
            "type": 4,  # TEXT_INPUT
            "custom_id": f"field_{self.name}",
            "label": self.label,
            "style": 2 if self.max_length > 200 else 1,  # PARAGRAPH vs SHORT
            "placeholder": self.placeholder,
            "required": self.required,
            "max_length": self.max_length,
        }


# ============================================================================
# SlackMessageBuilder: Block Kit Rich Messages
# ============================================================================


class SlackMessageBuilder:
    """Baut Slack Block Kit Nachrichten.

    Unterstützt:
    - Rich Text mit Markdown
    - Buttons und Actions
    - Sections mit Feldern
    - Modals / Formulare
    - Fortschritts-Anzeigen
    """

    def __init__(self) -> None:
        self._blocks: list[dict[str, Any]] = []
        self._text: str = ""

    def text(self, text: str) -> SlackMessageBuilder:
        """Setzt den Fallback-Text."""
        self._text = text
        return self

    def section(self, text: str, *, accessory: dict[str, Any] | None = None) -> SlackMessageBuilder:
        """Fügt eine Markdown-Section hinzu."""
        block: dict[str, Any] = {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        }
        if accessory:
            block["accessory"] = accessory
        self._blocks.append(block)
        return self

    def fields(self, field_pairs: list[tuple[str, str]]) -> SlackMessageBuilder:
        """Fügt eine Section mit Feldern hinzu (Key-Value-Paare)."""
        fields = []
        for label, value in field_pairs:
            fields.append({"type": "mrkdwn", "text": f"*{label}*\n{value}"})
        self._blocks.append({"type": "section", "fields": fields})
        return self

    def divider(self) -> SlackMessageBuilder:
        self._blocks.append({"type": "divider"})
        return self

    def header(self, text: str) -> SlackMessageBuilder:
        self._blocks.append({
            "type": "header",
            "text": {"type": "plain_text", "text": text},
        })
        return self

    def context(self, elements: list[str]) -> SlackMessageBuilder:
        """Fügt einen Context-Block hinzu (kleine Schrift)."""
        self._blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": e} for e in elements],
        })
        return self

    def button(
        self,
        text: str,
        action_id: str,
        value: str = "",
        style: ButtonStyle = ButtonStyle.DEFAULT,
        url: str = "",
    ) -> SlackMessageBuilder:
        """Fügt einen Standalone-Button hinzu."""
        btn: dict[str, Any] = {
            "type": "button",
            "text": {"type": "plain_text", "text": text},
            "action_id": action_id,
            "value": value or action_id,
        }
        if style != ButtonStyle.DEFAULT:
            btn["style"] = style.value
        if url:
            btn["url"] = url

        # In letzten Actions-Block einfügen oder neuen erstellen
        if self._blocks and self._blocks[-1].get("type") == "actions":
            self._blocks[-1]["elements"].append(btn)
        else:
            self._blocks.append({"type": "actions", "elements": [btn]})
        return self

    def image(self, url: str, alt_text: str, title: str = "") -> SlackMessageBuilder:
        block: dict[str, Any] = {
            "type": "image",
            "image_url": url,
            "alt_text": alt_text,
        }
        if title:
            block["title"] = {"type": "plain_text", "text": title}
        self._blocks.append(block)
        return self

    def progress_bar(self, percent: int, label: str = "") -> SlackMessageBuilder:
        """Simulierte Fortschrittsanzeige mit Emoji-Blöcken."""
        filled = percent // 10
        empty = 10 - filled
        bar = "█" * filled + "░" * empty
        text = f"{label + ': ' if label else ''}`{bar}` {percent}%"
        self._blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        })
        return self

    def build(self) -> dict[str, Any]:
        """Baut die finale Nachricht."""
        msg: dict[str, Any] = {"blocks": list(self._blocks)}
        if self._text:
            msg["text"] = self._text
        return msg

    def build_modal(
        self,
        title: str,
        callback_id: str,
        submit_label: str = "Absenden",
        form_fields: list[FormField] | None = None,
    ) -> dict[str, Any]:
        """Baut ein Slack Modal/Dialog."""
        blocks = list(self._blocks)
        if form_fields:
            blocks.extend(f.to_slack_block() for f in form_fields)

        return {
            "type": "modal",
            "callback_id": callback_id,
            "title": {"type": "plain_text", "text": title[:24]},
            "submit": {"type": "plain_text", "text": submit_label},
            "close": {"type": "plain_text", "text": "Abbrechen"},
            "blocks": blocks,
        }

    @property
    def block_count(self) -> int:
        return len(self._blocks)


# ============================================================================
# DiscordMessageBuilder: Embeds + Components
# ============================================================================


class DiscordColor(Enum):
    """Standard-Farben für Discord-Embeds."""

    SUCCESS = 0x2ECC71
    WARNING = 0xF39C12
    ERROR = 0xE74C3C
    INFO = 0x3498DB
    JARVIS = 0x5865F2  # Discord-Blau


class DiscordMessageBuilder:
    """Baut Discord Rich Messages mit Embeds und Components.

    Unterstützt:
    - Embeds (Titel, Beschreibung, Felder, Footer, Thumbnail)
    - Buttons mit Styles
    - Select-Menus
    - Action-Rows
    - Fortschritts-Anzeigen
    """

    def __init__(self) -> None:
        self._embeds: list[dict[str, Any]] = []
        self._components: list[dict[str, Any]] = []
        self._content: str = ""
        self._current_embed: dict[str, Any] | None = None

    def content(self, text: str) -> DiscordMessageBuilder:
        """Setzt den Text-Content (außerhalb Embed)."""
        self._content = text
        return self

    # --- Embed Builder ---

    def embed(
        self,
        title: str = "",
        description: str = "",
        color: DiscordColor = DiscordColor.JARVIS,
        url: str = "",
    ) -> DiscordMessageBuilder:
        """Startet ein neues Embed."""
        self._finalize_embed()
        self._current_embed = {"color": color.value}
        if title:
            self._current_embed["title"] = title
        if description:
            self._current_embed["description"] = description
        if url:
            self._current_embed["url"] = url
        return self

    def embed_field(
        self,
        name: str,
        value: str,
        inline: bool = True,
    ) -> DiscordMessageBuilder:
        """Fügt ein Feld zum aktuellen Embed hinzu."""
        if not self._current_embed:
            self.embed()
        assert self._current_embed is not None
        if "fields" not in self._current_embed:
            self._current_embed["fields"] = []
        self._current_embed["fields"].append({
            "name": name,
            "value": value,
            "inline": inline,
        })
        return self

    def embed_footer(self, text: str, icon_url: str = "") -> DiscordMessageBuilder:
        if not self._current_embed:
            self.embed()
        assert self._current_embed is not None
        footer: dict[str, Any] = {"text": text}
        if icon_url:
            footer["icon_url"] = icon_url
        self._current_embed["footer"] = footer
        return self

    def embed_thumbnail(self, url: str) -> DiscordMessageBuilder:
        if not self._current_embed:
            self.embed()
        assert self._current_embed is not None
        self._current_embed["thumbnail"] = {"url": url}
        return self

    def embed_author(self, name: str, url: str = "", icon_url: str = "") -> DiscordMessageBuilder:
        if not self._current_embed:
            self.embed()
        assert self._current_embed is not None
        author: dict[str, str] = {"name": name}
        if url:
            author["url"] = url
        if icon_url:
            author["icon_url"] = icon_url
        self._current_embed["author"] = author
        return self

    def embed_timestamp(self) -> DiscordMessageBuilder:
        if not self._current_embed:
            self.embed()
        assert self._current_embed is not None
        from datetime import datetime, timezone
        self._current_embed["timestamp"] = datetime.now(timezone.utc).isoformat()
        return self

    # --- Components ---

    def button(
        self,
        label: str,
        custom_id: str,
        style: ButtonStyle = ButtonStyle.PRIMARY,
        emoji: str = "",
        disabled: bool = False,
        url: str = "",
    ) -> DiscordMessageBuilder:
        """Fügt einen Button hinzu."""
        STYLE_MAP = {
            ButtonStyle.PRIMARY: 1,
            ButtonStyle.DANGER: 4,
            ButtonStyle.DEFAULT: 2,
        }
        btn: dict[str, Any] = {
            "type": 2,  # BUTTON
            "label": label,
            "custom_id": custom_id if not url else None,
            "style": 5 if url else STYLE_MAP.get(style, 2),
        }
        if emoji:
            btn["emoji"] = {"name": emoji}
        if disabled:
            btn["disabled"] = True
        if url:
            btn["url"] = url
            btn.pop("custom_id", None)

        self._add_to_action_row(btn)
        return self

    def select_menu(
        self,
        custom_id: str,
        placeholder: str,
        options: list[dict[str, str]],
        *,
        min_values: int = 1,
        max_values: int = 1,
    ) -> DiscordMessageBuilder:
        """Fügt ein Select-Menu hinzu."""
        menu: dict[str, Any] = {
            "type": 3,  # STRING_SELECT
            "custom_id": custom_id,
            "placeholder": placeholder,
            "options": [
                {"label": o.get("text", o.get("label", "")), "value": o["value"]}
                for o in options[:25]
            ],
            "min_values": min_values,
            "max_values": max_values,
        }
        # Select-Menu braucht eigene Action-Row
        self._components.append({"type": 1, "components": [menu]})
        return self

    def progress_bar(self, percent: int, label: str = "") -> DiscordMessageBuilder:
        """Fortschrittsanzeige als Embed-Feld."""
        filled = percent // 10
        empty = 10 - filled
        bar = "▓" * filled + "░" * empty
        desc = f"{label + ': ' if label else ''}[{bar}] {percent}%"
        if not self._current_embed:
            self.embed(description=desc)
        else:
            self._current_embed["description"] = (
                self._current_embed.get("description", "") + "\n" + desc
            )
        return self

    # --- Build ---

    def build(self) -> dict[str, Any]:
        """Baut die finale Discord-Nachricht."""
        self._finalize_embed()
        msg: dict[str, Any] = {}
        if self._content:
            msg["content"] = self._content
        if self._embeds:
            msg["embeds"] = self._embeds
        if self._components:
            msg["components"] = self._components
        return msg

    def build_modal(
        self,
        title: str,
        custom_id: str,
        form_fields: list[FormField] | None = None,
    ) -> dict[str, Any]:
        """Baut einen Discord Modal (Popup-Formular)."""
        components = []
        if form_fields:
            for f in form_fields[:5]:  # Discord: max 5 Felder
                components.append({
                    "type": 1,  # ACTION_ROW
                    "components": [f.to_discord_component()],
                })
        return {
            "title": title[:45],
            "custom_id": custom_id,
            "components": components,
        }

    @property
    def embed_count(self) -> int:
        count = len(self._embeds)
        if self._current_embed:
            count += 1
        return count

    @property
    def component_count(self) -> int:
        return len(self._components)

    def _finalize_embed(self) -> None:
        if self._current_embed:
            self._embeds.append(self._current_embed)
            self._current_embed = None

    def _add_to_action_row(self, component: dict[str, Any]) -> None:
        """Fügt Component in letzte Action-Row ein oder erstellt neue."""
        if (
            self._components
            and self._components[-1].get("type") == 1
            and len(self._components[-1].get("components", [])) < 5
            and self._components[-1]["components"][0].get("type") == 2  # Nur Buttons zusammen
        ):
            self._components[-1]["components"].append(component)
        else:
            self._components.append({"type": 1, "components": [component]})


# ============================================================================
# ProgressTracker: Echtzeit-Fortschritt
# ============================================================================


@dataclass
class ProgressStep:
    """Ein Schritt in einem Fortschritts-Tracker."""

    name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    message: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def duration_ms(self) -> int:
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at) * 1000)
        return 0

    @property
    def status_emoji(self) -> str:
        return {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️",
        }.get(self.status, "❓")


class ProgressTracker:
    """Multi-Step Progress-Tracker für UI-Channels.

    Erstellt formatierte Fortschritts-Nachrichten für
    Slack (Block Kit) und Discord (Embeds).
    """

    def __init__(self, title: str, steps: list[str]) -> None:
        self._title = title
        self._steps = [ProgressStep(name=s) for s in steps]
        self._current_index = -1
        self._started_at = time.time()

    def start_step(self, index: int | None = None) -> ProgressStep | None:
        """Startet den nächsten (oder angegebenen) Schritt."""
        idx = index if index is not None else self._current_index + 1
        if 0 <= idx < len(self._steps):
            self._current_index = idx
            step = self._steps[idx]
            step.status = "running"
            step.started_at = time.time()
            return step
        return None

    def complete_step(self, index: int | None = None, message: str = "") -> ProgressStep | None:
        """Schließt aktuellen Schritt ab."""
        idx = index if index is not None else self._current_index
        if 0 <= idx < len(self._steps):
            step = self._steps[idx]
            step.status = "completed"
            step.completed_at = time.time()
            step.message = message
            return step
        return None

    def fail_step(self, index: int | None = None, error: str = "") -> ProgressStep | None:
        idx = index if index is not None else self._current_index
        if 0 <= idx < len(self._steps):
            step = self._steps[idx]
            step.status = "failed"
            step.completed_at = time.time()
            step.message = error
            return step
        return None

    def skip_step(self, index: int) -> ProgressStep | None:
        if 0 <= index < len(self._steps):
            self._steps[index].status = "skipped"
            return self._steps[index]
        return None

    @property
    def percent_complete(self) -> int:
        done = sum(1 for s in self._steps if s.status in ("completed", "skipped"))
        return int(done / len(self._steps) * 100) if self._steps else 0

    @property
    def is_complete(self) -> bool:
        return all(s.status in ("completed", "failed", "skipped") for s in self._steps)

    @property
    def has_failures(self) -> bool:
        return any(s.status == "failed" for s in self._steps)

    @property
    def steps(self) -> list[ProgressStep]:
        return list(self._steps)

    def to_slack_blocks(self) -> list[dict[str, Any]]:
        """Generiert Slack Block Kit Blocks für Fortschritts-Anzeige."""
        builder = SlackMessageBuilder()
        builder.header(self._title)

        for step in self._steps:
            line = f"{step.status_emoji}  *{step.name}*"
            if step.message:
                line += f"  --  _{step.message}_"
            if step.duration_ms > 0:
                line += f"  ({step.duration_ms}ms)"
            builder.section(line)

        builder.progress_bar(self.percent_complete, "Fortschritt")
        return builder.build()["blocks"]

    def to_discord_embed(self) -> dict[str, Any]:
        """Generiert Discord Embed für Fortschritts-Anzeige."""
        color = DiscordColor.SUCCESS if self.is_complete and not self.has_failures else (
            DiscordColor.ERROR if self.has_failures else DiscordColor.INFO
        )

        builder = DiscordMessageBuilder()
        builder.embed(title=self._title, color=color)

        for step in self._steps:
            value = step.message or step.status
            if step.duration_ms > 0:
                value += f" ({step.duration_ms}ms)"
            builder.embed_field(
                f"{step.status_emoji} {step.name}",
                value,
                inline=False,
            )

        builder.progress_bar(self.percent_complete)
        builder.embed_footer(f"Fortschritt: {self.percent_complete}%")
        return builder.build()


# ============================================================================
# AdaptiveCard: Plattform-übergreifend
# ============================================================================


class AdaptiveCard:
    """Plattform-übergreifende Rich-Message.

    Kann in Slack Block Kit und Discord Embeds gerendert werden.
    Abstrahiert die Unterschiede zwischen den Plattformen.
    """

    def __init__(self, title: str = "", body: str = "") -> None:
        self._title = title
        self._body = body
        self._fields: list[tuple[str, str]] = []
        self._actions: list[dict[str, Any]] = []
        self._footer: str = ""
        self._color: str = "info"  # info, success, warning, error

    def add_field(self, label: str, value: str) -> AdaptiveCard:
        self._fields.append((label, value))
        return self

    def add_button(
        self,
        label: str,
        action_id: str,
        style: ButtonStyle = ButtonStyle.DEFAULT,
    ) -> AdaptiveCard:
        self._actions.append({"label": label, "action_id": action_id, "style": style})
        return self

    def set_footer(self, text: str) -> AdaptiveCard:
        self._footer = text
        return self

    def set_color(self, color: str) -> AdaptiveCard:
        self._color = color
        return self

    def to_slack(self) -> dict[str, Any]:
        """Rendert als Slack Block Kit Message."""
        builder = SlackMessageBuilder()
        if self._title:
            builder.header(self._title)
        if self._body:
            builder.section(self._body)
        if self._fields:
            builder.fields(self._fields)
        for action in self._actions:
            builder.button(action["label"], action["action_id"], style=action["style"])
        if self._footer:
            builder.context([self._footer])
        return builder.build()

    def to_discord(self) -> dict[str, Any]:
        """Rendert als Discord Embed + Components."""
        COLOR_MAP = {
            "info": DiscordColor.INFO,
            "success": DiscordColor.SUCCESS,
            "warning": DiscordColor.WARNING,
            "error": DiscordColor.ERROR,
        }
        builder = DiscordMessageBuilder()
        builder.embed(
            title=self._title,
            description=self._body,
            color=COLOR_MAP.get(self._color, DiscordColor.JARVIS),
        )
        for label, value in self._fields:
            builder.embed_field(label, value)
        if self._footer:
            builder.embed_footer(self._footer)
        for action in self._actions:
            STYLE_MAP = {
                ButtonStyle.PRIMARY: ButtonStyle.PRIMARY,
                ButtonStyle.DANGER: ButtonStyle.DANGER,
                ButtonStyle.DEFAULT: ButtonStyle.DEFAULT,
            }
            builder.button(
                action["label"],
                action["action_id"],
                style=STYLE_MAP.get(action["style"], ButtonStyle.DEFAULT),
            )
        return builder.build()

    @property
    def field_count(self) -> int:
        return len(self._fields)

    @property
    def action_count(self) -> int:
        return len(self._actions)


# ============================================================================
# Slash-Command Handler
# ============================================================================


@dataclass
class SlashCommand:
    """Definition eines Slash-Commands."""

    name: str               # z.B. "/jarvis", "/approve"
    description: str
    handler: str = ""       # Handler-Funktion-Name
    options: list[dict[str, Any]] = field(default_factory=list)
    channels: list[str] = field(default_factory=lambda: ["slack", "discord"])

    def to_slack(self) -> dict[str, Any]:
        return {
            "command": self.name,
            "description": self.description,
        }

    def to_discord(self) -> dict[str, Any]:
        cmd: dict[str, Any] = {
            "name": self.name.lstrip("/"),
            "description": self.description,
            "type": 1,  # CHAT_INPUT
        }
        if self.options:
            cmd["options"] = self.options
        return cmd


class SlashCommandRegistry:
    """Registriert und dispatcht Slash-Commands.

    Commands wie /jarvis schedule, /jarvis briefing, /approve usw.
    """

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._handlers: dict[str, Any] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Any = None,
        *,
        options: list[dict[str, Any]] | None = None,
    ) -> SlashCommand:
        """Registriert einen neuen Slash-Command."""
        cmd = SlashCommand(
            name=name, description=description,
            handler=handler.__name__ if handler else "",
            options=options or [],
        )
        self._commands[name] = cmd
        if handler:
            self._handlers[name] = handler
        return cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def dispatch(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatcht einen Slash-Command an seinen Handler.

        Returns:
            Response-Dict für den Channel.
        """
        handler = self._handlers.get(name)
        if not handler:
            return {"error": f"Unknown command: {name}", "ephemeral": True}
        try:
            return handler(payload)
        except Exception as exc:
            return {"error": str(exc), "ephemeral": True}

    def list_commands(self) -> list[SlashCommand]:
        return list(self._commands.values())

    @property
    def command_count(self) -> int:
        return len(self._commands)

    def to_slack_manifest(self) -> list[dict[str, Any]]:
        """Generiert Slack Slash-Command-Definitionen."""
        return [c.to_slack() for c in self._commands.values() if "slack" in c.channels]

    def to_discord_commands(self) -> list[dict[str, Any]]:
        """Generiert Discord Application-Command-Definitionen."""
        return [c.to_discord() for c in self._commands.values() if "discord" in c.channels]


# ============================================================================
# Modal/Form Submission Handler
# ============================================================================


@dataclass
class ModalSubmission:
    """Eingereichte Modal-Daten."""

    callback_id: str
    user_id: str
    channel: str  # "slack" oder "discord"
    values: dict[str, Any] = field(default_factory=dict)
    trigger_id: str = ""
    response_url: str = ""


class ModalHandler:
    """Verarbeitet view_submission-Payloads von Slack/Discord.

    Jedes Modal hat einen callback_id, der den Handler bestimmt.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}

    def register(self, callback_id: str, handler: Any) -> None:
        self._handlers[callback_id] = handler

    def handle(self, submission: ModalSubmission) -> dict[str, Any]:
        """Verarbeitet eine Modal-Submission."""
        handler = self._handlers.get(submission.callback_id)
        if not handler:
            return {"error": f"Unknown callback: {submission.callback_id}"}
        try:
            return handler(submission)
        except Exception as exc:
            return {"error": str(exc)}

    @property
    def handler_count(self) -> int:
        return len(self._handlers)

    def has_handler(self, callback_id: str) -> bool:
        return callback_id in self._handlers


# ============================================================================
# Interaction Signature Verification
# ============================================================================


class SignatureVerifier:
    """Verifiziert Signaturen von Slack/Discord Interaction-Payloads.

    Slack: HMAC-SHA256 mit Signing-Secret
    Discord: Ed25519 mit Public Key
    """

    def __init__(
        self,
        slack_signing_secret: str = "",
        discord_public_key: str = "",
    ) -> None:
        self._slack_secret = slack_signing_secret
        self._discord_key = discord_public_key

    def verify_slack(
        self,
        body: bytes,
        timestamp: str,
        signature: str,
    ) -> bool:
        """Verifiziert eine Slack-Signatur (X-Slack-Signature)."""
        if not self._slack_secret:
            return False
        import hashlib
        import hmac

        basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected = "v0=" + hmac.new(
            self._slack_secret.encode(), basestring.encode(), hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def verify_discord(
        self,
        body: bytes,
        timestamp: str,
        signature: str,
    ) -> bool:
        """Verifiziert eine Discord-Signatur (Ed25519)."""
        if not self._discord_key:
            return False
        try:
            from nacl.signing import VerifyKey
            verify_key = VerifyKey(bytes.fromhex(self._discord_key))
            verify_key.verify(timestamp.encode() + body, bytes.fromhex(signature))
            return True
        except Exception:
            return False

    @property
    def has_slack_secret(self) -> bool:
        return bool(self._slack_secret)

    @property
    def has_discord_key(self) -> bool:
        return bool(self._discord_key)


# ============================================================================
# Interaction State Management
# ============================================================================


@dataclass
class InteractionState:
    """Temporärer State für Button-Callbacks und Multi-Step-Workflows."""

    interaction_id: str
    user_id: str
    action_type: str       # "approval", "config", "install", ...
    context: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__("time").time())
    expires_at: float = 0.0
    completed: bool = False
    response_url: str = ""


class InteractionStateStore:
    """Persistiert temporäre Interaction-States.

    Stellt sicher dass nach einem Button-Klick der richtige
    Kontext wiederhergestellt werden kann.
    """

    DEFAULT_TTL = 3600  # 1 Stunde

    def __init__(self, ttl: int = DEFAULT_TTL) -> None:
        self._states: dict[str, InteractionState] = {}
        self._ttl = ttl

    def create(
        self,
        interaction_id: str,
        user_id: str,
        action_type: str,
        context: dict[str, Any] | None = None,
        response_url: str = "",
    ) -> InteractionState:
        import time
        now = time.time()
        state = InteractionState(
            interaction_id=interaction_id,
            user_id=user_id,
            action_type=action_type,
            context=context or {},
            created_at=now,
            expires_at=now + self._ttl,
            response_url=response_url,
        )
        self._states[interaction_id] = state
        return state

    def get(self, interaction_id: str) -> InteractionState | None:
        import time
        state = self._states.get(interaction_id)
        if state and state.expires_at < time.time():
            del self._states[interaction_id]
            return None
        return state

    def complete(self, interaction_id: str) -> bool:
        state = self._states.get(interaction_id)
        if not state:
            return False
        state.completed = True
        return True

    def cleanup_expired(self) -> int:
        """Entfernt abgelaufene States."""
        import time
        now = time.time()
        expired = [k for k, v in self._states.items() if v.expires_at < now]
        for k in expired:
            del self._states[k]
        return len(expired)

    @property
    def state_count(self) -> int:
        return len(self._states)


# ============================================================================
# Fallback-Renderer: Für Clients ohne Interaktivität
# ============================================================================


class FallbackRenderer:
    """Rendert interaktive Inhalte für Nicht-Interaktive-Clients.

    Wandelt Buttons, Formulare und Cards in reinen Text um,
    z.B. für E-Mail-Benachrichtigungen oder einfache Chat-Clients.
    """

    @staticmethod
    def render_card(card: AdaptiveCard) -> str:
        """Rendert eine AdaptiveCard als Plaintext."""
        lines: list[str] = []
        if card._title:
            lines.append(f"=== {card._title} ===")
        if card._body:
            lines.append(card._body)
        for label, value in card._fields:
            lines.append(f"  {label}: {value}")
        if card._actions:
            lines.append("")
            lines.append("Aktionen:")
            for i, action in enumerate(card._actions, 1):
                lines.append(f"  [{i}] {action['label']}")
        if card._footer:
            lines.append(f"\n-- {card._footer}")
        return "\n".join(lines)

    @staticmethod
    def render_progress(tracker: ProgressTracker) -> str:
        """Rendert einen ProgressTracker als Plaintext."""
        lines: list[str] = [f"Fortschritt: {tracker._title}"]
        for step in tracker._steps:
            emoji = {"pending": "○", "running": "►", "completed": "✓",
                     "failed": "✗", "skipped": "--"}.get(step.status, "?")
            lines.append(f"  {emoji} {step.name}")
        return "\n".join(lines)

    @staticmethod
    def render_buttons(buttons: list[dict[str, str]]) -> str:
        """Rendert Buttons als nummerierte Textoptionen."""
        lines = ["Bitte antworten Sie mit der Nummer:"]
        for i, btn in enumerate(buttons, 1):
            lines.append(f"  [{i}] {btn.get('label', btn.get('text', '?'))}")
        return "\n".join(lines)

    @staticmethod
    def render_form(fields: list[FormField]) -> str:
        """Rendert ein Formular als Text-Vorlage."""
        lines = ["Bitte füllen Sie aus:"]
        for f in fields:
            required = " *" if f.required else ""
            lines.append(f"  {f.label}{required}: ___________")
            if f.placeholder:
                lines.append(f"    (Beispiel: {f.placeholder})")
        return "\n".join(lines)
