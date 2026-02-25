"""Konfigurations-Assistenten (Wizards).

Schritt-für-Schritt-Assistenten für komplexe Konfigurationsthemen:
  - HeartbeatWizard: Heartbeat-Aufgaben erstellen
  - BindingWizard: Routing-Regeln definieren
  - AgentWizard: Agenten-Profile konfigurieren
  - SandboxWizard: Sandbox-Profile einrichten

Jeder Wizard liefert Templates, validiert Eingaben schrittweise
und generiert daraus fertige Konfigurationen.

Bibel-Referenz: §12 (Konfiguration)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Wizard-Framework
# ============================================================================


class WizardStepType(Enum):
    """Eingabe-Typen für Wizard-Schritte."""

    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"          # Dropdown
    MULTI_SELECT = "multi_select"
    CRON = "cron"              # Cron-Expression
    KEY_VALUE = "key_value"    # Schlüssel-Wert-Paare
    CONFIRM = "confirm"        # Zusammenfassung + Bestätigung


@dataclass
class WizardStep:
    """Ein einzelner Schritt im Wizard."""

    step_id: str
    title: str
    description: str
    field_type: WizardStepType
    field_name: str            # Konfig-Schlüssel
    required: bool = True
    default: Any = None
    options: list[dict[str, str]] = field(default_factory=list)
    validation_hint: str = ""  # z.B. "1-1440"
    tooltip: str = ""
    depends_on: str = ""       # Step-ID von der dieser abhängt

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validiert die Eingabe für diesen Schritt."""
        if self.required and value is None:
            return False, f"'{self.title}' ist erforderlich"

        if value is None:
            return True, ""

        if self.field_type == WizardStepType.NUMBER:
            try:
                float(value)
            except (TypeError, ValueError):
                return False, f"'{self.title}' muss eine Zahl sein"

        if self.field_type == WizardStepType.SELECT and self.options:
            valid_values = [o.get("value", o.get("label", "")) for o in self.options]
            if str(value) not in valid_values:
                return False, f"Ungültige Auswahl für '{self.title}'"

        if self.field_type == WizardStepType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"'{self.title}' muss true/false sein"

        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "description": self.description,
            "field_type": self.field_type.value,
            "field_name": self.field_name,
            "required": self.required,
            "default": self.default,
            "options": self.options,
            "validation_hint": self.validation_hint,
            "tooltip": self.tooltip,
            "depends_on": self.depends_on,
        }


@dataclass
class WizardTemplate:
    """Vordefinierte Vorlage für einen Wizard."""

    template_id: str
    name: str
    description: str
    icon: str = ""
    preset_values: dict[str, Any] = field(default_factory=dict)


@dataclass
class WizardResult:
    """Ergebnis eines abgeschlossenen Wizards."""

    wizard_type: str
    values: dict[str, Any]
    config_patch: dict[str, Any]   # Generierte Konfiguration
    warnings: list[str] = field(default_factory=list)
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "wizard_type": self.wizard_type,
            "values": self.values,
            "config_patch": self.config_patch,
            "warnings": self.warnings,
            "valid": self.valid,
        }


class BaseWizard:
    """Basis-Klasse für alle Wizards."""

    wizard_type: str = "base"

    def __init__(self) -> None:
        self._steps: list[WizardStep] = []
        self._templates: list[WizardTemplate] = []
        self._setup_steps()
        self._setup_templates()

    def _setup_steps(self) -> None:
        """Override: Definiert die Schritte."""

    def _setup_templates(self) -> None:
        """Override: Definiert Templates."""

    @property
    def steps(self) -> list[WizardStep]:
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def templates(self) -> list[WizardTemplate]:
        return list(self._templates)

    def get_template(self, template_id: str) -> WizardTemplate | None:
        return next((t for t in self._templates if t.template_id == template_id), None)

    def validate_step(self, step_id: str, value: Any) -> tuple[bool, str]:
        """Validiert einen einzelnen Schritt."""
        step = next((s for s in self._steps if s.step_id == step_id), None)
        if not step:
            return False, f"Schritt '{step_id}' nicht gefunden"
        return step.validate(value)

    def validate_all(self, values: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validiert alle Schritte."""
        errors: list[str] = []
        for step in self._steps:
            val = values.get(step.field_name)
            ok, msg = step.validate(val)
            if not ok:
                errors.append(msg)
        return len(errors) == 0, errors

    def generate_config(self, values: dict[str, Any]) -> WizardResult:
        """Override: Generiert Konfiguration aus Wizard-Eingaben."""
        return WizardResult(
            wizard_type=self.wizard_type,
            values=values,
            config_patch={},
        )

    def apply_template(self, template_id: str) -> dict[str, Any]:
        """Wendet ein Template an und gibt vorausgefüllte Werte zurück."""
        template = self.get_template(template_id)
        if not template:
            return {}
        return dict(template.preset_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wizard_type": self.wizard_type,
            "steps": [s.to_dict() for s in self._steps],
            "templates": [
                {
                    "template_id": t.template_id,
                    "name": t.name,
                    "description": t.description,
                    "icon": t.icon,
                }
                for t in self._templates
            ],
            "step_count": self.step_count,
        }


# ============================================================================
# HeartbeatWizard
# ============================================================================


class HeartbeatWizard(BaseWizard):
    """Schritt-für-Schritt-Assistent für Heartbeat-Aufgaben."""

    wizard_type = "heartbeat"

    def _setup_steps(self) -> None:
        self._steps = [
            WizardStep(
                step_id="hb_enabled",
                title="Heartbeat aktivieren",
                description="Soll Jarvis regelmäßig eigenständig Aufgaben ausführen?",
                field_type=WizardStepType.BOOLEAN,
                field_name="enabled",
                default=True,
                tooltip="Der Heartbeat prüft periodisch, ob Aufgaben anstehen.",
            ),
            WizardStep(
                step_id="hb_interval",
                title="Intervall (Minuten)",
                description="Wie oft soll der Heartbeat laufen?",
                field_type=WizardStepType.NUMBER,
                field_name="interval_minutes",
                default=30,
                validation_hint="1-1440",
                tooltip="Empfohlen: 15-60 Min. Kürzere Intervalle = mehr Ressourcen.",
            ),
            WizardStep(
                step_id="hb_channel",
                title="Kanal",
                description="Über welchen Kanal sollen Ergebnisse gemeldet werden?",
                field_type=WizardStepType.SELECT,
                field_name="channel",
                default="cli",
                options=[
                    {"value": "cli", "label": "CLI (Terminal)"},
                    {"value": "webui", "label": "Web-Dashboard"},
                    {"value": "slack", "label": "Slack"},
                    {"value": "discord", "label": "Discord"},
                    {"value": "telegram", "label": "Telegram"},
                ],
            ),
            WizardStep(
                step_id="hb_checklist",
                title="Checkliste",
                description="Welche Aufgaben soll der Heartbeat prüfen?",
                field_type=WizardStepType.MULTI_SELECT,
                field_name="tasks",
                required=False,
                options=[
                    {"value": "briefing", "label": "Tägliches Briefing"},
                    {"value": "crm_check", "label": "CRM-Auswertung"},
                    {"value": "email_digest", "label": "E-Mail-Zusammenfassung"},
                    {"value": "skill_updates", "label": "Skill-Updates prüfen"},
                    {"value": "security_scan", "label": "Sicherheits-Scan"},
                    {"value": "backup", "label": "Backup-Prüfung"},
                ],
            ),
            WizardStep(
                step_id="hb_confirm",
                title="Bestätigung",
                description="Prüfe die Einstellungen und bestätige.",
                field_type=WizardStepType.CONFIRM,
                field_name="_confirm",
                required=False,
            ),
        ]

    def _setup_templates(self) -> None:
        self._templates = [
            WizardTemplate(
                template_id="daily_briefing",
                name="Tägliches Briefing",
                description="Morgens um 8:00 Zusammenfassung des Tages",
                icon="📋",
                preset_values={
                    "enabled": True,
                    "interval_minutes": 60,
                    "channel": "slack",
                    "tasks": ["briefing", "email_digest"],
                },
            ),
            WizardTemplate(
                template_id="crm_weekly",
                name="Wöchentliche CRM-Auswertung",
                description="Jeden Montag CRM-Daten analysieren",
                icon="📊",
                preset_values={
                    "enabled": True,
                    "interval_minutes": 1440,
                    "channel": "webui",
                    "tasks": ["crm_check"],
                },
            ),
            WizardTemplate(
                template_id="security_watch",
                name="Sicherheits-Wächter",
                description="Alle 15 Minuten Sicherheits-Scans",
                icon="🔒",
                preset_values={
                    "enabled": True,
                    "interval_minutes": 15,
                    "channel": "cli",
                    "tasks": ["security_scan", "skill_updates"],
                },
            ),
        ]

    def generate_config(self, values: dict[str, Any]) -> WizardResult:
        valid, errors = self.validate_all(values)
        interval = values.get("interval_minutes", 30)

        warnings = list(errors)
        if isinstance(interval, (int, float)) and interval < 5:
            warnings.append("Kurze Intervalle (<5 Min.) können Performance beeinträchtigen.")

        config_patch = {
            "heartbeat": {
                "enabled": values.get("enabled", True),
                "interval_minutes": int(interval) if interval else 30,
                "channel": values.get("channel", "cli"),
            },
        }

        return WizardResult(
            wizard_type=self.wizard_type,
            values=values,
            config_patch=config_patch,
            warnings=warnings,
            valid=valid,
        )


# ============================================================================
# BindingWizard
# ============================================================================


class BindingWizard(BaseWizard):
    """Schritt-für-Schritt-Assistent für Binding-Regeln."""

    wizard_type = "binding"

    def _setup_steps(self) -> None:
        self._steps = [
            WizardStep(
                step_id="b_name",
                title="Regel-Name",
                description="Eindeutiger Name für diese Routing-Regel.",
                field_type=WizardStepType.TEXT,
                field_name="name",
                tooltip="Z.B. 'slack_code_requests' oder 'discord_briefing'.",
            ),
            WizardStep(
                step_id="b_agent",
                title="Ziel-Agent",
                description="Welcher Agent soll die Nachrichten verarbeiten?",
                field_type=WizardStepType.SELECT,
                field_name="target_agent",
                options=[
                    {"value": "jarvis", "label": "Jarvis (Standard)"},
                    {"value": "coder", "label": "Coder (Programmierung)"},
                    {"value": "researcher", "label": "Researcher (Recherche)"},
                    {"value": "assistant", "label": "Assistent (Büro)"},
                ],
                tooltip="Der Agent erhält alle Nachrichten die dieser Regel entsprechen.",
            ),
            WizardStep(
                step_id="b_channels",
                title="Quell-Kanäle",
                description="Aus welchen Kanälen sollen Nachrichten geroutet werden?",
                field_type=WizardStepType.MULTI_SELECT,
                field_name="channels",
                required=False,
                options=[
                    {"value": "cli", "label": "CLI"},
                    {"value": "slack", "label": "Slack"},
                    {"value": "discord", "label": "Discord"},
                    {"value": "telegram", "label": "Telegram"},
                    {"value": "webui", "label": "Web UI"},
                ],
                tooltip="Leer = alle Kanäle.",
            ),
            WizardStep(
                step_id="b_prefix",
                title="Befehlspräfix",
                description="Optional: Nachrichten mit diesem Präfix werden geroutet.",
                field_type=WizardStepType.TEXT,
                field_name="command_prefix",
                required=False,
                default="",
                tooltip="Z.B. '/code' oder '!recherche'. Leer = alle Nachrichten.",
            ),
            WizardStep(
                step_id="b_regex",
                title="Regex-Muster",
                description="Optional: Regulärer Ausdruck als Filter.",
                field_type=WizardStepType.TEXT,
                field_name="regex_pattern",
                required=False,
                tooltip="Z.B. '(?i)bug|fehler|error' für Bug-Reports.",
            ),
            WizardStep(
                step_id="b_confirm",
                title="Bestätigung",
                description="Prüfe die Regel und bestätige.",
                field_type=WizardStepType.CONFIRM,
                field_name="_confirm",
                required=False,
            ),
        ]

    def _setup_templates(self) -> None:
        self._templates = [
            WizardTemplate(
                template_id="slash_code",
                name="/code → Coder-Agent",
                description="Alle /code-Befehle an den Coder-Agenten",
                icon="💻",
                preset_values={
                    "name": "slash_code",
                    "target_agent": "coder",
                    "command_prefix": "/code",
                },
            ),
            WizardTemplate(
                template_id="slack_to_assistant",
                name="Slack → Assistent",
                description="Alle Slack-Nachrichten an den Büro-Assistenten",
                icon="💬",
                preset_values={
                    "name": "slack_assistant",
                    "target_agent": "assistant",
                    "channels": ["slack"],
                },
            ),
            WizardTemplate(
                template_id="bug_reports",
                name="Bug-Reports → Coder",
                description="Nachrichten mit 'Bug/Fehler/Error' an den Coder",
                icon="🐛",
                preset_values={
                    "name": "bug_routing",
                    "target_agent": "coder",
                    "regex_pattern": "(?i)bug|fehler|error|crash",
                },
            ),
        ]

    def generate_config(self, values: dict[str, Any]) -> WizardResult:
        valid, errors = self.validate_all(values)

        config_patch = {
            "bindings": [{
                "name": values.get("name", ""),
                "target_agent": values.get("target_agent", "jarvis"),
                "channels": values.get("channels", []),
                "command_prefixes": [values["command_prefix"]] if values.get("command_prefix") else [],
                "regex_pattern": values.get("regex_pattern", ""),
            }],
        }

        return WizardResult(
            wizard_type=self.wizard_type,
            values=values,
            config_patch=config_patch,
            warnings=errors,
            valid=valid,
        )


# ============================================================================
# AgentWizard
# ============================================================================


class AgentWizard(BaseWizard):
    """Schritt-für-Schritt-Assistent für Agenten-Profile."""

    wizard_type = "agent"

    def _setup_steps(self) -> None:
        self._steps = [
            WizardStep(
                step_id="a_name",
                title="Agent-Name",
                description="Eindeutiger Name für den Agenten.",
                field_type=WizardStepType.TEXT,
                field_name="name",
                tooltip="Z.B. 'coder', 'researcher', 'family_assistant'.",
            ),
            WizardStep(
                step_id="a_system",
                title="System-Prompt",
                description="Beschreibung der Persönlichkeit und Aufgabe.",
                field_type=WizardStepType.TEXT,
                field_name="system_prompt",
                tooltip="Definiert wie der Agent sich verhält.",
            ),
            WizardStep(
                step_id="a_model",
                title="Bevorzugtes Modell",
                description="Welches LLM soll der Agent nutzen?",
                field_type=WizardStepType.SELECT,
                field_name="preferred_model",
                default="",
                options=[
                    {"value": "", "label": "Standard (vom ModelRouter)"},
                    {"value": "qwen3:8b", "label": "Qwen 3 8B (schnell)"},
                    {"value": "qwen3:32b", "label": "Qwen 3 32B (ausgewogen)"},
                    {"value": "deepseek-r1:70b", "label": "DeepSeek R1 70B (stark)"},
                    {"value": "codestral:22b", "label": "Codestral 22B (Code)"},
                ],
            ),
            WizardStep(
                step_id="a_tools",
                title="Tool-Zugriff",
                description="Welche Tools darf der Agent verwenden?",
                field_type=WizardStepType.MULTI_SELECT,
                field_name="allowed_tools",
                required=False,
                options=[
                    {"value": "shell", "label": "Shell-Befehle"},
                    {"value": "filesystem", "label": "Dateisystem"},
                    {"value": "web", "label": "Web-Suche/Fetch"},
                    {"value": "browser", "label": "Browser-Steuerung"},
                    {"value": "memory", "label": "Langzeit-Gedächtnis"},
                    {"value": "calendar", "label": "Kalender"},
                    {"value": "email", "label": "E-Mail"},
                ],
                tooltip="Leer = alle Tools erlaubt.",
            ),
            WizardStep(
                step_id="a_sandbox",
                title="Sandbox-Profil",
                description="Wie stark soll der Agent eingeschränkt sein?",
                field_type=WizardStepType.SELECT,
                field_name="sandbox_profile",
                default="standard",
                options=[
                    {"value": "minimal", "label": "Minimal (kein Netzwerk, 256MB RAM)"},
                    {"value": "standard", "label": "Standard (Netzwerk erlaubt, 512MB)"},
                    {"value": "full", "label": "Vollzugriff (8GB RAM, keine Limits)"},
                ],
            ),
            WizardStep(
                step_id="a_delegation",
                title="Delegation",
                description="An welche Agenten darf delegiert werden?",
                field_type=WizardStepType.MULTI_SELECT,
                field_name="can_delegate_to",
                required=False,
                options=[
                    {"value": "jarvis", "label": "Jarvis"},
                    {"value": "coder", "label": "Coder"},
                    {"value": "researcher", "label": "Researcher"},
                ],
            ),
        ]

    def _setup_templates(self) -> None:
        self._templates = [
            WizardTemplate(
                template_id="coder",
                name="Coder-Agent",
                description="Programmierung, Code-Reviews, DevOps",
                icon="💻",
                preset_values={
                    "name": "coder",
                    "system_prompt": "Du bist ein erfahrener Entwickler. Schreibe sauberen, getesteten Code.",
                    "preferred_model": "codestral:22b",
                    "allowed_tools": ["shell", "filesystem", "web"],
                    "sandbox_profile": "standard",
                },
            ),
            WizardTemplate(
                template_id="researcher",
                name="Recherche-Agent",
                description="Web-Recherche, Zusammenfassungen, Faktencheck",
                icon="🔍",
                preset_values={
                    "name": "researcher",
                    "system_prompt": "Du bist ein gründlicher Rechercheur. Prüfe Fakten und fasse zusammen.",
                    "preferred_model": "",
                    "allowed_tools": ["web", "browser", "memory"],
                    "sandbox_profile": "minimal",
                },
            ),
            WizardTemplate(
                template_id="family",
                name="Familien-Assistent",
                description="Termine, Einkaufslisten, Alltagsorganisation",
                icon="👨‍👩‍👧",
                preset_values={
                    "name": "family_assistant",
                    "system_prompt": "Du hilfst bei Familienorganisation: Termine, Einkäufe, Erinnerungen.",
                    "preferred_model": "qwen3:8b",
                    "allowed_tools": ["calendar", "memory"],
                    "sandbox_profile": "minimal",
                },
            ),
        ]

    def generate_config(self, values: dict[str, Any]) -> WizardResult:
        valid, errors = self.validate_all(values)

        sandbox_profiles = {
            "minimal": {"network": "block", "max_memory_mb": 256, "max_processes": 16},
            "standard": {"network": "allow", "max_memory_mb": 512, "max_processes": 64},
            "full": {"network": "allow", "max_memory_mb": 8192, "max_processes": 256},
        }
        profile = sandbox_profiles.get(values.get("sandbox_profile", "standard"), sandbox_profiles["standard"])

        config_patch = {
            "agents": [{
                "name": values.get("name", ""),
                "system_prompt": values.get("system_prompt", ""),
                "preferred_model": values.get("preferred_model", ""),
                "allowed_tools": values.get("allowed_tools"),
                "sandbox_network": profile["network"],
                "sandbox_max_memory_mb": profile["max_memory_mb"],
                "sandbox_max_processes": profile["max_processes"],
                "can_delegate_to": values.get("can_delegate_to", []),
            }],
        }

        return WizardResult(
            wizard_type=self.wizard_type,
            values=values,
            config_patch=config_patch,
            warnings=errors,
            valid=valid,
        )


# ============================================================================
# RBAC: Rollenbasierte Zugriffskontrolle
# ============================================================================


class UserRole(Enum):
    """Benutzerrollen für das Admin-Dashboard."""

    OWNER = "owner"         # Voller Zugriff, kann alles
    ADMIN = "admin"         # Kann Konfiguration ändern
    OPERATOR = "operator"   # Kann Monitoring sehen, Agents steuern
    USER = "user"           # Kann eigene Agents und Tasks sehen
    VIEWER = "viewer"       # Nur Lesezugriff


@dataclass
class Permission:
    """Eine einzelne Berechtigung."""

    resource: str    # z.B. "config", "agents", "skills", "monitoring"
    action: str      # z.B. "read", "write", "delete", "execute"

    @property
    def key(self) -> str:
        return f"{self.resource}:{self.action}"


# Vordefinierte Berechtigungen pro Rolle
ROLE_PERMISSIONS: dict[UserRole, list[Permission]] = {
    UserRole.OWNER: [
        Permission("config", "read"), Permission("config", "write"), Permission("config", "delete"),
        Permission("agents", "read"), Permission("agents", "write"), Permission("agents", "delete"),
        Permission("skills", "read"), Permission("skills", "write"), Permission("skills", "delete"),
        Permission("monitoring", "read"), Permission("monitoring", "write"),
        Permission("users", "read"), Permission("users", "write"), Permission("users", "delete"),
        Permission("credentials", "read"), Permission("credentials", "write"),
        Permission("audit", "read"),
    ],
    UserRole.ADMIN: [
        Permission("config", "read"), Permission("config", "write"),
        Permission("agents", "read"), Permission("agents", "write"),
        Permission("skills", "read"), Permission("skills", "write"),
        Permission("monitoring", "read"), Permission("monitoring", "write"),
        Permission("credentials", "read"), Permission("credentials", "write"),
        Permission("audit", "read"),
    ],
    UserRole.OPERATOR: [
        Permission("config", "read"),
        Permission("agents", "read"), Permission("agents", "execute"),
        Permission("skills", "read"),
        Permission("monitoring", "read"),
        Permission("audit", "read"),
    ],
    UserRole.USER: [
        Permission("agents", "read"), Permission("agents", "execute"),
        Permission("skills", "read"),
        Permission("monitoring", "read"),
    ],
    UserRole.VIEWER: [
        Permission("config", "read"),
        Permission("agents", "read"),
        Permission("monitoring", "read"),
    ],
}


@dataclass
class DashboardUser:
    """Ein Dashboard-Benutzer."""

    user_id: str
    display_name: str
    role: UserRole
    email: str = ""
    agent_scope: list[str] = field(default_factory=list)  # Leer = alle Agents

    def has_permission(self, resource: str, action: str) -> bool:
        """Prüft ob der User eine Berechtigung hat."""
        perms = ROLE_PERMISSIONS.get(self.role, [])
        return any(p.resource == resource and p.action == action for p in perms)

    def can_access_agent(self, agent_id: str) -> bool:
        """Prüft ob der User Zugriff auf einen bestimmten Agenten hat."""
        if self.role in (UserRole.OWNER, UserRole.ADMIN):
            return True  # Admins sehen alles
        if not self.agent_scope:
            return True  # Kein Scope = alle
        return agent_id in self.agent_scope

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "role": self.role.value,
            "email": self.email,
            "agent_scope": self.agent_scope,
            "permissions": [p.key for p in ROLE_PERMISSIONS.get(self.role, [])],
        }


class RBACManager:
    """Verwaltet Benutzer und Rollen für das Admin-Dashboard."""

    def __init__(self) -> None:
        self._users: dict[str, DashboardUser] = {}

    def add_user(
        self,
        user_id: str,
        display_name: str,
        role: UserRole,
        email: str = "",
        agent_scope: list[str] | None = None,
    ) -> DashboardUser:
        user = DashboardUser(
            user_id=user_id,
            display_name=display_name,
            role=role,
            email=email,
            agent_scope=agent_scope or [],
        )
        self._users[user_id] = user
        return user

    def get_user(self, user_id: str) -> DashboardUser | None:
        return self._users.get(user_id)

    def remove_user(self, user_id: str) -> bool:
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False

    def update_role(self, user_id: str, role: UserRole) -> bool:
        user = self._users.get(user_id)
        if not user:
            return False
        user.role = role
        return True

    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Prüft ob ein User eine Berechtigung hat."""
        user = self._users.get(user_id)
        if not user:
            return False
        return user.has_permission(resource, action)

    def list_users(self, role: UserRole | None = None) -> list[DashboardUser]:
        users = list(self._users.values())
        if role:
            users = [u for u in users if u.role == role]
        return users

    @property
    def user_count(self) -> int:
        return len(self._users)

    def roles_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for user in self._users.values():
            counts[user.role.value] = counts.get(user.role.value, 0) + 1
        return counts


# ============================================================================
# WizardRegistry: Alle Wizards zentral
# ============================================================================


class WizardRegistry:
    """Zentrales Register aller verfügbaren Wizards."""

    def __init__(self) -> None:
        self._wizards: dict[str, BaseWizard] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(HeartbeatWizard())
        self.register(BindingWizard())
        self.register(AgentWizard())

    def register(self, wizard: BaseWizard) -> None:
        self._wizards[wizard.wizard_type] = wizard

    def get(self, wizard_type: str) -> BaseWizard | None:
        return self._wizards.get(wizard_type)

    def list_wizards(self) -> list[dict[str, Any]]:
        return [w.to_dict() for w in self._wizards.values()]

    def run_wizard(self, wizard_type: str, values: dict[str, Any]) -> WizardResult | None:
        wizard = self._wizards.get(wizard_type)
        if not wizard:
            return None
        return wizard.generate_config(values)

    @property
    def wizard_count(self) -> int:
        return len(self._wizards)
