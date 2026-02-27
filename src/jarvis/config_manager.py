"""Jarvis · Konfigurations-Manager.

Stellt eine sichere API für Lesen, Ändern und Speichern der
JarvisConfig bereit. Unterstützt:

  - Lesen der gesamten Konfiguration (ohne Secrets)
  - Partielles Update einzelner Sektionen
  - Validierung via Pydantic vor dem Speichern
  - Persistierung in config.yaml
  - Live-Reload-Callback

Architektur-Bibel: §12
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from jarvis.config import JarvisConfig, load_config
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


def _is_secret_field(field_name: str) -> bool:
    """Return True if *field_name* looks like it holds sensitive data.

    Checks the explicit ``_SECRET_FIELDS`` set first, then falls back
    to a substring-based heuristic using ``_SECRET_PATTERNS`` (while
    respecting the exclusion list).
    """
    if field_name in _SECRET_FIELDS:
        return True
    lower = field_name.lower()
    # Exclusions take priority over pattern matches
    if any(excl in lower for excl in _SECRET_PATTERN_EXCLUSIONS):
        return False
    return any(pat in lower for pat in _SECRET_PATTERNS)

# Felder die NIEMALS via API zurückgegeben werden
_SECRET_FIELDS = frozenset({
    "openai_api_key",
    "anthropic_api_key",
    "slack_token",
    "slack_app_token",
    "telegram_token",
    "discord_token",
    "whatsapp_token",
    "whatsapp_verify_token",
    "matrix_token",
    "teams_token",
    "signal_token",
    "pg_password",
    "api_key",
    "secret_key",
    "webhook_secret",
})

# Patterns in field names that indicate sensitive data.
# Any field whose name contains one of these substrings (case-insensitive)
# will be masked -- unless the name also matches an exclusion pattern.
_SECRET_PATTERNS = ("token", "secret", "password", "key")
_SECRET_PATTERN_EXCLUSIONS = (
    "key_file",
    "keyboard",
    # Numeric *token* fields that are NOT secrets
    "_tokens",        # chunk_size_tokens, chunk_overlap_tokens
    "token_budget",   # response_token_budget
    "max_tokens",     # anthropic_max_tokens
)

# Sektionen die über die API editierbar sind
_EDITABLE_SECTIONS = frozenset({
    "ollama",
    "models",
    "gatekeeper",
    "planner",
    "memory",
    "channels",
    "sandbox",
    "logging",
    "security",
    "heartbeat",
    "plugins",
    "dashboard",
    "model_overrides",
    "web",
    "database",
})

# Top-Level-Felder die editierbar sind
_EDITABLE_TOP_LEVEL = frozenset({
    "owner_name",
    "llm_backend_type",
    "operation_mode",
    "cost_tracking_enabled",
    "daily_budget_usd",
    "monthly_budget_usd",
    "vision_model",
    "vision_model_detail",
    "openai_base_url",
    "anthropic_max_tokens",
    # API keys — the UI sends these; the update handler skips masked "***" values
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "groq_api_key",
    "deepseek_api_key",
    "mistral_api_key",
    "together_api_key",
    "openrouter_api_key",
    "xai_api_key",
    "cerebras_api_key",
    "github_api_key",
    "bedrock_api_key",
    "huggingface_api_key",
    "moonshot_api_key",
})


class ConfigManager:
    """Verwaltet die Jarvis-Konfiguration mit sicherem Read/Write.

    Verwendung::

        mgr = ConfigManager()
        config_dict = mgr.read()
        mgr.update_section("planner", {"temperature": 0.9})
        mgr.save()
    """

    def __init__(
        self,
        config: JarvisConfig | None = None,
        config_path: Path | None = None,
        on_reload: Callable[[JarvisConfig], None] | None = None,
    ) -> None:
        self._config_path = config_path
        self._on_reload = on_reload

        if config is not None:
            self._config = config
        else:
            self._config = load_config(config_path)

    @property
    def config(self) -> JarvisConfig:
        """Aktuelle Konfiguration (read-only Zugriff)."""
        return self._config

    # ------------------------------------------------------------------
    # Lesen
    # ------------------------------------------------------------------

    def read(self, *, include_secrets: bool = False) -> dict[str, Any]:
        """Gibt die Konfiguration als Dictionary zurück.

        Args:
            include_secrets: Wenn False, werden API-Keys maskiert.

        Returns:
            Vollständiges Config-Dict (serialisierbar).
        """
        data = self._config.model_dump(mode="json")

        if not include_secrets:
            self._mask_secrets(data)

        # Path-Objekte → Strings
        for key in ("jarvis_home",):
            if key in data:
                data[key] = str(data[key])

        return data

    def read_section(self, section: str) -> dict[str, Any] | None:
        """Gibt eine einzelne Sektion zurück.

        Args:
            section: Name der Sektion (z.B. "planner", "memory").

        Returns:
            Dict der Sektion oder None wenn nicht vorhanden.
        """
        full = self.read()
        return full.get(section)

    # ------------------------------------------------------------------
    # Schreiben
    # ------------------------------------------------------------------

    def update_section(self, section: str, values: dict[str, Any]) -> JarvisConfig:
        """Aktualisiert eine Konfigurations-Sektion.

        Validiert die Änderungen über Pydantic, bevor sie angewendet werden.
        Bei Validierungsfehlern wird eine ValueError geworfen.

        Args:
            section: Name der Sektion.
            values: Neue Werte (partielles Update).

        Returns:
            Aktualisierte JarvisConfig.

        Raises:
            ValueError: Ungültige Sektion oder Validierungsfehler.
        """
        if section not in _EDITABLE_SECTIONS:
            msg = f"Sektion '{section}' ist nicht editierbar. Erlaubt: {sorted(_EDITABLE_SECTIONS)}"
            raise ValueError(msg)

        # Aktuellen State als Dict holen
        current = self._config.model_dump(mode="json")

        # Sektion mergen
        if section not in current:
            msg = f"Sektion '{section}' existiert nicht in der Konfiguration"
            raise ValueError(msg)

        if isinstance(current[section], dict):
            from jarvis.config import _deep_merge
            merged = _deep_merge(current[section], values)
        else:
            merged = values
        current[section] = merged

        # Über Pydantic validieren
        try:
            new_config = JarvisConfig(**current)
        except ValidationError as exc:
            msg = f"Validierungsfehler: {exc}"
            raise ValueError(msg) from exc

        self._config = new_config
        log.info("config_section_updated", section=section, keys=list(values.keys()))
        return new_config

    def update_top_level(self, key: str, value: Any) -> JarvisConfig:
        """Aktualisiert ein Top-Level-Feld.

        Args:
            key: Feldname (z.B. "owner_name").
            value: Neuer Wert.

        Returns:
            Aktualisierte JarvisConfig.

        Raises:
            ValueError: Feld nicht editierbar oder Validierungsfehler.
        """
        if key not in _EDITABLE_TOP_LEVEL:
            msg = f"Feld '{key}' ist nicht editierbar. Erlaubt: {sorted(_EDITABLE_TOP_LEVEL)}"
            raise ValueError(msg)

        current = self._config.model_dump(mode="json")
        current[key] = value

        try:
            new_config = JarvisConfig(**current)
        except ValidationError as exc:
            msg = f"Validierungsfehler: {exc}"
            raise ValueError(msg) from exc

        self._config = new_config
        log.info("config_top_level_updated", key=key)
        return new_config

    # ------------------------------------------------------------------
    # Persistieren
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Speichert die Konfiguration in config.yaml.

        Args:
            path: Ziel-Pfad. Default: config_path oder config.config_file.

        Returns:
            Pfad der gespeicherten Datei.
        """
        target = path or self._config_path or self._config.config_file

        # Serialisieren
        data = self._config.model_dump(mode="json")

        # Pfade serialisieren
        for key in ("jarvis_home",):
            if key in data:
                data[key] = str(data[key])

        # Secrets nicht im Klartext speichern wenn sie "***" sind
        self._strip_masked_secrets(data)

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        log.info("config_saved", path=str(target))

        # Callback auslösen
        if self._on_reload:
            self._on_reload(self._config)

        return target

    def reload(self) -> JarvisConfig:
        """Lädt die Konfiguration neu aus der Datei.

        Returns:
            Neu geladene JarvisConfig.
        """
        self._config = load_config(self._config_path)
        log.info("config_reloaded")

        if self._on_reload:
            self._on_reload(self._config)

        return self._config

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_secrets(data: dict[str, Any], *, _depth: int = 0) -> None:
        """Recursively mask secret values in *data* with ``'***'``.

        Handles both flat keys and nested dicts/sections so that
        secrets buried inside sub-sections are also redacted.
        """
        if _depth > 5:
            return  # safety guard against deeply nested structures
        for key in list(data.keys()):
            value = data[key]
            if isinstance(value, dict):
                ConfigManager._mask_secrets(value, _depth=_depth + 1)
            elif _is_secret_field(key) and value:
                data[key] = "***"

    @staticmethod
    def _strip_masked_secrets(data: dict[str, Any], *, _depth: int = 0) -> None:
        """Remove keys whose value is the mask placeholder ``'***'``.

        Prevents writing masked placeholders back to config.yaml.
        """
        if _depth > 5:
            return
        keys_to_delete: list[str] = []
        for key in list(data.keys()):
            value = data[key]
            if isinstance(value, dict):
                ConfigManager._strip_masked_secrets(value, _depth=_depth + 1)
            elif value == "***" and _is_secret_field(key):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del data[key]

    @staticmethod
    def editable_sections() -> list[str]:
        """Liste der editierbaren Sektionen."""
        return sorted(_EDITABLE_SECTIONS)

    @staticmethod
    def editable_top_level_fields() -> list[str]:
        """Liste der editierbaren Top-Level-Felder."""
        return sorted(_EDITABLE_TOP_LEVEL)
