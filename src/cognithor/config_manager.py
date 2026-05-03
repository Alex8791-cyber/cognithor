"""Cognithor · Configuration Manager.

Provides a secure API for reading, modifying and saving the
CognithorConfig. Supports:

  - Reading the entire configuration (without secrets)
  - Partial update of individual sections
  - Validation via Pydantic before saving
  - Persistence in config.yaml
  - Live reload callback

Architecture Bible: §12
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from cognithor.config import CognithorConfig, load_config
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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


# Fields that are NEVER returned via API
_SECRET_FIELDS = frozenset(
    {
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
    }
)

# Patterns in field names that indicate sensitive data.
# Any field whose name contains one of these substrings (case-insensitive)
# will be masked -- unless the name also matches an exclusion pattern.
_SECRET_PATTERNS = ("token", "secret", "password", "key")
_SECRET_PATTERN_EXCLUSIONS = (
    "key_file",
    "keyboard",
    # Numeric *token* fields that are NOT secrets
    "_tokens",  # chunk_size_tokens, chunk_overlap_tokens
    "token_budget",  # response_token_budget
    "max_tokens",  # anthropic_max_tokens
)

# Sections that are editable via API
_EDITABLE_SECTIONS = frozenset(
    {
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
        "prompt_evolution",
        "improvement",
        "executor",
        "tools",
        "audit",
        "browser",
        "calendar",
        "email",
        "recovery",
        "identity",
        "personality",
        "evolution",
        "atl",
        "social",
        "kanban",
        "vault",
    }
)

# Top-level fields that are editable
_EDITABLE_TOP_LEVEL = frozenset(
    {
        "owner_name",
        "language",
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
        "lmstudio_api_key",
        "lmstudio_base_url",
    }
)


class ConfigManager:
    """Manages the Cognithor configuration with secure read/write.

    Usage::

        mgr = ConfigManager()
        config_dict = mgr.read()
        mgr.update_section("planner", {"temperature": 0.9})
        mgr.save()
    """

    def __init__(
        self,
        config: CognithorConfig | None = None,
        config_path: Path | None = None,
        on_reload: Callable[[CognithorConfig], None] | None = None,
    ) -> None:
        self._config_path = config_path
        self._on_reload = on_reload

        if config is not None:
            self._config = config
        else:
            self._config = load_config(config_path)

    @property
    def config(self) -> CognithorConfig:
        """Current configuration (read-only access)."""
        return self._config

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, *, include_secrets: bool = False) -> dict[str, Any]:
        """Returns the configuration as a dictionary.

        Args:
            include_secrets: If False, API keys are masked.

        Returns:
            Complete config dict (serializable).
        """
        data = self._config.model_dump(mode="json")

        # Version always from the package -- never from config.yaml
        from cognithor import __version__

        data["version"] = __version__

        if not include_secrets:
            self._mask_secrets(data)

        # Path objects -> strings
        for key in ("cognithor_home",):
            if key in data:
                data[key] = str(data[key])

        return data

    def read_section(self, section: str) -> dict[str, Any] | None:
        """Returns a single section.

        Args:
            section: Name of the section (e.g. "planner", "memory").

        Returns:
            Dict of the section or None if not found.
        """
        full = self.read()
        return full.get(section)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update_section(self, section: str, values: dict[str, Any]) -> CognithorConfig:
        """Aktualisiert eine Konfigurations-Sektion.

        Validiert die Aenderungen ueber Pydantic, bevor sie angewendet werden.
        Bei Validierungsfehlern wird eine ValueError geworfen.

        Args:
            section: Name der Sektion.
            values: Neue Werte (partielles Update).

        Returns:
            Aktualisierte CognithorConfig.

        Raises:
            ValueError: Ungueltige Sektion oder Validierungsfehler.
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
            from cognithor.config import _deep_merge

            merged = _deep_merge(current[section], values)
        else:
            merged = values
        current[section] = merged

        # Ueber Pydantic validieren
        try:
            new_config = CognithorConfig(**current)
        except ValidationError as exc:
            msg = f"Validierungsfehler: {exc}"
            raise ValueError(msg) from exc

        self._config = new_config
        log.info("config_section_updated", section=section, keys=list(values.keys()))
        return new_config

    def update_top_level(self, key: str, value: Any) -> CognithorConfig:
        """Aktualisiert ein Top-Level-Feld.

        Args:
            key: Feldname (z.B. "owner_name").
            value: Neuer Wert.

        Returns:
            Aktualisierte CognithorConfig.

        Raises:
            ValueError: Feld nicht editierbar oder Validierungsfehler.
        """
        if key not in _EDITABLE_TOP_LEVEL:
            msg = f"Feld '{key}' ist nicht editierbar. Erlaubt: {sorted(_EDITABLE_TOP_LEVEL)}"
            raise ValueError(msg)

        current = self._config.model_dump(mode="json")
        old_value = current.get(key)
        current[key] = value

        # Log secret field changes (without exposing actual values)
        if _is_secret_field(key):
            had_value = bool(old_value and old_value != "")
            gets_value = bool(value and value != "" and value != "***")
            if had_value and not gets_value:
                log.warning(
                    "config_secret_field_cleared",
                    key=key,
                    msg="A secret field with a real value is being cleared — "
                    "this may indicate a UI bug sending default values",
                )
            else:
                log.info(
                    "config_secret_field_update",
                    key=key,
                    had_value=had_value,
                    gets_value=gets_value,
                )

        try:
            new_config = CognithorConfig(**current)
        except ValidationError as exc:
            msg = f"Validierungsfehler: {exc}"
            raise ValueError(msg) from exc

        self._config = new_config
        log.info("config_top_level_updated", key=key)
        return new_config

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Saves the configuration to config.yaml.

        Writes atomically: first to a temporary file, then rename.
        This way a crash during writing cannot corrupt the existing
        config.yaml.

        Args:
            path: Ziel-Pfad. Default: config_path oder config.config_file.

        Returns:
            Pfad der gespeicherten Datei.
        """
        import os
        import tempfile

        from cognithor.core.config_versioning import save_config_revision

        target = path or self._config_path or self._config.config_file

        # Save a revision of the current config before overwriting
        try:
            revision_data = self._config.model_dump(mode="json")
            # Convert Path objects for JSON serialization
            for k in ("cognithor_home",):
                if k in revision_data:
                    revision_data[k] = str(revision_data[k])
            save_config_revision(revision_data, reason="pre-save snapshot")
        except Exception:
            log.warning("config_revision_save_failed", exc_info=True)

        # Serialize
        data = self._config.model_dump(mode="json")

        # Serialize paths
        for key in ("cognithor_home",):
            if key in data:
                data[key] = str(data[key])

        # Do not save version in config.yaml -- always read from the package
        data.pop("version", None)

        # Do not save secrets in plaintext when they are "***"
        self._strip_masked_secrets(data)

        target.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file in same directory → rename
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent),
                prefix=".config_",
                suffix=".yaml.tmp",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            # On Windows os.rename fails if target exists; use os.replace
            os.replace(tmp_path, str(target))
        except Exception:
            # Clean up temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        log.info("config_saved", path=str(target))

        # Trigger callback
        if self._on_reload:
            self._on_reload(self._config)

        return target

    def reload(self) -> CognithorConfig:
        """Reloads the configuration from the file.

        Returns:
            Newly loaded CognithorConfig.
        """
        self._config = load_config(self._config_path)
        log.info("config_reloaded")

        if self._on_reload:
            self._on_reload(self._config)

        return self._config

    # ------------------------------------------------------------------
    # Helper methods
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
        """List of editable sections."""
        return sorted(_EDITABLE_SECTIONS)

    @staticmethod
    def editable_top_level_fields() -> list[str]:
        """List of editable top-level fields."""
        return sorted(_EDITABLE_TOP_LEVEL)
