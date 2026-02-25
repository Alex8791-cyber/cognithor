"""Tests für ConfigManager und Config-API-Routes.

Testet:
  - ConfigManager: read, update_section, update_top_level, save, reload
  - Secret-Masking
  - Validierung (ungültige Werte, unbekannte Sektionen)
  - Config-API-Routes: GET/PATCH config, presets, status
  - Credential-Endpoints
  - Agent-Endpoints
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jarvis.config import JarvisConfig
from jarvis.config_manager import ConfigManager


# ===========================================================================
# ConfigManager: Lesen
# ===========================================================================

class TestConfigManagerRead:
    def test_read_returns_dict(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        result = mgr.read()
        assert isinstance(result, dict)
        assert "planner" in result
        assert "memory" in result
        assert "channels" in result

    def test_read_masks_secrets(self, tmp_path: Path) -> None:
        config = JarvisConfig(
            jarvis_home=tmp_path / ".jarvis",
            openai_api_key="sk-real-key-12345",
            anthropic_api_key="sk-ant-real-key",
        )
        mgr = ConfigManager(config=config)

        result = mgr.read(include_secrets=False)
        assert result["openai_api_key"] == "***"
        assert result["anthropic_api_key"] == "***"

    def test_read_includes_secrets_when_requested(self, tmp_path: Path) -> None:
        config = JarvisConfig(
            jarvis_home=tmp_path / ".jarvis",
            openai_api_key="sk-real-key-12345",
        )
        mgr = ConfigManager(config=config)

        result = mgr.read(include_secrets=True)
        assert result["openai_api_key"] == "sk-real-key-12345"

    def test_read_section(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        planner = mgr.read_section("planner")
        assert planner is not None
        assert "temperature" in planner
        assert "max_iterations" in planner

    def test_read_section_unknown(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        result = mgr.read_section("nonexistent")
        assert result is None

    def test_jarvis_home_is_string(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        result = mgr.read()
        assert isinstance(result["jarvis_home"], str)


# ===========================================================================
# ConfigManager: Schreiben
# ===========================================================================

class TestConfigManagerUpdate:
    def test_update_section_planner(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        mgr.update_section("planner", {"temperature": 0.9})
        assert mgr.config.planner.temperature == 0.9

    def test_update_section_memory(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        mgr.update_section("memory", {"search_top_k": 12})
        assert mgr.config.memory.search_top_k == 12

    def test_update_section_channels(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        mgr.update_section("channels", {"telegram_enabled": True, "slack_enabled": True})
        assert mgr.config.channels.telegram_enabled is True
        assert mgr.config.channels.slack_enabled is True

    def test_update_section_heartbeat(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        mgr.update_section("heartbeat", {"enabled": True, "interval_minutes": 15})
        assert mgr.config.heartbeat.enabled is True
        assert mgr.config.heartbeat.interval_minutes == 15

    def test_update_section_preserves_other_values(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        original_iterations = config.planner.max_iterations
        mgr.update_section("planner", {"temperature": 0.5})

        assert mgr.config.planner.temperature == 0.5
        assert mgr.config.planner.max_iterations == original_iterations

    def test_update_invalid_section_raises(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        with pytest.raises(ValueError, match="nicht editierbar"):
            mgr.update_section("jarvis_home", {"value": "/tmp"})

    def test_update_with_invalid_value_raises(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        # temperature muss zwischen 0.0 und 2.0 sein
        with pytest.raises(ValueError, match="Validierungsfehler"):
            mgr.update_section("planner", {"temperature": 99.0})

    def test_update_top_level_owner_name(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        mgr.update_top_level("owner_name", "Alexander")
        assert mgr.config.owner_name == "Alexander"

    def test_update_top_level_llm_backend(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        mgr.update_top_level("llm_backend_type", "anthropic")
        assert mgr.config.llm_backend_type == "anthropic"

    def test_update_top_level_invalid_field(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        with pytest.raises(ValueError, match="nicht editierbar"):
            mgr.update_top_level("version", "99.0")


# ===========================================================================
# ConfigManager: Persistenz
# ===========================================================================

class TestConfigManagerPersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)

        target = tmp_path / "test_config.yaml"
        result = mgr.save(target)

        assert result == target
        assert target.exists()
        content = target.read_text()
        assert "planner" in content
        assert "memory" in content

    def test_save_does_not_store_masked_secrets(self, tmp_path: Path) -> None:
        config = JarvisConfig(
            jarvis_home=tmp_path / ".jarvis",
            openai_api_key="sk-real",
        )
        mgr = ConfigManager(config=config)

        target = tmp_path / "test_config.yaml"
        mgr.save(target)

        content = target.read_text()
        # Echter Key sollte gespeichert sein
        assert "sk-real" in content

    def test_save_triggers_on_reload(self, tmp_path: Path) -> None:
        reloaded: list[JarvisConfig] = []

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config, on_reload=reloaded.append)

        mgr.save(tmp_path / "test.yaml")
        assert len(reloaded) == 1

    def test_reload_from_file(self, tmp_path: Path) -> None:
        # Config-Datei anlegen
        config_path = tmp_path / ".jarvis" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("owner_name: TestUser\n")

        mgr = ConfigManager(config_path=config_path)
        assert mgr.config.owner_name == "TestUser"

        # Datei ändern
        config_path.write_text("owner_name: NewUser\n")
        mgr.reload()
        assert mgr.config.owner_name == "NewUser"


# ===========================================================================
# ConfigManager: Metadaten
# ===========================================================================

class TestConfigManagerMeta:
    def test_editable_sections(self) -> None:
        sections = ConfigManager.editable_sections()
        assert "planner" in sections
        assert "memory" in sections
        assert "channels" in sections
        assert "heartbeat" in sections

    def test_editable_top_level_fields(self) -> None:
        fields = ConfigManager.editable_top_level_fields()
        assert "owner_name" in fields
        assert "llm_backend_type" in fields

    def test_config_property(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        mgr = ConfigManager(config=config)
        assert mgr.config is config


# ===========================================================================
# Config API Routes (Unit-Tests ohne HTTP)
# ===========================================================================

class TestConfigRoutes:
    """Testet die Route-Funktionen direkt (ohne FastAPI-Server)."""

    @pytest.fixture()
    def mgr(self, tmp_path: Path) -> ConfigManager:
        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        return ConfigManager(config=config)

    @pytest.fixture()
    def routes(self, mgr: ConfigManager) -> dict[str, Any]:
        """Registriert Routes auf einem Mock-App und gibt Route-Handler zurück."""
        handlers: dict[str, Any] = {}

        class MockApp:
            def get(self, path: str, **kwargs: Any) -> Any:
                def deco(func: Any) -> Any:
                    handlers[f"GET {path}"] = func
                    return func
                return deco

            def patch(self, path: str, **kwargs: Any) -> Any:
                def deco(func: Any) -> Any:
                    handlers[f"PATCH {path}"] = func
                    return func
                return deco

            def post(self, path: str, **kwargs: Any) -> Any:
                def deco(func: Any) -> Any:
                    handlers[f"POST {path}"] = func
                    return func
                return deco

            def delete(self, path: str, **kwargs: Any) -> Any:
                def deco(func: Any) -> Any:
                    handlers[f"DELETE {path}"] = func
                    return func
                return deco

        from jarvis.channels.config_routes import create_config_routes
        create_config_routes(MockApp(), mgr)
        return handlers

    @pytest.mark.asyncio()
    async def test_get_config(self, routes: dict[str, Any]) -> None:
        result = await routes["GET /api/v1/config"]()
        assert "planner" in result
        assert "memory" in result
        assert "_meta" in result
        assert "editable_sections" in result["_meta"]

    @pytest.mark.asyncio()
    async def test_get_config_section(self, routes: dict[str, Any]) -> None:
        result = await routes["GET /api/v1/config/{section}"](section="planner")
        assert result["section"] == "planner"
        assert "temperature" in result["values"]

    @pytest.mark.asyncio()
    async def test_get_config_section_unknown(self, routes: dict[str, Any]) -> None:
        result = await routes["GET /api/v1/config/{section}"](section="nonexistent")
        assert result["status"] == 404

    @pytest.mark.asyncio()
    async def test_update_config_section(
        self, routes: dict[str, Any], mgr: ConfigManager, tmp_path: Path,
    ) -> None:
        # Save-Pfad setzen damit save() nicht fehlschlägt
        mgr._config_path = tmp_path / ".jarvis" / "config.yaml"
        mgr._config_path.parent.mkdir(parents=True, exist_ok=True)

        result = await routes["PATCH /api/v1/config/{section}"](
            section="planner", values={"temperature": 0.3},
        )
        assert result["status"] == "ok"
        assert mgr.config.planner.temperature == 0.3

    @pytest.mark.asyncio()
    async def test_update_config_invalid_section(
        self, routes: dict[str, Any],
    ) -> None:
        result = await routes["PATCH /api/v1/config/{section}"](
            section="version", values={"value": "2.0"},
        )
        assert result["status"] == 400

    @pytest.mark.asyncio()
    async def test_reload_config(
        self, routes: dict[str, Any], mgr: ConfigManager, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / ".jarvis" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("owner_name: Reloaded\n")
        mgr._config_path = config_path

        result = await routes["POST /api/v1/config/reload"]()
        assert result["status"] == "ok"

    @pytest.mark.asyncio()
    async def test_get_status(self, routes: dict[str, Any]) -> None:
        result = await routes["GET /api/v1/status"]()
        assert "timestamp" in result
        assert "models" in result
        assert "active_channels" in result
        assert "llm_backend" in result

    @pytest.mark.asyncio()
    async def test_list_presets(self, routes: dict[str, Any]) -> None:
        result = await routes["GET /api/v1/config/presets"]()
        assert "presets" in result
        names = [p["name"] for p in result["presets"]]
        assert "minimal" in names
        assert "standard" in names
        assert "full" in names

    @pytest.mark.asyncio()
    async def test_apply_preset(
        self, routes: dict[str, Any], mgr: ConfigManager, tmp_path: Path,
    ) -> None:
        mgr._config_path = tmp_path / ".jarvis" / "config.yaml"
        mgr._config_path.parent.mkdir(parents=True, exist_ok=True)

        result = await routes["POST /api/v1/config/presets/{preset_name}"](
            preset_name="standard",
        )
        assert result["preset"] == "standard"
        assert mgr.config.heartbeat.enabled is True

    @pytest.mark.asyncio()
    async def test_apply_unknown_preset(self, routes: dict[str, Any]) -> None:
        result = await routes["POST /api/v1/config/presets/{preset_name}"](
            preset_name="nonexistent",
        )
        assert result["status"] == 404

    @pytest.mark.asyncio()
    async def test_list_agents(self, routes: dict[str, Any]) -> None:
        result = await routes["GET /api/v1/agents"]()
        assert "agents" in result
        assert len(result["agents"]) >= 1

    @pytest.mark.asyncio()
    async def test_list_credentials(self, routes: dict[str, Any]) -> None:
        result = await routes["GET /api/v1/credentials"]()
        assert "credentials" in result
