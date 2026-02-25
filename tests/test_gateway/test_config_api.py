"""Tests für Config-API: ConfigManager.

Testet CRUD-Operationen für Heartbeat, Agents, Bindings, Sandbox,
Presets und Konfigurationsübersicht.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis.gateway.config_api import (
    AgentProfileDTO,
    BindingRuleDTO,
    ConfigManager,
    ConfigOverview,
    HeartbeatUpdate,
    PresetInfo,
    SandboxUpdate,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock-JarvisConfig mit realistischen Defaults."""
    config = MagicMock()
    config.version = "0.9.0"
    config.owner_name = "Alexander"
    config.llm_backend_type = "ollama"

    # Heartbeat
    hb = MagicMock()
    hb.enabled = False
    hb.interval_minutes = 30
    hb.checklist_file = "HEARTBEAT.md"
    hb.channel = "cli"
    hb.model = "qwen3:8b"
    config.heartbeat = hb

    # Sandbox
    sb = MagicMock()
    sb.enabled = True
    sb.network = "allow"
    sb.max_memory_mb = 512
    sb.max_processes = 64
    sb.timeout_seconds = 30
    sb.allowed_paths = ["/tmp"]
    sb.blocked_paths = ["/etc/shadow"]
    config.sandbox = sb

    # Channels
    ch = MagicMock()
    ch.cli_enabled = True
    ch.telegram_enabled = True
    ch.webui_enabled = False
    ch.slack_enabled = False
    ch.discord_enabled = False
    config.channels = ch

    # Keine Agents (werden über ConfigManager verwaltet)
    config.agents = []

    return config


@pytest.fixture
def mgr(mock_config: MagicMock) -> ConfigManager:
    """ConfigManager mit Mock-Config."""
    return ConfigManager(mock_config)


# ============================================================================
# 1. Übersicht
# ============================================================================


class TestConfigOverview:
    def test_overview_returns_all_fields(self, mgr: ConfigManager) -> None:
        overview = mgr.get_overview()
        assert isinstance(overview, ConfigOverview)
        assert overview.version == "0.9.0"
        assert overview.owner_name == "Alexander"
        assert overview.llm_backend == "ollama"
        assert overview.heartbeat_enabled is False
        assert overview.heartbeat_interval == 30
        assert overview.agent_count == 0
        assert overview.binding_count == 0
        assert overview.sandbox_enabled is True

    def test_overview_shows_active_channels(self, mgr: ConfigManager) -> None:
        overview = mgr.get_overview()
        assert "cli" in overview.channels_active
        assert "telegram" in overview.channels_active
        assert "webui" not in overview.channels_active

    def test_overview_updates_after_adding_agents(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(name="coder"))
        overview = mgr.get_overview()
        assert overview.agent_count == 1


# ============================================================================
# 2. Heartbeat CRUD
# ============================================================================


class TestHeartbeatConfig:
    def test_get_heartbeat(self, mgr: ConfigManager) -> None:
        hb = mgr.get_heartbeat()
        assert hb["enabled"] is False
        assert hb["interval_minutes"] == 30
        assert hb["channel"] == "cli"
        assert hb["checklist_file"] == "HEARTBEAT.md"

    def test_update_heartbeat_enabled(self, mgr: ConfigManager) -> None:
        result = mgr.update_heartbeat(HeartbeatUpdate(enabled=True))
        assert result["enabled"] is True

    def test_update_heartbeat_interval(self, mgr: ConfigManager) -> None:
        result = mgr.update_heartbeat(HeartbeatUpdate(interval_minutes=60))
        assert result["interval_minutes"] == 60

    def test_update_heartbeat_partial(self, mgr: ConfigManager) -> None:
        """Nur gesetzte Felder werden aktualisiert."""
        mgr.update_heartbeat(HeartbeatUpdate(channel="telegram"))
        hb = mgr.get_heartbeat()
        assert hb["channel"] == "telegram"
        assert hb["interval_minutes"] == 30  # Unverändert

    def test_update_heartbeat_multiple_fields(self, mgr: ConfigManager) -> None:
        result = mgr.update_heartbeat(HeartbeatUpdate(
            enabled=True,
            interval_minutes=15,
            channel="slack",
            model="llama3:8b",
        ))
        assert result["enabled"] is True
        assert result["interval_minutes"] == 15
        assert result["channel"] == "slack"

    def test_heartbeat_validation_min_interval(self) -> None:
        """interval_minutes < 1 wird abgelehnt."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            HeartbeatUpdate(interval_minutes=0)

    def test_heartbeat_validation_max_interval(self) -> None:
        """interval_minutes > 1440 wird abgelehnt."""
        with pytest.raises(Exception):
            HeartbeatUpdate(interval_minutes=1441)


# ============================================================================
# 3. Agent-Profile CRUD
# ============================================================================


class TestAgentProfiles:
    def test_list_agents_initially_empty(self, mgr: ConfigManager) -> None:
        assert mgr.list_agents() == []

    def test_create_agent(self, mgr: ConfigManager) -> None:
        result = mgr.upsert_agent(AgentProfileDTO(
            name="coder",
            display_name="Code-Agent",
            description="Schreibt und debuggt Code",
            trigger_keywords=["code", "python", "debug"],
            priority=10,
        ))
        assert result["name"] == "coder"
        assert result["display_name"] == "Code-Agent"
        assert "python" in result["trigger_keywords"]

    def test_read_agent(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(name="researcher"))
        agent = mgr.get_agent("researcher")
        assert agent is not None
        assert agent["name"] == "researcher"

    def test_read_nonexistent_agent(self, mgr: ConfigManager) -> None:
        assert mgr.get_agent("ghost") is None

    def test_update_agent(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(name="coder", priority=5))
        mgr.upsert_agent(AgentProfileDTO(name="coder", priority=20))
        agent = mgr.get_agent("coder")
        assert agent is not None
        assert agent["priority"] == 20

    def test_delete_agent(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(name="temp"))
        assert mgr.delete_agent("temp") is True
        assert mgr.get_agent("temp") is None

    def test_delete_nonexistent_agent(self, mgr: ConfigManager) -> None:
        assert mgr.delete_agent("ghost") is False

    def test_cannot_delete_default_agent(self, mgr: ConfigManager) -> None:
        """jarvis-Agent kann nicht gelöscht werden."""
        mgr.upsert_agent(AgentProfileDTO(name="jarvis"))
        assert mgr.delete_agent("jarvis") is False
        assert mgr.get_agent("jarvis") is not None

    def test_agent_with_credential_scope(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(
            name="coder",
            credential_scope="coder",
            credential_mappings={"api_key": "github:token"},
        ))
        agent = mgr.get_agent("coder")
        assert agent is not None
        assert agent["credential_scope"] == "coder"
        assert agent["credential_mappings"]["api_key"] == "github:token"

    def test_agent_with_sandbox_config(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(
            name="untrusted",
            sandbox_network="block",
            sandbox_max_memory_mb=256,
            sandbox_timeout=10,
        ))
        agent = mgr.get_agent("untrusted")
        assert agent is not None
        assert agent["sandbox_network"] == "block"
        assert agent["sandbox_max_memory_mb"] == 256

    def test_agent_with_delegation(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(
            name="lead",
            can_delegate_to=["coder", "researcher"],
            max_delegation_depth=3,
        ))
        agent = mgr.get_agent("lead")
        assert agent is not None
        assert "coder" in agent["can_delegate_to"]
        assert agent["max_delegation_depth"] == 3

    def test_multiple_agents(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(name="agent_a"))
        mgr.upsert_agent(AgentProfileDTO(name="agent_b"))
        mgr.upsert_agent(AgentProfileDTO(name="agent_c"))
        assert len(mgr.list_agents()) == 3


# ============================================================================
# 4. Binding-Regeln CRUD
# ============================================================================


class TestBindingRules:
    def test_list_bindings_initially_empty(self, mgr: ConfigManager) -> None:
        assert mgr.list_bindings() == []

    def test_create_binding(self, mgr: ConfigManager) -> None:
        result = mgr.upsert_binding(BindingRuleDTO(
            name="slash_code",
            target_agent="coder",
            command_prefixes=["/code", "/debug"],
        ))
        assert result["name"] == "slash_code"
        assert result["target_agent"] == "coder"
        assert "/code" in result["command_prefixes"]

    def test_read_binding(self, mgr: ConfigManager) -> None:
        mgr.upsert_binding(BindingRuleDTO(name="test_b", target_agent="jarvis"))
        b = mgr.get_binding("test_b")
        assert b is not None
        assert b["target_agent"] == "jarvis"

    def test_read_nonexistent_binding(self, mgr: ConfigManager) -> None:
        assert mgr.get_binding("ghost") is None

    def test_update_binding(self, mgr: ConfigManager) -> None:
        mgr.upsert_binding(BindingRuleDTO(name="b1", target_agent="a", priority=50))
        mgr.upsert_binding(BindingRuleDTO(name="b1", target_agent="b", priority=100))
        b = mgr.get_binding("b1")
        assert b is not None
        assert b["target_agent"] == "b"
        assert b["priority"] == 100

    def test_delete_binding(self, mgr: ConfigManager) -> None:
        mgr.upsert_binding(BindingRuleDTO(name="temp", target_agent="x"))
        assert mgr.delete_binding("temp") is True
        assert mgr.get_binding("temp") is None

    def test_delete_nonexistent_binding(self, mgr: ConfigManager) -> None:
        assert mgr.delete_binding("ghost") is False

    def test_binding_with_channels(self, mgr: ConfigManager) -> None:
        mgr.upsert_binding(BindingRuleDTO(
            name="tg_only",
            target_agent="coder",
            channels=["telegram"],
        ))
        b = mgr.get_binding("tg_only")
        assert b is not None
        assert b["channels"] == ["telegram"]

    def test_binding_with_regex(self, mgr: ConfigManager) -> None:
        mgr.upsert_binding(BindingRuleDTO(
            name="regex_b",
            target_agent="coder",
            message_patterns=[r"(?i)python\b", r"(?i)bug\b"],
        ))
        b = mgr.get_binding("regex_b")
        assert b is not None
        assert len(b["message_patterns"]) == 2

    def test_binding_with_metadata(self, mgr: ConfigManager) -> None:
        mgr.upsert_binding(BindingRuleDTO(
            name="meta_b",
            target_agent="jarvis",
            metadata_conditions={"priority": "high"},
        ))
        b = mgr.get_binding("meta_b")
        assert b is not None
        assert b["metadata_conditions"]["priority"] == "high"

    def test_binding_negation(self, mgr: ConfigManager) -> None:
        mgr.upsert_binding(BindingRuleDTO(
            name="not_cli",
            target_agent="jarvis",
            channels=["cli"],
            negate=True,
        ))
        b = mgr.get_binding("not_cli")
        assert b is not None
        assert b["negate"] is True

    def test_multiple_bindings_preserved(self, mgr: ConfigManager) -> None:
        for i in range(5):
            mgr.upsert_binding(BindingRuleDTO(name=f"b{i}", target_agent="jarvis"))
        assert len(mgr.list_bindings()) == 5


# ============================================================================
# 5. Sandbox CRUD
# ============================================================================


class TestSandboxConfig:
    def test_get_sandbox(self, mgr: ConfigManager) -> None:
        sb = mgr.get_sandbox()
        assert sb["enabled"] is True
        assert sb["network"] == "allow"
        assert sb["max_memory_mb"] == 512

    def test_update_sandbox_network(self, mgr: ConfigManager) -> None:
        result = mgr.update_sandbox(SandboxUpdate(network="block"))
        assert result["network"] == "block"

    def test_update_sandbox_memory(self, mgr: ConfigManager) -> None:
        result = mgr.update_sandbox(SandboxUpdate(max_memory_mb=1024))
        assert result["max_memory_mb"] == 1024

    def test_update_sandbox_partial(self, mgr: ConfigManager) -> None:
        """Nur gesetzte Felder werden aktualisiert."""
        mgr.update_sandbox(SandboxUpdate(max_processes=128))
        sb = mgr.get_sandbox()
        assert sb["max_processes"] == 128
        assert sb["network"] == "allow"  # Unverändert

    def test_sandbox_validation_min_memory(self) -> None:
        with pytest.raises(Exception):
            SandboxUpdate(max_memory_mb=32)  # < 64

    def test_sandbox_validation_max_memory(self) -> None:
        with pytest.raises(Exception):
            SandboxUpdate(max_memory_mb=16384)  # > 8192


# ============================================================================
# 6. Presets
# ============================================================================


class TestPresets:
    def test_list_presets(self, mgr: ConfigManager) -> None:
        presets = mgr.list_presets()
        assert len(presets) >= 3
        names = {p.name for p in presets}
        assert "office" in names
        assert "developer" in names
        assert "family" in names

    def test_preset_info_structure(self, mgr: ConfigManager) -> None:
        presets = mgr.list_presets()
        dev = next(p for p in presets if p.name == "developer")
        assert isinstance(dev, PresetInfo)
        assert "coder" in dev.agents
        assert dev.heartbeat_enabled is False

    def test_apply_office_preset(self, mgr: ConfigManager) -> None:
        result = mgr.apply_preset("office")
        assert "error" not in result
        assert "heartbeat" in result["applied"]
        hb = mgr.get_heartbeat()
        assert hb["enabled"] is True
        assert hb["channel"] == "telegram"

    def test_apply_developer_preset(self, mgr: ConfigManager) -> None:
        result = mgr.apply_preset("developer")
        assert "error" not in result
        # Coder-Agent angelegt
        coder = mgr.get_agent("coder")
        assert coder is not None
        assert coder["credential_scope"] == "coder"
        # Slash-Code Binding
        b = mgr.get_binding("slash_code")
        assert b is not None
        assert "/code" in b["command_prefixes"]

    def test_apply_family_preset(self, mgr: ConfigManager) -> None:
        result = mgr.apply_preset("family")
        assert "error" not in result
        hb = mgr.get_heartbeat()
        assert hb["enabled"] is True
        assert hb["interval_minutes"] == 60

    def test_apply_nonexistent_preset(self, mgr: ConfigManager) -> None:
        result = mgr.apply_preset("unicorn")
        assert "error" in result

    def test_preset_preserves_existing_agents(self, mgr: ConfigManager) -> None:
        """Presets ergänzen, überschreiben nicht alles."""
        mgr.upsert_agent(AgentProfileDTO(name="custom_agent"))
        mgr.apply_preset("developer")
        assert mgr.get_agent("custom_agent") is not None
        assert mgr.get_agent("coder") is not None


# ============================================================================
# 7. Export
# ============================================================================


class TestConfigExport:
    def test_export_contains_all_sections(self, mgr: ConfigManager) -> None:
        export = mgr.export_config()
        assert "heartbeat" in export
        assert "agents" in export
        assert "bindings" in export
        assert "sandbox" in export

    def test_export_reflects_changes(self, mgr: ConfigManager) -> None:
        mgr.upsert_agent(AgentProfileDTO(name="test_agent"))
        mgr.upsert_binding(BindingRuleDTO(name="test_bind", target_agent="test_agent"))
        export = mgr.export_config()
        assert len(export["agents"]) == 1
        assert len(export["bindings"]) == 1
        assert export["agents"][0]["name"] == "test_agent"
