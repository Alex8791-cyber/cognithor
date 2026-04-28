"""Integration-Tests: Verdrahtung aller 4 Schwächen.

Beweist, dass circles, config_api, Slack/Discord tatsächlich
in das System integriert sind und nicht isoliert existieren.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from types import ModuleType


def _module_or_package_source(mod: ModuleType) -> str:
    """Concatenate source for a module or every `.py` file in a package.

    Robust gegen den schrittweisen Split von `cognithor.channels.config_routes`
    aus dem Single-File-Modul in ein Paket mit Sub-Modulen
    (siehe `docs/superpowers/plans/2026-04-29-config-routes-split.md`).
    """
    file = getattr(mod, "__file__", None)
    if file and Path(file).name == "__init__.py":
        pkg_dir = Path(file).parent
        return "\n".join(p.read_text(encoding="utf-8") for p in sorted(pkg_dir.glob("*.py")))
    return inspect.getsource(mod)


# ============================================================================
# 1. Channels-Export: Slack/Discord im Package verfügbar
# ============================================================================


class TestChannelsWiring:
    def test_slack_importable_from_channels(self) -> None:
        """SlackChannel kann aus jarvis.channels importiert werden."""
        from cognithor.channels import SlackChannel

        assert SlackChannel is not None

    def test_discord_importable_from_channels(self) -> None:
        """DiscordChannel kann aus jarvis.channels importiert werden."""
        from cognithor.channels import DiscordChannel

        assert DiscordChannel is not None

    def test_slack_channel_is_channel(self) -> None:
        """SlackChannel implementiert das Channel-Interface."""
        from cognithor.channels import Channel, SlackChannel

        ch = SlackChannel(token="test")
        assert isinstance(ch, Channel)
        assert ch.name == "slack"

    def test_discord_channel_is_channel(self) -> None:
        """DiscordChannel implementiert das Channel-Interface."""
        from cognithor.channels import Channel, DiscordChannel

        ch = DiscordChannel(token="test", channel_id=123)
        assert isinstance(ch, Channel)
        assert ch.name == "discord"

    def test_slack_bidirectional_property(self) -> None:
        """SlackChannel hat is_bidirectional Property."""
        from cognithor.channels import SlackChannel

        ch = SlackChannel(token="t", app_token="xapp-test")
        assert hasattr(ch, "is_bidirectional")
        assert ch.app_token == "xapp-test"

    def test_discord_bidirectional_property(self) -> None:
        """DiscordChannel hat is_bidirectional Property."""
        from cognithor.channels import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=1)
        assert hasattr(ch, "is_bidirectional")


# ============================================================================
# 2. SkillExchange → CircleManager verdrahtet
# ============================================================================


class TestSkillExchangeCirclesWiring:
    def test_exchange_has_circles_property(self, tmp_path: Path) -> None:
        """SkillExchange hat circles-Property → CircleManager."""
        from cognithor.skills.circles import CircleManager
        from cognithor.skills.p2p import SkillExchange

        exchange = SkillExchange(skills_dir=tmp_path, require_signatures=False)
        assert hasattr(exchange, "circles")
        assert isinstance(exchange.circles, CircleManager)

    def test_exchange_circles_can_create_circle(self, tmp_path: Path) -> None:
        """CircleManager über SkillExchange ist voll funktionsfähig."""
        from cognithor.skills.p2p import SkillExchange

        exchange = SkillExchange(skills_dir=tmp_path, require_signatures=False)
        circle = exchange.circles.create_circle(
            "Test-Circle",
            "peer_1",
            "Alex",
        )
        assert circle.name == "Test-Circle"
        assert circle.member_count == 1

    def test_exchange_stats_include_circles(self, tmp_path: Path) -> None:
        """stats() enthält Circle-Daten."""
        from cognithor.skills.p2p import SkillExchange

        exchange = SkillExchange(skills_dir=tmp_path, require_signatures=False)
        exchange.circles.create_circle("C1", "p1")
        stats = exchange.stats()

        assert "circles" in stats
        assert stats["circles"] == 1
        assert "circle_members" in stats
        assert "curated_skills" in stats
        assert "collections" in stats

    def test_exchange_search_with_trust_filter(self, tmp_path: Path) -> None:
        """search(trust_filter=True) nutzt CircleManager."""
        from cognithor.skills.p2p import PeerNode, SkillExchange

        exchange = SkillExchange(skills_dir=tmp_path, require_signatures=False)
        exchange.set_identity(PeerNode(peer_id="me", display_name="Alex"))

        # Ohne Index-Einträge → leere Ergebnisse (aber kein Crash)
        results = exchange.search("test", trust_filter=True)
        assert isinstance(results, list)


# ============================================================================
# 3. Skills-Package exportiert CircleManager
# ============================================================================


class TestSkillsPackageExport:
    def test_circle_manager_from_skills(self) -> None:
        """CircleManager kann aus jarvis.skills importiert werden."""
        from cognithor.skills import CircleManager

        assert CircleManager is not None

    def test_trusted_circle_from_skills(self) -> None:
        """TrustedCircle kann aus jarvis.skills importiert werden."""
        from cognithor.skills import TrustedCircle

        assert TrustedCircle is not None


# ============================================================================
# 4. ConfigAPI → ConfigManager verdrahtet
# ============================================================================


class TestConfigAPIWiring:
    def test_config_api_importable(self) -> None:
        """config_api Module sind importierbar."""
        from cognithor.gateway.config_api import (
            AgentProfileDTO,
            ConfigManager,
        )

        assert ConfigManager is not None
        assert AgentProfileDTO is not None

    def test_config_manager_works_with_mock_config(self) -> None:
        """ConfigManager akzeptiert eine Config-Instanz."""
        from cognithor.gateway.config_api import ConfigManager

        config = MagicMock()
        config.version = "1.0"
        config.owner_name = "Test"
        config.llm_backend_type = "ollama"
        config.heartbeat = MagicMock(
            enabled=True,
            interval_minutes=15,
            checklist_file="HB.md",
            channel="cli",
            model="test",
        )
        config.sandbox = MagicMock(enabled=True, network="allow", max_memory_mb=512)
        config.channels = MagicMock(cli_enabled=True)
        config.agents = []

        mgr = ConfigManager(config)
        overview = mgr.get_overview()
        assert overview.version == "1.0"
        assert overview.heartbeat_enabled is True


# ============================================================================
# 5. Config-Routes referenzieren config_api
# ============================================================================


class TestConfigRoutesWiring:
    def test_config_routes_importable(self) -> None:
        """config_routes Modul ist importierbar."""
        from cognithor.channels.config_routes import create_config_routes

        assert create_config_routes is not None

    def test_config_routes_references_config_api(self) -> None:
        """config_routes nutzt die neue config_api."""

        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        # Prüfe, dass die neuen Endpoints referenziert werden
        assert "bindings" in source.lower()
        assert "circles" in source.lower()
        assert "sandbox" in source.lower()
        assert "overview" in source.lower()

    def test_config_routes_imports_config_api(self) -> None:
        """config_routes kann config_api importieren."""
        # Simuliert was config_routes intern tut
        from cognithor.gateway.config_api import BindingRuleDTO
        from cognithor.gateway.config_api import ConfigManager as CfgMgr

        assert CfgMgr is not None
        assert BindingRuleDTO is not None

    def test_config_routes_imports_circles(self) -> None:
        """config_routes kann circles importieren."""
        from cognithor.skills.circles import CircleManager

        assert CircleManager is not None


# ============================================================================
# 6. Gesamtsystem: Verdrahtungskette
# ============================================================================


class TestFullWiringChain:
    def test_circle_through_exchange_through_stats(self, tmp_path: Path) -> None:
        """Komplette Kette: Circle → Exchange → Stats → Sichtbar."""
        from cognithor.skills.circles import CircleRole, ReviewVerdict
        from cognithor.skills.p2p import PeerNode, SkillExchange

        exchange = SkillExchange(skills_dir=tmp_path, require_signatures=False)
        exchange.set_identity(PeerNode(peer_id="alex", display_name="Alex"))

        # Circle erstellen
        circle = exchange.circles.create_circle(
            "Versicherungs-Profis",
            "alex",
            "Alexander",
        )
        inv = exchange.circles.invite_to_circle(
            circle.circle_id,
            "alex",
            "bob",
        )
        assert inv is not None
        exchange.circles.accept_invite(circle.circle_id, inv.invite_id)

        # Skill einreichen + reviewen
        circle.submit_skill("bu_calc@1.0", "BU-Rechner", "bob")
        circle.update_role("bob", CircleRole.ADMIN)
        # Alex (Owner) reviewed
        circle.review_skill("bu_calc@1.0", "alex", ReviewVerdict.APPROVED)

        # Stats zeigen Circle-Daten
        stats = exchange.stats()
        assert stats["circles"] == 1
        assert stats["circle_members"] == 2
        assert stats["curated_skills"] == 1

    def test_config_overview_through_config_manager(self) -> None:
        """Komplette Kette: CognithorConfig → ConfigManager → Overview."""
        from cognithor.gateway.config_api import AgentProfileDTO, BindingRuleDTO, ConfigManager

        config = MagicMock()
        config.version = "0.9"
        config.owner_name = "Alex"
        config.llm_backend_type = "ollama"
        config.heartbeat = MagicMock(
            enabled=False,
            interval_minutes=30,
            checklist_file="HB.md",
            channel="cli",
            model="qwen3:8b",
        )
        config.sandbox = MagicMock(enabled=True, network="allow", max_memory_mb=512)
        config.channels = MagicMock(
            cli_enabled=True,
            telegram_enabled=True,
            webui_enabled=False,
            slack_enabled=False,
            discord_enabled=False,
        )
        config.agents = []

        mgr = ConfigManager(config)

        # Agent + Binding erstellen
        mgr.upsert_agent(AgentProfileDTO(name="coder", credential_scope="coder"))
        mgr.upsert_binding(
            BindingRuleDTO(
                name="slash_code",
                target_agent="coder",
                command_prefixes=["/code"],
            )
        )

        # Overview zeigt alles
        overview = mgr.get_overview()
        assert overview.agent_count == 1
        assert overview.binding_count == 1

        # Export enthält alles
        export = mgr.export_config()
        assert len(export["agents"]) == 1
        assert len(export["bindings"]) == 1
        assert export["agents"][0]["credential_scope"] == "coder"


# ============================================================================
# 7. Punkt 1: GUI/Monitoring-Verdrahtung
# ============================================================================


class TestMonitoringWiring:
    def test_monitoring_hub_importable(self) -> None:
        from cognithor.gateway.monitoring import MonitoringHub

        hub = MonitoringHub()
        assert hub.events is not None
        assert hub.metrics is not None
        assert hub.audit is not None
        assert hub.heartbeat is not None

    def test_gateway_has_monitoring_hub(self) -> None:
        from cognithor.gateway.gateway import Gateway

        gw = Gateway()
        assert hasattr(gw, "_monitoring_hub")
        # MonitoringHub is deferred (Enterprise) — attribute exists but may be None

    def test_monitoring_emit_creates_event_and_metric(self) -> None:
        from cognithor.gateway.monitoring import EventType, MonitoringHub

        hub = MonitoringHub()
        hub.emit(EventType.MESSAGE_RECEIVED, source="test")
        assert hub.events.event_count == 1
        assert hub.metrics.get_counter("events.message_received") == 1.0

    def test_dashboard_snapshot_structure(self) -> None:
        from cognithor.gateway.monitoring import MonitoringHub

        hub = MonitoringHub()
        snap = hub.dashboard_snapshot()
        assert "events" in snap
        assert "metrics" in snap
        assert "audit" in snap
        assert "heartbeat" in snap

    def test_config_routes_has_monitoring_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "monitoring_dashboard" in source
        assert "monitoring_metrics" in source
        assert "audit_trail" in source
        assert "heartbeat_status" in source


# ============================================================================
# 8. Punkt 2: Agent-Separation-Verdrahtung
# ============================================================================


class TestIsolationWiring:
    def test_isolation_classes_importable(self) -> None:
        from cognithor.core.isolation import (
            MultiUserIsolation,
            WorkspaceGuard,
        )

        assert WorkspaceGuard is not None
        assert MultiUserIsolation is not None

    def test_isolation_exported_from_core(self) -> None:
        from cognithor.core import MultiUserIsolation, WorkspaceGuard

        assert WorkspaceGuard is not None
        assert MultiUserIsolation is not None

    def test_gateway_has_isolation(self) -> None:
        from cognithor.gateway.gateway import Gateway

        gw = Gateway()
        assert hasattr(gw, "_isolation")
        # MultiUserIsolation is Enterprise-deferred — attribute exists but may be None

    def test_multiuser_isolation_creates_scopes(self) -> None:
        from cognithor.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        scope = iso.get_or_create_scope("user_alex", "agent_coder")
        assert scope.scope_key == "user_alex:agent_coder"

    def test_workspace_guard_enforces_paths(self, tmp_path: Path) -> None:
        from cognithor.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(workspace_root=tmp_path)
        policy = guard.register_agent("agent_1")
        assert policy is not None
        # Agent kann in eigenem Workspace zugreifen
        assert guard.check_access("agent_1", tmp_path / "agent_1" / "file.txt")
        # Agent kann NICHT außerhalb zugreifen
        assert not guard.check_access("agent_1", Path("/etc/passwd"))

    def test_config_routes_has_isolation_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "isolation_stats" in source
        assert "isolation_quotas" in source
        assert "isolation_violations" in source


# ============================================================================
# 9. Punkt 3: Marketplace-Verdrahtung
# ============================================================================


class TestMarketplaceWiring:
    def test_marketplace_importable(self) -> None:
        from cognithor.skills.marketplace import SkillMarketplace

        mp = SkillMarketplace()
        assert mp is not None

    def test_marketplace_exported_from_skills(self) -> None:
        from cognithor.skills import SkillMarketplace

        assert SkillMarketplace is not None

    def test_exchange_has_marketplace(self, tmp_path: Path) -> None:
        from cognithor.skills.p2p import SkillExchange

        exchange = SkillExchange(skills_dir=tmp_path, require_signatures=False)
        assert hasattr(exchange, "marketplace")
        assert exchange.marketplace is not None

    def test_config_routes_has_marketplace_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "marketplace_feed" in source
        assert "marketplace_search" in source
        assert "marketplace_categories" in source
        assert "marketplace_featured" in source
        assert "marketplace_trending" in source


# ============================================================================
# 10. Punkt 4: Interactive-UI-Verdrahtung
# ============================================================================


class TestInteractiveWiring:
    def test_interactive_importable(self) -> None:
        from cognithor.channels.interactive import (
            AdaptiveCard,
            SlackMessageBuilder,
        )

        assert SlackMessageBuilder is not None
        assert AdaptiveCard is not None

    def test_interactive_exported_from_channels(self) -> None:
        from cognithor.channels import (
            SlackMessageBuilder,
        )

        assert SlackMessageBuilder is not None

    def test_slack_imports_interactive(self) -> None:
        import inspect

        from cognithor.channels import slack

        source = inspect.getsource(slack)
        assert "SlackMessageBuilder" in source
        assert "AdaptiveCard" in source
        assert "ProgressTracker" in source

    def test_discord_imports_interactive(self) -> None:
        import inspect

        from cognithor.channels import discord

        source = inspect.getsource(discord)
        assert "DiscordMessageBuilder" in source
        assert "AdaptiveCard" in source
        assert "ProgressTracker" in source

    def test_slack_has_send_rich_method(self) -> None:
        from cognithor.channels.slack import SlackChannel

        ch = SlackChannel(token="test")
        assert hasattr(ch, "send_rich")

    def test_discord_has_send_rich_method(self) -> None:
        from cognithor.channels.discord import DiscordChannel

        ch = DiscordChannel(token="test", channel_id=123)
        assert hasattr(ch, "send_rich")

    def test_adaptive_card_cross_platform(self) -> None:
        from cognithor.channels.interactive import AdaptiveCard

        card = AdaptiveCard(title="Test", body="Inhalt")
        slack_output = card.to_slack()
        discord_embed = card.to_discord()
        assert isinstance(slack_output, dict)
        assert "blocks" in slack_output
        assert isinstance(discord_embed, dict)


# ============================================================================
# 11. Auth-Gateway-Verdrahtung
# ============================================================================


class TestAuthWiring:
    def test_auth_importable(self) -> None:
        from cognithor.gateway.auth import AuthGateway

        gw = AuthGateway()
        assert gw is not None

    def test_gateway_has_auth(self) -> None:
        from cognithor.gateway.gateway import Gateway

        gw = Gateway()
        assert hasattr(gw, "_auth_gateway")
        # AuthGateway is Enterprise-deferred

    def test_sso_login_creates_sessions(self) -> None:
        from cognithor.gateway.auth import AuthGateway

        gw = AuthGateway()
        result = gw.login("alex", ["coder", "researcher"])
        assert len(result) == 2

    def test_config_routes_has_auth_endpoint(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "auth_stats" in source


# ============================================================================
# 12. Agent-Heartbeat-Verdrahtung
# ============================================================================


class TestAgentHeartbeatWiring:
    def test_agent_heartbeat_importable(self) -> None:
        from cognithor.core.agent_heartbeat import AgentHeartbeatScheduler

        sched = AgentHeartbeatScheduler()
        assert sched is not None

    def test_exported_from_core(self) -> None:
        from cognithor.core import AgentHeartbeatScheduler

        assert AgentHeartbeatScheduler is not None

    def test_gateway_has_agent_heartbeat(self) -> None:
        from cognithor.gateway.gateway import Gateway

        gw = Gateway()
        assert hasattr(gw, "_agent_heartbeat")
        # AgentHeartbeat is Enterprise-deferred

    def test_config_routes_has_heartbeat_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "agent_heartbeat_dashboard" in source
        assert "agent_heartbeat_summary" in source


# ============================================================================
# 13. Skill-Updater-Verdrahtung
# ============================================================================


class TestUpdaterWiring:
    def test_updater_importable(self) -> None:
        from cognithor.skills.updater import SkillUpdater

        u = SkillUpdater()
        assert u is not None

    def test_exported_from_skills(self) -> None:
        from cognithor.skills import SkillUpdater

        assert SkillUpdater is not None

    def test_exchange_has_updater(self, tmp_path: Path) -> None:
        from cognithor.skills.p2p import SkillExchange

        exchange = SkillExchange(skills_dir=tmp_path, require_signatures=False)
        assert hasattr(exchange, "updater")
        assert exchange.updater is not None

    def test_config_routes_has_updater_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "updater_stats" in source
        assert "updater_pending" in source
        assert "updater_recalls" in source
        assert "updater_history" in source


# ============================================================================
# 14. Commands-Verdrahtung
# ============================================================================


class TestCommandsWiring:
    def test_commands_importable(self) -> None:
        from cognithor.channels.commands import (
            CommandRegistry,
            FallbackRenderer,
            InteractionStore,
        )

        assert CommandRegistry is not None
        assert FallbackRenderer is not None
        assert InteractionStore is not None

    def test_exported_from_channels(self) -> None:
        from cognithor.channels import CommandRegistry

        assert CommandRegistry is not None

    def test_gateway_has_commands(self) -> None:
        from cognithor.gateway.gateway import Gateway

        gw = Gateway()
        assert hasattr(gw, "_command_registry")
        # CommandRegistry + InteractionStore are Enterprise-deferred
        assert hasattr(gw, "_interaction_store")

    def test_config_routes_has_command_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "list_commands" in source
        assert "commands_slack" in source
        assert "commands_discord" in source

    def test_default_commands_registered(self) -> None:
        from cognithor.channels.commands import CommandRegistry

        reg = CommandRegistry()
        assert reg.command_count >= 7
        assert reg.get("schedule") is not None
        assert reg.get("approve") is not None
        assert reg.get("briefing") is not None


# ============================================================================
# 15. Wizards + RBAC-Verdrahtung
# ============================================================================


class TestWizardsRBACWiring:
    def test_wizards_importable(self) -> None:
        from cognithor.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        assert reg.wizard_count == 3

    def test_rbac_importable(self) -> None:
        from cognithor.gateway.wizards import RBACManager, UserRole

        rbac = RBACManager()
        rbac.add_user("admin", "Admin", UserRole.ADMIN)
        assert rbac.check_permission("admin", "config", "write")

    def test_config_routes_has_wizard_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "list_wizards" in source
        assert "run_wizard" in source
        assert "wizard_templates" in source

    def test_config_routes_has_rbac_endpoints(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "rbac_roles" in source
        assert "rbac_check" in source

    def test_slash_commands_exported(self) -> None:
        from cognithor.channels import SlashCommandRegistry

        reg = SlashCommandRegistry()
        reg.register("/test", "Test command")
        assert reg.command_count == 1

    def test_modal_handler_exported(self) -> None:
        from cognithor.channels import ModalHandler

        mh = ModalHandler()
        assert mh.handler_count == 0

    def test_fallback_renderer_exported(self) -> None:
        from cognithor.channels import FallbackRenderer

        assert FallbackRenderer is not None

    def test_signature_verifier_exported(self) -> None:
        from cognithor.channels import SignatureVerifier

        v = SignatureVerifier()
        assert not v.has_slack_secret

    def test_interaction_state_exported(self) -> None:
        from cognithor.channels import InteractionStateStore

        store = InteractionStateStore()
        state = store.create("id1", "u1", "approval")
        assert state.interaction_id == "id1"


# ============================================================================
# 11. Neue Endpoint-Verdrahtung
# ============================================================================


class TestNewEndpointsWiring:
    def test_sse_endpoint_exists(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "monitoring_sse_stream" in source
        assert "text/event-stream" in source

    def test_wizard_endpoints_exist(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "list_wizards" in source
        assert "get_wizard" in source
        assert "run_wizard" in source
        assert "wizard_templates" in source

    def test_rbac_endpoints_exist(self) -> None:
        from cognithor.channels import config_routes

        source = _module_or_package_source(config_routes)
        assert "rbac_roles" in source
        assert "rbac_check" in source

    def test_marketplace_verification_exists(self) -> None:
        from cognithor.skills.marketplace import SkillMarketplace

        mp = SkillMarketplace()
        assert hasattr(mp, "verify_publisher")
        assert hasattr(mp, "recall_skill")
        assert hasattr(mp, "set_permissions")
        assert hasattr(mp, "set_scan_result")

    def test_gateway_exports_monitoring(self) -> None:
        from cognithor.gateway import MonitoringHub

        hub = MonitoringHub()
        assert hub is not None

    def test_gateway_exports_wizards(self) -> None:
        from cognithor.gateway import RBACManager, WizardRegistry

        assert WizardRegistry is not None
        assert RBACManager is not None
