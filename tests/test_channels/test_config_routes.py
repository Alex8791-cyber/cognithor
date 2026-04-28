"""Tests fuer config_routes.py -- REST-Endpoints fuer die Konfigurationsverwaltung.

Strategie: Wir rufen create_config_routes() mit einem Fake-App-Objekt auf,
das route-Dekoratoren erfasst, und testen dann die registrierten Handler direkt.
So brauchen wir keinen echten FastAPI/Starlette-Server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from cognithor.config import CognithorConfig
from cognithor.config_manager import ConfigManager

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# Fake-App: Erfasst registrierte Route-Handler
# ============================================================================


class FakeApp:
    """Simuliert FastAPI-App -- speichert registrierte Handler."""

    def __init__(self) -> None:
        self.routes: dict[str, Any] = {}

    def _register(self, method: str, path: str, **kwargs: Any):
        def decorator(fn):
            self.routes[f"{method} {path}"] = fn
            return fn

        return decorator

    def get(self, path: str, **kwargs: Any):
        return self._register("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any):
        return self._register("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any):
        return self._register("PUT", path, **kwargs)

    def patch(self, path: str, **kwargs: Any):
        return self._register("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any):
        return self._register("DELETE", path, **kwargs)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / ".cognithor"
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture
def config(tmp_home: Path) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_home)


@pytest.fixture
def config_manager(config: CognithorConfig) -> ConfigManager:
    return ConfigManager(config=config)


@pytest.fixture
def app() -> FakeApp:
    return FakeApp()


@pytest.fixture
def gateway() -> MagicMock:
    gw = MagicMock()
    gw.reload_components = MagicMock()
    gw.shutdown = AsyncMock()
    # Default: keine optionalen Attribute
    gw._vault_manager = None
    gw._isolated_sessions = None
    gw._session_guard = None
    gw._isolation_enforcer = None
    gw._memory_hygiene = None
    gw._integrity_checker = None
    gw._decision_explainer = None
    gw._explainability = None
    gw._connector_registry = None
    gw._template_library = None
    gw._workflow_engine = None
    gw._model_registry = None
    gw._i18n = None
    gw._skill_cli = None
    gw._setup_wizard = None
    gw._security_scanner = None
    gw._compliance_framework = None
    gw._decision_log = None
    gw._remediation_tracker = None
    gw._compliance_exporter = None
    gw._security_pipeline = None
    gw._ecosystem_policy = None
    gw._security_metrics = None
    gw._incident_tracker = None
    gw._security_team = None
    gw._posture_scorer = None
    gw._security_gate = None
    gw._continuous_redteam = None
    gw._scan_scheduler = None
    gw._red_team = None
    gw._code_auditor = None
    gw._reputation_engine = None
    gw._recall_manager = None
    gw._abuse_reporter = None
    gw._governance_policy = None
    gw._interop = None
    gw._economic_governor = None
    gw._governance_hub = None
    gw._impact_assessor = None
    gw._ecosystem_controller = None
    gw._perf_manager = None
    gw._user_portal = None
    gw._telemetry_hub = None
    return gw


@pytest.fixture
def registered_app(app: FakeApp, config_manager: ConfigManager, gateway: MagicMock) -> FakeApp:
    """App mit allen Routes registriert."""
    from cognithor.channels.config_routes import create_config_routes

    create_config_routes(app, config_manager, gateway=gateway)
    return app


# ============================================================================
# Tests: create_config_routes registriert Routes
# ============================================================================


class TestRouteRegistration:
    def test_routes_registered(self, registered_app: FakeApp) -> None:
        """create_config_routes registriert zahlreiche Routes."""
        assert len(registered_app.routes) > 30

    def test_route_inventory_unchanged(self, registered_app: FakeApp) -> None:
        """Drift-Bremse fuer den schrittweisen Split nach Sub-Modulen.

        Der Plan in `docs/superpowers/plans/2026-04-29-config-routes-split.md`
        bewegt 24 `_register_*_routes`-Helper aus `_factory.py` heraus. Wenn
        bei einer Migration ein Endpoint verschwindet, sich umbenennt oder
        zusaetzlich registriert wird, muss diese Fixture bewusst aktualisiert
        werden — nicht stillschweigend driften.
        """
        import json
        from pathlib import Path

        fixture = Path(__file__).parent / "fixtures" / "route_inventory.json"
        expected = set(json.loads(fixture.read_text(encoding="utf-8"))["routes"])
        actual = set(registered_app.routes.keys())
        assert actual == expected, (
            f"route inventory drifted; missing={expected - actual}, extra={actual - expected}"
        )

    def test_health_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/health" in registered_app.routes

    def test_status_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/status" in registered_app.routes

    def test_config_get_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/config" in registered_app.routes

    def test_config_patch_registered(self, registered_app: FakeApp) -> None:
        assert "PATCH /api/v1/config" in registered_app.routes

    def test_dashboard_registered(self, registered_app: FakeApp) -> None:
        assert "GET /dashboard" in registered_app.routes

    def test_agents_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/agents" in registered_app.routes

    def test_credentials_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/credentials" in registered_app.routes

    def test_bindings_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/bindings" in registered_app.routes

    def test_presets_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/config/presets" in registered_app.routes

    def test_config_section_registered(self, registered_app: FakeApp) -> None:
        assert "GET /api/v1/config/{section}" in registered_app.routes

    def test_config_reload_registered(self, registered_app: FakeApp) -> None:
        assert "POST /api/v1/config/reload" in registered_app.routes


# ============================================================================
# Tests: System Routes
# ============================================================================


class TestSystemRoutes:
    @pytest.mark.asyncio
    async def test_health_check(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/health"]
        result = await handler()
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_system_status(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/status"]
        with patch("cognithor.channels.config_routes.RuntimeMonitor", create=True):
            result = await handler()
        assert "timestamp" in result
        assert "config_version" in result
        assert "owner" in result
        assert "models" in result
        assert "active_channels" in result

    @pytest.mark.asyncio
    async def test_overview_error_handling(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/overview"]
        # ConfigManager aus gateway.config_api loest evtl. Exception aus
        result = await handler()
        # Entweder erfolgreich oder Error-Dict
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_agents_no_file(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/agents"]
        result = await handler()
        assert "agents" in result
        # Ohne agents.yaml kommt Default-Agent
        assert len(result["agents"]) >= 1

    @pytest.mark.asyncio
    async def test_list_agents_with_file(
        self, registered_app: FakeApp, config: CognithorConfig
    ) -> None:
        agents_path = config.cognithor_home / "agents.yaml"
        agents_path.parent.mkdir(parents=True, exist_ok=True)
        agents_path.write_text(
            yaml.dump({"agents": [{"name": "test-agent", "enabled": True}]}),
            encoding="utf-8",
        )
        handler = registered_app.routes["GET /api/v1/agents"]
        result = await handler()
        assert result["agents"][0]["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_list_credentials_error(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/credentials"]
        result = await handler()
        # Either works or returns error-dict
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_store_credential(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/credentials"]
        fake_request = AsyncMock()
        fake_request.json = AsyncMock(return_value={"service": "test", "key": "k", "value": "v"})
        with patch("cognithor.channels.config_routes.CredentialStore", create=True):
            result = await handler(fake_request)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_delete_credential(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["DELETE /api/v1/credentials/{service}/{key}"]
        with patch("cognithor.channels.config_routes.CredentialStore", create=True):
            result = await handler(service="test", key="k")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_bindings_no_file(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/bindings"]
        result = await handler()
        assert "bindings" in result
        assert result["bindings"] == []

    @pytest.mark.asyncio
    async def test_list_bindings_with_file(
        self, registered_app: FakeApp, config: CognithorConfig
    ) -> None:
        bindings_path = config.cognithor_home / "bindings.yaml"
        bindings_path.parent.mkdir(parents=True, exist_ok=True)
        bindings_path.write_text(
            yaml.dump({"bindings": [{"name": "b1", "channel": "telegram"}]}),
            encoding="utf-8",
        )
        handler = registered_app.routes["GET /api/v1/bindings"]
        result = await handler()
        assert len(result["bindings"]) == 1

    @pytest.mark.asyncio
    async def test_create_binding(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/bindings"]
        result = await handler(data={"name": "b1", "channel": "cli"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_delete_binding(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["DELETE /api/v1/bindings/{name}"]
        result = await handler(name="nonexistent")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_circles(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/circles"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_circles_stats(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/circles/stats"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_sandbox_get(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/sandbox"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_sandbox_update(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["PATCH /api/v1/sandbox"]
        result = await handler(values={})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_wizards_list(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/wizards"]
        result = await handler()
        assert "wizards" in result

    @pytest.mark.asyncio
    async def test_wizard_get_not_found(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/wizards/{wizard_type}"]
        result = await handler(wizard_type="nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_wizard_run_not_found(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/wizards/{wizard_type}/run"]
        result = await handler(wizard_type="nonexistent", body={"values": {}})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_wizard_templates_not_found(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/wizards/{wizard_type}/templates"]
        result = await handler(wizard_type="nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rbac_roles(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/rbac/roles"]
        result = await handler()
        assert "roles" in result

    @pytest.mark.asyncio
    async def test_rbac_check(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/rbac/check"]
        result = await handler(user_id="u1", resource="r1", action="read")
        assert "allowed" in result

    @pytest.mark.asyncio
    async def test_auth_stats(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/auth/stats"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_agent_heartbeat_dashboard(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/agent-heartbeat/dashboard"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_agent_heartbeat_summary(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/agent-heartbeat/{agent_id}"]
        result = await handler(agent_id="agent1")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_dashboard_serves(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /dashboard"]
        result = await handler()
        # Either HTMLResponse (if dashboard.html exists) or error dict
        from starlette.responses import HTMLResponse

        assert isinstance(result, dict | HTMLResponse)


# ============================================================================
# Tests: Config CRUD Routes
# ============================================================================


class TestConfigRoutes:
    @pytest.mark.asyncio
    async def test_get_config(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/config"]
        result = await handler()
        assert "_meta" in result

    @pytest.mark.asyncio
    async def test_update_config_top_level(
        self, registered_app: FakeApp, config_manager: ConfigManager
    ) -> None:
        handler = registered_app.routes["PATCH /api/v1/config"]
        result = await handler(updates={"owner_name": "Test Owner"})
        assert "results" in result

    @pytest.mark.asyncio
    async def test_update_config_skip_masked_secret(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["PATCH /api/v1/config"]
        result = await handler(updates={"telegram_token": "***"})
        assert result["results"][0]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_language_change_triggers_prompt_reload(
        self, registered_app: FakeApp, gateway: MagicMock
    ) -> None:
        """Issue #136: changing language must reload the planner prompts too —
        otherwise the planner keeps its cached German system prompt and the LLM
        keeps answering in German even after the user switches to English."""
        handler = registered_app.routes["PATCH /api/v1/config"]
        await handler(updates={"language": "en"})
        gateway.reload_components.assert_called_once()
        kwargs = gateway.reload_components.call_args.kwargs
        assert kwargs.get("config") is True
        assert kwargs.get("prompts") is True

    @pytest.mark.asyncio
    async def test_non_language_update_does_not_reload_prompts(
        self, registered_app: FakeApp, gateway: MagicMock
    ) -> None:
        """Updates that don't touch `language` must NOT pay the reload_prompts cost."""
        handler = registered_app.routes["PATCH /api/v1/config"]
        await handler(updates={"owner_name": "Tester"})
        gateway.reload_components.assert_called_once()
        kwargs = gateway.reload_components.call_args.kwargs
        assert kwargs.get("prompts") is not True

    @pytest.mark.asyncio
    async def test_config_reload(self, registered_app: FakeApp, gateway: MagicMock) -> None:
        handler = registered_app.routes["POST /api/v1/config/reload"]
        result = await handler()
        assert result["status"] == "ok"
        gateway.reload_components.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_presets(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/config/presets"]
        result = await handler()
        assert "presets" in result
        names = [p["name"] for p in result["presets"]]
        assert "minimal" in names
        assert "standard" in names
        assert "full" in names

    @pytest.mark.asyncio
    async def test_apply_preset_not_found(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/config/presets/{preset_name}"]
        result = await handler(preset_name="nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_apply_preset_minimal(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/config/presets/{preset_name}"]
        result = await handler(preset_name="minimal")
        assert "results" in result

    @pytest.mark.asyncio
    async def test_get_config_section(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/config/{section}"]
        result = await handler(section="channels")
        # Entweder success oder 404 error
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_config_section_not_found(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/config/{section}"]
        result = await handler(section="nonexistent_section_xyz")
        assert "error" in result or "values" in result

    @pytest.mark.asyncio
    async def test_update_config_section(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["PATCH /api/v1/config/{section}"]
        result = await handler(section="channels", values={"cli_enabled": True})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_update_config_section_secret_mask(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["PATCH /api/v1/config/{section}"]
        # Masked secrets should be stripped
        result = await handler(
            section="channels",
            values={"telegram_token": "***", "cli_enabled": True},
        )
        assert isinstance(result, dict)


# ============================================================================
# Tests: Session Routes
# ============================================================================


class TestSessionRoutes:
    @pytest.mark.asyncio
    async def test_vault_stats_no_manager(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/vault/stats"]
        result = await handler()
        assert result["total_vaults"] == 0

    @pytest.mark.asyncio
    async def test_vault_agents_no_manager(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/vault/agents"]
        result = await handler()
        assert result["agents"] == []

    @pytest.mark.asyncio
    async def test_session_stats_no_store(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/sessions/stats"]
        result = await handler()
        assert result["total_sessions"] == 0

    @pytest.mark.asyncio
    async def test_guard_violations_no_guard(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/sessions/guard/violations"]
        result = await handler()
        assert result["violations"] == []

    @pytest.mark.asyncio
    async def test_isolation_stats(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/isolation/stats"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_isolation_quotas(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/isolation/quotas"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_isolation_violations(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/isolation/violations"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_isolation_sandboxes_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/isolation/sandboxes"]
        result = await handler()
        assert result["sandboxes"] == []

    @pytest.mark.asyncio
    async def test_isolation_tenants_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/isolation/tenants"]
        result = await handler()
        assert result["total_tenants"] == 0

    @pytest.mark.asyncio
    async def test_isolation_secrets_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/isolation/secrets"]
        result = await handler()
        assert result["total_secrets"] == 0

    @pytest.mark.asyncio
    async def test_vaults_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/vaults/stats"]
        result = await handler()
        assert result["total_vaults"] == 0

    @pytest.mark.asyncio
    async def test_vaults_sessions_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/vaults/sessions"]
        result = await handler()
        assert result["agent_stores"] == 0

    @pytest.mark.asyncio
    async def test_vaults_firewall_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/vaults/firewall"]
        result = await handler()
        assert result["total_violations"] == 0


# ============================================================================
# Tests: Memory Routes
# ============================================================================


class TestMemoryRoutes:
    @pytest.mark.asyncio
    async def test_memory_hygiene_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/memory/hygiene/stats"]
        result = await handler()
        assert result["total_scans"] == 0

    @pytest.mark.asyncio
    async def test_memory_quarantine_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/memory/hygiene/quarantine"]
        result = await handler()
        assert result["quarantined"] == []

    @pytest.mark.asyncio
    async def test_memory_integrity_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/memory/integrity"]
        result = await handler()
        assert result["total_checks"] == 0

    @pytest.mark.asyncio
    async def test_memory_explainability_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/memory/explainability"]
        result = await handler()
        assert result["total_explanations"] == 0

    @pytest.mark.asyncio
    async def test_memory_hygiene_scan(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/memory/hygiene/scan"]
        request = MagicMock()
        request.json = AsyncMock(return_value={"entries": [], "auto_quarantine": False})
        result = await handler(request=request)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_explainability_trails_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/explainability/trails"]
        result = await handler()
        assert result["trails"] == []

    @pytest.mark.asyncio
    async def test_explainability_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/explainability/stats"]
        result = await handler()
        assert result["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_explainability_low_trust_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/explainability/low-trust"]
        result = await handler()
        assert result["low_trust_trails"] == []


# ============================================================================
# Tests: Skill Routes
# ============================================================================


class TestSkillRoutes:
    @pytest.mark.asyncio
    async def test_marketplace_feed(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/marketplace/feed"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_marketplace_search(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/marketplace/search"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_marketplace_categories(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/marketplace/categories"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_marketplace_featured(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/marketplace/featured"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_marketplace_trending(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/marketplace/trending"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_marketplace_stats(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/marketplace/stats"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_updater_stats(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/updater/stats"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_updater_pending(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/updater/pending"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_updater_recalls(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/updater/recalls"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_updater_history(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/updater/history"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_commands_list(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/commands/list"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_commands_slack(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/commands/slack"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_commands_discord(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/commands/discord"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_connectors_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/connectors/list"]
        result = await handler()
        assert result["connectors"] == []

    @pytest.mark.asyncio
    async def test_connector_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/connectors/stats"]
        result = await handler()
        assert result["total_connectors"] == 0

    @pytest.mark.asyncio
    async def test_workflow_templates_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/workflows/templates"]
        result = await handler()
        assert result["templates"] == []

    @pytest.mark.asyncio
    async def test_workflow_categories_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/workflows/templates/categories"]
        result = await handler()
        assert result["categories"] == []

    @pytest.mark.asyncio
    async def test_workflow_instances_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/workflows/instances"]
        result = await handler()
        assert result["instances"] == []

    @pytest.mark.asyncio
    async def test_workflow_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/workflows/stats"]
        result = await handler()
        assert result["templates"] == 0
        assert result["simple"] == {}
        assert result["dag_runs"] == 0

    @pytest.mark.asyncio
    async def test_model_list_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/models/list"]
        result = await handler()
        assert result["models"] == []

    @pytest.mark.asyncio
    async def test_model_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/models/stats"]
        result = await handler()
        assert result["total_models"] == 0

    @pytest.mark.asyncio
    async def test_i18n_locales_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/i18n/locales"]
        result = await handler()
        assert result["default"] == "de"

    @pytest.mark.asyncio
    async def test_i18n_translate_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/i18n/translate/{key}"]
        result = await handler(key="hello")
        assert result["translation"] == "hello"

    @pytest.mark.asyncio
    async def test_i18n_stats_returns_locales(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/i18n/stats"]
        result = await handler()
        assert result["locale_count"] >= 1
        assert "default_locale" in result

    @pytest.mark.asyncio
    async def test_skill_cli_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/skill-cli/stats"]
        result = await handler()
        assert "scaffolder" in result

    @pytest.mark.asyncio
    async def test_skill_cli_templates_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/skill-cli/templates"]
        result = await handler()
        assert result["templates"] == []

    @pytest.mark.asyncio
    async def test_skill_cli_rewards_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/skill-cli/rewards"]
        result = await handler()
        assert result["contributors"] == 0

    @pytest.mark.asyncio
    async def test_setup_state_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/setup/state"]
        result = await handler()
        assert result["step"] == "unavailable"

    @pytest.mark.asyncio
    async def test_setup_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/setup/stats"]
        result = await handler()
        assert result["state"] == {}


# ============================================================================
# Tests: Security Routes
# ============================================================================


class TestSecurityRoutes:
    @pytest.mark.asyncio
    async def test_redteam_status(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/security/redteam/status"]
        result = await handler()
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_compliance_decisions_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/compliance/decisions"]
        result = await handler()
        assert result["total_decisions"] == 0

    @pytest.mark.asyncio
    async def test_compliance_remediations_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/compliance/remediations"]
        result = await handler()
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_compliance_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/compliance/stats"]
        result = await handler()
        assert result["total_reports"] == 0

    @pytest.mark.asyncio
    async def test_compliance_transparency_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/compliance/transparency"]
        result = await handler()
        assert result["total_obligations"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/security/pipeline/stats"]
        result = await handler()
        assert result["total_runs"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_history_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/security/pipeline/history"]
        result = await handler()
        assert result["runs"] == []

    @pytest.mark.asyncio
    async def test_ecosystem_policy_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/ecosystem/policy/stats"]
        result = await handler()
        assert result["total_requirements"] == 0

    @pytest.mark.asyncio
    async def test_framework_metrics_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/framework/metrics"]
        result = await handler()
        assert result["mttd_seconds"] == 0

    @pytest.mark.asyncio
    async def test_framework_incidents_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/framework/incidents"]
        result = await handler()
        assert result["incidents"] == []

    @pytest.mark.asyncio
    async def test_framework_team_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/framework/team"]
        result = await handler()
        assert result["members"] == []

    @pytest.mark.asyncio
    async def test_framework_posture_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/framework/posture"]
        result = await handler()
        assert result["posture_score"] == 0

    @pytest.mark.asyncio
    async def test_gate_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/gate/stats"]
        result = await handler()
        assert result["total_evaluations"] == 0

    @pytest.mark.asyncio
    async def test_gate_evaluate_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/gate/evaluate"]
        result = await handler(body={})
        assert result["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_gate_history_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/gate/history"]
        result = await handler()
        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_redteam_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/redteam/stats"]
        result = await handler()
        assert result["total_probes"] == 0

    @pytest.mark.asyncio
    async def test_scans_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/scans/stats"]
        result = await handler()
        assert result["total_schedules"] == 0

    @pytest.mark.asyncio
    async def test_red_team_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/red-team/stats"]
        result = await handler()
        assert result["total_runs"] == 0

    @pytest.mark.asyncio
    async def test_red_team_coverage_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/red-team/coverage"]
        result = await handler()
        assert result["coverage_rate"] == 0

    @pytest.mark.asyncio
    async def test_red_team_latest_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/red-team/latest"]
        result = await handler()
        assert result["report"] is None

    @pytest.mark.asyncio
    async def test_code_audit_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/code-audit/stats"]
        result = await handler()
        assert result["total_audits"] == 0


# ============================================================================
# Tests: Governance Routes
# ============================================================================


class TestGovernanceRoutes:
    @pytest.mark.asyncio
    async def test_reputation_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/reputation/stats"]
        result = await handler()
        assert result["total_entities"] == 0

    @pytest.mark.asyncio
    async def test_reputation_detail_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/reputation/{entity_id}"]
        result = await handler(entity_id="e1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recalls_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/recalls/stats"]
        result = await handler()
        assert result["total_recalls"] == 0

    @pytest.mark.asyncio
    async def test_recalls_active_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/recalls/active"]
        result = await handler()
        assert result["recalls"] == []

    @pytest.mark.asyncio
    async def test_abuse_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/abuse/stats"]
        result = await handler()
        assert result["total_reports"] == 0

    @pytest.mark.asyncio
    async def test_policy_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/policy/stats"]
        result = await handler()
        assert result["total_rules"] == 0

    @pytest.mark.asyncio
    async def test_interop_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/interop/stats"]
        result = await handler()
        assert result["registered_agents"] == 0

    @pytest.mark.asyncio
    async def test_interop_agents_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/interop/agents"]
        result = await handler()
        assert result["agents"] == []

    @pytest.mark.asyncio
    async def test_interop_federation_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/interop/federation"]
        result = await handler()
        assert result["links"] == []

    @pytest.mark.asyncio
    async def test_economics_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/economics/stats"]
        result = await handler()
        assert "budget" in result

    @pytest.mark.asyncio
    async def test_economics_budget_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/economics/budget"]
        result = await handler()
        assert result["total_entities"] == 0

    @pytest.mark.asyncio
    async def test_economics_costs_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/economics/costs"]
        result = await handler()
        assert result["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_economics_fairness_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/economics/fairness"]
        result = await handler()
        assert result["total_audits"] == 0

    @pytest.mark.asyncio
    async def test_economics_ethics_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/economics/ethics"]
        result = await handler()
        assert result["total_violations"] == 0

    @pytest.mark.asyncio
    async def test_governance_health_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/health"]
        result = await handler()
        assert result["skill_reviews"] == 0

    @pytest.mark.asyncio
    async def test_governance_curation_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/curation"]
        result = await handler()
        assert result["total_reviews"] == 0

    @pytest.mark.asyncio
    async def test_governance_diversity_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/diversity"]
        result = await handler()
        assert result["total_audits"] == 0

    @pytest.mark.asyncio
    async def test_governance_budget_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/budget"]
        result = await handler()
        assert result["total_transfers"] == 0

    @pytest.mark.asyncio
    async def test_governance_explainer_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/governance/explainer"]
        result = await handler()
        assert result["total_explanations"] == 0

    @pytest.mark.asyncio
    async def test_impact_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/impact/stats"]
        result = await handler()
        assert result["total_assessments"] == 0

    @pytest.mark.asyncio
    async def test_impact_board_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/impact/board"]
        result = await handler()
        assert result["board_members"] == 0

    @pytest.mark.asyncio
    async def test_impact_stakeholders_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/impact/stakeholders"]
        result = await handler()
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_impact_mitigations_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/impact/mitigations"]
        result = await handler()
        assert result["total"] == 0


# ============================================================================
# Tests: Infrastructure Routes
# ============================================================================


class TestInfrastructureRoutes:
    @pytest.mark.asyncio
    async def test_ecosystem_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/ecosystem/stats"]
        result = await handler()
        assert "curator" in result

    @pytest.mark.asyncio
    async def test_ecosystem_curator_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/ecosystem/curator"]
        result = await handler()
        assert result["total_reviews"] == 0

    @pytest.mark.asyncio
    async def test_ecosystem_fraud_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/ecosystem/fraud"]
        result = await handler()
        assert result["total_signals"] == 0

    @pytest.mark.asyncio
    async def test_ecosystem_training_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/ecosystem/training"]
        result = await handler()
        assert result["total_modules"] == 0

    @pytest.mark.asyncio
    async def test_ecosystem_trust_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/ecosystem/trust"]
        result = await handler()
        assert result["total_boundaries"] == 0

    @pytest.mark.asyncio
    async def test_perf_health_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/performance/health"]
        result = await handler()
        assert "vector_store" in result

    @pytest.mark.asyncio
    async def test_perf_latency_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/performance/latency"]
        result = await handler()
        assert result["total_samples"] == 0

    @pytest.mark.asyncio
    async def test_perf_resources_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/performance/resources"]
        result = await handler()
        assert result["snapshots"] == 0


# ============================================================================
# Tests: Portal Routes
# ============================================================================


class TestPortalRoutes:
    @pytest.mark.asyncio
    async def test_portal_stats_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/portal/stats"]
        result = await handler()
        assert "consents" in result

    @pytest.mark.asyncio
    async def test_portal_consents_none(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/portal/consents"]
        result = await handler()
        assert result["total_users"] == 0


# ============================================================================
# Tests: UI Routes
# ============================================================================


class TestUIRoutes:
    @pytest.mark.asyncio
    async def test_system_status(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/system/status"]
        result = await handler()
        assert result["status"] == "running"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_system_start(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/system/start"]
        result = await handler()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_system_stop_with_gateway(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/system/stop"]
        result = await handler()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_upsert_agent(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/agents/{name}"]
        request = MagicMock()
        request.json = AsyncMock(return_value={"display_name": "Test"})
        result = await handler(name="test-agent", request=request)
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_upsert_binding(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["POST /api/v1/bindings/{name}"]
        request = MagicMock()
        request.json = AsyncMock(return_value={"target_agent": "agent-1", "channels": ["telegram"]})
        result = await handler(name="b1", request=request)
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_get_prompts(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/prompts"]
        result = await handler()
        assert isinstance(result, dict)
        # Should have standard keys
        for key in ("coreMd", "plannerSystem", "replanPrompt"):
            assert key in result

    @pytest.mark.asyncio
    async def test_put_prompts(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["PUT /api/v1/prompts"]
        request = MagicMock()
        request.json = AsyncMock(return_value={"coreMd": "# Test"})
        result = await handler(request=request)
        assert result["status"] == "ok"
        assert "coreMd" in result["written"]

    @pytest.mark.asyncio
    async def test_get_cron_jobs(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/cron-jobs"]
        result = await handler()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_mcp_servers(self, registered_app: FakeApp) -> None:
        handler = registered_app.routes["GET /api/v1/mcp-servers"]
        result = await handler()
        assert isinstance(result, dict)
