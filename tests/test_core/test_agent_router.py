"""Tests für den Multi-Agent Router.

Jarvis ist ein universelles Agent-OS — keine hardcodierten
Branchen-Agenten. Alle Spezialisten werden vom Nutzer definiert.
Diese Tests prüfen die dynamische Agent-Verwaltung.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.core.agent_router import (
    AgentProfile,
    AgentRouter,
    DelegationRequest,
    RouteDecision,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def router() -> AgentRouter:
    """Router nur mit Default-Agent (Jarvis)."""
    r = AgentRouter()
    r.initialize()
    return r


@pytest.fixture
def custom_agents() -> list[AgentProfile]:
    """Vom Nutzer definierte Agenten (nicht hardcoded!)."""
    return [
        AgentProfile(
            name="researcher",
            display_name="Recherche",
            trigger_keywords=["recherche", "suche", "finde"],
            trigger_patterns=[r"recherchiere?\s+(?:zu|über)"],
            allowed_tools=["web_search", "read_file"],
            sandbox_network="allow",
        ),
        AgentProfile(
            name="coder",
            display_name="Entwickler",
            trigger_keywords=["code", "programmier", "script", "debug"],
            trigger_patterns=[r"(?:schreib|erstell)\s+ein"],
            allowed_tools=["exec_command", "write_file", "read_file"],
            sandbox_network="block",
            sandbox_max_memory_mb=1024,
            sandbox_timeout=120,
            can_delegate_to=["researcher"],
            max_delegation_depth=2,
        ),
    ]


@pytest.fixture
def full_router(custom_agents: list[AgentProfile]) -> AgentRouter:
    """Router mit nutzerdefinierten Agenten."""
    r = AgentRouter()
    r.initialize(custom_agents=custom_agents)
    return r


# ============================================================================
# Initialisierung
# ============================================================================


class TestInit:
    def test_only_jarvis_default(self, router: AgentRouter) -> None:
        agents = router.list_agents()
        assert len(agents) == 1
        assert agents[0].name == "jarvis"

    def test_jarvis_is_universal(self, router: AgentRouter) -> None:
        jarvis = router.get_agent("jarvis")
        assert jarvis is not None
        assert jarvis.shared_workspace is True
        assert jarvis.allowed_tools is None  # Keine Einschränkung

    def test_custom_agents_added(self, full_router: AgentRouter) -> None:
        agents = full_router.list_agents()
        names = [a.name for a in agents]
        assert "jarvis" in names
        assert "researcher" in names
        assert "coder" in names
        assert len(agents) == 3

    def test_custom_overrides_default(self) -> None:
        custom_jarvis = AgentProfile(
            name="jarvis",
            display_name="Mein Assistent",
            system_prompt="Du bist mein persönlicher Assistent.",
        )
        r = AgentRouter()
        r.initialize(custom_agents=[custom_jarvis])
        assert r.get_agent("jarvis").display_name == "Mein Assistent"


# ============================================================================
# Routing
# ============================================================================


class TestRouting:
    def test_route_to_custom_agent(self, full_router: AgentRouter) -> None:
        decision = full_router.route("Recherchiere zu KI-Sicherheit")
        assert decision.agent.name == "researcher"
        assert decision.confidence >= 0.7

    def test_route_to_coder(self, full_router: AgentRouter) -> None:
        decision = full_router.route("Schreib ein Python-Script")
        assert decision.agent.name == "coder"

    def test_fallback_to_jarvis(self, full_router: AgentRouter) -> None:
        decision = full_router.route("Was denkst du über das Wetter?")
        assert decision.agent.name == "jarvis"
        assert decision.confidence == 0.3

    def test_empty_query(self, router: AgentRouter) -> None:
        decision = router.route("")
        assert decision.agent.name == "jarvis"

    def test_no_custom_agents_all_goes_to_jarvis(self, router: AgentRouter) -> None:
        decision = router.route("Recherchiere was")
        assert decision.agent.name == "jarvis"

    def test_pattern_match(self, full_router: AgentRouter) -> None:
        decision = full_router.route("Recherchiere zu den neuesten Trends")
        assert decision.agent.name == "researcher"
        assert decision.confidence >= 0.8

    def test_confidence_range(self, full_router: AgentRouter) -> None:
        decision = full_router.route("code debug python")
        assert 0.0 <= decision.confidence <= 1.0


# ============================================================================
# Tool-Filterung
# ============================================================================


class TestToolFilter:
    def test_jarvis_no_restrictions(self, full_router: AgentRouter) -> None:
        jarvis = full_router.get_agent("jarvis")
        all_tools = {"exec_command": {}, "web_search": {}, "read_file": {}}
        filtered = jarvis.filter_tools(all_tools)
        assert filtered == all_tools

    def test_researcher_whitelist(self, full_router: AgentRouter) -> None:
        researcher = full_router.get_agent("researcher")
        all_tools = {"web_search": {}, "exec_command": {}, "read_file": {}}
        filtered = researcher.filter_tools(all_tools)
        assert "web_search" in filtered
        assert "read_file" in filtered
        assert "exec_command" not in filtered

    def test_blocked_tools(self) -> None:
        agent = AgentProfile(name="safe", blocked_tools=["exec_command"])
        all_tools = {"exec_command": {}, "web_search": {}}
        filtered = agent.filter_tools(all_tools)
        assert "exec_command" not in filtered
        assert "web_search" in filtered


# ============================================================================
# Workspace-Isolation
# ============================================================================


class TestWorkspaceIsolation:
    def test_jarvis_shared_workspace(self, full_router: AgentRouter, tmp_path: Path) -> None:
        workspace = full_router.resolve_agent_workspace("jarvis", tmp_path)
        assert workspace == tmp_path

    def test_custom_agent_isolated(self, full_router: AgentRouter, tmp_path: Path) -> None:
        workspace = full_router.resolve_agent_workspace("researcher", tmp_path)
        assert workspace != tmp_path
        assert "agents" in str(workspace)
        assert "researcher" in str(workspace)
        assert workspace.exists()

    def test_different_agents_different_dirs(self, full_router: AgentRouter, tmp_path: Path) -> None:
        ws_r = full_router.resolve_agent_workspace("researcher", tmp_path)
        ws_c = full_router.resolve_agent_workspace("coder", tmp_path)
        assert ws_r != ws_c

    def test_unknown_agent_gets_base(self, full_router: AgentRouter, tmp_path: Path) -> None:
        workspace = full_router.resolve_agent_workspace("nonexistent", tmp_path)
        assert workspace == tmp_path

    def test_custom_subdir(self, tmp_path: Path) -> None:
        agent = AgentProfile(name="custom", workspace_subdir="mein_bereich")
        workspace = agent.resolve_workspace(tmp_path)
        assert "mein_bereich" in str(workspace)

    def test_effective_workspace_subdir(self) -> None:
        agent = AgentProfile(name="test")
        assert agent.effective_workspace_subdir == "test"

        agent2 = AgentProfile(name="test", workspace_subdir="custom")
        assert agent2.effective_workspace_subdir == "custom"

        agent3 = AgentProfile(name="test", shared_workspace=True)
        assert agent3.effective_workspace_subdir == ""


# ============================================================================
# Per-Agent Sandbox
# ============================================================================


class TestPerAgentSandbox:
    def test_custom_sandbox_config(self, full_router: AgentRouter) -> None:
        coder = full_router.get_agent("coder")
        config = coder.get_sandbox_config()
        assert config["network"] == "block"
        assert config["max_memory_mb"] == 1024
        assert config["timeout"] == 120

    def test_researcher_allows_network(self, full_router: AgentRouter) -> None:
        researcher = full_router.get_agent("researcher")
        config = researcher.get_sandbox_config()
        assert config["network"] == "allow"

    def test_default_sandbox_values(self) -> None:
        agent = AgentProfile(name="fresh")
        config = agent.get_sandbox_config()
        assert config["network"] == "allow"
        assert config["max_memory_mb"] == 512
        assert config["max_processes"] == 64
        assert config["timeout"] == 30


# ============================================================================
# Delegation
# ============================================================================


class TestDelegation:
    def test_delegation_allowed(self, full_router: AgentRouter) -> None:
        assert full_router.can_delegate("coder", "researcher") is True

    def test_delegation_not_configured(self, full_router: AgentRouter) -> None:
        assert full_router.can_delegate("researcher", "coder") is False

    def test_delegation_to_nonexistent(self, full_router: AgentRouter) -> None:
        assert full_router.can_delegate("coder", "nonexistent") is False

    def test_create_delegation(self, full_router: AgentRouter) -> None:
        req = full_router.create_delegation("coder", "researcher", "Finde Infos zu Python 3.12")
        assert req is not None
        assert req.from_agent == "coder"
        assert req.to_agent == "researcher"
        assert req.depth == 1
        assert req.target_profile.name == "researcher"

    def test_delegation_blocked(self, full_router: AgentRouter) -> None:
        req = full_router.create_delegation("researcher", "coder", "Unmöglich")
        assert req is None

    def test_delegation_depth_exceeded(self, full_router: AgentRouter) -> None:
        req = full_router.create_delegation("coder", "researcher", "Test", depth=2)
        assert req is None

    def test_get_delegation_targets(self, full_router: AgentRouter) -> None:
        targets = full_router.get_delegation_targets("coder")
        names = [t.name for t in targets]
        assert "researcher" in names

    def test_no_delegation_targets(self, full_router: AgentRouter) -> None:
        targets = full_router.get_delegation_targets("researcher")
        assert targets == []


# ============================================================================
# Verwaltung
# ============================================================================


class TestManagement:
    def test_add_agent_runtime(self, router: AgentRouter) -> None:
        new = AgentProfile(name="custom_agent", display_name="Custom")
        router.add_agent(new)
        assert router.get_agent("custom_agent") is not None

    def test_remove_agent(self, full_router: AgentRouter) -> None:
        assert full_router.remove_agent("researcher") is True
        assert full_router.get_agent("researcher") is None

    def test_cannot_remove_jarvis(self, router: AgentRouter) -> None:
        assert router.remove_agent("jarvis") is False

    def test_stats(self, full_router: AgentRouter) -> None:
        stats = full_router.stats()
        assert stats["total_agents"] >= 3
        assert stats["default"] == "jarvis"

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = (
            "agents:\n"
            "  - name: yaml_agent\n"
            "    display_name: YAML Agent\n"
            "    trigger_keywords: [yaml, test]\n"
            "    sandbox_network: block\n"
            "    can_delegate_to: [jarvis]\n"
        )
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        router = AgentRouter.from_yaml(yaml_file)
        agent = router.get_agent("yaml_agent")
        assert agent is not None
        assert agent.sandbox_network == "block"
        assert "jarvis" in agent.can_delegate_to

    def test_from_yaml_nonexistent(self, tmp_path: Path) -> None:
        router = AgentRouter.from_yaml(tmp_path / "nonexistent.yaml")
        assert router.get_agent("jarvis") is not None
        assert len(router.list_agents()) == 1


# ============================================================================
# Dynamische Nutzer-Konfiguration
# ============================================================================


class TestDynamicConfig:
    """Verifiziert dass Agenten vollständig dynamisch konfigurierbar sind."""

    def test_user_defines_insurance_agent(self) -> None:
        insurance = AgentProfile(
            name="tarif_berater",
            display_name="Tarif-Berater",
            system_prompt="Du bist ein Experte für Versicherungstarife.",
            trigger_keywords=["tarif", "versicherung", "police", "BU"],
            allowed_tools=["web_search", "read_file"],
            can_delegate_to=["jarvis"],
        )
        r = AgentRouter()
        r.initialize(custom_agents=[insurance])
        decision = r.route("Vergleiche die BU-Tarife")
        assert decision.agent.name == "tarif_berater"

    def test_user_defines_dev_agent(self) -> None:
        dev = AgentProfile(
            name="my_coder",
            system_prompt="Du schreibst TypeScript und React.",
            trigger_keywords=["typescript", "react", "component"],
            sandbox_network="block",
            sandbox_timeout=180,
        )
        r = AgentRouter()
        r.initialize(custom_agents=[dev])
        decision = r.route("Erstelle eine React-Component")
        assert decision.agent.name == "my_coder"
        assert decision.agent.get_sandbox_config()["timeout"] == 180

    def test_user_defines_multilingual_agent(self) -> None:
        en_agent = AgentProfile(
            name="english",
            trigger_keywords=["english", "translate", "EN"],
            language="en",
        )
        r = AgentRouter()
        r.initialize(custom_agents=[en_agent])
        decision = r.route("Translate this to english please")
        assert decision.agent.language == "en"

    def test_completely_custom_setup(self) -> None:
        """Ein Nutzer baut ein komplett eigenes Agent-System."""
        agents = [
            AgentProfile(
                name="analyst",
                trigger_keywords=["analyse", "daten", "report"],
                can_delegate_to=["writer"],
            ),
            AgentProfile(
                name="writer",
                trigger_keywords=["schreib", "dokument", "bericht"],
                can_delegate_to=["analyst"],
            ),
        ]
        r = AgentRouter()
        r.initialize(custom_agents=agents)

        assert len(r.list_agents()) == 3  # jarvis + 2 custom
        assert r.can_delegate("analyst", "writer") is True
        assert r.can_delegate("writer", "analyst") is True


# ============================================================================
# Auto-Agent-Erstellung
# ============================================================================


class TestAutoCreateAgent:
    """Jarvis erstellt seine Agenten selbst zur Laufzeit."""

    def test_auto_create_basic(self, router: AgentRouter) -> None:
        agent = router.auto_create_agent(
            name="sales_bot",
            description="Verkaufsassistent für Kaltakquise",
            trigger_keywords=["akquise", "verkauf", "kaltanruf"],
        )
        assert agent.name == "sales_bot"
        assert router.get_agent("sales_bot") is not None

        # Sofort routbar
        decision = router.route("Starte die Akquise-Kampagne")
        assert decision.agent.name == "sales_bot"

    def test_auto_create_with_full_config(self, router: AgentRouter) -> None:
        agent = router.auto_create_agent(
            name="secure_coder",
            description="Sicherer Code-Agent",
            system_prompt="Du schreibst nur sicheren Code.",
            allowed_tools=["write_file", "read_file"],
            sandbox_network="block",
            can_delegate_to=["jarvis"],
        )
        assert agent.sandbox_network == "block"
        assert agent.can_delegate_to == ["jarvis"]
        assert agent.allowed_tools == ["write_file", "read_file"]

    def test_auto_create_and_persist(self, router: AgentRouter, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config" / "agents.yaml"

        router.auto_create_agent(
            name="persisted_agent",
            description="Test Persistenz",
            trigger_keywords=["test"],
            persist_path=yaml_path,
        )

        # YAML wurde geschrieben
        assert yaml_path.exists()
        content = yaml_path.read_text(encoding="utf-8")
        assert "persisted_agent" in content

        # Neuer Router kann es laden
        router2 = AgentRouter.from_yaml(yaml_path)
        assert router2.get_agent("persisted_agent") is not None

    def test_save_agents_yaml(self, full_router: AgentRouter, tmp_path: Path) -> None:
        yaml_path = tmp_path / "agents.yaml"
        full_router.save_agents_yaml(yaml_path)

        assert yaml_path.exists()
        content = yaml_path.read_text(encoding="utf-8")
        assert "researcher" in content
        assert "coder" in content

    def test_auto_create_overwrites_existing(self, router: AgentRouter) -> None:
        router.auto_create_agent(name="bot", description="v1")
        router.auto_create_agent(name="bot", description="v2")
        assert router.get_agent("bot").description == "v2"


# ============================================================================
# Per-Agent Sessions
# ============================================================================


class TestPerAgentSessions:
    """Verschiedene Agenten bekommen getrennte Sessions."""

    def test_session_includes_agent_name(self) -> None:
        from jarvis.models import SessionContext

        session = SessionContext(
            user_id="alex",
            channel="telegram",
            agent_name="researcher",
        )
        assert session.agent_name == "researcher"

    def test_default_agent_is_jarvis(self) -> None:
        from jarvis.models import SessionContext

        session = SessionContext(user_id="alex", channel="telegram")
        assert session.agent_name == "jarvis"

    def test_cronjob_has_agent_field(self) -> None:
        from jarvis.models import CronJob

        job = CronJob(
            name="morning_briefing",
            schedule="0 7 * * 1-5",
            prompt="Erstelle ein Morgen-Briefing",
            agent="organizer",
        )
        assert job.agent == "organizer"

    def test_cronjob_default_no_agent(self) -> None:
        from jarvis.models import CronJob

        job = CronJob(name="test", schedule="* * * * *", prompt="Test")
        assert job.agent == ""
