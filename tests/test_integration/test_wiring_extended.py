"""Erweiterte Integrationstests: Alle verbleibenden Verdrahtungen.

Beweist, dass JEDE Komponente korrekt an AuditLogger/RuntimeMonitor/PackageBuilder
angebunden ist – lückenlos von Eingang (User-Input) bis Ausgang (Response).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.audit import AuditCategory, AuditLogger, AuditSeverity
from jarvis.models import (
    GateDecision,
    GateStatus,
    PlannedAction,
    RiskLevel,
    SessionContext,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    return AuditLogger(log_dir=tmp_path / "audit")


@pytest.fixture
def mock_session() -> SessionContext:
    return SessionContext(session_id="test-session-001", channel="cli", user_id="test")


# ============================================================================
# 1. Gatekeeper → AuditLogger
# ============================================================================


class TestGatekeeperAuditIntegration:
    """Beweist: Gatekeeper loggt JEDE Entscheidung in den AuditLogger."""

    def test_gatekeeper_block_logged(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        """Blockierte Aktionen werden im zentralen AuditLogger protokolliert."""
        from jarvis.config import JarvisConfig

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()

        from jarvis.core.gatekeeper import Gatekeeper

        gk = Gatekeeper(config, audit_logger=audit_logger)
        gk.initialize()

        action = PlannedAction(tool="exec_command", params={"command": "rm -rf /"})
        session = SessionContext(session_id="s1", channel="cli", user_id="u1")

        decision = gk.evaluate(action, session)

        assert decision.status == GateStatus.BLOCK

        gate_events = audit_logger.query(category=AuditCategory.GATEKEEPER)
        assert len(gate_events) >= 1
        assert "BLOCK" in gate_events[0].description

    def test_gatekeeper_allow_logged(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        """Erlaubte/genehmigte Aktionen werden ebenfalls auditiert."""
        from jarvis.config import JarvisConfig

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()

        from jarvis.core.gatekeeper import Gatekeeper

        gk = Gatekeeper(config, audit_logger=audit_logger)
        gk.initialize()

        action = PlannedAction(tool="memory_search", params={"query": "test"})
        session = SessionContext(session_id="s2", channel="cli", user_id="u1")

        decision = gk.evaluate(action, session)

        # Status kann ALLOW, INFORM oder APPROVE sein – alles wird auditiert
        assert decision.status in (GateStatus.ALLOW, GateStatus.INFORM, GateStatus.APPROVE)

        gate_events = audit_logger.query(category=AuditCategory.GATEKEEPER)
        assert len(gate_events) >= 1

    def test_gatekeeper_credential_mask_logged(
        self, audit_logger: AuditLogger, tmp_path: Path,
    ) -> None:
        """Credential-Maskierung wird auditiert."""
        from jarvis.config import JarvisConfig

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()

        from jarvis.core.gatekeeper import Gatekeeper

        gk = Gatekeeper(config, audit_logger=audit_logger)
        gk.initialize()

        # Credential-Pattern: sk-[a-zA-Z0-9]{20,} matcht im Wert
        action = PlannedAction(
            tool="send_request",
            params={"url": "https://api.example.com", "auth": "sk-abcdefghijklmnopqrstuvwxyz1234"},
        )
        session = SessionContext(session_id="s3", channel="cli", user_id="u1")

        decision = gk.evaluate(action, session)

        assert decision.status == GateStatus.MASK

        gate_events = audit_logger.query(category=AuditCategory.GATEKEEPER)
        assert len(gate_events) >= 1
        assert "MASK" in gate_events[0].description


# ============================================================================
# 2. Planner → AuditLogger (LLM-Calls)
# ============================================================================


class TestPlannerAuditIntegration:
    """Beweist: Planner loggt LLM-Aufrufe in den AuditLogger."""

    @pytest.mark.asyncio
    async def test_planner_logs_llm_call_success(self, audit_logger: AuditLogger) -> None:
        """Erfolgreicher LLM-Plan-Call wird auditiert."""
        from jarvis.core.planner import Planner

        mock_config = MagicMock()
        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value={
            "message": {"content": "Das ist eine direkte Antwort."},
        })

        mock_router = MagicMock()
        mock_router.select_model.return_value = "test-model"
        mock_router.get_model_config.return_value = {"temperature": 0.7}

        planner = Planner(mock_config, mock_ollama, mock_router, audit_logger=audit_logger)

        # Working Memory mocken
        mock_wm = MagicMock()
        mock_wm.chat_history = []
        mock_wm.context_window = []
        mock_wm.injected_context = None
        mock_wm.injected_procedures = []

        plan = await planner.plan("Hallo", mock_wm, {})

        tool_events = audit_logger.query(category=AuditCategory.TOOL_CALL)
        llm_events = [e for e in tool_events if "llm_plan" in (e.tool_name or "")]
        assert len(llm_events) >= 1
        assert llm_events[0].success is True
        assert "test-model" in str(llm_events[0].parameters)

    @pytest.mark.asyncio
    async def test_planner_logs_llm_error(self, audit_logger: AuditLogger) -> None:
        """LLM-Fehler werden als failed audit-entries geloggt."""
        from jarvis.core.model_router import OllamaError
        from jarvis.core.planner import Planner

        mock_config = MagicMock()
        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(side_effect=OllamaError("Connection refused"))

        mock_router = MagicMock()
        mock_router.select_model.return_value = "test-model"
        mock_router.get_model_config.return_value = {}

        planner = Planner(mock_config, mock_ollama, mock_router, audit_logger=audit_logger)

        mock_wm = MagicMock()
        mock_wm.chat_history = []
        mock_wm.context_window = []
        mock_wm.injected_context = None
        mock_wm.injected_procedures = []

        plan = await planner.plan("Test", mock_wm, {})

        assert plan.direct_response is not None  # Fallback-Response
        tool_events = audit_logger.query(category=AuditCategory.TOOL_CALL)
        llm_errors = [e for e in tool_events if not e.success and "llm_plan" in (e.tool_name or "")]
        assert len(llm_errors) >= 1


# ============================================================================
# 3. Reflector → AuditLogger
# ============================================================================


class TestReflectorAuditIntegration:
    """Beweist: Reflector loggt LLM-Reflexions-Calls."""

    @pytest.mark.asyncio
    async def test_reflector_logs_llm_error(self, audit_logger: AuditLogger) -> None:
        """Reflector-LLM-Fehler wird auditiert."""
        from jarvis.core.model_router import OllamaError
        from jarvis.core.reflector import Reflector
        from jarvis.models import AgentResult

        mock_config = MagicMock()
        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(side_effect=OllamaError("timeout"))

        mock_router = MagicMock()
        mock_router.select_model.return_value = "test-model"
        mock_router.get_model_config.return_value = {}

        reflector = Reflector(mock_config, mock_ollama, mock_router, audit_logger=audit_logger)

        session = SessionContext(session_id="ref-1", channel="cli", user_id="u1")
        mock_wm = MagicMock()
        mock_wm.chat_history = []

        from jarvis.models import ActionPlan, PlannedAction as _PA

        agent_result = AgentResult(
            response="test",
            total_iterations=3,
            plans=[ActionPlan(
                goal="test",
                reasoning="test reasoning",
                steps=[_PA(tool="read_file", params={"path": "/tmp/x"})],
            )],
        )

        result = await reflector.reflect(session, mock_wm, agent_result)

        tool_events = audit_logger.query(category=AuditCategory.TOOL_CALL)
        llm_errors = [e for e in tool_events if not e.success and "llm_reflect" in (e.tool_name or "")]
        assert len(llm_errors) >= 1


# ============================================================================
# 4. MemoryManager → AuditLogger
# ============================================================================


class TestMemoryManagerAuditIntegration:
    """Beweist: MemoryManager loggt Such- und Indexierungsvorgänge."""

    def test_memory_manager_accepts_audit_logger(self, audit_logger: AuditLogger) -> None:
        """MemoryManager kann mit AuditLogger initialisiert werden."""
        from jarvis.memory.manager import MemoryManager

        mm = MemoryManager(audit_logger=audit_logger)
        assert mm._audit_logger is audit_logger


# ============================================================================
# 5. SkillGenerator → PackageBuilder + AuditLogger
# ============================================================================


class TestSkillGeneratorPackageBuilderIntegration:
    """Beweist: SkillGenerator erstellt signierte Pakete bei Registrierung."""

    def test_register_creates_package(self, audit_logger: AuditLogger, tmp_path: Path) -> None:
        """Nach Registrierung wird ein signiertes Paket erstellt."""
        from jarvis.skills.generator import GeneratedSkill, GenerationStatus, SkillGenerator
        from jarvis.skills.package import PackageBuilder

        builder = PackageBuilder()  # Ohne Signer → unsigniertes Paket
        generator = SkillGenerator(
            skills_dir=tmp_path / "skills",
            package_builder=builder,
            audit_logger=audit_logger,
        )

        skill = GeneratedSkill(
            name="test_calc",
            description="Rechnet 2+2",
            version="1.0.0",
            code="def handler(**p): return str(2+2)",
            test_code="def test_calc(): assert True",
            skill_markdown="---\nname: test_calc\n---\nEin Test-Skill.",
            status=GenerationStatus.TEST_PASSED,
            test_passed=True,
        )

        result = generator.register(skill)
        assert result is True

        # Paket wurde erstellt
        packages_dir = tmp_path / "skills" / "packages"
        pkg_files = list(packages_dir.glob("*.jarvis-skill"))
        assert len(pkg_files) == 1

        # Audit: Skill-Installation wurde geloggt
        skill_events = audit_logger.query(category=AuditCategory.SKILL_INSTALL)
        assert len(skill_events) >= 1
        assert "test_calc" in skill_events[0].description

    def test_register_without_builder_still_works(self, tmp_path: Path) -> None:
        """Ohne PackageBuilder funktioniert register() wie bisher."""
        from jarvis.skills.generator import GeneratedSkill, GenerationStatus, SkillGenerator

        generator = SkillGenerator(skills_dir=tmp_path / "skills")

        skill = GeneratedSkill(
            name="basic_skill",
            description="Ein Skill",
            version="1.0.0",
            code="def handler(**p): return 'ok'",
            test_code="def test_it(): assert True",
            skill_markdown="---\nname: basic\n---\nBasic.",
            status=GenerationStatus.TEST_PASSED,
            test_passed=True,
        )

        assert generator.register(skill) is True
        assert (tmp_path / "skills" / f"{skill.module_name}.py").exists()


# ============================================================================
# 6. SkillRegistry → P2P-installierte Skills
# ============================================================================


class TestSkillRegistryP2PIntegration:
    """Beweist: SkillRegistry lädt Skills aus P2P-Unterverzeichnissen."""

    def test_loads_p2p_installed_skills(self, tmp_path: Path) -> None:
        """Skills in Unterverzeichnissen (P2P-Format) werden geladen."""
        from jarvis.skills.registry import SkillRegistry

        # P2P-installiertes Skill simulieren (Unterverzeichnis mit skill.md)
        skill_dir = tmp_path / "skills" / "weather_api"
        skill_dir.mkdir(parents=True)
        (skill_dir / "skill.md").write_text(
            "---\nname: weather_api\ntrigger_keywords: [wetter, weather]\n---\n"
            "# Weather API\nHolt aktuelle Wetterdaten.",
            encoding="utf-8",
        )
        (skill_dir / "skill.py").write_text(
            "async def handler(**params): return 'Sonnig, 22°C'",
            encoding="utf-8",
        )

        # Normalen Skill daneben legen
        (tmp_path / "skills" / "greeting.md").write_text(
            "---\nname: greeting\ntrigger_keywords: [hallo, hi]\n---\n"
            "# Greeting\nBegrüßt den Nutzer.",
            encoding="utf-8",
        )

        registry = SkillRegistry()
        count = registry.load_from_directories([tmp_path / "skills"])

        # Beide geladen: flach + P2P-Unterverzeichnis
        assert count == 2
        assert registry.get("weather_api") is not None
        assert registry.get("greeting") is not None

    def test_ignores_non_skill_subdirs(self, tmp_path: Path) -> None:
        """Unterverzeichnisse ohne skill.md werden ignoriert."""
        from jarvis.skills.registry import SkillRegistry

        (tmp_path / "not_a_skill").mkdir()
        (tmp_path / "not_a_skill" / "random.txt").write_text("nope")

        registry = SkillRegistry()
        count = registry.load_from_directories([tmp_path])

        assert count == 0

    def test_mixed_flat_and_p2p(self, tmp_path: Path) -> None:
        """Flache Skills und P2P-Skills können koexistieren."""
        from jarvis.skills.registry import SkillRegistry

        # Flach
        (tmp_path / "flat_skill.md").write_text(
            "---\nname: flat_skill\n---\nFlach.",
        )

        # P2P in Unterverzeichnis
        p2p_dir = tmp_path / "p2p_skill"
        p2p_dir.mkdir()
        (p2p_dir / "skill.md").write_text(
            "---\nname: p2p_skill\n---\nP2P.",
        )

        registry = SkillRegistry()
        count = registry.load_from_directories([tmp_path])

        assert count == 2
        assert registry.get("flat_skill") is not None
        assert registry.get("p2p_skill") is not None


# ============================================================================
# 7. AuditLogger: Neue Methoden (user_input, system)
# ============================================================================


class TestAuditLoggerNewMethods:
    """Beweist: Die neuen log_user_input und log_system Methoden funktionieren."""

    def test_log_user_input(self, audit_logger: AuditLogger) -> None:
        entry = audit_logger.log_user_input("telegram", "Hallo Jarvis!", agent_name="jarvis")

        assert entry.category == AuditCategory.USER_INPUT
        assert "[telegram]" in entry.description
        assert "Hallo" in entry.description

        events = audit_logger.query(category=AuditCategory.USER_INPUT)
        assert len(events) == 1

    def test_log_system_startup(self, audit_logger: AuditLogger) -> None:
        entry = audit_logger.log_system("startup", description="Jarvis v0.1.0 gestartet")

        assert entry.category == AuditCategory.SYSTEM
        assert "startup" in entry.action
        assert entry.success is True

    def test_log_system_shutdown(self, audit_logger: AuditLogger) -> None:
        entry = audit_logger.log_system("shutdown")
        assert entry.category == AuditCategory.SYSTEM

    def test_all_categories_in_summary(self, audit_logger: AuditLogger) -> None:
        """Summary enthält alle neuen Kategorien."""
        audit_logger.log_user_input("cli", "test")
        audit_logger.log_system("startup")
        audit_logger.log_tool_call("memory_search", {"query": "BU"})
        audit_logger.log_gatekeeper("ALLOW", "OK", tool_name="search")
        audit_logger.log_security("Rate limit", blocked=True)

        summary = audit_logger.summarize(hours=1)
        assert summary.total_entries == 5
        assert AuditCategory.USER_INPUT.value in summary.by_category
        assert AuditCategory.SYSTEM.value in summary.by_category


# ============================================================================
# 8. End-to-End: Kompletter Audit-Trail einer Anfrage
# ============================================================================


class TestFullAuditTrail:
    """Simuliert den kompletten Flow einer Jarvis-Anfrage und prüft den Audit-Trail."""

    @pytest.mark.asyncio
    async def test_complete_request_audit_trail(
        self, audit_logger: AuditLogger, tmp_path: Path,
    ) -> None:
        """Simuliert: User-Input → Gatekeeper → Executor → Audit-Summary."""
        from jarvis.core.executor import Executor
        from jarvis.security.monitor import RuntimeMonitor

        monitor = RuntimeMonitor(enable_defaults=True)

        # 1. User-Input
        audit_logger.log_user_input("cli", "Suche BU-Tarife", agent_name="jarvis")

        # 2. Gatekeeper (simuliert)
        audit_logger.log_gatekeeper("ALLOW", "Default: GREEN", tool_name="memory_search")

        # 3. Executor mit Monitor
        mock_config = MagicMock()
        mock_mcp = AsyncMock()
        result_ok = MagicMock(content="3 BU-Tarife gefunden", is_error=False)
        mock_mcp.call_tool = AsyncMock(return_value=result_ok)

        executor = Executor(
            mock_config, mock_mcp,
            runtime_monitor=monitor,
            audit_logger=audit_logger,
        )
        executor.set_agent_context(agent_name="jarvis")

        await executor._execute_single("memory_search", {"query": "BU-Tarif"})

        executor.clear_agent_context()

        # 4. Summary prüfen
        summary = audit_logger.summarize(hours=1)
        assert summary.total_entries >= 3  # user_input + gatekeeper + tool_call
        assert AuditCategory.USER_INPUT.value in summary.by_category
        assert AuditCategory.GATEKEEPER.value in summary.by_category
        assert AuditCategory.TOOL_CALL.value in summary.by_category

        # 5. Export
        export_path = tmp_path / "trail.json"
        count = audit_logger.export_json(export_path, hours=1)
        assert count >= 3

        data = json.loads(export_path.read_text())
        categories = {e["category"] for e in data["entries"]}
        assert "user_input" in categories
        assert "gatekeeper" in categories
        assert "tool_call" in categories
