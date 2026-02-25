"""Tests für den Auto-Skill-Generator.

Testet:
  - GapDetector: Lücken erkennen, priorisieren, deduplizieren
  - SkillGenerator: Code-Generierung, Testing, Registrierung
  - Versionierung und Roll-Back
  - End-to-End Pipeline
  - Approval-Workflow
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.skills.generator import (
    GapDetector,
    GeneratedSkill,
    GenerationStatus,
    SkillGap,
    SkillGapType,
    SkillGenerator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def gap_detector() -> GapDetector:
    return GapDetector()


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "auto_skills"
    d.mkdir()
    return d


@pytest.fixture
def generator(skills_dir: Path) -> SkillGenerator:
    return SkillGenerator(skills_dir)


# ============================================================================
# GapDetector
# ============================================================================


class TestGapDetector:
    """Erkennung und Priorisierung von Skill-Lücken."""

    def test_empty_detector(self, gap_detector: GapDetector) -> None:
        assert gap_detector.gap_count == 0
        assert gap_detector.get_all_gaps() == []
        assert gap_detector.get_actionable_gaps() == []

    def test_report_unknown_tool(self, gap_detector: GapDetector) -> None:
        gap = gap_detector.report_unknown_tool("web_scraper", "User wollte scrapen")
        assert gap.gap_type == SkillGapType.UNKNOWN_TOOL
        assert gap.tool_name == "web_scraper"
        assert gap.frequency == 1
        assert gap_detector.gap_count == 1

    def test_report_no_skill_match(self, gap_detector: GapDetector) -> None:
        gap = gap_detector.report_no_skill_match("Erstelle eine Steuererklärung")
        assert gap.gap_type == SkillGapType.NO_SKILL_MATCH
        assert "Steuererklärung" in gap.description

    def test_report_low_success_rate(self, gap_detector: GapDetector) -> None:
        gap = gap_detector.report_low_success_rate("email_skill", 0.2)
        assert gap.gap_type == SkillGapType.LOW_SUCCESS_RATE
        assert "20%" in gap.description

    def test_report_user_request(self, gap_detector: GapDetector) -> None:
        gap = gap_detector.report_user_request("PDF-Generator Tool")
        assert gap.gap_type == SkillGapType.USER_REQUEST

    def test_report_repeated_failure(self, gap_detector: GapDetector) -> None:
        gap = gap_detector.report_repeated_failure("Kalender-Import", "API Error 403")
        assert gap.gap_type == SkillGapType.REPEATED_FAILURE
        assert "Kalender-Import" in gap.description

    def test_frequency_counting(self, gap_detector: GapDetector) -> None:
        """Gleiche Lücke mehrfach melden erhöht Frequenz."""
        gap_detector.report_unknown_tool("scraper")
        gap_detector.report_unknown_tool("scraper")
        gap_detector.report_unknown_tool("scraper")

        gaps = gap_detector.get_all_gaps()
        assert len(gaps) == 1
        assert gaps[0].frequency == 3

    def test_actionable_threshold(self, gap_detector: GapDetector) -> None:
        """Gaps werden erst ab Schwellwert actionable."""
        # Einmal gemeldet → noch nicht actionable (Schwellwert=2)
        gap_detector.report_unknown_tool("rare_tool")
        assert len(gap_detector.get_actionable_gaps()) == 0

        # Zweimal → jetzt actionable
        gap_detector.report_unknown_tool("rare_tool")
        assert len(gap_detector.get_actionable_gaps()) == 1

    def test_user_request_immediately_actionable(self, gap_detector: GapDetector) -> None:
        """User-Requests sind sofort actionable (auch bei frequency=1)."""
        gap_detector.report_user_request("Neues Tool erstellen")
        assert len(gap_detector.get_actionable_gaps()) == 1

    def test_priority_ordering(self, gap_detector: GapDetector) -> None:
        """Gaps werden nach Priorität sortiert."""
        gap_detector.report_unknown_tool("low_freq")
        gap_detector.report_user_request("high_prio")  # Sofort hoch
        gap_detector.report_unknown_tool("med_freq")
        gap_detector.report_unknown_tool("med_freq")

        gaps = gap_detector.get_all_gaps()
        # User-Request hat höchste Priorität (2.0 weight × 1 freq)
        assert gaps[0].gap_type == SkillGapType.USER_REQUEST

    def test_clear_gap(self, gap_detector: GapDetector) -> None:
        gap = gap_detector.report_user_request("Test")
        assert gap_detector.gap_count == 1
        assert gap_detector.clear_gap(gap.id) is True
        assert gap_detector.gap_count == 0

    def test_clear_nonexistent(self, gap_detector: GapDetector) -> None:
        assert gap_detector.clear_gap("ghost") is False

    def test_context_update(self, gap_detector: GapDetector) -> None:
        """Kontext wird bei wiederholtem Melden aktualisiert."""
        gap_detector.report_unknown_tool("tool", "Kontext 1")
        gap_detector.report_unknown_tool("tool", "Kontext 2")
        gap = gap_detector.get_all_gaps()[0]
        assert gap.context == "Kontext 2"


# ============================================================================
# SkillGenerator (ohne LLM)
# ============================================================================


class TestSkillGeneratorStub:
    """Generator im Stub-Modus (kein LLM)."""

    @pytest.mark.asyncio
    async def test_generate_stub(self, generator: SkillGenerator) -> None:
        gap = SkillGap(
            id="test",
            gap_type=SkillGapType.USER_REQUEST,
            description="CSV-Parser Tool",
        )
        skill = await generator.generate(gap)

        assert skill.name
        assert skill.code  # Stub-Code generiert
        assert skill.test_code  # Stub-Test generiert
        assert skill.skill_markdown  # Markdown generiert
        assert skill.status == GenerationStatus.GENERATING
        assert skill.version == 1

    @pytest.mark.asyncio
    async def test_generate_increments_version(self, generator: SkillGenerator) -> None:
        gap = SkillGap(
            id="test",
            gap_type=SkillGapType.USER_REQUEST,
            description="Test Tool",
        )
        skill1 = await generator.generate(gap)
        assert skill1.version == 1

        skill2 = await generator.generate(gap)
        assert skill2.version == 2

    @pytest.mark.asyncio
    async def test_test_stub_passes_syntax(self, generator: SkillGenerator) -> None:
        """Stub-Code besteht Syntax-Check."""
        gap = SkillGap(
            id="test",
            gap_type=SkillGapType.USER_REQUEST,
            description="Simple Tool",
        )
        skill = await generator.generate(gap)
        passed = await generator.test(skill)

        assert passed is True
        assert skill.test_passed is True
        assert skill.status == GenerationStatus.TEST_PASSED

    @pytest.mark.asyncio
    async def test_test_invalid_code_fails(self, generator: SkillGenerator) -> None:
        """Ungültiger Code besteht Syntax-Check nicht."""
        skill = GeneratedSkill(
            name="broken",
            code="def foo(:\n  pass",  # Syntaxfehler
            test_code="",
        )
        passed = await generator.test(skill)

        assert passed is False
        assert skill.status == GenerationStatus.TEST_FAILED
        assert len(skill.test_errors) > 0


class TestSkillRegistration:
    """Skill-Registrierung und Dateisystem."""

    @pytest.mark.asyncio
    async def test_register_creates_files(
        self,
        generator: SkillGenerator,
        skills_dir: Path,
    ) -> None:
        gap = SkillGap(
            id="test",
            gap_type=SkillGapType.USER_REQUEST,
            description="File Tool",
        )
        skill = await generator.generate(gap)
        await generator.test(skill)

        success = generator.register(skill)
        assert success is True
        assert skill.status == GenerationStatus.REGISTERED

        # Dateien existieren
        assert (skills_dir / f"{skill.module_name}.py").exists()
        assert (skills_dir / f"{skill.module_name}.md").exists()
        assert (skills_dir / f"test_{skill.module_name}.py").exists()

    @pytest.mark.asyncio
    async def test_register_rejects_untested(self, generator: SkillGenerator) -> None:
        skill = GeneratedSkill(name="untested", code="x = 1")
        success = generator.register(skill)
        assert success is False

    @pytest.mark.asyncio
    async def test_register_with_approval_required(
        self,
        skills_dir: Path,
    ) -> None:
        gen = SkillGenerator(skills_dir, require_approval=True)
        gap = SkillGap(
            id="test",
            gap_type=SkillGapType.USER_REQUEST,
            description="Protected Tool",
        )
        skill = await gen.generate(gap)
        await gen.test(skill)

        # Ohne Approval → wartet
        success = gen.register(skill)
        assert success is False
        assert skill.status == GenerationStatus.AWAITING_APPROVAL

        # Mit Approval → registriert
        gen.approve(skill.name, approved_by="alex")
        success = gen.register(skill)
        assert success is True
        assert skill.status == GenerationStatus.REGISTERED

    def test_approve_nonexistent(self, generator: SkillGenerator) -> None:
        assert generator.approve("ghost") is False


class TestVersioning:
    """Versionierung und Roll-Back."""

    @pytest.mark.asyncio
    async def test_archive_previous_version(
        self,
        generator: SkillGenerator,
        skills_dir: Path,
    ) -> None:
        gap = SkillGap(id="t", gap_type=SkillGapType.USER_REQUEST, description="V1")
        skill_v1 = await generator.generate(gap)
        await generator.test(skill_v1)
        generator.register(skill_v1)

        # Nochmal generieren → v2
        gap2 = SkillGap(id="t2", gap_type=SkillGapType.USER_REQUEST, description="V1")
        skill_v2 = await generator.generate(gap2)
        await generator.test(skill_v2)
        generator.register(skill_v2)

        # History-Datei existiert
        history_dir = skills_dir / "history"
        history_files = list(history_dir.glob("*.py"))
        assert len(history_files) >= 1

    @pytest.mark.asyncio
    async def test_rollback(
        self,
        generator: SkillGenerator,
        skills_dir: Path,
    ) -> None:
        gap = SkillGap(id="t", gap_type=SkillGapType.USER_REQUEST, description="Rollback Test")
        skill_v1 = await generator.generate(gap)
        await generator.test(skill_v1)
        generator.register(skill_v1)
        v1_content = (skills_dir / f"{skill_v1.module_name}.py").read_text()

        # v2 generieren und registrieren
        skill_v2 = await generator.generate(gap)
        await generator.test(skill_v2)
        generator.register(skill_v2)

        # Rollback
        success = generator.rollback(skill_v2.name)
        assert success is True
        assert skill_v2.status == GenerationStatus.ROLLED_BACK

    def test_rollback_v1_impossible(self, generator: SkillGenerator) -> None:
        """Roll-Back auf v1 wenn schon v1 → unmöglich."""
        skill = GeneratedSkill(name="v1_only", version=1)
        generator._generated["v1_only"] = skill
        assert generator.rollback("v1_only") is False


class TestEndToEnd:
    """Vollständiger Workflow: Gap → Generate → Test → Register."""

    @pytest.mark.asyncio
    async def test_process_gap_success(
        self,
        generator: SkillGenerator,
        skills_dir: Path,
    ) -> None:
        gap = SkillGap(
            id="e2e",
            gap_type=SkillGapType.USER_REQUEST,
            description="E2E Test Tool",
        )
        skill = await generator.process_gap(gap)

        assert skill.status == GenerationStatus.REGISTERED
        assert (skills_dir / f"{skill.module_name}.py").exists()

    @pytest.mark.asyncio
    async def test_process_all_gaps(self, generator: SkillGenerator) -> None:
        detector = generator.gap_detector
        detector.report_user_request("Tool A")
        detector.report_user_request("Tool B")

        results = await generator.process_all_gaps()
        assert len(results) == 2
        assert all(s.status == GenerationStatus.REGISTERED for s in results)

        # Gaps wurden entfernt
        assert detector.gap_count == 0


class TestSkillNameDerivation:
    """Skill-Name-Ableitung aus Gap."""

    @pytest.mark.asyncio
    async def test_name_from_tool(self, generator: SkillGenerator) -> None:
        gap = SkillGap(
            id="t",
            gap_type=SkillGapType.UNKNOWN_TOOL,
            description="X",
            tool_name="web-scraper",
        )
        skill = await generator.generate(gap)
        assert skill.name == "web_scraper"

    @pytest.mark.asyncio
    async def test_name_from_description(self, generator: SkillGenerator) -> None:
        gap = SkillGap(
            id="t",
            gap_type=SkillGapType.USER_REQUEST,
            description="CSV Parser erstellen",
        )
        skill = await generator.generate(gap)
        assert "csv" in skill.name.lower()


class TestCodeExtraction:
    """LLM-Antwort Code-Block Extraktion."""

    def test_extract_python_block(self) -> None:
        text = "Hier ist der Code:\n```python\nx = 1\n```\nFertig."
        assert SkillGenerator._extract_code_block(text) == "x = 1"

    def test_extract_generic_block(self) -> None:
        text = "```\ny = 2\n```"
        assert SkillGenerator._extract_code_block(text) == "y = 2"

    def test_extract_no_block(self) -> None:
        text = "x = 3"
        assert SkillGenerator._extract_code_block(text) == "x = 3"


class TestStats:
    """Generator-Statistiken."""

    @pytest.mark.asyncio
    async def test_stats(self, generator: SkillGenerator) -> None:
        generator.gap_detector.report_user_request("Test")
        gap = SkillGap(id="s", gap_type=SkillGapType.USER_REQUEST, description="X")
        await generator.generate(gap)

        stats = generator.stats()
        assert stats["total_generated"] >= 1
        assert stats["gaps_detected"] >= 1
        assert "by_status" in stats
