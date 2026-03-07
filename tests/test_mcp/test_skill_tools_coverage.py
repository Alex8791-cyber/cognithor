"""Coverage-Tests fuer skill_tools.py -- fehlende Pfade abdecken."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.mcp.skill_tools import SkillTools, _slugify, register_skill_tools


# ============================================================================
# _slugify
# ============================================================================


class TestSlugify:
    def test_basic(self) -> None:
        assert _slugify("PDF Export") == "pdf_export"

    def test_special_chars(self) -> None:
        assert _slugify("Hello! World?") == "hello_world"

    def test_leading_trailing(self) -> None:
        assert _slugify("  --test-- ") == "test"

    def test_empty_gives_unnamed(self) -> None:
        assert _slugify("") == "unnamed_skill"

    def test_only_special_chars(self) -> None:
        assert _slugify("!!!") == "unnamed_skill"

    def test_underscores_and_hyphens(self) -> None:
        assert _slugify("my-cool_skill") == "my_cool_skill"

    def test_multiple_spaces(self) -> None:
        assert _slugify("a   b   c") == "a_b_c"


# ============================================================================
# SkillTools
# ============================================================================


@pytest.fixture
def registry() -> MagicMock:
    reg = MagicMock()
    reg._skills = {}
    reg.load_from_directories = MagicMock(return_value=1)
    return reg


@pytest.fixture
def skill_tools(registry: MagicMock, tmp_path: Path) -> SkillTools:
    dirs = [tmp_path / "skills"]
    return SkillTools(registry, dirs)


class TestCreateSkill:
    def test_success(self, skill_tools: SkillTools, tmp_path: Path) -> None:
        result = skill_tools.create_skill(
            name="Test Skill",
            description="A test skill",
            trigger_keywords="test,demo",
            body="Do something",
            category="testing",
        )
        assert "erfolgreich" in result
        assert "test_skill" in result

    def test_with_tools_required(self, skill_tools: SkillTools) -> None:
        result = skill_tools.create_skill(
            name="Tool Skill",
            description="Needs tools",
            trigger_keywords="tool",
            body="Use tool",
            tools_required="web_search,read_file",
        )
        assert "erfolgreich" in result

    def test_empty_name(self, skill_tools: SkillTools) -> None:
        result = skill_tools.create_skill(
            name="", description="desc", trigger_keywords="kw", body="body",
        )
        assert "Fehler" in result
        assert "name" in result

    def test_empty_description(self, skill_tools: SkillTools) -> None:
        result = skill_tools.create_skill(
            name="Test", description="", trigger_keywords="kw", body="body",
        )
        assert "Fehler" in result
        assert "description" in result

    def test_empty_keywords(self, skill_tools: SkillTools) -> None:
        result = skill_tools.create_skill(
            name="Test", description="desc", trigger_keywords="", body="body",
        )
        assert "Fehler" in result
        assert "trigger_keywords" in result

    def test_empty_body(self, skill_tools: SkillTools) -> None:
        result = skill_tools.create_skill(
            name="Test", description="desc", trigger_keywords="kw", body="",
        )
        assert "Fehler" in result
        assert "body" in result

    def test_whitespace_only_name(self, skill_tools: SkillTools) -> None:
        result = skill_tools.create_skill(
            name="   ", description="desc", trigger_keywords="kw", body="body",
        )
        assert "Fehler" in result

    def test_file_already_exists(self, skill_tools: SkillTools) -> None:
        # Create first
        skill_tools.create_skill(
            name="Existing", description="desc", trigger_keywords="kw", body="body",
        )
        # Try to create again
        result = skill_tools.create_skill(
            name="Existing", description="desc2", trigger_keywords="kw2", body="body2",
        )
        assert "existiert bereits" in result

    def test_registry_reload_failure(self, skill_tools: SkillTools, registry: MagicMock) -> None:
        registry.load_from_directories.side_effect = RuntimeError("reload error")
        result = skill_tools.create_skill(
            name="Broken Reload",
            description="desc",
            trigger_keywords="kw",
            body="body",
        )
        assert "WARNUNG" in result
        assert "Registry-Reload fehlgeschlagen" in result

    def test_write_dir_created(self, registry: MagicMock, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        st = SkillTools(registry, [deep])
        result = st.create_skill(
            name="Deep Skill",
            description="desc",
            trigger_keywords="kw",
            body="body",
        )
        assert "erfolgreich" in result
        assert deep.exists()


class TestListSkills:
    def test_empty(self, skill_tools: SkillTools) -> None:
        result = skill_tools.list_skills()
        assert "Keine Skills" in result

    def test_with_skills(self, skill_tools: SkillTools, registry: MagicMock) -> None:
        skill = MagicMock()
        skill.name = "Test Skill"
        skill.slug = "test_skill"
        skill.enabled = True
        skill.category = "general"
        skill.trigger_keywords = ["test", "demo"]
        skill.success_count = 5
        skill.failure_count = 1
        skill.success_rate = 5 / 6
        registry._skills = {"test_skill": skill}

        result = skill_tools.list_skills()
        assert "Test Skill" in result
        assert "test_skill" in result

    def test_filter_category(self, skill_tools: SkillTools, registry: MagicMock) -> None:
        skill_a = MagicMock()
        skill_a.name = "A"
        skill_a.slug = "a"
        skill_a.enabled = True
        skill_a.category = "daily"
        skill_a.trigger_keywords = ["a"]
        skill_a.success_count = 0
        skill_a.failure_count = 0
        skill_a.success_rate = 0.0

        skill_b = MagicMock()
        skill_b.name = "B"
        skill_b.slug = "b"
        skill_b.enabled = True
        skill_b.category = "general"
        skill_b.trigger_keywords = ["b"]
        skill_b.success_count = 0
        skill_b.failure_count = 0
        skill_b.success_rate = 0.0

        registry._skills = {"a": skill_a, "b": skill_b}

        result = skill_tools.list_skills(category="daily")
        assert "A" in result
        assert "B" not in result

    def test_filter_all_categories(self, skill_tools: SkillTools, registry: MagicMock) -> None:
        skill = MagicMock()
        skill.name = "X"
        skill.slug = "x"
        skill.enabled = True
        skill.category = "special"
        skill.trigger_keywords = ["x"]
        skill.success_count = 0
        skill.failure_count = 0
        registry._skills = {"x": skill}

        result = skill_tools.list_skills(category="all")
        assert "X" in result

    def test_disabled_filtered(self, skill_tools: SkillTools, registry: MagicMock) -> None:
        skill = MagicMock()
        skill.name = "Disabled"
        skill.slug = "disabled"
        skill.enabled = False
        skill.category = "general"
        skill.trigger_keywords = []
        registry._skills = {"disabled": skill}

        result = skill_tools.list_skills(enabled_only=True)
        assert "Keine Skills" in result

    def test_include_disabled(self, skill_tools: SkillTools, registry: MagicMock) -> None:
        skill = MagicMock()
        skill.name = "Disabled"
        skill.slug = "disabled"
        skill.enabled = False
        skill.category = "general"
        skill.trigger_keywords = ["d"]
        skill.success_count = 0
        skill.failure_count = 0
        skill.success_rate = 0.0
        registry._skills = {"disabled": skill}

        result = skill_tools.list_skills(enabled_only=False)
        assert "Disabled" in result
        assert "inaktiv" in result

    def test_no_filter_info(self, skill_tools: SkillTools) -> None:
        result = skill_tools.list_skills(enabled_only=False)
        assert "Keine Skills" in result


# ============================================================================
# register_skill_tools
# ============================================================================


class TestRegisterSkillTools:
    def test_registers_two_tools(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        reg = MagicMock()
        reg._skills = {}
        st = register_skill_tools(mock_client, reg, [tmp_path])
        assert isinstance(st, SkillTools)
        assert mock_client.register_builtin_handler.call_count == 5
        names = [call.args[0] for call in mock_client.register_builtin_handler.call_args_list]
        assert "create_skill" in names
        assert "list_skills" in names
        assert "install_community_skill" in names
        assert "search_community_skills" in names
        assert "report_skill" in names
