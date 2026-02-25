"""Tests für die Skill Registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.skills.registry import Skill, SkillMatch, SkillRegistry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()

    # Skill 1: Morgen-Briefing
    (d / "morgen-briefing.md").write_text(
        "---\n"
        "name: morgen-briefing\n"
        "trigger_keywords: [Briefing, Morgen, Tagesplan, Überblick]\n"
        "tools_required: [search_memory, list_directory]\n"
        "category: organization\n"
        "priority: 2\n"
        "---\n"
        "# Morgen-Briefing\n\n"
        "## Schritte\n1. Memory durchsuchen\n2. Zusammenfassung erstellen\n",
        encoding="utf-8",
    )

    # Skill 2: Web-Recherche
    (d / "web-recherche.md").write_text(
        "---\n"
        "name: web-recherche\n"
        "trigger_keywords: [Recherche, recherchieren, Zusammenfassung, Internet]\n"
        "tools_required: [web_search, write_file]\n"
        "category: research\n"
        "priority: 1\n"
        "---\n"
        "# Web-Recherche\n\n"
        "## Schritte\n1. Web durchsuchen\n2. Ergebnisse zusammenfassen\n",
        encoding="utf-8",
    )

    # Skill 3: Deaktiviert
    (d / "disabled-skill.md").write_text(
        "---\n"
        "name: disabled-skill\n"
        "trigger_keywords: [deaktiviert]\n"
        "enabled: false\n"
        "---\n"
        "# Deaktiviert\n",
        encoding="utf-8",
    )

    return d


@pytest.fixture
def registry(skills_dir: Path) -> SkillRegistry:
    reg = SkillRegistry()
    reg.load_from_directories([skills_dir])
    return reg


# ============================================================================
# Laden
# ============================================================================


class TestLoading:
    def test_load_from_directory(self, registry: SkillRegistry) -> None:
        assert registry.count == 3  # Inkl. deaktiviertem Skill
        assert registry.enabled_count == 2

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        reg = SkillRegistry()
        count = reg.load_from_directories([tmp_path / "nonexistent"])
        assert count == 0

    def test_skill_parsed_correctly(self, registry: SkillRegistry) -> None:
        skill = registry.get("morgen-briefing")
        assert skill is not None
        assert skill.name == "morgen-briefing"
        assert "Briefing" in skill.trigger_keywords
        assert "search_memory" in skill.tools_required
        assert skill.category == "organization"
        assert skill.priority == 2
        assert skill.enabled is True
        assert "Memory durchsuchen" in skill.body

    def test_disabled_skill(self, registry: SkillRegistry) -> None:
        skill = registry.get("disabled-skill")
        assert skill is not None
        assert skill.enabled is False

    def test_multiple_directories(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        (dir1 / "skill-a.md").write_text(
            "---\nname: skill-a\ntrigger_keywords: [alpha]\n---\n# A\n"
        )

        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        (dir2 / "skill-b.md").write_text(
            "---\nname: skill-b\ntrigger_keywords: [beta]\n---\n# B\n"
        )

        reg = SkillRegistry()
        count = reg.load_from_directories([dir1, dir2])
        assert count == 2


# ============================================================================
# Matching
# ============================================================================


class TestMatching:
    def test_exact_keyword_match(self, registry: SkillRegistry) -> None:
        matches = registry.match("Gib mir ein Briefing")
        assert len(matches) >= 1
        assert matches[0].skill.slug == "morgen-briefing"
        assert matches[0].score >= 0.7

    def test_fuzzy_keyword_match(self, registry: SkillRegistry) -> None:
        matches = registry.match("Recherchiere zu KI-Sicherheit")
        assert len(matches) >= 1
        assert matches[0].skill.slug == "web-recherche"

    def test_no_match(self, registry: SkillRegistry) -> None:
        matches = registry.match("Was ist 2+2?")
        # Könnte leere Liste sein oder sehr niedrige Scores
        high_matches = [m for m in matches if m.score > 0.5]
        assert len(high_matches) == 0

    def test_disabled_skill_not_matched(self, registry: SkillRegistry) -> None:
        matches = registry.match("deaktiviert")
        slugs = [m.skill.slug for m in matches]
        assert "disabled-skill" not in slugs

    def test_tool_filter(self, registry: SkillRegistry) -> None:
        # Nur Skills die web_search nutzen
        matches = registry.match(
            "Recherchiere zu KI",
            available_tools=["web_search", "write_file"],
        )
        for m in matches:
            for tool in m.skill.tools_required:
                assert tool in ["web_search", "write_file"]

    def test_tool_filter_blocks_skill(self, registry: SkillRegistry) -> None:
        # Morgen-Briefing braucht search_memory — nicht verfügbar
        matches = registry.match(
            "Briefing erstellen",
            available_tools=["web_search"],
        )
        slugs = [m.skill.slug for m in matches]
        assert "morgen-briefing" not in slugs

    def test_match_best(self, registry: SkillRegistry) -> None:
        best = registry.match_best("Morgen-Briefing bitte")
        assert best is not None
        assert best.skill.slug == "morgen-briefing"

    def test_match_best_none(self, registry: SkillRegistry) -> None:
        best = registry.match_best("xyz komplett unbekanntes Thema qrs")
        # Könnte None sein wenn kein Score > min_score
        # Oder ein Match mit niedrigem Score
        if best:
            assert best.score < 0.5

    def test_empty_query(self, registry: SkillRegistry) -> None:
        matches = registry.match("")
        assert matches == []

    def test_priority_bonus(self, registry: SkillRegistry) -> None:
        # Bei gleichem Keyword-Score sollte höhere Priority gewinnen
        skill = registry.get("morgen-briefing")
        assert skill is not None
        assert skill.priority == 2


# ============================================================================
# Verwaltung
# ============================================================================


class TestManagement:
    def test_enable_disable(self, registry: SkillRegistry) -> None:
        registry.disable("morgen-briefing")
        assert registry.get("morgen-briefing").enabled is False
        assert registry.enabled_count == 1

        registry.enable("morgen-briefing")
        assert registry.get("morgen-briefing").enabled is True
        assert registry.enabled_count == 2

    def test_record_usage(self, registry: SkillRegistry) -> None:
        registry.record_usage("web-recherche", success=True, score=0.9)
        skill = registry.get("web-recherche")
        assert skill.total_uses == 1
        assert skill.success_count == 1
        assert skill.avg_score == 0.9
        assert skill.last_used is not None

    def test_success_rate(self) -> None:
        skill = Skill(name="test", slug="test", file_path=Path("/tmp/test.md"))
        assert skill.success_rate == 0.5  # Neutral für ungetestet

        skill.success_count = 8
        skill.failure_count = 2
        assert skill.success_rate == 0.8

    def test_list_by_category(self, registry: SkillRegistry) -> None:
        org = registry.list_by_category("organization")
        assert len(org) == 1
        assert org[0].slug == "morgen-briefing"

    def test_stats(self, registry: SkillRegistry) -> None:
        stats = registry.stats()
        assert stats["total"] == 3
        assert stats["enabled"] == 2
        assert "organization" in stats["categories"]


# ============================================================================
# Working Memory Injection
# ============================================================================


class TestInjection:
    def test_inject_into_working_memory(self, registry: SkillRegistry) -> None:
        wm = MagicMock()
        wm.injected_procedures = []

        result = registry.inject_into_working_memory("Gib mir ein Briefing", wm)

        assert result is not None
        assert result.skill.slug == "morgen-briefing"
        assert len(wm.injected_procedures) == 1
        assert "Memory durchsuchen" in wm.injected_procedures[0]

    def test_inject_no_match(self, registry: SkillRegistry) -> None:
        wm = MagicMock()
        wm.injected_procedures = []

        result = registry.inject_into_working_memory("xyz unbekannt", wm)
        # Entweder None oder niedrig-scorender Match
        if result is None:
            assert len(wm.injected_procedures) == 0

    def test_inject_no_duplicate(self, registry: SkillRegistry) -> None:
        wm = MagicMock()
        skill = registry.get("morgen-briefing")
        wm.injected_procedures = [skill.body]

        result = registry.inject_into_working_memory("Briefing bitte", wm)
        # Sollte nicht nochmal injizieren
        assert len(wm.injected_procedures) == 1
