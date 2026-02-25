"""Tests für memory/core_memory.py · Tier 1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jarvis.memory.core_memory import CoreMemory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def core_file(tmp_path: Path) -> Path:
    return tmp_path / "CORE.md"


@pytest.fixture
def core(core_file: Path) -> CoreMemory:
    return CoreMemory(core_file)


class TestCoreMemory:
    def test_load_nonexistent(self, core: CoreMemory):
        result = core.load()
        assert result == ""
        assert core.content == ""
        assert core.sections == {}

    def test_create_default(self, core: CoreMemory):
        text = core.create_default()
        assert "Identität" in text
        assert "Regeln" in text
        assert core.path.exists()

    def test_load_existing(self, core: CoreMemory, core_file: Path):
        core_file.write_text("# Test\nHello\n", encoding="utf-8")
        result = core.load()
        assert result == "# Test\nHello\n"
        assert core.content == "# Test\nHello\n"

    def test_save_new_content(self, core: CoreMemory):
        core.save("# Neu\nNeuer Inhalt\n")
        assert core.path.exists()
        assert core.content == "# Neu\nNeuer Inhalt\n"

    def test_save_updates_sections(self, core: CoreMemory):
        core.save("# Titel\nInhalt A\n# Zweiter\nInhalt B\n")
        assert "Titel" in core.sections
        assert "Zweiter" in core.sections
        assert core.sections["Titel"] == "Inhalt A"
        assert core.sections["Zweiter"] == "Inhalt B"

    def test_save_none_keeps_current(self, core: CoreMemory):
        core.save("# Test\nHello\n")
        core.save(None)
        assert core.path.read_text(encoding="utf-8") == "# Test\nHello\n"

    def test_get_section(self, core: CoreMemory):
        core.save("# Identität\nIch bin Jarvis\n# Regeln\nKeine Logs\n")
        assert core.get_section("Identität") == "Ich bin Jarvis"
        assert core.get_section("Regeln") == "Keine Logs"

    def test_get_section_case_insensitive(self, core: CoreMemory):
        core.save("# Identität\nTest\n")
        assert core.get_section("identität") == "Test"
        assert core.get_section("IDENTITÄT") == "Test"

    def test_get_section_not_found(self, core: CoreMemory):
        core.save("# Test\nHello\n")
        assert core.get_section("nope") == ""

    def test_parse_h1_and_h2(self, core: CoreMemory):
        text = "# Main\nContent\n## Sub\nSub content\n"
        core.save(text)
        assert "Main" in core.sections
        assert "Sub" in core.sections

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "CORE.md"
        core = CoreMemory(deep_path)
        core.save("# Test\nHello\n")
        assert deep_path.exists()

    def test_path_property(self, core: CoreMemory, core_file: Path):
        assert core.path == core_file
