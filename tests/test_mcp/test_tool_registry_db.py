"""Tests fuer das datenbankgestuetzte Tool-Registry (tool_registry_db.py).

Testet CRUD-Operationen, Rollen-Filterung, lokalisierte Beschreibungen,
Prompt-Generierung, Beispiel-Verwaltung, MCP-Sync und Prozedur-Deduplizierung.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from jarvis.mcp.tool_registry_db import (
    DEFAULT_EXAMPLES,
    TOOL_CATEGORIES,
    TOOL_ROLE_DEFAULTS,
    ToolInfo,
    ToolRegistryDB,
    _ProcedureEntry,
    _jaccard,
    deduplicate_procedures,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Temporaerer Datenbankpfad."""
    return tmp_path / "test_tool_registry.db"


@pytest.fixture()
def registry(db_path: Path) -> ToolRegistryDB:
    """Frische ToolRegistryDB-Instanz."""
    reg = ToolRegistryDB(db_path)
    yield reg
    reg.close()


@pytest.fixture()
def populated_registry(registry: ToolRegistryDB) -> ToolRegistryDB:
    """Registry mit einigen Test-Tools."""
    registry.upsert_tool(
        name="read_file",
        description_de="Liest eine Datei.",
        description_en="Reads a file.",
        description_zh="读取文件。",
        input_schema={
            "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}},
            "required": ["path"],
        },
        example_input='read_file(path="/tmp/test.txt")',
        example_output='"Zeile 1\\nZeile 2"',
        category="filesystem",
        agent_roles=["executor"],
    )
    registry.upsert_tool(
        name="web_search",
        description_de="Websuche durchfuehren.",
        description_en="Perform a web search.",
        description_zh="执行网络搜索。",
        input_schema={
            "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
            "required": ["query"],
        },
        example_input='web_search(query="test")',
        example_output='[{"title": "..."}]',
        category="web",
        agent_roles=["planner", "executor", "researcher"],
    )
    registry.upsert_tool(
        name="browser_navigate",
        description_de="Navigiert zu einer URL.",
        description_en="Navigates to a URL.",
        description_zh="导航到URL。",
        input_schema={
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        example_input='browser_navigate(url="https://example.com")',
        example_output='{"title": "Example Domain"}',
        category="browser",
        agent_roles=["browser"],
    )
    registry.upsert_tool(
        name="search_memory",
        description_de="Durchsucht das Gedaechtnis.",
        description_en="Searches memory.",
        description_zh="搜索记忆。",
        input_schema={
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        category="memory",
        agent_roles=["planner", "researcher"],
    )
    return registry


def _make_mock_mcp_client(tools: dict[str, dict[str, Any]]) -> MagicMock:
    """Erstellt einen Mock-MCP-Client mit gegebenen Tool-Schemas."""
    client = MagicMock()
    client.get_tool_schemas.return_value = tools
    return client


# ============================================================================
# Test: Upsert und Retrieval
# ============================================================================


class TestUpsertAndRetrieval:
    """Tests fuer das Einfuegen und Abrufen von Tools."""

    def test_upsert_new_tool(self, registry: ToolRegistryDB) -> None:
        """Ein neues Tool wird korrekt eingefuegt."""
        registry.upsert_tool(
            name="test_tool",
            description_de="Test-Beschreibung",
            description_en="Test description",
            category="other",
        )
        info = registry.get_tool("test_tool")
        assert info is not None
        assert info.name == "test_tool"
        assert info.description == "Test description"  # default language = en
        assert info.category == "other"

    def test_upsert_update_existing(self, registry: ToolRegistryDB) -> None:
        """Ein bestehendes Tool wird korrekt aktualisiert."""
        registry.upsert_tool(name="tool_a", description_en="Version 1")
        registry.upsert_tool(name="tool_a", description_en="Version 2")
        info = registry.get_tool("tool_a")
        assert info is not None
        assert info.description == "Version 2"

    def test_upsert_preserves_examples_on_empty_update(self, registry: ToolRegistryDB) -> None:
        """Bestehende Beispiele werden nicht durch leere Werte ueberschrieben."""
        registry.upsert_tool(
            name="tool_b",
            description_en="Tool B",
            example_input="input_1",
            example_output="output_1",
        )
        # Update ohne Beispiele -> bestehende bleiben erhalten
        registry.upsert_tool(
            name="tool_b",
            description_en="Tool B updated",
            example_input="",
            example_output="",
        )
        info = registry.get_tool("tool_b")
        assert info is not None
        assert info.example_input == "input_1"
        assert info.example_output == "output_1"

    def test_get_nonexistent_tool(self, registry: ToolRegistryDB) -> None:
        """Nicht vorhandenes Tool gibt None zurueck."""
        assert registry.get_tool("does_not_exist") is None

    def test_tool_count(self, populated_registry: ToolRegistryDB) -> None:
        """tool_count gibt korrekte Anzahl zurueck."""
        assert populated_registry.tool_count() == 4

    def test_upsert_with_all_languages(self, registry: ToolRegistryDB) -> None:
        """Alle drei Sprach-Beschreibungen werden gespeichert."""
        registry.upsert_tool(
            name="multi_lang",
            description_de="Deutsch",
            description_en="English",
            description_zh="中文",
        )
        info = registry.get_tool("multi_lang")
        assert info is not None
        assert info.description == "English"


# ============================================================================
# Test: Role Filtering
# ============================================================================


class TestRoleFiltering:
    """Tests fuer die rollenbasierte Tool-Filterung."""

    def test_get_tools_for_executor(self, populated_registry: ToolRegistryDB) -> None:
        """Executor sieht nur ihm zugeordnete Tools."""
        tools = populated_registry.get_tools_for_role("executor", "en")
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "web_search" in names
        # browser_navigate ist nur fuer 'browser'
        assert "browser_navigate" not in names

    def test_get_tools_for_planner(self, populated_registry: ToolRegistryDB) -> None:
        """Planner sieht Planner-Tools."""
        tools = populated_registry.get_tools_for_role("planner", "en")
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "search_memory" in names
        assert "read_file" not in names

    def test_get_tools_for_browser(self, populated_registry: ToolRegistryDB) -> None:
        """Browser-Agent sieht Browser-Tools."""
        tools = populated_registry.get_tools_for_role("browser", "en")
        names = {t.name for t in tools}
        assert "browser_navigate" in names
        assert "read_file" not in names

    def test_get_tools_for_all(self, populated_registry: ToolRegistryDB) -> None:
        """Rolle 'all' sieht alle Tools."""
        tools = populated_registry.get_tools_for_role("all", "en")
        assert len(tools) == 4

    def test_tool_with_role_all_visible_everywhere(self, registry: ToolRegistryDB) -> None:
        """Ein Tool mit Rolle 'all' ist fuer jede Rolle sichtbar."""
        registry.upsert_tool(name="global_tool", agent_roles=["all"])
        for role in ["planner", "executor", "browser", "researcher"]:
            tools = registry.get_tools_for_role(role)
            assert any(t.name == "global_tool" for t in tools)


# ============================================================================
# Test: Localized Descriptions
# ============================================================================


class TestLocalizedDescriptions:
    """Tests fuer lokalisierte Beschreibungen."""

    def test_german_description(self, populated_registry: ToolRegistryDB) -> None:
        """Deutsche Beschreibung wird korrekt zurueckgegeben."""
        tools = populated_registry.get_tools_for_role("all", "de")
        rf = next(t for t in tools if t.name == "read_file")
        assert rf.description == "Liest eine Datei."

    def test_english_description(self, populated_registry: ToolRegistryDB) -> None:
        """Englische Beschreibung wird korrekt zurueckgegeben."""
        tools = populated_registry.get_tools_for_role("all", "en")
        rf = next(t for t in tools if t.name == "read_file")
        assert rf.description == "Reads a file."

    def test_chinese_description(self, populated_registry: ToolRegistryDB) -> None:
        """Chinesische Beschreibung wird korrekt zurueckgegeben."""
        tools = populated_registry.get_tools_for_role("all", "zh")
        rf = next(t for t in tools if t.name == "read_file")
        assert rf.description == "读取文件。"

    def test_fallback_to_english(self, registry: ToolRegistryDB) -> None:
        """Bei fehlender Sprache wird auf Englisch zurueckgefallen."""
        registry.upsert_tool(
            name="only_en", description_en="Only English", description_de="", description_zh="",
        )
        tools = registry.get_tools_for_role("all", "de")
        tool = next(t for t in tools if t.name == "only_en")
        # Fallback to en when de is empty
        assert tool.description == "Only English"


# ============================================================================
# Test: sync_from_mcp
# ============================================================================


class TestSyncFromMCP:
    """Tests fuer die Synchronisation mit dem MCP-Client."""

    def test_sync_basic(self, registry: ToolRegistryDB) -> None:
        """Grundlegende Synchronisation funktioniert."""
        mock_client = _make_mock_mcp_client({
            "read_file": {
                "description": "Reads a file from disk.",
                "inputSchema": {
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            "web_search": {
                "description": "Search the web.",
                "inputSchema": {
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        })
        count = registry.sync_from_mcp(mock_client)
        assert count == 2
        assert registry.tool_count() == 2

    def test_sync_assigns_categories(self, registry: ToolRegistryDB) -> None:
        """Sync weist korrekte Kategorien zu."""
        mock_client = _make_mock_mcp_client({
            "read_file": {"description": "Read", "inputSchema": {}},
            "browser_navigate": {"description": "Nav", "inputSchema": {}},
            "docker_ps": {"description": "PS", "inputSchema": {}},
        })
        registry.sync_from_mcp(mock_client)

        assert registry.get_tool("read_file").category == "filesystem"
        assert registry.get_tool("browser_navigate").category == "browser"
        assert registry.get_tool("docker_ps").category == "docker"

    def test_sync_assigns_roles(self, registry: ToolRegistryDB) -> None:
        """Sync weist korrekte Rollen basierend auf TOOL_ROLE_DEFAULTS zu."""
        mock_client = _make_mock_mcp_client({
            "web_search": {"description": "Search", "inputSchema": {}},
        })
        registry.sync_from_mcp(mock_client)
        tool = registry.get_tool("web_search")
        assert "planner" in tool.agent_roles
        assert "executor" in tool.agent_roles
        assert "researcher" in tool.agent_roles

    def test_sync_preserves_existing_examples(self, registry: ToolRegistryDB) -> None:
        """Sync ueberschreibt manuell gesetzte Beispiele nicht."""
        registry.upsert_tool(
            name="custom_tool",
            example_input="my_custom_input",
            example_output="my_custom_output",
        )
        mock_client = _make_mock_mcp_client({
            "custom_tool": {"description": "Custom", "inputSchema": {}},
        })
        registry.sync_from_mcp(mock_client)
        tool = registry.get_tool("custom_tool")
        assert tool.example_input == "my_custom_input"
        assert tool.example_output == "my_custom_output"

    def test_sync_adds_default_examples(self, registry: ToolRegistryDB) -> None:
        """Sync fuegt Default-Beispiele fuer bekannte Tools hinzu."""
        mock_client = _make_mock_mcp_client({
            "web_search": {"description": "Search", "inputSchema": {}},
        })
        registry.sync_from_mcp(mock_client)
        tool = registry.get_tool("web_search")
        assert tool.example_input != ""
        assert "query" in tool.example_input

    def test_sync_unknown_tool_gets_all_role(self, registry: ToolRegistryDB) -> None:
        """Unbekannte Tools bekommen Rolle 'all'."""
        mock_client = _make_mock_mcp_client({
            "totally_new_tool": {"description": "New", "inputSchema": {}},
        })
        registry.sync_from_mcp(mock_client)
        tool = registry.get_tool("totally_new_tool")
        assert "all" in tool.agent_roles


# ============================================================================
# Test: Prompt Section Generation
# ============================================================================


class TestPromptSectionGeneration:
    """Tests fuer die Prompt-Abschnitt-Generierung."""

    def test_prompt_section_contains_tool_names(self, populated_registry: ToolRegistryDB) -> None:
        """Generierter Prompt-Abschnitt enthaelt Tool-Namen."""
        section = populated_registry.get_tool_prompt_section("all", "en")
        assert "read_file" in section
        assert "web_search" in section
        assert "browser_navigate" in section

    def test_prompt_section_contains_examples(self, populated_registry: ToolRegistryDB) -> None:
        """Prompt-Abschnitt zeigt Beispiele an."""
        section = populated_registry.get_tool_prompt_section("all", "en")
        assert "Example Input:" in section
        assert 'read_file(path="/tmp/test.txt")' in section

    def test_prompt_section_groups_by_category(self, populated_registry: ToolRegistryDB) -> None:
        """Prompt-Abschnitt gruppiert nach Kategorie."""
        section = populated_registry.get_tool_prompt_section("all", "en")
        assert "**Filesystem:**" in section
        assert "**Web & Research:**" in section
        assert "**Browser:**" in section

    def test_prompt_section_german_headers(self, populated_registry: ToolRegistryDB) -> None:
        """Deutsche Header werden korrekt verwendet."""
        section = populated_registry.get_tool_prompt_section("all", "de")
        assert "Registrierte Tools" in section
        assert "Dateisystem" in section
        assert "Web & Recherche" in section

    def test_prompt_section_chinese_headers(self, populated_registry: ToolRegistryDB) -> None:
        """Chinesische Header werden korrekt verwendet."""
        section = populated_registry.get_tool_prompt_section("all", "zh")
        assert "已注册工具" in section

    def test_prompt_section_role_filtering(self, populated_registry: ToolRegistryDB) -> None:
        """Prompt-Abschnitt filtert nach Rolle."""
        section = populated_registry.get_tool_prompt_section("browser", "en")
        assert "browser_navigate" in section
        assert "read_file" not in section

    def test_prompt_section_param_format(self, populated_registry: ToolRegistryDB) -> None:
        """Parameter werden korrekt formatiert (required mit *)."""
        section = populated_registry.get_tool_prompt_section("all", "en")
        # read_file has path: string * (required)
        assert "path: string *" in section

    def test_prompt_section_no_tools(self, registry: ToolRegistryDB) -> None:
        """Leere DB generiert leeren Abschnitt."""
        section = registry.get_tool_prompt_section("executor", "en")
        assert "Registered Tools (0)" in section


# ============================================================================
# Test: Example Storage
# ============================================================================


class TestExampleStorage:
    """Tests fuer die Beispiel-Verwaltung."""

    def test_add_example(self, populated_registry: ToolRegistryDB) -> None:
        """Beispiel hinzufuegen funktioniert."""
        result = populated_registry.add_example(
            "search_memory",
            'search_memory(query="test")',
            '[{"text": "..."}]',
        )
        assert result is True
        tool = populated_registry.get_tool("search_memory")
        assert tool.example_input == 'search_memory(query="test")'
        assert tool.example_output == '[{"text": "..."}]'

    def test_add_example_nonexistent_tool(self, registry: ToolRegistryDB) -> None:
        """Beispiel fuer nicht vorhandenes Tool gibt False zurueck."""
        result = registry.add_example("no_such_tool", "input", "output")
        assert result is False

    def test_update_example(self, populated_registry: ToolRegistryDB) -> None:
        """Bestehendes Beispiel wird korrekt aktualisiert."""
        populated_registry.add_example("read_file", "new_input", "new_output")
        tool = populated_registry.get_tool("read_file")
        assert tool.example_input == "new_input"
        assert tool.example_output == "new_output"


# ============================================================================
# Test: set_agent_roles
# ============================================================================


class TestSetAgentRoles:
    """Tests fuer die Rollen-Zuweisung."""

    def test_set_roles(self, populated_registry: ToolRegistryDB) -> None:
        """Rollen werden korrekt gesetzt."""
        populated_registry.set_agent_roles("read_file", ["planner", "executor", "researcher"])
        tool = populated_registry.get_tool("read_file")
        assert set(tool.agent_roles) == {"planner", "executor", "researcher"}

    def test_set_roles_nonexistent_tool(self, registry: ToolRegistryDB) -> None:
        """Rollen-Zuweisung fuer nicht vorhandenes Tool gibt False zurueck."""
        result = registry.set_agent_roles("no_such_tool", ["planner"])
        assert result is False


# ============================================================================
# Test: Procedure Deduplication
# ============================================================================


class TestProcedureDeduplication:
    """Tests fuer die Prozedur-Deduplizierung."""

    def test_jaccard_identical(self) -> None:
        """Identische Mengen haben Jaccard 1.0."""
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_jaccard_disjoint(self) -> None:
        """Disjunkte Mengen haben Jaccard 0.0."""
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self) -> None:
        """Teilueberlappung berechnet korrekte Jaccard-Aehnlichkeit."""
        # {a, b} & {b, c} = {b}, union = {a, b, c} -> 1/3
        assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 0.01

    def test_jaccard_empty(self) -> None:
        """Leere Mengen haben Jaccard 1.0."""
        assert _jaccard(set(), set()) == 1.0

    def test_dedup_groups_similar(self) -> None:
        """Aehnliche Prozeduren werden gruppiert."""
        procs = [
            _ProcedureEntry(f"heartbeat-{i}", 1, ["heartbeat", "systemstatus", "check"])
            for i in range(10)
        ]
        lines = deduplicate_procedures(procs)
        # 10 Prozeduren mit identischen Keywords -> eine Gruppenzeile
        assert len(lines) == 1
        assert "Varianten" in lines[0] or "variants" in lines[0]

    def test_dedup_keeps_unique(self) -> None:
        """Einzigartige Prozeduren bleiben individuell."""
        procs = [
            _ProcedureEntry("backup-routine", 5, ["backup", "sicherung"]),
            _ProcedureEntry("deploy-check", 3, ["deploy", "release"]),
            _ProcedureEntry("email-digest", 2, ["email", "zusammenfassung"]),
        ]
        lines = deduplicate_procedures(procs)
        assert len(lines) == 3
        assert any("backup-routine" in line for line in lines)
        assert any("deploy-check" in line for line in lines)

    def test_dedup_mixed(self) -> None:
        """Mischung aus gruppierten und individuellen Prozeduren."""
        procs = [
            # 5 aehnliche Heartbeats
            _ProcedureEntry("heartbeat-1", 1, ["heartbeat", "status"]),
            _ProcedureEntry("heartbeat-2", 1, ["heartbeat", "status"]),
            _ProcedureEntry("heartbeat-3", 1, ["heartbeat", "status"]),
            _ProcedureEntry("heartbeat-4", 1, ["heartbeat", "status"]),
            _ProcedureEntry("heartbeat-5", 1, ["heartbeat", "status"]),
            # 1 einzigartiger
            _ProcedureEntry("backup", 10, ["backup", "sicherung", "daten"]),
        ]
        lines = deduplicate_procedures(procs)
        # 1 Gruppe + 1 individuell
        assert len(lines) == 2

    def test_dedup_empty_list(self) -> None:
        """Leere Prozedur-Liste ergibt leeres Ergebnis."""
        assert deduplicate_procedures([]) == []

    def test_dedup_english_output(self) -> None:
        """Englische Ausgabe verwendet englische Labels."""
        procs = [
            _ProcedureEntry(f"hb-{i}", 1, ["heartbeat", "check"])
            for i in range(5)
        ]
        lines = deduplicate_procedures(procs, language="en")
        assert len(lines) == 1
        assert "variants" in lines[0]

    def test_dedup_below_threshold(self) -> None:
        """Prozeduren unter dem Aehnlichkeits-Schwellenwert werden nicht gruppiert."""
        procs = [
            _ProcedureEntry("proc-a", 1, ["alpha", "beta", "gamma"]),
            _ProcedureEntry("proc-b", 1, ["delta", "epsilon", "zeta"]),
            _ProcedureEntry("proc-c", 1, ["eta", "theta", "iota"]),
            _ProcedureEntry("proc-d", 1, ["kappa", "lambda", "mu"]),
        ]
        lines = deduplicate_procedures(procs)
        # Alle disjunkt -> keine Gruppierung, alle individuell
        assert len(lines) == 4


# ============================================================================
# Test: Database Persistence
# ============================================================================


class TestDatabasePersistence:
    """Tests fuer die Datenbankpersistenz."""

    def test_data_survives_reconnect(self, db_path: Path) -> None:
        """Daten ueberleben das Schliessen und Wiederoeffen der DB."""
        reg1 = ToolRegistryDB(db_path)
        reg1.upsert_tool(name="persistent_tool", description_en="I persist")
        reg1.close()

        reg2 = ToolRegistryDB(db_path)
        tool = reg2.get_tool("persistent_tool")
        reg2.close()

        assert tool is not None
        assert tool.description == "I persist"

    def test_auto_creates_directory(self, tmp_path: Path) -> None:
        """DB-Verzeichnis wird automatisch erstellt."""
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        reg = ToolRegistryDB(deep_path)
        reg.upsert_tool(name="test", description_en="test")
        assert reg.tool_count() == 1
        reg.close()
        assert deep_path.exists()


# ============================================================================
# Test: Constants
# ============================================================================


class TestConstants:
    """Tests fuer die Konstanten-Definitionen."""

    def test_role_defaults_are_sets(self) -> None:
        """TOOL_ROLE_DEFAULTS enthaelt Sets."""
        for role, tools in TOOL_ROLE_DEFAULTS.items():
            assert isinstance(tools, set), f"Role {role} is not a set"

    def test_all_roles_present(self) -> None:
        """Alle erwarteten Rollen sind definiert."""
        assert "planner" in TOOL_ROLE_DEFAULTS
        assert "executor" in TOOL_ROLE_DEFAULTS
        assert "browser" in TOOL_ROLE_DEFAULTS
        assert "researcher" in TOOL_ROLE_DEFAULTS

    def test_default_examples_have_pairs(self) -> None:
        """Alle Default-Beispiele haben Input und Output."""
        for name, (ex_in, ex_out) in DEFAULT_EXAMPLES.items():
            assert ex_in, f"Example input missing for {name}"
            assert ex_out, f"Example output missing for {name}"

    def test_default_examples_count(self) -> None:
        """Mindestens 20 Default-Beispiele vorhanden."""
        assert len(DEFAULT_EXAMPLES) >= 20

    def test_tool_categories_coverage(self) -> None:
        """TOOL_CATEGORIES deckt die wichtigsten Tools ab."""
        assert "read_file" in TOOL_CATEGORIES
        assert "web_search" in TOOL_CATEGORIES
        assert "docker_ps" in TOOL_CATEGORIES
        # browser_navigate uses prefix detection ("browser_" -> "browser")


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests fuer Randfaelle."""

    def test_empty_input_schema(self, registry: ToolRegistryDB) -> None:
        """Tool ohne Input-Schema wird korrekt behandelt."""
        registry.upsert_tool(name="no_params", input_schema={})
        section = registry.get_tool_prompt_section("all", "en")
        assert "no_params()" in section

    def test_unicode_in_descriptions(self, registry: ToolRegistryDB) -> None:
        """Unicode-Zeichen in Beschreibungen werden korrekt gespeichert."""
        registry.upsert_tool(
            name="unicode_tool",
            description_de="Umlaute: ae, oe, ue, ss",
            description_en="English description",
            description_zh="中文描述测试",
        )
        # get_tool defaults to "en"
        tool = registry.get_tool("unicode_tool")
        assert tool.description == "English description"
        # Check zh via role query
        tools = registry.get_tools_for_role("all", "zh")
        zh_tool = next(t for t in tools if t.name == "unicode_tool")
        assert zh_tool.description == "中文描述测试"
        # Check de via role query
        tools_de = registry.get_tools_for_role("all", "de")
        de_tool = next(t for t in tools_de if t.name == "unicode_tool")
        assert de_tool.description == "Umlaute: ae, oe, ue, ss"

    def test_many_tools_performance(self, registry: ToolRegistryDB) -> None:
        """Viele Tools koennen ohne Fehler verarbeitet werden."""
        for i in range(200):
            registry.upsert_tool(
                name=f"tool_{i:03d}",
                description_en=f"Tool number {i}",
                category="other",
            )
        assert registry.tool_count() == 200
        tools = registry.get_tools_for_role("all", "en")
        assert len(tools) == 200

    def test_special_chars_in_tool_name(self, registry: ToolRegistryDB) -> None:
        """Tool-Namen mit Sonderzeichen werden korrekt behandelt."""
        registry.upsert_tool(name="my-tool_v2.1", description_en="Special name")
        tool = registry.get_tool("my-tool_v2.1")
        assert tool is not None
        assert tool.name == "my-tool_v2.1"
