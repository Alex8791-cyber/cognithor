"""Tests für mcp/memory_server.py · Memory-Tools für den Planner.

Testet:
  - MemoryTools: Alle 10 Tool-Funktionen
  - register_memory_tools: Korrekte MCP-Registration
  - Fehlerbehandlung: Leere Inputs, ungültige Tiers, fehlende Entitäten
  - Integration: Zusammenspiel mit MemoryManager
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from jarvis.config import JarvisConfig
from jarvis.mcp.memory_server import MemoryTools, register_memory_tools
from jarvis.memory.manager import MemoryManager
from jarvis.models import Entity, MemoryTier, Relation

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config(tmp_path: Path) -> JarvisConfig:
    """Config mit temporärem Home-Verzeichnis."""
    return JarvisConfig(jarvis_home=tmp_path / ".jarvis")


@pytest.fixture
def manager(config: JarvisConfig) -> MemoryManager:
    """Initialisierter MemoryManager."""
    mm = MemoryManager(config)
    mm.initialize_sync()
    return mm


@pytest.fixture
def tools(manager: MemoryManager) -> MemoryTools:
    """MemoryTools-Instanz zum Testen."""
    return MemoryTools(manager)


@pytest.fixture
def manager_with_data(manager: MemoryManager, config: JarvisConfig) -> MemoryManager:
    """MemoryManager mit vorindexierten Testdaten."""
    # Semantic data
    knowledge_dir = config.knowledge_dir
    (knowledge_dir / "kunden").mkdir(parents=True, exist_ok=True)
    test_file = knowledge_dir / "kunden" / "mueller-thomas.md"
    test_file.write_text(
        "# Müller, Thomas\n\n"
        "- **Beruf:** Softwareentwickler\n"
        "- **Risikoklasse:** 1+\n"
        "- **Budget:** 100€/Monat\n\n"
        "## Policen\n"
        "- Cloud Pro (TechCorp, seit 2024-07)\n"
        "- Monitoring (TechCorp, seit 2024-07)\n",
        encoding="utf-8",
    )
    manager.index_file(test_file, MemoryTier.SEMANTIC)

    # Entities
    e1 = Entity(type="person", name="Müller, Thomas", attributes={"beruf": "Softwareentwickler"})
    e2 = Entity(type="product", name="Cloud Pro", attributes={"versicherer": "TechCorp"})
    manager.index.upsert_entity(e1)
    manager.index.upsert_entity(e2)

    r = Relation(
        source_entity=e1.id,
        relation_type="hat_police",
        target_entity=e2.id,
    )
    manager.index.upsert_relation(r)

    return manager


@pytest.fixture
def tools_with_data(manager_with_data: MemoryManager) -> MemoryTools:
    """MemoryTools mit vorindexierten Testdaten."""
    return MemoryTools(manager_with_data)


# =============================================================================
# Tests: search_memory
# =============================================================================


class TestSearchMemory:
    """Tests für die Hybrid-Suche."""

    def test_search_finds_indexed_content(self, tools_with_data: MemoryTools):
        result = tools_with_data.search_memory("Müller Softwareentwickler")
        assert "Müller" in result
        assert "Ergebnis" in result

    def test_search_no_results(self, tools: MemoryTools):
        result = tools.search_memory("xyznonexistent12345")
        assert "Keine Ergebnisse" in result

    def test_search_empty_query(self, tools: MemoryTools):
        result = tools.search_memory("")
        assert "Fehler" in result

    def test_search_whitespace_query(self, tools: MemoryTools):
        result = tools.search_memory("   ")
        assert "Fehler" in result

    def test_search_with_tier_filter(self, tools_with_data: MemoryTools):
        result = tools_with_data.search_memory("Müller", tier="semantic")
        assert "Müller" in result or "Keine Ergebnisse" in result

    def test_search_invalid_tier(self, tools: MemoryTools):
        result = tools.search_memory("test", tier="invalid_tier")
        assert "Fehler" in result
        assert "Unbekannter Tier" in result

    def test_search_top_k_clamped(self, tools_with_data: MemoryTools):
        # top_k wird auf 1-20 begrenzt
        result = tools_with_data.search_memory("Müller", top_k=0)
        # Should still work (clamped to 1)
        assert isinstance(result, str)

        result = tools_with_data.search_memory("Müller", top_k=100)
        # Should still work (clamped to 20)
        assert isinstance(result, str)


# =============================================================================
# Tests: save_to_memory
# =============================================================================


class TestSaveToMemory:
    """Tests für das Speichern in verschiedene Tiers."""

    def test_save_episodic(self, tools: MemoryTools):
        result = tools.save_to_memory(
            content="Recherche-Bericht wurde erstellt",
            tier="episodic",
            topic="Beratung",
        )
        assert "Episodic Memory gespeichert" in result
        assert "Beratung" in result

    def test_save_episodic_default_topic(self, tools: MemoryTools):
        result = tools.save_to_memory(content="Test-Eintrag")
        assert "Notiz" in result

    def test_save_semantic(self, tools: MemoryTools):
        result = tools.save_to_memory(
            content="Cloud Platform Pro ist der beste Tarif",
            tier="semantic",
            source_path="produkte/wwk-bu.md",
        )
        assert "Semantic Memory indexiert" in result
        assert "Chunk" in result

    def test_save_semantic_auto_path(self, tools: MemoryTools):
        result = tools.save_to_memory(content="Test-Fakt", tier="semantic")
        assert "knowledge/auto/" in result

    def test_save_procedural(self, tools: MemoryTools, manager: MemoryManager):
        proc_content = (
            "---\nname: test-skill\ntrigger: [test]\n---\n"
            "# Test Skill\n\n## Ablauf\n1. Schritt eins\n"
        )
        result = tools.save_to_memory(
            content=proc_content,
            tier="procedural",
            source_path="test-skill.md",
        )
        assert "Procedural Memory gespeichert" in result
        # Datei muss existieren
        proc_file = manager.procedural._dir / "test-skill.md"
        assert proc_file.exists()

    def test_save_procedural_auto_extension(self, tools: MemoryTools):
        result = tools.save_to_memory(
            content="# Test",
            tier="procedural",
            source_path="my-proc",
        )
        assert "my-proc.md" in result

    def test_save_core_blocked(self, tools: MemoryTools):
        result = tools.save_to_memory(content="hack", tier="core")
        assert "Fehler" in result
        assert "nicht direkt beschreibbar" in result

    def test_save_invalid_tier(self, tools: MemoryTools):
        result = tools.save_to_memory(content="test", tier="invalid")
        assert "Fehler" in result
        assert "Unbekannter Tier" in result

    def test_save_empty_content(self, tools: MemoryTools):
        result = tools.save_to_memory(content="")
        assert "Fehler" in result


# =============================================================================
# Tests: Entity/Relation (Wissens-Graph)
# =============================================================================


class TestEntityOperations:
    """Tests für Entitäten und Relationen."""

    def test_get_entity(self, tools_with_data: MemoryTools):
        result = tools_with_data.get_entity("Müller")
        assert "Müller" in result
        assert "person" in result

    def test_get_entity_with_relations(self, tools_with_data: MemoryTools):
        result = tools_with_data.get_entity("Müller")
        assert "hat_police" in result
        assert "Cloud Pro" in result

    def test_get_entity_not_found(self, tools: MemoryTools):
        result = tools.get_entity("NichtExistent999")
        assert "Keine Entität" in result

    def test_get_entity_empty_name(self, tools: MemoryTools):
        result = tools.get_entity("")
        assert "Fehler" in result

    def test_add_entity(self, tools: MemoryTools, manager: MemoryManager):
        result = tools.add_entity(
            name="Schmidt, Anna",
            entity_type="person",
            attributes='{"beruf": "Ärztin"}',
        )
        assert "Entität erstellt" in result
        assert "Schmidt, Anna" in result

        # Verify in DB
        entities = manager.index.search_entities("Schmidt")
        assert len(entities) == 1
        assert entities[0].name == "Schmidt, Anna"

    def test_add_entity_empty_name(self, tools: MemoryTools):
        result = tools.add_entity(name="", entity_type="person")
        assert "Fehler" in result

    def test_add_entity_invalid_json(self, tools: MemoryTools):
        result = tools.add_entity(
            name="Test",
            entity_type="person",
            attributes="{broken json",
        )
        assert "Fehler" in result
        assert "Ungültiges JSON" in result

    def test_add_relation(self, tools_with_data: MemoryTools):
        result = tools_with_data.add_relation(
            source_name="Müller",
            relation_type="arbeitet_bei",
            target_name="Cloud Pro",
        )
        assert "Relation erstellt" in result
        assert "arbeitet_bei" in result

    def test_add_relation_source_not_found(self, tools: MemoryTools):
        result = tools.add_relation(
            source_name="NichtDa",
            relation_type="kennt",
            target_name="AuchNichtDa",
        )
        assert "Fehler" in result
        assert "nicht gefunden" in result

    def test_add_relation_target_not_found(self, tools_with_data: MemoryTools):
        result = tools_with_data.add_relation(
            source_name="Müller",
            relation_type="kennt",
            target_name="NichtDa999",
        )
        assert "Fehler" in result
        assert "Ziel-Entität" in result

    def test_add_relation_invalid_json(self, tools_with_data: MemoryTools):
        result = tools_with_data.add_relation(
            source_name="Müller",
            relation_type="test",
            target_name="Cloud Pro",
            attributes="{broken",
        )
        assert "Fehler" in result
        assert "Ungültiges JSON" in result


# =============================================================================
# Tests: Core Memory
# =============================================================================


class TestCoreMemory:
    """Tests für Core Memory Zugriff."""

    def test_get_core_memory(self, tools: MemoryTools):
        result = tools.get_core_memory()
        assert "Identität" in result  # Aus Default-CORE.md

    def test_get_core_memory_custom(self, tools: MemoryTools, manager: MemoryManager):
        manager.core._path.write_text(
            "# Custom Core\nIch bin ein Test-Agent.\n",
            encoding="utf-8",
        )
        manager.core._content = None  # Cache invalidieren
        result = tools.get_core_memory()
        assert "Test-Agent" in result


# =============================================================================
# Tests: Episodes
# =============================================================================


class TestEpisodes:
    """Tests für Episodic Memory Zugriff."""

    def test_get_recent_episodes_empty(self, tools: MemoryTools):
        result = tools.get_recent_episodes(days=3)
        assert "Keine Episodic-Einträge" in result

    def test_get_recent_episodes_with_data(self, tools: MemoryTools, manager: MemoryManager):
        manager.episodic.append_entry(topic="Test", content="Ein Testeintrag")
        result = tools.get_recent_episodes(days=1)
        assert date.today().isoformat() in result
        assert "Test" in result

    def test_get_recent_episodes_clamped(self, tools: MemoryTools):
        # Days wird auf 1-30 begrenzt
        result = tools.get_recent_episodes(days=0)
        assert isinstance(result, str)  # Clamped to 1

        result = tools.get_recent_episodes(days=100)
        assert isinstance(result, str)  # Clamped to 30


# =============================================================================
# Tests: Procedures
# =============================================================================


class TestProcedures:
    """Tests für Procedural Memory Zugriff."""

    def test_search_procedures_empty(self, tools: MemoryTools):
        result = tools.search_procedures("Recherche-Bericht")
        assert "Keine Prozeduren" in result

    def test_search_procedures_with_data(self, tools: MemoryTools, manager: MemoryManager):
        # Prozedur erstellen
        from jarvis.memory.procedural import ProcedureMetadata

        meta = ProcedureMetadata(
            name="bu-angebot",
            trigger_keywords=["Recherche", "Bericht"],
        )
        manager.procedural.save_procedure(
            name="bu-angebot",
            body="# Recherche-Bericht\n\n## Ablauf\n1. Beruf klären\n2. Tarif wählen\n",
            metadata=meta,
        )
        result = tools.search_procedures("BU")
        assert "bu-angebot" in result
        assert "Prozedur" in result

    def test_search_procedures_empty_query(self, tools: MemoryTools):
        result = tools.search_procedures("")
        assert "Fehler" in result

    def test_record_usage_not_found(self, tools: MemoryTools):
        result = tools.record_procedure_usage(name="nichtda", success=True)
        assert "Fehler" in result or "nicht gefunden" in result

    def test_record_usage_success(self, tools: MemoryTools, manager: MemoryManager):
        from jarvis.memory.procedural import ProcedureMetadata

        meta = ProcedureMetadata(
            name="test-proc",
            trigger_keywords=["test"],
        )
        manager.procedural.save_procedure(name=meta.name, body="# Test\n", metadata=meta)

        result = tools.record_procedure_usage(name="test-proc", success=True, score=0.9)
        assert "Erfolg" in result
        assert "1x" in result

    def test_record_usage_failure(self, tools: MemoryTools, manager: MemoryManager):
        from jarvis.memory.procedural import ProcedureMetadata

        meta = ProcedureMetadata(name="fail-proc", trigger_keywords=["fail"])
        manager.procedural.save_procedure(name=meta.name, body="# Fail\n", metadata=meta)

        result = tools.record_procedure_usage(name="fail-proc", success=False)
        assert "Fehlschlag" in result

    def test_record_usage_empty_name(self, tools: MemoryTools):
        result = tools.record_procedure_usage(name="", success=True)
        assert "Fehler" in result


# =============================================================================
# Tests: Stats
# =============================================================================


class TestStats:
    """Tests für Memory-Statistiken."""

    def test_memory_stats(self, tools: MemoryTools):
        result = tools.memory_stats()
        assert "Memory-System Status" in result
        assert "Chunks" in result
        assert "Entitäten" in result
        assert "Prozeduren" in result

    def test_memory_stats_with_data(self, tools_with_data: MemoryTools):
        result = tools_with_data.memory_stats()
        assert "Memory-System Status" in result
        # Sollte Chunks > 0 anzeigen
        assert "Chunks:" in result


# =============================================================================
# Tests: Registration
# =============================================================================


class TestRegistration:
    """Tests für die MCP-Tool-Registration."""

    def test_register_memory_tools(self, manager: MemoryManager):
        """Alle 10 Tools werden korrekt registriert."""

        class MockMCPClient:
            def __init__(self):
                self.handlers: dict[str, Any] = {}

            def register_builtin_handler(self, name, fn, description="", input_schema=None):
                self.handlers[name] = {
                    "fn": fn,
                    "description": description,
                    "schema": input_schema,
                }

        from typing import Any

        client = MockMCPClient()
        mt = register_memory_tools(client, manager)

        assert isinstance(mt, MemoryTools)

        expected_tools = [
            "search_memory",
            "save_to_memory",
            "get_entity",
            "add_entity",
            "add_relation",
            "get_core_memory",
            "get_recent_episodes",
            "search_procedures",
            "record_procedure_usage",
            "memory_stats",
        ]
        for tool_name in expected_tools:
            assert tool_name in client.handlers, f"Tool '{tool_name}' nicht registriert"
            handler = client.handlers[tool_name]
            assert handler["description"], f"Tool '{tool_name}' hat keine Beschreibung"
            assert handler["schema"], f"Tool '{tool_name}' hat kein Schema"

    def test_registered_handlers_callable(self, manager: MemoryManager):
        """Registrierte Handler sind aufrufbar."""

        class MockMCPClient:
            def __init__(self):
                self.handlers = {}

            def register_builtin_handler(self, name, fn, **kwargs):
                self.handlers[name] = fn

        client = MockMCPClient()
        register_memory_tools(client, manager)

        # Alle Handler aufrufen
        result = client.handlers["memory_stats"]()
        assert "Memory-System Status" in result

        result = client.handlers["get_core_memory"]()
        assert isinstance(result, str)

        result = client.handlers["search_memory"](query="test")
        assert isinstance(result, str)


# =============================================================================
# Tests: Format-Qualität
# =============================================================================


class TestFormatQuality:
    """Tests für die Ausgabe-Formatierung."""

    def test_search_results_contain_score(self, tools_with_data: MemoryTools):
        result = tools_with_data.search_memory("Müller")
        if "Ergebnis" in result:
            assert "Score:" in result

    def test_search_results_contain_tier(self, tools_with_data: MemoryTools):
        result = tools_with_data.search_memory("Müller")
        if "Ergebnis" in result:
            assert "Tier:" in result

    def test_entity_format_shows_type(self, tools_with_data: MemoryTools):
        result = tools_with_data.get_entity("Müller")
        assert "Typ:" in result

    def test_stats_format_shows_sections(self, tools: MemoryTools):
        result = tools.memory_stats()
        assert "Chunks:" in result
        assert "Embeddings:" in result
        assert "Entitäten:" in result
        assert "Relationen:" in result
        assert "Prozeduren:" in result
