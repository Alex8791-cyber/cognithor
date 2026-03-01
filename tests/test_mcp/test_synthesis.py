"""Tests für Knowledge Synthesis Tools (synthesis.py).

Prüft:
  - Tool-Registration (4 Tools, Schemas, Descriptions)
  - Dependency Injection (LLM, Memory, Vault, Web)
  - _check_ready Logik
  - Hilfsfunktionen (_truncate, _extract_keywords, _filter_relevant_text)
  - Tool-Methoden mit gemockten Abhängigkeiten
  - Wiring in tools.py (register_*-Rückgabewerte erfasst)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.mcp.synthesis import (
    KnowledgeSynthesizer,
    _extract_keywords,
    _filter_relevant_text,
    _truncate,
    register_synthesis_tools,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


class MockMCPClient:
    """Simpler Mock für den JarvisMCPClient."""

    def __init__(self) -> None:
        self.registered: dict[str, dict[str, Any]] = {}

    def register_builtin_handler(
        self,
        name: str,
        handler: object,
        *,
        description: str = "",
        input_schema: dict | None = None,
    ) -> None:
        self.registered[name] = {
            "handler": handler,
            "description": description,
            "input_schema": input_schema,
        }


@pytest.fixture()
def mock_client() -> MockMCPClient:
    return MockMCPClient()


@pytest.fixture()
def synthesizer() -> KnowledgeSynthesizer:
    return KnowledgeSynthesizer()


@pytest.fixture()
def wired_synthesizer() -> KnowledgeSynthesizer:
    """Synthesizer mit allen Abhängigkeiten gemockt."""
    synth = KnowledgeSynthesizer()

    async def fake_llm(prompt: str, model: str = "") -> str:
        return f"LLM-Antwort für: {prompt[:50]}"

    synth._set_llm_fn(fake_llm, "test-model")

    memory = MagicMock()
    memory.search_memory.return_value = "Tesla ist ein Elektroautohersteller."
    memory.get_entity.return_value = "Entität: Tesla | Typ: Unternehmen"
    memory.get_recent_episodes.return_value = "2026-03-01: Tesla Recherche durchgeführt."
    synth._set_memory_tools(memory)

    vault = AsyncMock()
    vault.vault_search.return_value = "Notiz: Tesla Q4 2025 Ergebnisse"
    vault.vault_save.return_value = "Gespeichert als 'Synthese: Tesla'"
    synth._set_vault_tools(vault)

    web = AsyncMock()
    web.search_and_read.return_value = "Tesla meldete Rekordumsatz in Q4 2025."
    synth._set_web_tools(web)

    return synth


# ── Test: Registration ───────────────────────────────────────────────────


class TestRegisterSynthesisTools:
    """Tests für register_synthesis_tools()."""

    EXPECTED_TOOLS = [
        "knowledge_synthesize",
        "knowledge_contradictions",
        "knowledge_timeline",
        "knowledge_gaps",
    ]

    def test_registers_four_tools(self, mock_client: MockMCPClient) -> None:
        """Alle 4 Synthesis-Tools werden registriert."""
        synth = register_synthesis_tools(mock_client)

        assert isinstance(synth, KnowledgeSynthesizer)
        assert len(mock_client.registered) == 4

        for name in self.EXPECTED_TOOLS:
            assert name in mock_client.registered, f"Tool '{name}' nicht registriert"

    def test_returns_synthesizer_instance(self, mock_client: MockMCPClient) -> None:
        """Gibt KnowledgeSynthesizer-Instanz zurück."""
        synth = register_synthesis_tools(mock_client)
        assert isinstance(synth, KnowledgeSynthesizer)

    def test_handlers_are_callable(self, mock_client: MockMCPClient) -> None:
        """Alle Handler sind aufrufbar."""
        register_synthesis_tools(mock_client)

        for name, entry in mock_client.registered.items():
            assert callable(entry["handler"]), f"Handler für '{name}' nicht aufrufbar"

    def test_descriptions_non_empty(self, mock_client: MockMCPClient) -> None:
        """Alle Tools haben eine nicht-leere Beschreibung."""
        register_synthesis_tools(mock_client)

        for name, entry in mock_client.registered.items():
            assert entry["description"], f"Description für '{name}' ist leer"
            assert len(entry["description"]) > 20, f"Description für '{name}' zu kurz"

    def test_schemas_present_and_valid(self, mock_client: MockMCPClient) -> None:
        """Alle Tools haben ein gültiges JSON-Schema."""
        register_synthesis_tools(mock_client)

        for name, entry in mock_client.registered.items():
            schema = entry["input_schema"]
            assert schema is not None, f"Schema für '{name}' fehlt"
            assert schema.get("type") == "object", f"Schema für '{name}' hat falschen type"
            assert "properties" in schema, f"Schema für '{name}' hat keine properties"
            assert "required" in schema, f"Schema für '{name}' hat keine required"
            assert "topic" in schema["required"], f"'topic' fehlt in required für '{name}'"

    def test_schema_topic_is_string(self, mock_client: MockMCPClient) -> None:
        """Das 'topic'-Feld in allen Schemas ist vom Typ 'string'."""
        register_synthesis_tools(mock_client)

        for name, entry in mock_client.registered.items():
            props = entry["input_schema"]["properties"]
            assert "topic" in props, f"'topic' fehlt in properties für '{name}'"
            assert props["topic"]["type"] == "string"

    def test_synthesize_schema_has_depth(self, mock_client: MockMCPClient) -> None:
        """knowledge_synthesize hat depth-Parameter mit enum."""
        register_synthesis_tools(mock_client)

        schema = mock_client.registered["knowledge_synthesize"]["input_schema"]
        assert "depth" in schema["properties"]
        assert schema["properties"]["depth"]["enum"] == ["quick", "standard", "deep"]

    def test_synthesize_schema_has_save_to_vault(self, mock_client: MockMCPClient) -> None:
        """knowledge_synthesize hat save_to_vault-Parameter."""
        register_synthesis_tools(mock_client)

        schema = mock_client.registered["knowledge_synthesize"]["input_schema"]
        assert "save_to_vault" in schema["properties"]
        assert schema["properties"]["save_to_vault"]["type"] == "boolean"

    def test_config_optional(self, mock_client: MockMCPClient) -> None:
        """Config-Parameter ist optional."""
        synth = register_synthesis_tools(mock_client, config=None)
        assert isinstance(synth, KnowledgeSynthesizer)

    def test_with_magicmock_client(self) -> None:
        """Funktioniert auch mit MagicMock."""
        client = MagicMock()
        synth = register_synthesis_tools(client)

        assert isinstance(synth, KnowledgeSynthesizer)
        assert client.register_builtin_handler.call_count == 4

        registered_names = [
            call.args[0]
            for call in client.register_builtin_handler.call_args_list
        ]
        for name in self.EXPECTED_TOOLS:
            assert name in registered_names


# ── Test: Dependency Injection ───────────────────────────────────────────


class TestDependencyInjection:
    """Tests für Setter-Methoden und _check_ready."""

    def test_initial_state(self, synthesizer: KnowledgeSynthesizer) -> None:
        """Alle Abhängigkeiten sind initial None."""
        assert synthesizer._llm_fn is None
        assert synthesizer._llm_model == ""
        assert synthesizer._memory_tools is None
        assert synthesizer._vault_tools is None
        assert synthesizer._web_tools is None

    def test_check_ready_no_llm(self, synthesizer: KnowledgeSynthesizer) -> None:
        """Fehlermeldung wenn kein LLM."""
        error = synthesizer._check_ready()
        assert error is not None
        assert "LLM" in error

    def test_check_ready_no_memory(self, synthesizer: KnowledgeSynthesizer) -> None:
        """Fehlermeldung wenn kein Memory."""
        synthesizer._set_llm_fn(lambda p, m="": "test", "model")
        error = synthesizer._check_ready()
        assert error is not None
        assert "Memory" in error

    def test_check_ready_ok(self, synthesizer: KnowledgeSynthesizer) -> None:
        """None wenn LLM + Memory gesetzt."""
        synthesizer._set_llm_fn(lambda p, m="": "test", "model")
        synthesizer._set_memory_tools(MagicMock())
        assert synthesizer._check_ready() is None

    def test_set_llm_fn(self, synthesizer: KnowledgeSynthesizer) -> None:
        """LLM-Funktion und Modellname werden gesetzt."""
        fn = lambda p, m="": "result"
        synthesizer._set_llm_fn(fn, "qwen3:32b")
        assert synthesizer._llm_fn is fn
        assert synthesizer._llm_model == "qwen3:32b"

    def test_set_memory_tools(self, synthesizer: KnowledgeSynthesizer) -> None:
        """Memory-Tools werden gesetzt."""
        mem = MagicMock()
        synthesizer._set_memory_tools(mem)
        assert synthesizer._memory_tools is mem

    def test_set_vault_tools(self, synthesizer: KnowledgeSynthesizer) -> None:
        """Vault-Tools werden gesetzt."""
        vault = AsyncMock()
        synthesizer._set_vault_tools(vault)
        assert synthesizer._vault_tools is vault

    def test_set_web_tools(self, synthesizer: KnowledgeSynthesizer) -> None:
        """Web-Tools werden gesetzt."""
        web = AsyncMock()
        synthesizer._set_web_tools(web)
        assert synthesizer._web_tools is web


# ── Test: Hilfsfunktionen ────────────────────────────────────────────────


class TestHelpers:
    """Tests für _truncate, _extract_keywords, _filter_relevant_text."""

    # _truncate

    def test_truncate_short_text(self) -> None:
        """Kurzer Text wird nicht gekürzt."""
        text = "Kurzer Text."
        assert _truncate(text, 100) == text

    def test_truncate_long_text(self) -> None:
        """Langer Text wird an Satzende gekürzt."""
        text = "Erster Satz. Zweiter Satz. Dritter Satz. Vierter Satz."
        result = _truncate(text, 30)
        assert len(result) < len(text)
        assert "[... gekürzt]" in result

    def test_truncate_exact_limit(self) -> None:
        """Text genau am Limit wird nicht gekürzt."""
        text = "Genau fünf."  # 11 Zeichen
        assert _truncate(text, 11) == text

    def test_truncate_preserves_sentence_boundary(self) -> None:
        """Kürzung am Satzende, nicht mitten im Wort."""
        text = "Erster Satz. Zweiter Satz ist länger und hat viele Wörter."
        result = _truncate(text, 25)
        # Sollte nach "Erster Satz." kürzen
        assert result.startswith("Erster Satz.")

    # _extract_keywords

    def test_extract_keywords_basic(self) -> None:
        """Extrahiert Schlüsselwörter, filtert Stoppwörter."""
        keywords = _extract_keywords("Was wissen wir über Tesla und Elektroautos")
        assert "Tesla" in keywords
        assert "Elektroautos" in keywords
        # Stoppwörter sollten nicht enthalten sein
        assert "wir" not in [k.lower() for k in keywords]
        assert "über" not in [k.lower() for k in keywords]

    def test_extract_keywords_deduplicate(self) -> None:
        """Keine Duplikate in Keywords."""
        keywords = _extract_keywords("Tesla Tesla Tesla Aktie Aktie")
        # Groß/Kleinschreibung normalisiert, nur eindeutige
        lower_kws = [k.lower() for k in keywords]
        assert len(lower_kws) == len(set(lower_kws))

    def test_extract_keywords_max_10(self) -> None:
        """Maximal 10 Keywords."""
        long_text = " ".join(f"Keyword{i}" for i in range(30))
        keywords = _extract_keywords(long_text)
        assert len(keywords) <= 10

    def test_extract_keywords_empty(self) -> None:
        """Leerer Text gibt leere Liste."""
        assert _extract_keywords("") == []

    def test_extract_keywords_only_stopwords(self) -> None:
        """Nur Stoppwörter gibt leere Liste."""
        result = _extract_keywords("der die das und oder")
        assert result == []

    # _filter_relevant_text

    def test_filter_relevant_basic(self) -> None:
        """Filtert relevante Absätze basierend auf Keywords."""
        text = "Tesla meldet Rekordumsatz.\n\nDas Wetter ist schön.\n\nTesla Aktie steigt."
        result = _filter_relevant_text(text, "Tesla Aktie")
        assert "Rekordumsatz" in result
        assert "Aktie steigt" in result
        assert "Wetter" not in result

    def test_filter_relevant_no_match(self) -> None:
        """Kein passender Absatz ergibt leeren String."""
        text = "Abschnitt über Äpfel.\n\nAbschnitt über Birnen."
        result = _filter_relevant_text(text, "Quantencomputer")
        assert result == ""

    def test_filter_relevant_all_match(self) -> None:
        """Alle Absätze relevant → alles zurückgeben."""
        text = "Tesla meldet Gewinn.\n\nTesla expandiert."
        result = _filter_relevant_text(text, "Tesla Gewinn")
        assert "meldet Gewinn" in result
        assert "expandiert" in result


# ── Test: Tool-Methoden ──────────────────────────────────────────────────


class TestKnowledgeSynthesize:
    """Tests für knowledge_synthesize()."""

    @pytest.mark.asyncio
    async def test_empty_topic(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        """Leeres Thema gibt Fehlermeldung."""
        result = await wired_synthesizer.knowledge_synthesize(topic="")
        assert "Fehler" in result

    @pytest.mark.asyncio
    async def test_whitespace_topic(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        """Nur Whitespace-Thema gibt Fehlermeldung."""
        result = await wired_synthesizer.knowledge_synthesize(topic="   ")
        assert "Fehler" in result

    @pytest.mark.asyncio
    async def test_not_ready(self, synthesizer: KnowledgeSynthesizer) -> None:
        """Ohne Abhängigkeiten gibt Fehlermeldung."""
        result = await synthesizer.knowledge_synthesize(topic="Tesla")
        assert "nicht verfügbar" in result

    @pytest.mark.asyncio
    async def test_basic_synthesis(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        """Grundlegende Synthese wird durchgeführt."""
        result = await wired_synthesizer.knowledge_synthesize(topic="Tesla")
        assert "LLM-Antwort" in result
        assert "Synthese erstellt:" in result
        assert "Tiefe: standard" in result

    @pytest.mark.asyncio
    async def test_depth_quick(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        """depth=quick ruft kein Web auf."""
        result = await wired_synthesizer.knowledge_synthesize(
            topic="Tesla", depth="quick"
        )
        assert "Tiefe: quick" in result
        # Web-Tools sollten nicht aufgerufen worden sein
        wired_synthesizer._web_tools.search_and_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_depth_deep(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        """depth=deep funktioniert."""
        result = await wired_synthesizer.knowledge_synthesize(
            topic="Tesla", depth="deep"
        )
        assert "Tiefe: deep" in result

    @pytest.mark.asyncio
    async def test_save_to_vault(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        """save_to_vault speichert im Vault."""
        result = await wired_synthesizer.knowledge_synthesize(
            topic="Tesla", save_to_vault=True
        )
        wired_synthesizer._vault_tools.vault_save.assert_called_once()
        assert "Im Vault gespeichert" in result

    @pytest.mark.asyncio
    async def test_no_sources_found(self) -> None:
        """Wenn keine Quellen gefunden: Fehlermeldung."""
        synth = KnowledgeSynthesizer()

        async def fake_llm(prompt: str, model: str = "") -> str:
            return "Result"

        synth._set_llm_fn(fake_llm, "model")

        # Memory gibt nichts zurück
        memory = MagicMock()
        memory.search_memory.return_value = "Keine Ergebnisse"
        memory.get_entity.return_value = "Keine Entität gefunden"
        memory.get_recent_episodes.return_value = "Keine Episodic"
        synth._set_memory_tools(memory)

        # Kein Vault, kein Web
        result = await synth.knowledge_synthesize(topic="Unbekannt", depth="quick")
        assert "Keine Informationen" in result

    @pytest.mark.asyncio
    async def test_llm_failure(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        """LLM-Fehler wird abgefangen."""
        async def failing_llm(prompt: str, model: str = "") -> str:
            raise RuntimeError("LLM timeout")

        wired_synthesizer._set_llm_fn(failing_llm, "model")
        result = await wired_synthesizer.knowledge_synthesize(topic="Tesla")
        assert "Fehler" in result
        assert "LLM timeout" in result


class TestKnowledgeContradictions:
    """Tests für knowledge_contradictions()."""

    @pytest.mark.asyncio
    async def test_empty_topic(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        result = await wired_synthesizer.knowledge_contradictions(topic="")
        assert "Fehler" in result

    @pytest.mark.asyncio
    async def test_not_ready(self, synthesizer: KnowledgeSynthesizer) -> None:
        result = await synthesizer.knowledge_contradictions(topic="Tesla")
        assert "nicht verfügbar" in result

    @pytest.mark.asyncio
    async def test_basic_contradiction_check(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        result = await wired_synthesizer.knowledge_contradictions(topic="Tesla")
        assert "LLM-Antwort" in result

    @pytest.mark.asyncio
    async def test_no_stored_data(self) -> None:
        """Ohne gespeicherte Daten: Fehlermeldung."""
        synth = KnowledgeSynthesizer()

        async def fake_llm(prompt: str, model: str = "") -> str:
            return "Result"

        synth._set_llm_fn(fake_llm, "m")
        memory = MagicMock()
        memory.search_memory.return_value = "Keine Ergebnisse"
        memory.get_entity.return_value = "Keine Entität gefunden"
        memory.get_recent_episodes.return_value = "Keine Episodic"
        synth._set_memory_tools(memory)

        result = await synth.knowledge_contradictions(topic="Unbekannt")
        assert "Keine gespeicherten Informationen" in result


class TestKnowledgeTimeline:
    """Tests für knowledge_timeline()."""

    @pytest.mark.asyncio
    async def test_empty_topic(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        result = await wired_synthesizer.knowledge_timeline(topic="")
        assert "Fehler" in result

    @pytest.mark.asyncio
    async def test_not_ready(self, synthesizer: KnowledgeSynthesizer) -> None:
        result = await synthesizer.knowledge_timeline(topic="Tesla")
        assert "nicht verfügbar" in result

    @pytest.mark.asyncio
    async def test_basic_timeline(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        result = await wired_synthesizer.knowledge_timeline(topic="Tesla")
        assert "LLM-Antwort" in result


class TestKnowledgeGaps:
    """Tests für knowledge_gaps()."""

    @pytest.mark.asyncio
    async def test_empty_topic(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        result = await wired_synthesizer.knowledge_gaps(topic="")
        assert "Fehler" in result

    @pytest.mark.asyncio
    async def test_not_ready(self, synthesizer: KnowledgeSynthesizer) -> None:
        result = await synthesizer.knowledge_gaps(topic="Tesla")
        assert "nicht verfügbar" in result

    @pytest.mark.asyncio
    async def test_basic_gaps(self, wired_synthesizer: KnowledgeSynthesizer) -> None:
        result = await wired_synthesizer.knowledge_gaps(topic="Tesla")
        assert "LLM-Antwort" in result

    @pytest.mark.asyncio
    async def test_gaps_no_data(self) -> None:
        """Ohne gespeicherte Daten: Analyse mit 'keine Informationen'."""
        synth = KnowledgeSynthesizer()

        async def fake_llm(prompt: str, model: str = "") -> str:
            return "Keine Daten vorhanden — Recherche-Roadmap: ..."

        synth._set_llm_fn(fake_llm, "m")
        memory = MagicMock()
        memory.search_memory.return_value = "Keine Ergebnisse"
        memory.get_entity.return_value = "Keine Entität gefunden"
        memory.get_recent_episodes.return_value = "Keine Episodic"
        synth._set_memory_tools(memory)

        result = await synth.knowledge_gaps(topic="Quantencomputer")
        # Sollte trotzdem funktionieren (gibt LLM-Antwort mit leerem Kontext)
        assert "Recherche-Roadmap" in result


# ── Test: Source Gathering ───────────────────────────────────────────────


class TestGatherSources:
    """Tests für _gather_sources()."""

    @pytest.mark.asyncio
    async def test_gathers_all_sources(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        """Sammelt von allen 5 Quelltypen."""
        sources = await wired_synthesizer._gather_sources("Tesla")

        assert "memory" in sources
        assert "entities" in sources
        assert "episodes" in sources
        assert "vault" in sources
        assert "web" in sources

    @pytest.mark.asyncio
    async def test_exclude_web(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        """include_web=False schließt Web aus."""
        sources = await wired_synthesizer._gather_sources(
            "Tesla", include_web=False
        )
        assert "web" not in sources

    @pytest.mark.asyncio
    async def test_exclude_vault(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        """include_vault=False schließt Vault aus."""
        sources = await wired_synthesizer._gather_sources(
            "Tesla", include_vault=False
        )
        assert "vault" not in sources

    @pytest.mark.asyncio
    async def test_exclude_memory(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        """include_memory=False schließt Memory + Entities aus."""
        sources = await wired_synthesizer._gather_sources(
            "Tesla", include_memory=False
        )
        assert "memory" not in sources
        assert "entities" not in sources

    @pytest.mark.asyncio
    async def test_exclude_episodes(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        """include_episodes=False schließt Episoden aus."""
        sources = await wired_synthesizer._gather_sources(
            "Tesla", include_episodes=False
        )
        assert "episodes" not in sources

    @pytest.mark.asyncio
    async def test_memory_error_handled(self) -> None:
        """Fehler in Memory werden abgefangen, andere Quellen funktionieren."""
        synth = KnowledgeSynthesizer()
        synth._set_llm_fn(AsyncMock(return_value="ok"), "m")

        memory = MagicMock()
        memory.search_memory.side_effect = RuntimeError("DB locked")
        memory.get_entity.side_effect = RuntimeError("DB locked")
        memory.get_recent_episodes.side_effect = RuntimeError("DB locked")
        synth._set_memory_tools(memory)

        vault = AsyncMock()
        vault.vault_search.return_value = "Vault-Treffer"
        synth._set_vault_tools(vault)

        sources = await synth._gather_sources("Tesla", include_web=False)
        # Memory fehlt, aber Vault ist da
        assert "memory" not in sources
        assert "vault" in sources


# ── Test: Format Source Context ──────────────────────────────────────────


class TestFormatSourceContext:
    """Tests für _format_source_context()."""

    def test_formats_all_sections(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        sources = {
            "memory": "Memory-Daten",
            "entities": "Entity-Daten",
            "vault": "Vault-Daten",
            "web": "Web-Daten",
        }
        result = wired_synthesizer._format_source_context(sources)

        assert "GESPEICHERTES WISSEN" in result
        assert "BEKANNTE ENTITÄTEN" in result
        assert "VAULT-NOTIZEN" in result
        assert "AKTUELLE WEB-RECHERCHE" in result
        assert "---" in result

    def test_empty_sources(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        result = wired_synthesizer._format_source_context({})
        assert result == ""

    def test_truncates_large_context(
        self, wired_synthesizer: KnowledgeSynthesizer
    ) -> None:
        sources = {"memory": "X" * 30000}
        result = wired_synthesizer._format_source_context(sources)
        assert len(result) <= 26000  # MAX_CONTEXT_CHARS + header + truncation notice


# ── Test: Wiring in tools.py ────────────────────────────────────────────


class TestToolsWiring:
    """Tests dass tools.py die Rückgabewerte korrekt erfasst."""

    def test_tools_py_captures_web_tools(self) -> None:
        """tools.py speichert register_web_tools Rückgabewert."""
        import inspect
        from jarvis.gateway.phases.tools import init_tools

        source = inspect.getsource(init_tools)
        assert "web_tools = register_web_tools" in source

    def test_tools_py_captures_memory_tools(self) -> None:
        """tools.py speichert register_memory_tools Rückgabewert."""
        import inspect
        from jarvis.gateway.phases.tools import init_tools

        source = inspect.getsource(init_tools)
        assert "memory_tools = register_memory_tools" in source

    def test_tools_py_registers_synthesis(self) -> None:
        """tools.py importiert und registriert synthesis tools."""
        import inspect
        from jarvis.gateway.phases.tools import init_tools

        source = inspect.getsource(init_tools)
        assert "register_synthesis_tools" in source
        assert "synthesis_tools_registered" in source

    def test_tools_py_injects_dependencies(self) -> None:
        """tools.py injiziert alle 4 Abhängigkeiten in den Synthesizer."""
        import inspect
        from jarvis.gateway.phases.tools import init_tools

        source = inspect.getsource(init_tools)
        assert "_set_llm_fn" in source
        assert "_set_memory_tools" in source
        assert "_set_vault_tools" in source
        assert "_set_web_tools" in source


# ── Test: Skill-Datei ────────────────────────────────────────────────────


class TestWissensSyntheseSkill:
    """Tests dass die Skill-Prozedur existiert und gültig ist."""

    SKILL_PATH = Path("D:/Jarvis/jarvis complete v20/data/procedures/wissens-synthese.md")

    def test_skill_file_exists(self) -> None:
        assert self.SKILL_PATH.exists(), "wissens-synthese.md existiert nicht"

    def test_skill_has_frontmatter(self) -> None:
        content = self.SKILL_PATH.read_text(encoding="utf-8")
        assert content.startswith("---"), "Kein YAML-Frontmatter"
        # Zweites --- muss auch existieren (Frontmatter-Ende)
        assert content.count("---") >= 2

    def test_skill_has_trigger_keywords(self) -> None:
        content = self.SKILL_PATH.read_text(encoding="utf-8")
        assert "trigger_keywords:" in content
        assert "Synthese" in content
        assert "Überblick" in content

    def test_skill_has_tools_required(self) -> None:
        content = self.SKILL_PATH.read_text(encoding="utf-8")
        assert "tools_required:" in content
        assert "knowledge_synthesize" in content

    def test_skill_has_workflow(self) -> None:
        content = self.SKILL_PATH.read_text(encoding="utf-8")
        assert "## Ablauf" in content or "## Wann anwenden" in content
