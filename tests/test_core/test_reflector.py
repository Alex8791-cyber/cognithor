"""Tests für den Reflector – Lernen aus Erfahrung. [B§6]

Testet:
  - should_reflect(): Wann Reflexion sinnvoll ist
  - reflect(): LLM-basierte Reflexion (gemockt)
  - Parsing: JSON-Extraktion und Result-Building
  - Fallback: Reflexion ohne LLM
  - apply(): Schreiben in Memory-Tiers
  - Formatting: Session-Daten für Reflection-Prompt
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import JarvisConfig
from jarvis.core.model_router import ModelRouter, OllamaClient, OllamaError
from jarvis.core.reflector import Reflector
from jarvis.models import (
    ActionPlan,
    AgentResult,
    AuditEntry,
    Entity,
    ExtractedFact,
    GateStatus,
    Message,
    MessageRole,
    PlannedAction,
    ProcedureCandidate,
    ProcedureMetadata,
    ReflectionResult,
    SessionContext,
    SessionSummary,
    ToolResult,
    WorkingMemory,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    return JarvisConfig(jarvis_home=tmp_path)


@pytest.fixture()
def mock_ollama() -> AsyncMock:
    return AsyncMock(spec=OllamaClient)


@pytest.fixture()
def mock_router(config: JarvisConfig) -> MagicMock:
    router = MagicMock(spec=ModelRouter)
    router.select_model.return_value = "qwen3:32b"
    router.get_model_config.return_value = {
        "temperature": 0.3,
        "top_p": 0.9,
        "context_window": 32768,
    }
    return router


@pytest.fixture()
def reflector(config: JarvisConfig, mock_ollama: AsyncMock, mock_router: MagicMock) -> Reflector:
    return Reflector(config, mock_ollama, mock_router)


@pytest.fixture()
def session() -> SessionContext:
    return SessionContext(session_id="test-session-123", channel="cli", user_id="alex")


@pytest.fixture()
def working_memory() -> WorkingMemory:
    wm = WorkingMemory(session_id="test-session-123")
    wm.add_message(Message(role=MessageRole.USER, content="Erstelle einen Recherche-Bericht"))
    wm.add_message(Message(role=MessageRole.ASSISTANT, content="Ich suche die Kundendaten..."))
    return wm


def _make_agent_result(
    *,
    iterations: int = 2,
    success: bool = True,
    has_tools: bool = True,
    tool_count: int = 1,
) -> AgentResult:
    """Hilfsfunktion um AgentResults zu erzeugen."""
    plans = []
    tool_results = []

    if has_tools:
        steps = [
            PlannedAction(
                tool="memory_search",
                params={"query": "Müller"},
                rationale="Kundendaten laden",
            )
        ]
        plans.append(
            ActionPlan(
                goal="Recherche-Bericht erstellen",
                reasoning="Kunde braucht BU",
                steps=steps,
            )
        )
        for i in range(tool_count):
            tool_results.append(
                ToolResult(
                    tool_name="memory_search",
                    content=f"Ergebnis {i}",
                    is_error=False,
                )
            )
    else:
        plans.append(
            ActionPlan(
                goal="Guten Morgen",
                reasoning="Einfache Begrüßung",
                direct_response="Guten Morgen!",
            )
        )

    return AgentResult(
        response="Hier ist der Recherche-Bericht.",
        plans=plans,
        tool_results=tool_results,
        total_iterations=iterations,
        total_duration_ms=3500,
        model_used="qwen3:32b",
        success=success,
    )


GOOD_REFLECTION_JSON = """{
  "success_score": 0.85,
  "evaluation": "Ziel wurde erreicht. Recherche-Bericht erfolgreich erstellt.",
  "extracted_facts": [
    {
      "entity_name": "Thomas Müller",
      "entity_type": "person",
      "attribute_key": "beruf",
      "attribute_value": "Softwareentwickler",
      "relation_type": "hat_police",
      "relation_target": "PRJ-2024-001"
    }
  ],
  "procedure_candidate": {
    "name": "bu-angebot-erstellen",
    "trigger_keywords": ["BU", "Projektplanung"],
    "prerequisite_text": "Kundenname und Beruf",
    "steps_text": "1. Kundendaten laden\\n2. Risikoklasse bestimmen\\n3. Tarif wählen",
    "learned_text": "Kunden fragen immer nach Beispielrechnungen",
    "failure_patterns": [],
    "tools_required": ["memory_search", "file_write"],
    "is_update": false
  },
  "session_summary": {
    "goal": "Recherche-Bericht für Projekt Alpha erstellen",
    "outcome": "Angebot erstellt und per E-Mail versendet",
    "key_decisions": ["Cloud Platform Pro gewählt"],
    "open_items": ["Nachfass-Termin in 5 Tagen"],
    "tools_used": ["memory_search", "file_write", "email_draft"],
    "duration_ms": 3500
  },
  "failure_analysis": "",
  "improvement_suggestions": ["Beispielrechnung direkt beifügen"]
}"""


# ============================================================================
# TestShouldReflect
# ============================================================================


class TestShouldReflect:
    def test_no_reflection_for_zero_iterations(self, reflector: Reflector) -> None:
        """Keine Reflexion bei 0 Iterationen."""
        result = _make_agent_result(iterations=0)
        assert not reflector.should_reflect(result)

    def test_no_reflection_without_plans(self, reflector: Reflector) -> None:
        """Keine Reflexion ohne Pläne."""
        result = AgentResult(response="Hi", plans=[], total_iterations=2)
        assert not reflector.should_reflect(result)

    def test_no_reflection_for_direct_response(self, reflector: Reflector) -> None:
        """Keine Reflexion für einfache Frage-Antwort."""
        result = _make_agent_result(has_tools=False, iterations=1)
        assert not reflector.should_reflect(result)

    def test_reflection_for_tool_session(self, reflector: Reflector) -> None:
        """Reflexion wenn Tools aufgerufen wurden."""
        result = _make_agent_result(has_tools=True, iterations=2)
        assert reflector.should_reflect(result)

    def test_reflection_with_single_iteration(self, reflector: Reflector) -> None:
        """Reflexion auch bei einer Iteration mit Tools."""
        result = _make_agent_result(has_tools=True, iterations=1)
        assert reflector.should_reflect(result)


# ============================================================================
# TestReflect
# ============================================================================


class TestReflect:
    @pytest.mark.asyncio
    async def test_successful_reflection(
        self,
        reflector: Reflector,
        mock_ollama: AsyncMock,
        session: SessionContext,
        working_memory: WorkingMemory,
    ) -> None:
        """Vollständige Reflexion mit gemocktem LLM."""
        mock_ollama.chat.return_value = {"message": {"content": GOOD_REFLECTION_JSON}}
        agent_result = _make_agent_result()

        result = await reflector.reflect(session, working_memory, agent_result)

        assert result.session_id == "test-session-123"
        assert result.success_score == pytest.approx(0.85)
        assert result.was_successful
        assert result.has_procedure
        assert result.has_facts
        assert len(result.extracted_facts) == 1
        assert result.extracted_facts[0].entity_name == "Thomas Müller"
        assert result.procedure_candidate.name == "bu-angebot-erstellen"
        assert result.session_summary.goal == "Recherche-Bericht für Projekt Alpha erstellen"
        mock_ollama.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflection_uses_correct_model(
        self,
        reflector: Reflector,
        mock_ollama: AsyncMock,
        mock_router: MagicMock,
        session: SessionContext,
        working_memory: WorkingMemory,
    ) -> None:
        """Reflector wählt das richtige Modell."""
        mock_ollama.chat.return_value = {
            "message": {"content": '{"success_score": 0.5, "evaluation": "ok"}'}
        }
        await reflector.reflect(session, working_memory, _make_agent_result())

        mock_router.select_model.assert_called_with("reflection", "medium")

    @pytest.mark.asyncio
    async def test_reflection_on_llm_error(
        self,
        reflector: Reflector,
        mock_ollama: AsyncMock,
        session: SessionContext,
        working_memory: WorkingMemory,
    ) -> None:
        """LLM-Fehler → Fallback-Reflexion."""
        mock_ollama.chat.side_effect = OllamaError("Connection refused")
        agent_result = _make_agent_result()

        result = await reflector.reflect(session, working_memory, agent_result)

        assert result.session_id == "test-session-123"
        assert "Automatische Reflexion" in result.evaluation
        assert result.session_summary is not None
        assert result.success_score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_reflection_on_failed_session(
        self,
        reflector: Reflector,
        mock_ollama: AsyncMock,
        session: SessionContext,
        working_memory: WorkingMemory,
    ) -> None:
        """Fallback bei Fehler: niedrigerer Score für gescheiterte Session."""
        mock_ollama.chat.side_effect = OllamaError("timeout")
        agent_result = _make_agent_result(success=False)

        result = await reflector.reflect(session, working_memory, agent_result)

        assert result.success_score == pytest.approx(0.3)


# ============================================================================
# TestParseReflection
# ============================================================================


class TestParseReflection:
    def test_parse_valid_json(self, reflector: Reflector) -> None:
        """Valides JSON wird korrekt geparst."""
        result = reflector._parse_reflection(GOOD_REFLECTION_JSON, "sess-1")
        assert result.success_score == pytest.approx(0.85)
        assert result.has_procedure
        assert result.has_facts

    def test_parse_json_with_markdown_fences(self, reflector: Reflector) -> None:
        """JSON in Markdown-Code-Block wird korrekt extrahiert."""
        text = f"```json\n{GOOD_REFLECTION_JSON}\n```"
        result = reflector._parse_reflection(text, "sess-2")
        assert result.success_score == pytest.approx(0.85)

    def test_parse_minimal_json(self, reflector: Reflector) -> None:
        """Minimales JSON mit nur Score + Evaluation."""
        text = '{"success_score": 0.9, "evaluation": "Perfekt"}'
        result = reflector._parse_reflection(text, "sess-3")
        assert result.success_score == pytest.approx(0.9)
        assert result.evaluation == "Perfekt"
        assert not result.has_procedure
        assert not result.has_facts

    def test_parse_invalid_json_returns_default(self, reflector: Reflector) -> None:
        """Ungültiges JSON → Default-Result."""
        result = reflector._parse_reflection("Das ist kein JSON", "sess-4")
        assert result.success_score == pytest.approx(0.5)
        assert "nicht geparst" in result.evaluation

    def test_parse_score_clamped_to_range(self, reflector: Reflector) -> None:
        """Score wird auf 0–1 geklemmt."""
        text = '{"success_score": 1.5, "evaluation": "Übertrieben"}'
        result = reflector._parse_reflection(text, "sess-5")
        assert result.success_score == pytest.approx(1.0)

    def test_parse_negative_score(self, reflector: Reflector) -> None:
        """Negativer Score wird auf 0 geklemmt."""
        text = '{"success_score": -0.5, "evaluation": "Falsch"}'
        result = reflector._parse_reflection(text, "sess-6")
        assert result.success_score == pytest.approx(0.0)

    def test_parse_empty_facts_list(self, reflector: Reflector) -> None:
        """Leere Fakten-Liste wird korrekt verarbeitet."""
        text = '{"success_score": 0.7, "extracted_facts": []}'
        result = reflector._parse_reflection(text, "sess-7")
        assert not result.has_facts

    def test_parse_fact_without_name_is_skipped(self, reflector: Reflector) -> None:
        """Fakten ohne entity_name werden übersprungen."""
        text = '{"success_score": 0.7, "extracted_facts": [{"entity_type": "person"}]}'
        result = reflector._parse_reflection(text, "sess-8")
        assert not result.has_facts

    def test_parse_procedure_without_name_is_null(self, reflector: Reflector) -> None:
        """Prozedur ohne Name wird ignoriert."""
        text = '{"success_score": 0.7, "procedure_candidate": {"trigger_keywords": ["BU"]}}'
        result = reflector._parse_reflection(text, "sess-9")
        assert not result.has_procedure

    def test_parse_procedure_null_value(self, reflector: Reflector) -> None:
        """Explizites null für procedure_candidate."""
        text = '{"success_score": 0.7, "procedure_candidate": null}'
        result = reflector._parse_reflection(text, "sess-10")
        assert not result.has_procedure


# ============================================================================
# TestExtractJson
# ============================================================================


class TestExtractJson:
    def test_plain_json(self, reflector: Reflector) -> None:
        """Reines JSON-Objekt."""
        assert reflector._extract_json('{"key": "value"}') == {"key": "value"}

    def test_json_with_surrounding_text(self, reflector: Reflector) -> None:
        """JSON eingebettet in Text."""
        text = 'Hier meine Analyse: {"score": 0.8} Ende.'
        assert reflector._extract_json(text) == {"score": 0.8}

    def test_json_in_markdown_fence(self, reflector: Reflector) -> None:
        """JSON in ```json ... ```."""
        text = '```json\n{"key": "val"}\n```'
        assert reflector._extract_json(text) == {"key": "val"}

    def test_not_json(self, reflector: Reflector) -> None:
        """Kein JSON vorhanden → None."""
        assert reflector._extract_json("Kein JSON hier") is None

    def test_json_array_ignored(self, reflector: Reflector) -> None:
        """JSON-Array wird ignoriert (nur Dict erwartet)."""
        assert reflector._extract_json("[1, 2, 3]") is None

    def test_empty_string(self, reflector: Reflector) -> None:
        """Leerer String → None."""
        assert reflector._extract_json("") is None

    def test_nested_json(self, reflector: Reflector) -> None:
        """Verschachteltes JSON."""
        text = '{"a": {"b": 1}, "c": [1, 2]}'
        result = reflector._extract_json(text)
        assert result == {"a": {"b": 1}, "c": [1, 2]}


# ============================================================================
# TestFallbackReflection
# ============================================================================


class TestFallbackReflection:
    def test_fallback_on_success(self, reflector: Reflector, session: SessionContext) -> None:
        """Fallback bei erfolgreicher Session → Score 0.7."""
        result = reflector._fallback_reflection(session, _make_agent_result(success=True))
        assert result.success_score == pytest.approx(0.7)
        assert result.session_summary is not None
        assert result.session_summary.goal == "Recherche-Bericht erstellen"

    def test_fallback_on_failure(self, reflector: Reflector, session: SessionContext) -> None:
        """Fallback bei gescheiterter Session → Score 0.3."""
        result = reflector._fallback_reflection(session, _make_agent_result(success=False))
        assert result.success_score == pytest.approx(0.3)

    def test_fallback_captures_tools(self, reflector: Reflector, session: SessionContext) -> None:
        """Fallback erfasst genutzte Tools."""
        agent = _make_agent_result(tool_count=3)
        result = reflector._fallback_reflection(session, agent)
        assert "memory_search" in result.session_summary.tools_used

    def test_fallback_captures_duration(
        self, reflector: Reflector, session: SessionContext
    ) -> None:
        """Fallback erfasst Session-Dauer."""
        result = reflector._fallback_reflection(session, _make_agent_result())
        assert result.session_summary.duration_ms == 3500


# ============================================================================
# TestFormatSession
# ============================================================================


class TestFormatSession:
    def test_includes_goals(self, reflector: Reflector, working_memory: WorkingMemory) -> None:
        """Formatierung enthält Ziele."""
        agent = _make_agent_result()
        text = reflector._format_session_for_reflection(working_memory, agent)
        assert "Recherche-Bericht erstellen" in text

    def test_includes_chat_history(
        self, reflector: Reflector, working_memory: WorkingMemory
    ) -> None:
        """Formatierung enthält Chat-History."""
        agent = _make_agent_result()
        text = reflector._format_session_for_reflection(working_memory, agent)
        assert "USER" in text
        assert "ASSISTANT" in text

    def test_includes_tool_results(
        self, reflector: Reflector, working_memory: WorkingMemory
    ) -> None:
        """Formatierung enthält Tool-Ergebnisse."""
        agent = _make_agent_result(tool_count=2)
        text = reflector._format_session_for_reflection(working_memory, agent)
        assert "memory_search" in text
        assert "OK" in text

    def test_includes_meta_info(self, reflector: Reflector, working_memory: WorkingMemory) -> None:
        """Formatierung enthält Meta-Informationen."""
        agent = _make_agent_result()
        text = reflector._format_session_for_reflection(working_memory, agent)
        assert "META" in text
        assert "3500ms" in text

    def test_truncation_on_long_input(self, reflector: Reflector) -> None:
        """Sehr lange Eingaben werden gekürzt."""
        wm = WorkingMemory(session_id="long")
        for _i in range(100):
            wm.add_message(Message(role=MessageRole.USER, content="x" * 500))
        agent = _make_agent_result()
        text = reflector._format_session_for_reflection(wm, agent)
        # Should not exceed limit + meta
        assert len(text) < 15_000

    def test_includes_blocked_actions(
        self, reflector: Reflector, working_memory: WorkingMemory
    ) -> None:
        """Formatierung enthält blockierte Aktionen."""
        audit = AuditEntry(
            session_id="test",
            action_tool="shell_exec",
            action_params_hash="abc123",
            decision_status=GateStatus.BLOCK,
            decision_reason="Destruktiver Befehl",
        )
        agent = _make_agent_result()
        agent_with_blocks = AgentResult(
            response=agent.response,
            plans=agent.plans,
            tool_results=agent.tool_results,
            audit_entries=[audit],
            total_iterations=agent.total_iterations,
            total_duration_ms=agent.total_duration_ms,
            success=agent.success,
        )
        text = reflector._format_session_for_reflection(working_memory, agent_with_blocks)
        assert "BLOCKIERT" in text


# ============================================================================
# TestApply
# ============================================================================


class TestApply:
    @pytest.fixture()
    def mock_manager(self, tmp_path) -> MagicMock:
        """Mock MemoryManager mit echten Subkomponenten."""
        manager = MagicMock()

        # Episodic Mock
        manager.episodic = MagicMock()
        manager.episodic.add_entry = MagicMock()

        # Index Mock (für Semantic)
        manager.index = MagicMock()
        manager.index.find_entity.return_value = None
        manager.index.add_entity.return_value = "entity-123"
        manager.index.add_relation.return_value = "rel-123"

        # Procedural Mock
        manager.procedural = MagicMock()
        manager.procedural.save_procedure = MagicMock()
        manager.procedural.record_usage = MagicMock()
        manager.procedural.load_procedure.return_value = None

        return manager

    @pytest.mark.asyncio
    async def test_apply_full_result(self, reflector: Reflector, mock_manager: MagicMock) -> None:
        """Vollständige Reflexion wird in alle Tiers geschrieben."""
        result = ReflectionResult(
            session_id="sess-1",
            success_score=0.85,
            evaluation="Gut",
            extracted_facts=[
                ExtractedFact(
                    entity_name="Müller",
                    entity_type="person",
                    attribute_key="beruf",
                    attribute_value="Entwickler",
                    source_session="sess-1",
                )
            ],
            procedure_candidate=ProcedureCandidate(
                name="bu-angebot",
                trigger_keywords=["BU"],
                steps_text="1. Suchen\n2. Erstellen",
                tools_required=["memory_search"],
            ),
            session_summary=SessionSummary(
                goal="Recherche-Bericht",
                outcome="Erstellt",
            ),
        )

        counts = await reflector.apply(result, mock_manager)

        assert counts["episodic"] == 1
        assert counts["semantic"] >= 1
        assert counts["procedural"] == 1
        mock_manager.episodic.add_entry.assert_called_once()
        mock_manager.procedural.save_procedure.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_empty_result(self, reflector: Reflector, mock_manager: MagicMock) -> None:
        """Leere Reflexion schreibt nichts."""
        result = ReflectionResult(session_id="sess-2", success_score=0.5)

        counts = await reflector.apply(result, mock_manager)

        assert counts == {"episodic": 0, "semantic": 0, "procedural": 0}
        mock_manager.episodic.add_entry.assert_not_called()
        mock_manager.procedural.save_procedure.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_only_episodic(self, reflector: Reflector, mock_manager: MagicMock) -> None:
        """Nur Session-Summary → nur Episodic."""
        result = ReflectionResult(
            session_id="sess-3",
            session_summary=SessionSummary(goal="Test", outcome="OK"),
        )

        counts = await reflector.apply(result, mock_manager)

        assert counts["episodic"] == 1
        assert counts["semantic"] == 0
        assert counts["procedural"] == 0

    @pytest.mark.asyncio
    async def test_apply_semantic_with_relation(
        self, reflector: Reflector, mock_manager: MagicMock
    ) -> None:
        """Fakt mit Relation erzeugt Entity + Relation."""
        result = ReflectionResult(
            session_id="sess-4",
            extracted_facts=[
                ExtractedFact(
                    entity_name="Müller",
                    entity_type="person",
                    relation_type="hat_police",
                    relation_target="PRJ-2024-001",
                    source_session="sess-4",
                )
            ],
        )

        counts = await reflector.apply(result, mock_manager)

        assert counts["semantic"] >= 2  # Entity + Relation
        mock_manager.index.add_entity.assert_called()
        mock_manager.index.add_relation.assert_called()

    @pytest.mark.asyncio
    async def test_apply_semantic_updates_existing_entity(
        self, reflector: Reflector, mock_manager: MagicMock
    ) -> None:
        """Existierende Entität wird aktualisiert statt neu angelegt."""
        existing = Entity(
            id="ent-existing",
            type="person",
            name="Müller",
            attributes={"alter": "35"},
            source_file="old.md",
        )
        mock_manager.index.find_entity.return_value = existing

        result = ReflectionResult(
            session_id="sess-5",
            extracted_facts=[
                ExtractedFact(
                    entity_name="Müller",
                    attribute_key="beruf",
                    attribute_value="Entwickler",
                )
            ],
        )

        await reflector.apply(result, mock_manager)

        mock_manager.index.update_entity.assert_called_once()
        call_args = mock_manager.index.update_entity.call_args
        assert call_args[0][0] == "ent-existing"
        attrs = call_args[1]["attributes"]
        assert attrs["beruf"] == "Entwickler"
        assert attrs["alter"] == "35"  # Alte Attribute bleiben erhalten

    @pytest.mark.asyncio
    async def test_apply_procedure_update(
        self, reflector: Reflector, mock_manager: MagicMock
    ) -> None:
        """Update einer existierenden Prozedur mergt Inhalte."""
        old_meta = ProcedureMetadata(
            name="bu-angebot",
            trigger_keywords=["BU"],
            tools_required=["memory_search"],
            success_count=5,
            total_uses=7,
            source_file="proc.md",
        )
        mock_manager.procedural.load_procedure.return_value = (
            old_meta,
            "# bu-angebot\n\n## Ablauf\n1. Alt",
        )

        result = ReflectionResult(
            session_id="sess-6",
            success_score=0.9,
            procedure_candidate=ProcedureCandidate(
                name="bu-angebot",
                trigger_keywords=["Projektplanung"],
                steps_text="1. Neu",
                tools_required=["file_write"],
                is_update=True,
            ),
        )

        await reflector.apply(result, mock_manager)

        save_call = mock_manager.procedural.save_procedure.call_args
        # Keywords wurden zusammengeführt
        saved_meta = save_call[1]["metadata"]
        assert "BU" in saved_meta.trigger_keywords
        assert "Projektplanung" in saved_meta.trigger_keywords
        # Tools zusammengeführt
        assert "memory_search" in saved_meta.tools_required
        assert "file_write" in saved_meta.tools_required
        # Body enthält Alt + Neu
        saved_body = save_call[1]["body"]
        assert "Alt" in saved_body
        assert "Neu" in saved_body


# ============================================================================
# TestReflectionResult Properties
# ============================================================================


class TestReflectionResultProperties:
    def test_was_successful_above_threshold(self) -> None:
        """Score >= 0.6 gilt als erfolgreich."""
        r = ReflectionResult(session_id="s", success_score=0.6)
        assert r.was_successful

    def test_was_unsuccessful_below_threshold(self) -> None:
        """Score < 0.6 gilt als nicht erfolgreich."""
        r = ReflectionResult(session_id="s", success_score=0.59)
        assert not r.was_successful

    def test_has_procedure_true(self) -> None:
        r = ReflectionResult(
            session_id="s",
            procedure_candidate=ProcedureCandidate(name="test"),
        )
        assert r.has_procedure

    def test_has_procedure_false(self) -> None:
        r = ReflectionResult(session_id="s")
        assert not r.has_procedure

    def test_has_facts_true(self) -> None:
        r = ReflectionResult(
            session_id="s",
            extracted_facts=[ExtractedFact(entity_name="Test")],
        )
        assert r.has_facts

    def test_has_facts_false(self) -> None:
        r = ReflectionResult(session_id="s")
        assert not r.has_facts


# ============================================================================
# TestExtractKeywords
# ============================================================================


class TestExtractKeywords:
    def test_german_sentence(self) -> None:
        """Deutsche Stopwörter werden gefiltert."""
        keywords = Reflector.extract_keywords("Bitte erstelle mir ein Recherche-Bericht für Herrn Müller")
        assert "bitte" not in keywords
        assert "mir" not in keywords
        assert "ein" not in keywords
        # Inhaltswörter bleiben
        assert any("recherche" in k or "bericht" in k for k in keywords)
        assert "müller" in keywords

    def test_english_sentence(self) -> None:
        """Englische Stopwörter werden gefiltert."""
        keywords = Reflector.extract_keywords(
            "Please create a new insurance offer for the customer"
        )
        assert "please" not in keywords
        assert "the" not in keywords
        assert "insurance" in keywords
        assert "customer" in keywords

    def test_short_words_filtered(self) -> None:
        """Wörter unter 3 Zeichen werden entfernt."""
        keywords = Reflector.extract_keywords("ab cd BU ja nein")
        assert "ab" not in keywords
        assert "cd" not in keywords
        assert "nein" in keywords

    def test_max_8_keywords(self) -> None:
        """Maximal 8 Keywords werden zurückgegeben."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        keywords = Reflector.extract_keywords(text)
        assert len(keywords) <= 8

    def test_longest_first(self) -> None:
        """Längere Keywords werden bevorzugt."""
        keywords = Reflector.extract_keywords("Projektplanungsversicherung BU Angebot")
        assert keywords[0] == "projektplanungsversicherung"

    def test_empty_input(self) -> None:
        """Leerer Input → leere Liste."""
        assert Reflector.extract_keywords("") == []

    def test_only_stopwords(self) -> None:
        """Nur Stopwörter → leere Liste."""
        assert Reflector.extract_keywords("ich bin ein und oder") == []

    def test_punctuation_removed(self) -> None:
        """Satzzeichen werden entfernt."""
        keywords = Reflector.extract_keywords("Angebot? Nein! Wirklich.")
        assert "angebot" in keywords
        assert "wirklich" in keywords


# ============================================================================
# TestMatchProcedures
# ============================================================================


class TestMatchProcedures:
    @pytest.fixture()
    def mock_procedural(self) -> MagicMock:
        proc = MagicMock()
        proc.find_by_keywords.return_value = [
            (
                ProcedureMetadata(
                    name="bu-angebot",
                    trigger_keywords=["Recherche", "Bericht"],
                    total_uses=5,
                    success_count=4,
                    failure_count=1,
                ),
                "# bu-angebot\n\n## Ablauf\n1. Suchen\n2. Erstellen",
                0.8,
            )
        ]
        return proc

    def test_match_returns_body(self, reflector: Reflector, mock_procedural: MagicMock) -> None:
        """Matching findet passende Prozedur und gibt Body zurück."""
        results = reflector.match_procedures("Erstelle einen Recherche-Bericht", mock_procedural)
        assert len(results) == 1
        assert "bu-angebot" in results[0]
        mock_procedural.find_by_keywords.assert_called_once()

    def test_match_filters_low_score(
        self, reflector: Reflector, mock_procedural: MagicMock
    ) -> None:
        """Niedrige Scores werden gefiltert."""
        mock_procedural.find_by_keywords.return_value = [
            (
                ProcedureMetadata(name="low-score", total_uses=0),
                "body",
                0.1,  # Unter min_score
            )
        ]
        results = reflector.match_procedures("test", mock_procedural, min_score=0.3)
        assert len(results) == 0

    def test_match_filters_unreliable(
        self, reflector: Reflector, mock_procedural: MagicMock
    ) -> None:
        """Prozeduren mit schlechter Erfolgsquote (< 50%) werden übersprungen."""
        mock_procedural.find_by_keywords.return_value = [
            (
                ProcedureMetadata(
                    name="unreliable",
                    total_uses=5,
                    success_count=2,  # 40% success < 50% threshold
                    failure_count=3,
                ),
                "body",
                0.8,
            )
        ]
        results = reflector.match_procedures("test", mock_procedural)
        assert len(results) == 0

    def test_match_allows_new_procedures(
        self, reflector: Reflector, mock_procedural: MagicMock
    ) -> None:
        """Neue Prozeduren (< 3 Nutzungen) werden erlaubt auch wenn unreliable."""
        mock_procedural.find_by_keywords.return_value = [
            (
                ProcedureMetadata(
                    name="new-proc",
                    total_uses=2,
                    success_count=0,
                    failure_count=2,
                ),
                "body",
                0.8,
            )
        ]
        results = reflector.match_procedures("test", mock_procedural)
        assert len(results) == 1

    def test_match_empty_message(self, reflector: Reflector, mock_procedural: MagicMock) -> None:
        """Leere Nachricht → keine Ergebnisse."""
        results = reflector.match_procedures("", mock_procedural)
        assert len(results) == 0
        mock_procedural.find_by_keywords.assert_not_called()

    def test_match_max_results(self, reflector: Reflector, mock_procedural: MagicMock) -> None:
        """max_results begrenzt die Ausgabe."""
        mock_procedural.find_by_keywords.return_value = [
            (ProcedureMetadata(name=f"proc-{i}"), f"body-{i}", 0.9) for i in range(5)
        ]
        results = reflector.match_procedures("test query", mock_procedural, max_results=2)
        assert len(results) <= 2
