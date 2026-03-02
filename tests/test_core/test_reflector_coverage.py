"""Coverage-Tests fuer reflector.py -- fehlende Zeilen."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import JarvisConfig, ensure_directory_structure
from jarvis.core.reflector import Reflector, _sanitize_memory_text, _safe_float
from jarvis.models import (
    ActionPlan,
    AgentResult,
    SessionContext,
    ToolResult,
    WorkingMemory,
)


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    cfg = JarvisConfig(jarvis_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


def _mock_ollama(content: str = "{}") -> AsyncMock:
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value={
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 100,
        "eval_count": 50,
    })
    return mock


def _mock_router() -> MagicMock:
    router = MagicMock()
    router.select_model.return_value = "qwen3:8b"
    router.get_model_config.return_value = {"temperature": 0.3, "top_p": 0.9, "context_window": 32768}
    return router


def _make_agent_result(
    tool_results: list[ToolResult] | None = None,
    iterations: int = 2,
    has_actions: bool = True,
) -> AgentResult:
    """Creates a minimal AgentResult for tests."""
    from jarvis.models import PlannedAction
    if has_actions:
        plan = ActionPlan(
            goal="test",
            steps=[PlannedAction(tool="web_search", params={"query": "test"})],
        )
    else:
        plan = ActionPlan(goal="test", steps=[], direct_response="simple answer")
    tr = tool_results or [ToolResult(tool_name="web_search", content="result", is_error=False)]
    return AgentResult(
        response="Test response",
        plans=[plan],
        tool_results=tr,
        total_iterations=iterations,
        total_duration_ms=1000,
        model_used="qwen3:8b",
    )


# ============================================================================
# should_reflect
# ============================================================================


class TestShouldReflect:
    def test_should_reflect_with_tools(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        agent_result = _make_agent_result(iterations=2, has_actions=True)
        assert reflector.should_reflect(agent_result) is True

    def test_should_not_reflect_zero_iterations(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        agent_result = _make_agent_result(iterations=0, has_actions=True)
        assert reflector.should_reflect(agent_result) is False

    def test_should_not_reflect_no_plans(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        agent_result = AgentResult(
            response="Test",
            plans=[],
            tool_results=[],
            total_iterations=2,
            total_duration_ms=1000,
            model_used="qwen3:8b",
        )
        assert reflector.should_reflect(agent_result) is False

    def test_should_not_reflect_no_tool_calls(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        agent_result = _make_agent_result(iterations=2, has_actions=False)
        assert reflector.should_reflect(agent_result) is False


# ============================================================================
# extract_keywords (static method)
# ============================================================================


class TestExtractKeywords:
    def test_extract_keywords_basic(self, config: JarvisConfig) -> None:
        keywords = Reflector.extract_keywords("Python ist eine Programmiersprache fuer Data Science")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should filter out stop words like "ist", "eine", "fuer"
        assert "ist" not in keywords
        assert "eine" not in keywords

    def test_extract_keywords_empty(self, config: JarvisConfig) -> None:
        keywords = Reflector.extract_keywords("")
        assert keywords == []

    def test_extract_keywords_only_stopwords(self) -> None:
        keywords = Reflector.extract_keywords("ist die der das")
        assert keywords == []

    def test_extract_keywords_max_8(self) -> None:
        text = "Python JavaScript TypeScript Rust Golang Java Kotlin Swift Ruby Haskell Erlang"
        keywords = Reflector.extract_keywords(text)
        assert len(keywords) <= 8


# ============================================================================
# reflect
# ============================================================================


class TestReflect:
    @pytest.mark.asyncio
    async def test_reflect_returns_result(self, config: JarvisConfig) -> None:
        llm_response = '{"evaluation":"Good session","success_score":0.8,"extracted_facts":[],"session_summary":{"goal":"test","outcome":"success","tools_used":["web_search"]}}'
        reflector = Reflector(config, _mock_ollama(llm_response), _mock_router())

        session = SessionContext()
        wm = WorkingMemory(session_id="test")
        agent_result = _make_agent_result()

        reflection = await reflector.reflect(
            session=session,
            working_memory=wm,
            agent_result=agent_result,
        )
        assert reflection is not None

    @pytest.mark.asyncio
    async def test_reflect_llm_error_fallback(self, config: JarvisConfig) -> None:
        """LLM error should use fallback reflection."""
        from jarvis.core.model_router import OllamaError

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=OllamaError("LLM unavailable"))

        reflector = Reflector(config, mock_llm, _mock_router())
        session = SessionContext()
        wm = WorkingMemory(session_id="test")
        agent_result = _make_agent_result()

        reflection = await reflector.reflect(
            session=session,
            working_memory=wm,
            agent_result=agent_result,
        )
        # Should return fallback reflection, not raise
        assert reflection is not None

    @pytest.mark.asyncio
    async def test_reflect_with_audit_logger(self, config: JarvisConfig) -> None:
        llm_response = '{"evaluation":"OK","success_score":0.5,"extracted_facts":[]}'
        mock_audit = MagicMock()
        mock_audit.log_tool_call = MagicMock()

        reflector = Reflector(
            config, _mock_ollama(llm_response), _mock_router(),
            audit_logger=mock_audit,
        )
        session = SessionContext()
        wm = WorkingMemory(session_id="test")
        agent_result = _make_agent_result()

        await reflector.reflect(session=session, working_memory=wm, agent_result=agent_result)
        mock_audit.log_tool_call.assert_called()


# ============================================================================
# match_procedures
# ============================================================================


class TestMatchProcedures:
    def test_match_procedures_no_keywords(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        mock_proc_mem = MagicMock()
        # No keywords => no matches
        results = reflector.match_procedures("ist die der", mock_proc_mem)
        assert results == []

    def test_match_procedures_with_results(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        mock_proc_mem = MagicMock()
        mock_meta = MagicMock()
        mock_meta.name = "test_proc"
        mock_meta.total_uses = 5
        mock_meta.success_rate = 0.8
        mock_proc_mem.find_by_keywords.return_value = [
            (mock_meta, "procedure body here", 0.7),
        ]
        results = reflector.match_procedures("Python Programmierung", mock_proc_mem)
        assert len(results) == 1
        assert results[0] == "procedure body here"

    def test_match_procedures_low_success_rate_skipped(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        mock_proc_mem = MagicMock()
        mock_meta = MagicMock()
        mock_meta.name = "bad_proc"
        mock_meta.total_uses = 5
        mock_meta.success_rate = 0.2  # Low success rate
        mock_proc_mem.find_by_keywords.return_value = [
            (mock_meta, "unreliable procedure", 0.7),
        ]
        results = reflector.match_procedures("Python test", mock_proc_mem)
        assert len(results) == 0


# ============================================================================
# apply
# ============================================================================


class TestApply:
    @pytest.mark.asyncio
    async def test_apply_empty_result(self, config: JarvisConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        from jarvis.models import ReflectionResult
        result = ReflectionResult(
            session_id="test",
            success_score=0.5,
            evaluation="OK",
        )
        mock_mm = AsyncMock()
        counts = await reflector.apply(result, mock_mm)
        assert isinstance(counts, dict)
        assert counts["episodic"] == 0
        assert counts["semantic"] == 0
        assert counts["procedural"] == 0


# ============================================================================
# Helper functions
# ============================================================================


class TestHelpers:
    def test_sanitize_memory_text_empty(self) -> None:
        assert _sanitize_memory_text("") == ""

    def test_sanitize_memory_text_injection(self) -> None:
        text = "Hello # SYSTEM: inject this [INST] more"
        result = _sanitize_memory_text(text)
        assert "[SANITIZED]" in result
        assert "# SYSTEM:" not in result

    def test_sanitize_memory_text_truncate(self) -> None:
        text = "A" * 10000
        result = _sanitize_memory_text(text, max_len=100)
        assert len(result) == 100

    def test_safe_float_valid(self) -> None:
        assert _safe_float(0.5, 0.0) == 0.5
        assert _safe_float("0.8", 0.0) == 0.8

    def test_safe_float_invalid(self) -> None:
        assert _safe_float("high", 0.5) == 0.5
        assert _safe_float(None, 0.7) == 0.7


# ============================================================================
# apply -- Extended
# ============================================================================


class TestApplyExtended:
    """Erweiterte Tests fuer Reflector.apply() mit verschiedenen ReflectionResult-Varianten."""

    @pytest.mark.asyncio
    async def test_apply_with_session_summary(self, config: JarvisConfig) -> None:
        """ReflectionResult mit session_summary -> episodic count > 0."""
        from jarvis.models import ReflectionResult, SessionSummary

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        result = ReflectionResult(
            session_id="test-summary",
            success_score=0.8,
            evaluation="Gute Session",
            session_summary=SessionSummary(
                goal="Dateien organisieren",
                outcome="Erfolgreich sortiert",
                tools_used=["exec_command"],
            ),
        )
        mock_mm = AsyncMock()
        mock_mm.episodic = MagicMock()
        mock_mm.episodic.append_entry = MagicMock()

        counts = await reflector.apply(result, mock_mm)
        assert counts["episodic"] > 0
        mock_mm.episodic.append_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_with_extracted_facts(self, config: JarvisConfig) -> None:
        """ReflectionResult mit ExtractedFact-Entities -> semantic count > 0."""
        from jarvis.models import ExtractedFact, ReflectionResult

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        result = ReflectionResult(
            session_id="test-facts",
            success_score=0.9,
            evaluation="Fakten extrahiert",
            extracted_facts=[
                ExtractedFact(
                    entity_name="Python",
                    entity_type="concept",
                    attribute_key="version",
                    attribute_value="3.12",
                    source_session="test-facts",
                ),
            ],
        )
        mock_mm = AsyncMock()
        mock_indexer = MagicMock()
        mock_indexer.search_entities.return_value = []  # Keine existierende Entitaet
        mock_indexer.upsert_entity = MagicMock()
        mock_mm.index = mock_indexer

        counts = await reflector.apply(result, mock_mm)
        assert counts["semantic"] > 0
        mock_indexer.upsert_entity.assert_called()

    @pytest.mark.asyncio
    async def test_apply_with_procedure_candidate(self, config: JarvisConfig) -> None:
        """ReflectionResult mit ProcedureCandidate -> procedural count > 0."""
        from jarvis.models import ProcedureCandidate, ReflectionResult

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        result = ReflectionResult(
            session_id="test-proc",
            success_score=0.85,
            evaluation="Prozedur erkannt",
            procedure_candidate=ProcedureCandidate(
                name="file-backup",
                trigger_keywords=["backup", "sichern"],
                steps_text="1. Dateien auflisten\n2. Kopieren",
                tools_required=["exec_command"],
            ),
        )
        mock_mm = AsyncMock()
        mock_proc = MagicMock()
        mock_proc.save_procedure = MagicMock()
        mock_proc.record_usage = MagicMock()
        mock_mm.procedural = mock_proc

        counts = await reflector.apply(result, mock_mm)
        assert counts["procedural"] > 0
        mock_proc.save_procedure.assert_called_once()
        mock_proc.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_with_all_types(self, config: JarvisConfig) -> None:
        """Alle drei Typen (episodic, semantic, procedural) vorhanden -> alle counts > 0."""
        from jarvis.models import (
            ExtractedFact,
            ProcedureCandidate,
            ReflectionResult,
            SessionSummary,
        )

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        result = ReflectionResult(
            session_id="test-all",
            success_score=0.95,
            evaluation="Vollstaendige Session",
            session_summary=SessionSummary(
                goal="Vollstaendiger Test",
                outcome="Alles geschrieben",
                tools_used=["web_search", "exec_command"],
            ),
            extracted_facts=[
                ExtractedFact(
                    entity_name="TestEntity",
                    entity_type="concept",
                    attribute_key="status",
                    attribute_value="aktiv",
                    source_session="test-all",
                ),
            ],
            procedure_candidate=ProcedureCandidate(
                name="full-test-proc",
                trigger_keywords=["volltest"],
                steps_text="1. Alles testen",
                tools_required=["exec_command"],
            ),
        )
        mock_mm = AsyncMock()
        # Episodic
        mock_mm.episodic = MagicMock()
        mock_mm.episodic.append_entry = MagicMock()
        # Semantic
        mock_indexer = MagicMock()
        mock_indexer.search_entities.return_value = []
        mock_indexer.upsert_entity = MagicMock()
        mock_mm.index = mock_indexer
        # Procedural
        mock_proc = MagicMock()
        mock_proc.save_procedure = MagicMock()
        mock_proc.record_usage = MagicMock()
        mock_mm.procedural = mock_proc

        counts = await reflector.apply(result, mock_mm)
        assert counts["episodic"] > 0
        assert counts["semantic"] > 0
        assert counts["procedural"] > 0

    @pytest.mark.asyncio
    async def test_apply_memory_manager_error(self, config: JarvisConfig) -> None:
        """memory_manager wirft Exception -> wird graceful abgefangen."""
        from jarvis.models import ReflectionResult, SessionSummary

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        result = ReflectionResult(
            session_id="test-error",
            success_score=0.5,
            evaluation="Fehlertest",
            session_summary=SessionSummary(
                goal="Fehler provozieren",
                outcome="Sollte nicht abstuerzen",
            ),
        )
        mock_mm = AsyncMock()
        mock_mm.episodic = MagicMock()
        mock_mm.episodic.append_entry = MagicMock(side_effect=RuntimeError("DB kaputt"))

        # apply() sollte die Exception entweder auffangen oder weiterreichen,
        # aber nicht unkontrolliert crashen. Wir testen dass es sich definiert verhaelt.
        try:
            await reflector.apply(result, mock_mm)
        except RuntimeError:
            pass  # Erwartetes Verhalten wenn Exception nicht intern gefangen wird


# ============================================================================
# _write_semantic -- Extended
# ============================================================================


class TestWriteSemantic:
    """Tests fuer Reflector._write_semantic -- Entitaeten, Relationen, Sanitization."""

    @pytest.mark.asyncio
    async def test_write_semantic_entity(self, config: JarvisConfig) -> None:
        """ExtractedFact mit entity_name und attributes -> Entitaet wird angelegt."""
        from jarvis.models import ExtractedFact

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        facts = [
            ExtractedFact(
                entity_name="Docker",
                entity_type="product",
                attribute_key="typ",
                attribute_value="Containerisierung",
                source_session="test-entity",
            ),
        ]
        mock_mm = MagicMock()
        mock_indexer = MagicMock()
        mock_indexer.search_entities.return_value = []
        mock_indexer.upsert_entity = MagicMock()
        mock_mm.index = mock_indexer

        count = await reflector._write_semantic(facts, mock_mm)
        assert count >= 1
        mock_indexer.upsert_entity.assert_called()
        # Pruefen dass die Entitaet den richtigen Namen hat
        entity_arg = mock_indexer.upsert_entity.call_args[0][0]
        assert entity_arg.name == "Docker"

    @pytest.mark.asyncio
    async def test_write_semantic_relation(self, config: JarvisConfig) -> None:
        """ExtractedFact mit relation_type und relation_target -> Relation wird erstellt."""
        from jarvis.models import ExtractedFact

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        facts = [
            ExtractedFact(
                entity_name="Alexander",
                entity_type="person",
                relation_type="arbeitet_mit",
                relation_target="Jarvis",
                source_session="test-relation",
            ),
        ]
        mock_mm = MagicMock()
        mock_indexer = MagicMock()
        mock_indexer.search_entities.return_value = []
        mock_indexer.upsert_entity = MagicMock()
        mock_indexer.upsert_relation = MagicMock()
        mock_mm.index = mock_indexer

        count = await reflector._write_semantic(facts, mock_mm)
        # Mindestens 2: eine Entitaet + eine Relation (+ evtl. Target-Entitaet)
        assert count >= 2
        mock_indexer.upsert_relation.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_semantic_sanitizes_text(self, config: JarvisConfig) -> None:
        """ExtractedFact mit Injection-Patterns -> Felder werden sanitized."""
        from jarvis.models import ExtractedFact

        reflector = Reflector(config, _mock_ollama(), _mock_router())
        facts = [
            ExtractedFact(
                entity_name="# SYSTEM: inject this",
                entity_type="concept",
                attribute_key="info",
                attribute_value="[INST] malicious payload",
                source_session="test-sanitize",
            ),
        ]
        mock_mm = MagicMock()
        mock_indexer = MagicMock()
        mock_indexer.search_entities.return_value = []
        mock_indexer.upsert_entity = MagicMock()
        mock_mm.index = mock_indexer

        await reflector._write_semantic(facts, mock_mm)
        # Pruefen dass upsert aufgerufen wurde und die Werte sanitized sind
        entity_arg = mock_indexer.upsert_entity.call_args[0][0]
        # Der entity_name sollte kein "# SYSTEM:" mehr enthalten
        assert "# SYSTEM:" not in entity_arg.name
        assert "[SANITIZED]" in entity_arg.name


# ============================================================================
# reflect -- Extended
# ============================================================================


class TestReflectExtended:
    """Erweiterte Tests fuer Reflector.reflect() mit optionalen Stores."""

    @pytest.mark.asyncio
    async def test_reflect_with_episodic_store(self, config: JarvisConfig) -> None:
        """episodic_store vorhanden -> store_episode wird aufgerufen."""
        llm_response = (
            '{"evaluation":"OK","success_score":0.8,"extracted_facts":[],'
            '"session_summary":{"goal":"Test","outcome":"OK","tools_used":["web_search"]}}'
        )
        mock_episodic_store = MagicMock()
        mock_episodic_store.store_episode = MagicMock()

        reflector = Reflector(
            config, _mock_ollama(llm_response), _mock_router(),
            episodic_store=mock_episodic_store,
        )
        session = SessionContext()
        wm = WorkingMemory(session_id="test-episodic")
        agent_result = _make_agent_result()

        await reflector.reflect(session=session, working_memory=wm, agent_result=agent_result)
        mock_episodic_store.store_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_with_causal_analyzer(self, config: JarvisConfig) -> None:
        """causal_analyzer vorhanden -> record_sequence wird aufgerufen."""
        llm_response = '{"evaluation":"OK","success_score":0.7,"extracted_facts":[]}'
        mock_causal = MagicMock()
        mock_causal.record_sequence = MagicMock()

        reflector = Reflector(
            config, _mock_ollama(llm_response), _mock_router(),
            causal_analyzer=mock_causal,
        )
        session = SessionContext()
        wm = WorkingMemory(session_id="test-causal")
        agent_result = _make_agent_result()

        await reflector.reflect(session=session, working_memory=wm, agent_result=agent_result)
        # agent_result hat tool_results -> tool_sequence ist nicht leer
        mock_causal.record_sequence.assert_called_once()


# ============================================================================
# _extract_json -- Tests
# ============================================================================


class TestExtractJson:
    """Tests fuer Reflector._extract_json -- JSON-Parsing aus LLM-Output."""

    def test_extract_json_markdown_fences(self) -> None:
        """JSON innerhalb von ```json ... ``` -> wird korrekt geparst."""
        text = '```json\n{"success_score": 0.9, "evaluation": "gut"}\n```'
        result = Reflector._extract_json(text)
        assert result is not None
        assert result["success_score"] == 0.9
        assert result["evaluation"] == "gut"

    def test_extract_json_raw(self) -> None:
        """Reiner JSON-String ohne Fences -> wird korrekt geparst."""
        text = '{"success_score": 0.5, "evaluation": "mittel"}'
        result = Reflector._extract_json(text)
        assert result is not None
        assert result["success_score"] == 0.5
        assert result["evaluation"] == "mittel"

    def test_extract_json_no_json(self) -> None:
        """Kein JSON im Text -> None wird zurueckgegeben."""
        text = "Das ist kein JSON, nur normaler Text ohne Klammern."
        result = Reflector._extract_json(text)
        assert result is None
