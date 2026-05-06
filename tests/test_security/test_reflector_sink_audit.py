"""Per-sink REFLECTION audit-event tests (Operational-Trust PR-D/4).

Closes the gap PR-A (#494) left open: ``_write_episodic`` /
``_write_semantic`` / ``_write_procedural`` now emit one REFLECTION
audit event per successful write, plus a skip-event when the method
short-circuits with no actual write.

Locks the event-type IDs (the property suite asserts on them):

    +-------------------------+------------------------------+--------------------------------+
    | Method                  | event_type on success        | event_type on skip             |
    +-------------------------+------------------------------+--------------------------------+
    | _write_episodic         | episodic_appended            | episodic_skipped_empty_summary |
    | _write_semantic         | semantic_facts_extracted     | semantic_skipped_empty_facts   |
    | _write_procedural       | procedure_auto_created       | procedure_skipped_no_candidate |
    +-------------------------+------------------------------+--------------------------------+

The procedural-success event MUST carry ``learned_text_sha256``
computed via the canonical-form recipe shared with PR-A's helper
(NFC + ``json.dumps(..., sort_keys=True, ensure_ascii=False)`` + UTF-8)
— this closes the "Geister-Prozeduren"-loophole.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from cognithor.config import CognithorConfig, ensure_directory_structure
from cognithor.core.reflector import Reflector
from cognithor.models import (
    ExtractedFact,
    ProcedureCandidate,
    ReflectionResult,
    SessionSummary,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ollama() -> MagicMock:
    return MagicMock()


def _mock_router() -> MagicMock:
    router = MagicMock()
    router.select_model.return_value = "qwen3:8b"
    return router


def _mock_memory_manager() -> MagicMock:
    """Mock of the memory-manager surface accessed by the three sinks."""
    mgr = MagicMock()
    mgr.index.search_entities.return_value = []
    mgr.index.upsert_entity.return_value = None
    mgr.index.upsert_relation.return_value = None
    mgr.procedural.load_procedure.return_value = None
    mgr.procedural.save_procedure.return_value = None
    mgr.procedural.record_usage.return_value = None
    mgr.episodic.append_entry.return_value = None
    return mgr


def _expected_learned_text_sha256(learned_text: str) -> str:
    """Reference implementation of the canonical-form digest.

    Mirrors the recipe in ``Reflector._write_procedural``: NFC +
    ``json.dumps({"learned_text": ...}, sort_keys=True,
    ensure_ascii=False)`` + UTF-8, then SHA-256.
    """
    canonical = unicodedata.normalize(
        "NFC",
        json.dumps(
            {"learned_text": learned_text},
            sort_keys=True,
            ensure_ascii=False,
        ),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@pytest.fixture()
def config(tmp_path: Path) -> CognithorConfig:
    cfg = CognithorConfig(cognithor_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


def _reflector_with_audit(config: CognithorConfig) -> tuple[Reflector, MagicMock]:
    audit = MagicMock()
    reflector = Reflector(
        config,
        _mock_ollama(),
        _mock_router(),
        audit_logger=audit,
    )
    return reflector, audit


# ---------------------------------------------------------------------------
# 1. Episodic sink
# ---------------------------------------------------------------------------


class TestEpisodicSinkAudit:
    """``_write_episodic`` emits ``episodic_appended`` on success and
    ``episodic_skipped_empty_summary`` on early return."""

    @pytest.mark.asyncio
    async def test_episodic_write_emits_episodic_appended_event(
        self, config: CognithorConfig
    ) -> None:
        reflector, audit = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        summary = SessionSummary(
            goal="Test goal",
            outcome="Test outcome",
            tools_used=["tool_a", "tool_b"],
        )
        result = ReflectionResult(
            session_id="sess-episodic-1",
            success_score=0.8,
            session_summary=summary,
        )

        await reflector._write_episodic(result, memory_manager)

        # Exactly one REFLECTION emit, action=episodic_appended, payload
        # carries session_id + summary_chars + tags.
        assert audit.log_reflection_event.call_count == 1
        kwargs = audit.log_reflection_event.call_args.kwargs
        assert kwargs["action"] == "episodic_appended"
        payload = kwargs["payload"]
        assert payload["session_id"] == "sess-episodic-1"
        assert isinstance(payload["summary_chars"], int)
        assert payload["summary_chars"] > 0
        assert payload["tags"] == ["tool_a", "tool_b"]
        # File was actually written.
        memory_manager.episodic.append_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_episodic_skip_emits_skipped_event(self, config: CognithorConfig) -> None:
        reflector, audit = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        result = ReflectionResult(
            session_id="sess-episodic-skip",
            success_score=0.4,
            session_summary=None,  # triggers the skip branch
        )

        await reflector._write_episodic(result, memory_manager)

        assert audit.log_reflection_event.call_count == 1
        kwargs = audit.log_reflection_event.call_args.kwargs
        assert kwargs["action"] == "episodic_skipped_empty_summary"
        assert kwargs["payload"]["session_id"] == "sess-episodic-skip"
        memory_manager.episodic.append_entry.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Semantic sink
# ---------------------------------------------------------------------------


class TestSemanticSinkAudit:
    """``_write_semantic`` emits ``semantic_facts_extracted`` on success
    and ``semantic_skipped_empty_facts`` on early return."""

    @pytest.mark.asyncio
    async def test_semantic_write_emits_semantic_facts_extracted(
        self, config: CognithorConfig
    ) -> None:
        reflector, audit = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        facts = [
            ExtractedFact(
                entity_name="Alex",
                entity_type="person",
                attribute_key="role",
                attribute_value="owner",
                source_session="sess-sem-1",
            ),
            ExtractedFact(
                entity_name="Cognithor",
                entity_type="product",
                source_session="sess-sem-1",
            ),
        ]

        written = await reflector._write_semantic(facts, memory_manager)

        assert written >= 1
        assert audit.log_reflection_event.call_count == 1
        kwargs = audit.log_reflection_event.call_args.kwargs
        assert kwargs["action"] == "semantic_facts_extracted"
        payload = kwargs["payload"]
        assert payload["session_id"] == "sess-sem-1"
        assert payload["facts_count"] == written

    @pytest.mark.asyncio
    async def test_semantic_skip_emits_skipped_event(self, config: CognithorConfig) -> None:
        reflector, audit = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        written = await reflector._write_semantic([], memory_manager)
        assert written == 0
        assert audit.log_reflection_event.call_count == 1
        kwargs = audit.log_reflection_event.call_args.kwargs
        assert kwargs["action"] == "semantic_skipped_empty_facts"


# ---------------------------------------------------------------------------
# 3. Procedural sink
# ---------------------------------------------------------------------------


class TestProceduralSinkAudit:
    """``_write_procedural`` emits ``procedure_auto_created`` (with
    ``learned_text_sha256``) on success and
    ``procedure_skipped_no_candidate`` on early return."""

    @pytest.mark.asyncio
    async def test_procedural_write_emits_procedure_auto_created_with_hash(
        self, config: CognithorConfig
    ) -> None:
        reflector, audit = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        learned = "Wenn der Tool-Call X fehlschlaegt, retry mit Backoff."
        candidate = ProcedureCandidate(
            name="retry-on-x-fail",
            trigger_keywords=["retry"],
            steps_text="1. Detect failure\n2. Backoff\n3. Retry",
            learned_text=learned,
            is_update=False,
        )
        result = ReflectionResult(
            session_id="sess-proc-1",
            success_score=0.9,
            procedure_candidate=candidate,
        )

        await reflector._write_procedural(result, memory_manager)

        assert audit.log_reflection_event.call_count == 1
        kwargs = audit.log_reflection_event.call_args.kwargs
        assert kwargs["action"] == "procedure_auto_created"
        payload = kwargs["payload"]
        assert payload["session_id"] == "sess-proc-1"
        assert payload["procedure_name"] == "retry-on-x-fail"
        assert payload["is_update"] is False
        assert payload["learned_text_bytes"] == len(learned.encode("utf-8"))
        # Critical: the canonical-form hash matches the reference
        # recipe — closes the Geister-Prozeduren-loophole.
        assert payload["learned_text_sha256"] == _expected_learned_text_sha256(learned)
        # File-write actually happened.
        memory_manager.procedural.save_procedure.assert_called_once()

    @pytest.mark.asyncio
    async def test_procedural_skip_emits_skipped_event(self, config: CognithorConfig) -> None:
        reflector, audit = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        result = ReflectionResult(
            session_id="sess-proc-skip",
            success_score=0.5,
            procedure_candidate=None,  # triggers the skip branch
        )

        await reflector._write_procedural(result, memory_manager)

        assert audit.log_reflection_event.call_count == 1
        kwargs = audit.log_reflection_event.call_args.kwargs
        assert kwargs["action"] == "procedure_skipped_no_candidate"
        assert kwargs["payload"]["session_id"] == "sess-proc-skip"
        memory_manager.procedural.save_procedure.assert_not_called()

    @pytest.mark.asyncio
    async def test_procedural_learned_text_hash_is_nfc_stable(
        self, config: CognithorConfig
    ) -> None:
        """NFC normalisation: composed/decomposed Unicode produces same digest."""
        reflector, audit = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        composed = "Café"  # NFC
        decomposed = unicodedata.normalize("NFD", composed)
        assert composed != decomposed  # sanity: byte-level distinct

        for variant in (composed, decomposed):
            audit.reset_mock()
            candidate = ProcedureCandidate(name="cafe-skill", learned_text=variant)
            result = ReflectionResult(
                session_id="sess-nfc",
                success_score=0.7,
                procedure_candidate=candidate,
            )
            await reflector._write_procedural(result, memory_manager)
            payload = audit.log_reflection_event.call_args.kwargs["payload"]
            # Both variants produce the same canonical-form digest.
            assert payload["learned_text_sha256"] == _expected_learned_text_sha256(composed)


# ---------------------------------------------------------------------------
# 4. Best-effort discipline: audit failure doesn't crash runtime
# ---------------------------------------------------------------------------


class TestAuditFailureDoesNotCrashRuntime:
    """When the audit-emit helper raises, the underlying memory write
    must still complete (best-effort discipline)."""

    @pytest.mark.asyncio
    async def test_episodic_audit_failure_does_not_abort_write(
        self, config: CognithorConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reflector, _ = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        def raising_emit(_event_type: str, _payload: object) -> None:
            raise RuntimeError("audit backend down")

        monkeypatch.setattr(reflector, "_emit_reflection_audit_event", raising_emit)

        summary = SessionSummary(goal="g", outcome="o")
        result = ReflectionResult(
            session_id="sess-x",
            success_score=0.5,
            session_summary=summary,
        )
        # MUST NOT raise.
        await reflector._write_episodic(result, memory_manager)
        # The episodic write MUST have landed before the audit crashed.
        memory_manager.episodic.append_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_audit_failure_does_not_abort_write(
        self, config: CognithorConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reflector, _ = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        def raising_emit(_event_type: str, _payload: object) -> None:
            raise RuntimeError("audit backend down")

        monkeypatch.setattr(reflector, "_emit_reflection_audit_event", raising_emit)

        facts = [
            ExtractedFact(
                entity_name="Alex",
                entity_type="person",
                source_session="sess-y",
            )
        ]
        # MUST NOT raise; the upsert must still have been attempted.
        written = await reflector._write_semantic(facts, memory_manager)
        assert written >= 1
        memory_manager.index.upsert_entity.assert_called()

    @pytest.mark.asyncio
    async def test_procedural_audit_failure_does_not_abort_write(
        self, config: CognithorConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reflector, _ = _reflector_with_audit(config)
        memory_manager = _mock_memory_manager()

        def raising_emit(_event_type: str, _payload: object) -> None:
            raise RuntimeError("audit backend down")

        monkeypatch.setattr(reflector, "_emit_reflection_audit_event", raising_emit)

        candidate = ProcedureCandidate(name="proc-runtime-safe", learned_text="x")
        result = ReflectionResult(
            session_id="sess-z",
            success_score=0.5,
            procedure_candidate=candidate,
        )
        # MUST NOT raise.
        await reflector._write_procedural(result, memory_manager)
        memory_manager.procedural.save_procedure.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Migration-ledger entry confirmation
# ---------------------------------------------------------------------------


class TestMigrationLedgerEntry:
    """The PR-D closer entry MUST be in MIGRATION_LEDGER after AuditLogger
    construction."""

    def test_reflection_audit_completeness_v1_recorded(self, tmp_path: Path) -> None:
        from cognithor.audit import AuditLogger
        from cognithor.security.migration_ledger import MIGRATION_LEDGER

        # Constructing an AuditLogger triggers the idempotent record.
        AuditLogger(log_dir=tmp_path)

        step = MIGRATION_LEDGER.get("reflection-audit-completeness-v1")
        assert step is not None
        assert step.target_version == "v2-reflection-completeness"
        assert step.applied_by == "system"
        # Idempotent: a second construction does not raise + does not
        # duplicate the entry.
        AuditLogger(log_dir=tmp_path)
        target_id = "reflection-audit-completeness-v1"
        all_matching = [s for s in MIGRATION_LEDGER.steps() if s.migration_id == target_id]
        assert len(all_matching) == 1
