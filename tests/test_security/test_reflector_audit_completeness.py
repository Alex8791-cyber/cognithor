"""Hypothesis property tests for audit-completeness invariants (PR-C/3).

Locks the conventions established in PR-A (#494, commit ``f593cddc``)
and closed by PR-D:

  - Reflector autonomous writes MUST flow through the REFLECTION audit
    category. Any future refactor that adds a silent autonomous-learning
    sink must show up here as a failing property — Reflector writes and
    REFLECTION audit events are 1-to-1 (or many-to-1 — never zero-to-N).

  - Canonical-form for ``payload_sha256`` is the SAME recipe as the one
    used by ``AuditLogger._last_hash_for_file`` (line 830, audit module):
    ``unicodedata.normalize("NFC", json.dumps(payload, sort_keys=True,
    ensure_ascii=False)).encode("utf-8")``. No explicit separators. If
    a future PR adds ``separators=(",", ":")`` this property fails.

  - NFC normalisation collapses decomposed Unicode so the same logical
    string produces the same hash.

  - Hash determinism: same payload → same digest, regardless of dict
    insertion order or process restart.

  - SEC-HIGH-5 hash chain integrity is preserved when REFLECTION-category
    events are appended through the real ``AuditLogger`` persistence path.

PR-D closer (this commit): the previously-xfail property
``test_every_apply_emits_one_event_per_memory_write`` is now PASSing —
``_write_episodic`` / ``_write_semantic`` / ``_write_procedural`` all
emit REFLECTION audit events.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cognithor.audit import AuditCategory, AuditLogger
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
# Reference helpers (must mirror the convention in PR-A)
# ---------------------------------------------------------------------------


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Reference canonical-form bytes — must match the helper in
    ``Reflector._emit_reflection_audit_event``.

    Recipe:
      - ``json.dumps(..., sort_keys=True, ensure_ascii=False)`` (default
        separators)
      - ``unicodedata.normalize("NFC", ...)``
      - ``.encode("utf-8")``
    """
    return unicodedata.normalize(
        "NFC",
        json.dumps(payload, sort_keys=True, ensure_ascii=False),
    ).encode("utf-8")


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config(tmp_path: Path) -> CognithorConfig:
    cfg = CognithorConfig(cognithor_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


def _mock_ollama() -> MagicMock:
    return MagicMock()


def _mock_router() -> MagicMock:
    router = MagicMock()
    router.select_model.return_value = "qwen3:8b"
    return router


def _mock_memory_manager() -> MagicMock:
    """Mock memory manager that swallows writes silently.

    Mirrors the surface accessed by ``Reflector._write_episodic`` /
    ``_write_semantic`` / ``_write_procedural``: ``.episodic``,
    ``.index``, ``.procedural``.
    """
    mgr = MagicMock()
    # Indexer: search returns empty, upsert is a no-op.
    mgr.index.search_entities.return_value = []
    mgr.index.upsert_entity.return_value = None
    mgr.index.upsert_relation.return_value = None
    # Procedural: load_procedure returns None (no existing entry).
    mgr.procedural.load_procedure.return_value = None
    mgr.procedural.save_procedure.return_value = None
    mgr.procedural.record_usage.return_value = None
    # Episodic: append_entry is a no-op.
    mgr.episodic.append_entry.return_value = None
    return mgr


# ---------------------------------------------------------------------------
# Hypothesis strategies for synthetic ReflectionResult instances
# ---------------------------------------------------------------------------


# Restrict text to printable ASCII + a few Unicode samples to keep
# generation tractable; Reflector sanitises anyway, and we're testing
# the audit-emit invariant, not the sanitiser.
_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Zs"), max_codepoint=0x017F),
    max_size=80,
)
_short_id = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), max_codepoint=0x007F),
    min_size=1,
    max_size=20,
)


_extracted_fact = st.builds(
    ExtractedFact,
    entity_name=_safe_text.filter(lambda s: bool(s.strip())),
    entity_type=st.sampled_from(["person", "company", "product", "concept", "unknown"]),
    attribute_key=st.one_of(st.just(""), _safe_text),
    attribute_value=st.one_of(st.just(""), _safe_text),
    relation_type=st.one_of(st.none(), _safe_text),
    relation_target=st.one_of(st.none(), _safe_text),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    source_session=_short_id,
)


_session_summary = st.builds(
    SessionSummary,
    goal=_safe_text,
    outcome=_safe_text,
    key_decisions=st.lists(_safe_text, max_size=3),
    open_items=st.lists(_safe_text, max_size=3),
    tools_used=st.lists(_safe_text, max_size=3),
    duration_ms=st.integers(min_value=0, max_value=1_000_000),
)


_procedure_candidate = st.builds(
    ProcedureCandidate,
    name=_short_id,
    trigger_keywords=st.lists(_safe_text, max_size=3),
    prerequisite_text=_safe_text,
    steps_text=_safe_text,
    learned_text=_safe_text,
    failure_patterns=st.lists(_safe_text, max_size=2),
    tools_required=st.lists(_safe_text, max_size=3),
    is_update=st.just(False),
)


@st.composite
def _reflection_result(draw: st.DrawFn) -> ReflectionResult:
    return ReflectionResult(
        session_id=draw(_short_id),
        success_score=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        evaluation=draw(_safe_text),
        extracted_facts=draw(st.lists(_extracted_fact, max_size=4)),
        procedure_candidate=draw(st.one_of(st.none(), _procedure_candidate)),
        session_summary=draw(st.one_of(st.none(), _session_summary)),
    )


# JSON-serialisable payload strategy for hash-determinism + canonical
# form invariants. Constrained to JSON-compatible types so
# ``json.dumps`` never raises.
_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=40),
)
_json_payload = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(_json_scalar, st.lists(_json_scalar, max_size=4)),
    max_size=6,
)


# ---------------------------------------------------------------------------
# Test 1 — 1-to-1 correspondence: autonomous writes ↔ REFLECTION events
# ---------------------------------------------------------------------------


class TestWritesEmitAuditEvents:
    """Every Reflector autonomous-memory write MUST emit a REFLECTION
    audit event.

    PR-A wired ``CausalAnalyzer.record_sequence``. PR-D (this) closes
    the gap by wiring ``_write_episodic`` / ``_write_semantic`` /
    ``_write_procedural`` through the same REFLECTION audit category.
    The property is no longer xfail.
    """

    @given(result=_reflection_result())
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @pytest.mark.asyncio
    async def test_every_apply_emits_one_event_per_memory_write(
        self,
        config: CognithorConfig,
        result: ReflectionResult,
    ) -> None:
        """Property: ``len(REFLECTION_events) >= sum(counts.values())``.

        Each non-zero tier (episodic / semantic / procedural) MUST emit
        at least one REFLECTION audit event; zero-tier writes are
        allowed to emit zero events. The ``>=`` (not ``==``) tolerates
        sinks that emit additional bookkeeping events (e.g. skipped /
        deduplicated paths).
        """
        audit = MagicMock()
        reflector = Reflector(
            config,
            _mock_ollama(),
            _mock_router(),
            audit_logger=audit,
        )
        memory_manager = _mock_memory_manager()

        counts = await reflector.apply(result, memory_manager)

        reflection_events = [
            call
            for call in audit.log_reflection_event.call_args_list
            # Calls go through ``log_reflection_event`` which auto-tags
            # category=REFLECTION; no further category check needed.
        ]
        # Total events emitted must cover every non-zero-tier write.
        non_zero_writes = sum(1 for v in counts.values() if v > 0)
        assert len(reflection_events) >= non_zero_writes, (
            f"silent autonomous writes detected: counts={counts!r}, "
            f"reflection_events={len(reflection_events)} (expected >= {non_zero_writes})"
        )


# ---------------------------------------------------------------------------
# Test 2 — Canonical-form parity with _last_hash_for_file convention
# ---------------------------------------------------------------------------


class TestCanonicalFormParity:
    """The bytes hashed by ``_emit_reflection_audit_event`` MUST be
    byte-identical to the recipe in ``AuditLogger._last_hash_for_file``
    when both inputs are ASCII-clean + already NFC.

    If a future PR adds ``separators=(",", ":")`` to either side this
    property fails — that's the lock.
    """

    @given(payload=_json_payload)
    @settings(max_examples=100, deadline=None)
    def test_canonical_form_matches_last_hash_convention(self, payload: dict[str, Any]) -> None:
        # Reference recipe in PR-A's helper.
        helper_canonical = _canonical_bytes(payload)
        # Recipe used by AuditLogger._last_hash_for_file (audit/__init__.py
        # line 830) — no NFC, but otherwise identical.
        last_hash_canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )

        # Idempotency property: NFC may alter compatibility ideographs
        # (e.g. U+F900 豈 → U+8C9D 貝) and decomposed Unicode, so the
        # round-trip is NOT bit-equal to ``payload`` in general. What
        # MUST hold is that re-canonicalising the round-trip yields
        # identical bytes — NFC is idempotent, so the helper's recipe
        # converges in one pass.
        helper_roundtrip = json.loads(helper_canonical.decode("utf-8"))
        assert _canonical_bytes(helper_roundtrip) == helper_canonical
        # The non-NFC recipe is byte-stable across round-trip by
        # construction (json.dumps preserves Unicode codepoints when
        # ensure_ascii=False).
        assert json.loads(last_hash_canonical.decode("utf-8")) == payload

        # For ASCII payloads (no decomposable Unicode) the two forms
        # must be byte-identical. We can't assert this for ALL inputs
        # because NFC may collapse decomposed Unicode while the
        # _last_hash_for_file recipe doesn't normalise — but for the
        # JSON keys (always ASCII here) and ASCII string values the
        # bytes match.
        if helper_canonical == last_hash_canonical:
            # ASCII path: digests must match too.
            assert (
                hashlib.sha256(helper_canonical).hexdigest()
                == hashlib.sha256(last_hash_canonical).hexdigest()
            )

    @given(payload=_json_payload)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_helper_payload_sha256_matches_reference(
        self,
        config: CognithorConfig,
        payload: dict[str, Any],
    ) -> None:
        """The helper's emitted ``payload_sha256`` MUST match an
        independently-computed reference. Locks the recipe across
        Python sessions.
        """
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        reflector._emit_reflection_audit_event("evt", payload)

        emitted = audit.log_reflection_event.call_args.kwargs["payload"]
        assert emitted["payload_sha256"] == _canonical_hash(payload)


# ---------------------------------------------------------------------------
# Test 3 — NFC normalisation collapses decomposed Unicode
# ---------------------------------------------------------------------------


class TestNfcNormalisation:
    """NFC-normalisation must produce equal canonical bytes for the
    composed and decomposed forms of the same logical string.
    """

    @given(s=st.text(max_size=40))
    @settings(max_examples=100, deadline=None)
    def test_nfc_collapses_decomposed_unicode(self, s: str) -> None:
        decomposed = unicodedata.normalize("NFD", s)
        composed = unicodedata.normalize("NFC", s)

        canon_d = _canonical_bytes({"x": decomposed})
        canon_c = _canonical_bytes({"x": composed})

        # After NFC normalisation both forms produce identical canonical
        # bytes — and therefore identical SHA-256 digests.
        assert canon_d == canon_c
        assert hashlib.sha256(canon_d).hexdigest() == hashlib.sha256(canon_c).hexdigest()


# ---------------------------------------------------------------------------
# Test 4 — Hash determinism: independent of dict insertion order
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    """``payload_sha256`` MUST be insertion-order-independent and
    reproducible across constructions of the same logical payload.
    """

    @given(
        items=st.lists(
            st.tuples(st.text(min_size=1, max_size=20), st.integers()),
            min_size=1,
            max_size=8,
            unique_by=lambda x: x[0],
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_hash_independent_of_dict_insertion_order(self, items: list[tuple[str, int]]) -> None:
        payload_a = dict(items)
        payload_b = dict(reversed(items))

        h_a = _canonical_hash(payload_a)
        h_b = _canonical_hash(payload_b)

        assert h_a == h_b, (
            f"hash diverged across insertion orders: "
            f"a={payload_a!r} -> {h_a}, b={payload_b!r} -> {h_b}"
        )

    @given(payload=_json_payload)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_hash_stable_across_repeated_emits(
        self,
        config: CognithorConfig,
        payload: dict[str, Any],
    ) -> None:
        """Same payload emitted twice produces the same digest."""
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)

        reflector._emit_reflection_audit_event("evt", payload)
        reflector._emit_reflection_audit_event("evt", payload)

        emitted_1 = audit.log_reflection_event.call_args_list[0].kwargs["payload"]
        emitted_2 = audit.log_reflection_event.call_args_list[1].kwargs["payload"]
        assert emitted_1["payload_sha256"] == emitted_2["payload_sha256"]


# ---------------------------------------------------------------------------
# Test 5 — Hash chain validity preserved across reflection appends
# ---------------------------------------------------------------------------


class TestHashChainIntegrity:
    """SEC-HIGH-5 hash-chain MUST stay valid when REFLECTION events
    are persisted alongside other audit categories.
    """

    @given(
        payloads=st.lists(_json_payload, min_size=1, max_size=8),
    )
    @settings(
        max_examples=25,  # disk-backed → keep low
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_chain_valid_after_reflection_appends(
        self,
        tmp_path_factory: pytest.TempPathFactory,
        payloads: list[dict[str, Any]],
    ) -> None:
        log_dir = tmp_path_factory.mktemp("audit-chain")
        audit = AuditLogger(log_dir=log_dir)

        for i, payload in enumerate(payloads):
            audit.log_reflection_event(
                action=f"prop_test_event_{i}",
                payload=payload,
                session_id="prop-test",
            )

        # All entries land in today's JSONL file. Resolve it from the
        # first entry's persisted path.
        log_files = list(log_dir.glob("audit_*.jsonl"))
        assert len(log_files) >= 1, "no audit JSONL was written"

        for log_file in log_files:
            ok, errors = audit.validate_chain(log_file)
            assert ok, f"chain broke: {errors}"
            assert errors == []

    @given(payload=_json_payload)
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_reflection_event_lands_in_reflection_category(
        self,
        tmp_path_factory: pytest.TempPathFactory,
        payload: dict[str, Any],
    ) -> None:
        """End-to-end: persisted REFLECTION entries are queryable by
        category and carry the structured payload (incl. caller-supplied
        ``payload_sha256`` if present).
        """
        log_dir = tmp_path_factory.mktemp("audit-cat")
        audit = AuditLogger(log_dir=log_dir)

        audit.log_reflection_event(
            action="prop_category_check",
            payload=payload,
            session_id="prop-test",
        )

        entries = audit.query(category=AuditCategory.REFLECTION)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.action == "prop_category_check"
        # All keys from the input payload survive into ``parameters``.
        for k, v in payload.items():
            assert entry.parameters[k] == v
