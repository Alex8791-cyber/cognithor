"""Tests for the Reflector audit-event helper (Operational-Trust PR-A/3).

Verifies:
  - Helper produces stable, reproducible canonical-JSON SHA-256.
  - Helper no-ops when ``audit_logger`` is None.
  - Helper does NOT raise when the underlying ``log_system`` raises
    (best-effort discipline; runtime path stays alive).
  - ``payload_sha256`` is deterministic (NFC + sort_keys + compact).
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

if TYPE_CHECKING:
    from pathlib import Path


def _mock_ollama() -> MagicMock:
    return MagicMock()


def _mock_router() -> MagicMock:
    router = MagicMock()
    router.select_model.return_value = "qwen3:8b"
    return router


def _canonical_hash(payload: dict) -> str:
    """Reference implementation matching the helper."""
    canonical = unicodedata.normalize(
        "NFC",
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@pytest.fixture()
def config(tmp_path: Path) -> CognithorConfig:
    cfg = CognithorConfig(cognithor_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


class TestEmitReflectionAuditEvent:
    """``Reflector._emit_reflection_audit_event`` contract."""

    def test_no_op_when_audit_logger_none(self, config: CognithorConfig) -> None:
        reflector = Reflector(config, _mock_ollama(), _mock_router())
        # Should not raise, should not call anything.
        reflector._emit_reflection_audit_event(
            "test_event",
            {"foo": "bar"},
        )

    def test_calls_log_system_with_canonical_payload(self, config: CognithorConfig) -> None:
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        payload = {"session_id": "s1", "tools": ["a", "b"]}
        reflector._emit_reflection_audit_event("test_event", payload)

        audit.log_system.assert_called_once()
        call = audit.log_system.call_args
        assert call.kwargs["event"] == "test_event"
        # Description is canonical JSON containing payload_sha256.
        emitted = json.loads(call.kwargs["description"])
        assert emitted["session_id"] == "s1"
        assert emitted["tools"] == ["a", "b"]
        assert emitted["payload_sha256"] == _canonical_hash(payload)

    def test_payload_sha256_is_stable_across_calls(self, config: CognithorConfig) -> None:
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        payload = {"b": 2, "a": 1}
        reflector._emit_reflection_audit_event("evt", payload)
        reflector._emit_reflection_audit_event("evt", payload)

        emitted_1 = json.loads(audit.log_system.call_args_list[0].kwargs["description"])
        emitted_2 = json.loads(audit.log_system.call_args_list[1].kwargs["description"])
        assert emitted_1["payload_sha256"] == emitted_2["payload_sha256"]

    def test_payload_sha256_independent_of_key_order(self, config: CognithorConfig) -> None:
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        reflector._emit_reflection_audit_event("evt", {"a": 1, "b": 2})
        reflector._emit_reflection_audit_event("evt", {"b": 2, "a": 1})

        emitted_1 = json.loads(audit.log_system.call_args_list[0].kwargs["description"])
        emitted_2 = json.loads(audit.log_system.call_args_list[1].kwargs["description"])
        assert emitted_1["payload_sha256"] == emitted_2["payload_sha256"]

    def test_payload_sha256_reproducible_via_reference(self, config: CognithorConfig) -> None:
        """Hash matches an independently-computed reference (different
        Python session would compute the same digest)."""
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        payload = {"unicode": "café", "score": 0.5, "tools": ["read", "write"]}
        reflector._emit_reflection_audit_event("evt", payload)

        emitted = json.loads(audit.log_system.call_args.kwargs["description"])
        assert emitted["payload_sha256"] == _canonical_hash(payload)

    def test_does_not_raise_when_logger_raises(self, config: CognithorConfig) -> None:
        audit = MagicMock()
        audit.log_system.side_effect = RuntimeError("audit backend down")
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        # Best-effort: must NOT raise.
        reflector._emit_reflection_audit_event("evt", {"x": 1})

    def test_does_not_mutate_input_payload(self, config: CognithorConfig) -> None:
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        payload = {"a": 1}
        reflector._emit_reflection_audit_event("evt", payload)
        assert "payload_sha256" not in payload

    def test_handles_unicode_nfc_normalisation(self, config: CognithorConfig) -> None:
        audit = MagicMock()
        reflector = Reflector(config, _mock_ollama(), _mock_router(), audit_logger=audit)
        # "é" as composed (NFC) vs decomposed (NFD) should hash the same
        # after NFC normalisation.
        composed = {"name": "café"}
        decomposed = {"name": "café"}
        reflector._emit_reflection_audit_event("evt", composed)
        reflector._emit_reflection_audit_event("evt", decomposed)

        emitted_1 = json.loads(audit.log_system.call_args_list[0].kwargs["description"])
        emitted_2 = json.loads(audit.log_system.call_args_list[1].kwargs["description"])
        assert emitted_1["payload_sha256"] == emitted_2["payload_sha256"]
