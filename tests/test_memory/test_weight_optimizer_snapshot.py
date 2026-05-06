"""Tests for the Operational-Trust PR-B snapshot mechanism on
``SearchWeightOptimizer``.

Covers:

* deterministic ``weight_sha256`` (same plaintext → same hash, NFC + sort_keys)
* ``.fernet`` + ``.meta.json`` sidecar layout
* content-addressed deduplication (no rewrite on identical weights)
* audit-emit callback fires ``weight_snapshot_persisted`` with the
  correct triple ``(weight_sha256, session_id, snapshot_bytes)``
* graceful skip when EncryptedFileIO/snapshot_dir are not wired

The Fernet path is exercised via a real :class:`EncryptedFileIO` with
a transient ``COGNITHOR_DB_KEY`` env var so the keyring chain is never
touched during the test.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from cognithor.memory.weight_optimizer import SearchWeightOptimizer
from cognithor.security.encrypted_file import EncryptedFileIO

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def fernet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a deterministic encryption key via env var.

    ``EncryptedFileIO`` derives its Fernet key from this string via
    SHA-256, so the test never touches the real OS keyring.
    """
    monkeypatch.setenv("COGNITHOR_DB_KEY", "pr-b-snapshot-test-key")


@pytest.fixture()
def snapshot_dir(tmp_path: Path) -> Path:
    return tmp_path / "weight_snapshots"


@pytest.fixture()
def encrypted_io(fernet_env: None) -> EncryptedFileIO:
    return EncryptedFileIO()


class TestSnapshotPersistence:
    """End-to-end: weights → fernet+meta files on disk, deterministic hash."""

    def test_snapshot_files_created(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
        )
        try:
            opt.record_outcome(
                "test query",
                {"vector": 0.7, "bm25": 0.2, "graph": 0.1},
                feedback_score=0.9,
                session_id="sess-abc",
            )

            # Exactly one .fernet + one .meta.json should exist.
            fernet_files = list(snapshot_dir.glob("*.fernet"))
            meta_files = list(snapshot_dir.glob("*.meta.json"))
            assert len(fernet_files) == 1
            assert len(meta_files) == 1

            # Filename = weight_sha256 + extension.
            assert fernet_files[0].stem == meta_files[0].name.split(".meta.json")[0]
        finally:
            opt.close()

    def test_meta_sidecar_shape(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
        )
        try:
            opt.record_outcome(
                "test",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.8,
                session_id="sess-meta-shape",
            )

            meta_files = list(snapshot_dir.glob("*.meta.json"))
            assert len(meta_files) == 1
            meta = json.loads(meta_files[0].read_text(encoding="utf-8"))

            # Required keys (per PR-B brief).
            assert set(meta.keys()) == {
                "weight_sha256",
                "fernet_file",
                "created_at",
                "session_id",
                "snapshot_bytes",
            }
            assert meta["session_id"] == "sess-meta-shape"
            assert meta["fernet_file"] == f"{meta['weight_sha256']}.fernet"
            assert meta["snapshot_bytes"] > 0
            # Hash format = 64 hex chars.
            assert len(meta["weight_sha256"]) == 64
            assert all(c in "0123456789abcdef" for c in meta["weight_sha256"])
        finally:
            opt.close()

    def test_weight_sha256_matches_canonical_recipe(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        """The on-disk weight_sha256 must equal an independently-computed
        hash via the canonical-NFC-JSON-sort_keys recipe (same as PR-A).
        """
        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
            initial_weights=(0.5, 0.3, 0.2),
        )
        try:
            # feedback_score=0 ensures EMA does NOT mutate weights, so
            # the snapshot reflects the initial vector exactly.
            opt.record_outcome(
                "test",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.0,
                session_id="sess-deterministic",
            )

            expected = {"vector": 0.5, "bm25": 0.3, "graph": 0.2}
            canonical = unicodedata.normalize(
                "NFC",
                json.dumps(expected, sort_keys=True, ensure_ascii=False),
            ).encode("utf-8")
            expected_hash = hashlib.sha256(canonical).hexdigest()

            meta_files = list(snapshot_dir.glob("*.meta.json"))
            meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
            assert meta["weight_sha256"] == expected_hash
            assert meta["snapshot_bytes"] == len(canonical)
        finally:
            opt.close()

    def test_fernet_file_is_encrypted(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
        )
        try:
            opt.record_outcome(
                "test",
                {"vector": 0.7, "bm25": 0.2, "graph": 0.1},
                feedback_score=0.5,
                session_id="sess-encrypted",
            )

            fernet_path = next(snapshot_dir.glob("*.fernet"))
            # The encrypted file starts with the COGNITHOR_ENC_V1 magic
            # header, NOT raw JSON — proves Fernet was used.
            raw = fernet_path.read_bytes()
            assert raw.startswith(b"COGNITHOR_ENC_V1\n")

            # Round-trip via EncryptedFileIO should yield the canonical
            # plaintext we expect.
            decrypted = encrypted_io.read(fernet_path)
            decoded = json.loads(decrypted)
            assert set(decoded.keys()) == {"vector", "bm25", "graph"}
        finally:
            opt.close()

    def test_content_addressed_dedup(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        """Identical weight vectors → no rewrite of the .fernet file.

        The meta sidecar IS rewritten (different created_at + session_id
        per call) — that's intentional, the meta is per-event.
        """
        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
            initial_weights=(0.5, 0.3, 0.2),
        )
        try:
            # First call — feedback_score=0 keeps weights untouched.
            opt.record_outcome(
                "q1",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.0,
                session_id="sess-1",
            )
            fernet_files_1 = sorted(snapshot_dir.glob("*.fernet"))
            assert len(fernet_files_1) == 1
            mtime_1 = fernet_files_1[0].stat().st_mtime_ns

            # Second call — same weight vector → same hash → same file.
            opt.record_outcome(
                "q2",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.0,
                session_id="sess-2",
            )
            fernet_files_2 = sorted(snapshot_dir.glob("*.fernet"))
            assert len(fernet_files_2) == 1
            assert fernet_files_2[0] == fernet_files_1[0]
            # File was NOT rewritten — mtime unchanged.
            assert fernet_files_2[0].stat().st_mtime_ns == mtime_1
        finally:
            opt.close()


class TestAuditCallback:
    """``weight_snapshot_persisted`` event flows through the callback."""

    def test_audit_callback_invoked(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        captured: list[tuple[str, dict[str, Any]]] = []

        def cb(event_type: str, payload: dict[str, Any]) -> None:
            captured.append((event_type, payload))

        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
            audit_emit_callback=cb,
        )
        try:
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
                session_id="sess-audit-cb",
            )

            assert len(captured) == 1
            event_type, payload = captured[0]
            assert event_type == "weight_snapshot_persisted"
            assert payload["session_id"] == "sess-audit-cb"
            assert "weight_sha256" in payload
            assert payload["snapshot_bytes"] > 0
            # Hash format check.
            assert len(payload["weight_sha256"]) == 64
        finally:
            opt.close()

    def test_set_audit_emit_callback_late_bind(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        """Late-binding mirrors PR-A's CausalAnalyzer pattern."""
        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
        )
        captured: list[tuple[str, dict[str, Any]]] = []

        def cb(event_type: str, payload: dict[str, Any]) -> None:
            captured.append((event_type, payload))

        try:
            opt.set_audit_emit_callback(cb)
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
                session_id="sess-late-bind",
            )
            assert len(captured) == 1
            assert captured[0][0] == "weight_snapshot_persisted"
        finally:
            opt.close()

    def test_audit_callback_failure_does_not_break_record(
        self,
        encrypted_io: EncryptedFileIO,
        snapshot_dir: Path,
    ) -> None:
        """A throwing callback must not interrupt the EMA path."""
        cb = MagicMock(side_effect=RuntimeError("audit boom"))

        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=snapshot_dir,
            audit_emit_callback=cb,
        )
        try:
            # Should NOT raise.
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
                session_id="sess-cb-fails",
            )
            # And the snapshot files still exist.
            assert list(snapshot_dir.glob("*.fernet"))
            assert list(snapshot_dir.glob("*.meta.json"))
        finally:
            opt.close()


class TestGracefulSkip:
    """No EncryptedFileIO + no snapshot_dir → no error, no files."""

    def test_skip_when_io_missing(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "weight_snapshots"
        opt = SearchWeightOptimizer(
            encrypted_file_io=None,
            snapshot_dir=snapshot_dir,
        )
        try:
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
                session_id="sess",
            )
            assert not snapshot_dir.exists() or not list(snapshot_dir.iterdir())
        finally:
            opt.close()

    def test_skip_when_dir_missing(
        self,
        encrypted_io: EncryptedFileIO,
    ) -> None:
        opt = SearchWeightOptimizer(
            encrypted_file_io=encrypted_io,
            snapshot_dir=None,
        )
        try:
            # Should not raise.
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
                session_id="sess",
            )
        finally:
            opt.close()

    def test_skip_when_no_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        snapshot_dir: Path,
    ) -> None:
        """Without a Fernet key the snapshot is silently skipped (the
        encrypted-at-rest contract trumps best-effort persistence)."""
        # Ensure no key is reachable.
        monkeypatch.delenv("COGNITHOR_DB_KEY", raising=False)

        # A real EncryptedFileIO with no key + no keyring entry will
        # report ``is_available == False``.
        io = EncryptedFileIO()
        # Stub the keyring chain to ensure no key.
        io._initialized = True  # type: ignore[attr-defined]
        io._fernet = None  # type: ignore[attr-defined]

        opt = SearchWeightOptimizer(
            encrypted_file_io=io,
            snapshot_dir=snapshot_dir,
        )
        try:
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
                session_id="sess",
            )
            assert not snapshot_dir.exists() or not list(snapshot_dir.iterdir())
        finally:
            opt.close()


class TestBackwardCompat:
    """Existing callers without the new kwargs still work."""

    def test_legacy_construction_no_kwargs(self) -> None:
        opt = SearchWeightOptimizer()
        try:
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
            )
            # Weights still updated as before.
            w_v, w_b, w_g = opt.get_optimized_weights()
            assert abs((w_v + w_b + w_g) - 1.0) < 0.001
        finally:
            opt.close()

    def test_record_outcome_session_id_optional(self) -> None:
        """``session_id`` defaults to empty string."""
        opt = SearchWeightOptimizer()
        try:
            # Old signature still works without session_id.
            opt.record_outcome(
                "q",
                {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                feedback_score=0.7,
            )
        finally:
            opt.close()
