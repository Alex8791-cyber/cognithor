"""Tests for the TRUST-7 ToolFingerprint foundation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from cognithor.security.fingerprint import (
    FINGERPRINT_LEDGER,
    BinaryKind,
    FingerprintLedger,
    ToolFingerprint,
    fingerprint_python_tool,
    hash_bytes,
    hash_python_source,
)

if TYPE_CHECKING:
    from pathlib import Path

# A fixed valid SHA-256 hex string for unit tests where we need a
# concrete hash but don't care about the underlying bytes.
_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_REAL = hash_bytes(b"hello\n")


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# ToolFingerprint construction + validation
# ---------------------------------------------------------------------------


class TestToolFingerprintBasics:
    def test_minimal_fingerprint(self) -> None:
        fp = ToolFingerprint(
            name="web_fetch",
            kind=BinaryKind.TOOL,
            content_hash=_HEX_A,
        )
        assert fp.name == "web_fetch"
        assert fp.kind == BinaryKind.TOOL
        assert fp.content_hash == _HEX_A
        assert fp.short_hash == "a" * 12
        assert fp.version == ""
        assert fp.source_path == ""
        # captured_at defaults to now(UTC)
        assert fp.captured_at.tzinfo == UTC

    def test_short_hash_is_first_12(self) -> None:
        fp = ToolFingerprint(
            name="t",
            kind=BinaryKind.TOOL,
            content_hash=_HEX_REAL,
        )
        assert fp.short_hash == _HEX_REAL[:12]
        assert len(fp.short_hash) == 12

    def test_frozen_via_dataclass(self) -> None:
        fp = ToolFingerprint(name="t", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fp.name = "other"  # type: ignore[misc]


class TestToolFingerprintValidation:
    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ToolFingerprint(name="", kind=BinaryKind.TOOL, content_hash=_HEX_A)

    def test_short_hash_rejected(self) -> None:
        with pytest.raises(ValueError, match="64 lowercase-hex"):
            ToolFingerprint(name="t", kind=BinaryKind.TOOL, content_hash="a" * 16)

    def test_uppercase_hex_rejected(self) -> None:
        # The constructor enforces lowercase to keep the index stable.
        with pytest.raises(ValueError, match="64 lowercase-hex"):
            ToolFingerprint(name="t", kind=BinaryKind.TOOL, content_hash="A" * 64)

    def test_non_hex_rejected(self) -> None:
        with pytest.raises(ValueError, match="64 lowercase-hex"):
            ToolFingerprint(name="t", kind=BinaryKind.TOOL, content_hash="z" * 64)

    def test_equality_by_value(self) -> None:
        # Frozen dataclasses are equal field-wise. Two fingerprints
        # with the same hash but different captured_at values are
        # NOT equal — `captured_at` is part of the dataclass identity.
        captured = _utc(2026, 5, 4, 12, 0)
        a = ToolFingerprint(
            name="t",
            kind=BinaryKind.TOOL,
            content_hash=_HEX_A,
            captured_at=captured,
        )
        b = ToolFingerprint(
            name="t",
            kind=BinaryKind.TOOL,
            content_hash=_HEX_A,
            captured_at=captured,
        )
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# FingerprintLedger basic ops
# ---------------------------------------------------------------------------


class TestFingerprintLedgerBasic:
    def test_empty_ledger(self) -> None:
        ledger = FingerprintLedger()
        assert len(ledger) == 0
        assert ledger.get(_HEX_A) is None
        assert ledger.history("web_fetch") == ()
        assert ledger.names() == []
        assert _HEX_A not in ledger

    def test_register_returns_true_first_time(self) -> None:
        ledger = FingerprintLedger()
        fp = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        assert ledger.register(fp) is True
        assert _HEX_A in ledger
        assert ledger.get(_HEX_A) is fp
        assert ledger.history("web_fetch") == (fp,)

    def test_register_idempotent_for_same_hash(self) -> None:
        ledger = FingerprintLedger()
        fp1 = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        fp2 = ToolFingerprint(
            name="web_fetch",
            kind=BinaryKind.TOOL,
            content_hash=_HEX_A,
            notes="re-scan",
        )
        assert ledger.register(fp1) is True
        # Same hash → second register is a no-op even with different metadata.
        assert ledger.register(fp2) is False
        assert ledger.get(_HEX_A) is fp1
        assert ledger.history("web_fetch") == (fp1,)

    def test_register_new_hash_appends_to_name_history(self) -> None:
        ledger = FingerprintLedger()
        old = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        new = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_B)
        ledger.register(old)
        ledger.register(new)
        assert ledger.history("web_fetch") == (old, new)
        assert len(ledger) == 2

    def test_remove_existing(self) -> None:
        ledger = FingerprintLedger()
        fp = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        ledger.register(fp)
        assert ledger.remove(_HEX_A) is True
        assert _HEX_A not in ledger
        # name index also pruned
        assert ledger.history("web_fetch") == ()
        assert ledger.names() == []

    def test_remove_keeps_other_versions_under_same_name(self) -> None:
        ledger = FingerprintLedger()
        old = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        new = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_B)
        ledger.register(old)
        ledger.register(new)
        assert ledger.remove(_HEX_A) is True
        # The newer version stays under the same name.
        assert ledger.history("web_fetch") == (new,)
        assert _HEX_B in ledger

    def test_remove_missing_returns_false(self) -> None:
        ledger = FingerprintLedger()
        assert ledger.remove(_HEX_A) is False

    def test_clear(self) -> None:
        ledger = FingerprintLedger()
        ledger.register(ToolFingerprint(name="a", kind=BinaryKind.TOOL, content_hash=_HEX_A))
        ledger.register(ToolFingerprint(name="b", kind=BinaryKind.MODEL, content_hash=_HEX_B))
        ledger.clear()
        assert len(ledger) == 0
        assert ledger.names() == []

    def test_names_sorted(self) -> None:
        ledger = FingerprintLedger()
        for name, h in (("zeta", _HEX_A), ("alpha", _HEX_B), ("mu", _HEX_C)):
            ledger.register(ToolFingerprint(name=name, kind=BinaryKind.TOOL, content_hash=h))
        assert ledger.names() == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestFingerprintLedgerQueries:
    def test_filter_by_kind(self) -> None:
        ledger = FingerprintLedger()
        tool = ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        model = ToolFingerprint(name="qwen3:30b", kind=BinaryKind.MODEL, content_hash=_HEX_B)
        pack = ToolFingerprint(name="reddit-pro", kind=BinaryKind.PACK, content_hash=_HEX_C)
        ledger.register(tool)
        ledger.register(model)
        ledger.register(pack)
        assert ledger.filter_by_kind(BinaryKind.TOOL) == [tool]
        assert ledger.filter_by_kind(BinaryKind.MODEL) == [model]
        assert ledger.filter_by_kind(BinaryKind.PACK) == [pack]
        assert ledger.filter_by_kind(BinaryKind.SCHEMA) == []

    def test_filter_by_kind_sorted(self) -> None:
        ledger = FingerprintLedger()
        for name, h in (("zeta", _HEX_A), ("alpha", _HEX_B)):
            ledger.register(ToolFingerprint(name=name, kind=BinaryKind.TOOL, content_hash=h))
        result = ledger.filter_by_kind(BinaryKind.TOOL)
        assert [fp.name for fp in result] == ["alpha", "zeta"]

    def test_divergent_names_finds_only_multi_hash_names(self) -> None:
        ledger = FingerprintLedger()
        # web_fetch has two SHAs (drifted); read_file has one (stable).
        ledger.register(
            ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_A)
        )
        ledger.register(
            ToolFingerprint(name="web_fetch", kind=BinaryKind.TOOL, content_hash=_HEX_B)
        )
        ledger.register(
            ToolFingerprint(name="read_file", kind=BinaryKind.TOOL, content_hash=_HEX_C)
        )
        assert ledger.divergent_names() == ["web_fetch"]


# ---------------------------------------------------------------------------
# Snapshot serialisation
# ---------------------------------------------------------------------------


class TestFingerprintLedgerSnapshot:
    def test_snapshot_empty(self) -> None:
        assert FingerprintLedger().snapshot() == []

    def test_snapshot_round_trip_shape(self) -> None:
        ledger = FingerprintLedger()
        captured = _utc(2026, 5, 4, 12, 0)
        fp = ToolFingerprint(
            name="web_fetch",
            kind=BinaryKind.TOOL,
            content_hash=_HEX_REAL,
            version="1.4.1",
            captured_at=captured,
            source_path="/tmp/web_fetch.py",
            upstream_url="https://pypi.org/project/web-fetch/",
            notes="initial scan",
        )
        ledger.register(fp)
        snap = ledger.snapshot()
        assert len(snap) == 1
        entry = snap[0]
        assert entry["name"] == "web_fetch"
        assert entry["kind"] == "tool"
        assert entry["content_hash"] == _HEX_REAL
        assert entry["short_hash"] == _HEX_REAL[:12]
        assert entry["version"] == "1.4.1"
        assert entry["captured_at"] == captured.isoformat()
        assert entry["source_path"] == "/tmp/web_fetch.py"
        assert entry["upstream_url"] == "https://pypi.org/project/web-fetch/"
        assert entry["notes"] == "initial scan"

    def test_snapshot_sorted_by_name_then_captured_at(self) -> None:
        ledger = FingerprintLedger()
        early = _utc(2026, 5, 4, 10, 0)
        late = _utc(2026, 5, 4, 12, 0)
        # Same name, two timestamps — ascending captured_at expected.
        ledger.register(
            ToolFingerprint(
                name="web_fetch",
                kind=BinaryKind.TOOL,
                content_hash=_HEX_B,
                captured_at=late,
            )
        )
        ledger.register(
            ToolFingerprint(
                name="web_fetch",
                kind=BinaryKind.TOOL,
                content_hash=_HEX_A,
                captured_at=early,
            )
        )
        # Different name comes second alphabetically.
        ledger.register(
            ToolFingerprint(
                name="zzz_other",
                kind=BinaryKind.TOOL,
                content_hash=_HEX_C,
                captured_at=early,
            )
        )
        snap = ledger.snapshot()
        assert [(e["name"], e["content_hash"]) for e in snap] == [
            ("web_fetch", _HEX_A),
            ("web_fetch", _HEX_B),
            ("zzz_other", _HEX_C),
        ]


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


class TestHashHelpers:
    def test_hash_bytes_deterministic(self) -> None:
        h = hash_bytes(b"hello world\n")
        assert len(h) == 64
        assert h == hash_bytes(b"hello world\n")

    def test_hash_python_source_normalises_line_endings(self, tmp_path: Path) -> None:
        # Two files, identical content but Windows vs POSIX line endings,
        # must hash to the same value.
        unix = tmp_path / "unix.py"
        win = tmp_path / "win.py"
        unix.write_bytes(b"def x():\n    return 1\n")
        win.write_bytes(b"def x():\r\n    return 1\r\n")
        assert hash_python_source(unix) == hash_python_source(win)

    def test_hash_python_source_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            hash_python_source(tmp_path / "does_not_exist.py")

    def test_fingerprint_python_tool_builds_tool_kind(self, tmp_path: Path) -> None:
        src = tmp_path / "tool.py"
        src.write_bytes(b"def run():\n    return 'ok'\n")
        fp = fingerprint_python_tool(
            name="my_tool",
            path=src,
            version="0.1.0",
            upstream_url="https://example.test/my_tool",
        )
        assert fp.kind == BinaryKind.TOOL
        assert fp.name == "my_tool"
        assert fp.version == "0.1.0"
        assert fp.upstream_url == "https://example.test/my_tool"
        assert fp.source_path == str(src)
        assert fp.content_hash == hash_python_source(src)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestProcessLocalLedger:
    def test_default_ledger_is_a_fingerprint_ledger(self) -> None:
        assert isinstance(FINGERPRINT_LEDGER, FingerprintLedger)
