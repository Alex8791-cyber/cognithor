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
    fingerprint_native_binary,
    fingerprint_python_tool,
    hash_bytes,
    hash_native_binary,
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
# TRUST-7 BINARY — native-binary fingerprinting
# ---------------------------------------------------------------------------


class TestHashNativeBinary:
    """``hash_native_binary`` must produce the same SHA-256 a user gets
    from ``sha256sum`` — raw bytes, no line-ending normalisation."""

    def test_matches_raw_sha256(self, tmp_path: Path) -> None:
        # Hand-rolled bytes including \r\n so we can prove the hash is
        # NOT line-ending-normalised (which would diverge from
        # sha256sum on a binary).
        import hashlib

        payload = b"\x7fELF\x01\x02\x03\x00\r\n\x00\xff\xfe"
        binary = tmp_path / "fake.bin"
        binary.write_bytes(payload)

        expected = hashlib.sha256(payload).hexdigest()
        actual = hash_native_binary(binary)
        assert actual == expected
        # 64 lowercase-hex chars (matches ToolFingerprint contract)
        assert len(actual) == 64
        assert all(c in "0123456789abcdef" for c in actual)

    def test_does_not_normalise_line_endings(self, tmp_path: Path) -> None:
        """Crucial difference vs ``hash_python_source``: native binaries
        are raw bytes. CRLF and LF must produce DIFFERENT hashes here
        even though they'd produce the SAME hash for source code."""
        unix = tmp_path / "unix.bin"
        win = tmp_path / "win.bin"
        unix.write_bytes(b"AB\nCD\n")
        win.write_bytes(b"AB\r\nCD\r\n")
        assert hash_native_binary(unix) != hash_native_binary(win), (
            "native-binary hash must not collapse CRLF/LF — that would "
            "diverge from sha256sum and break audit reconstruction"
        )

    def test_handles_chunked_read_for_large_files(self, tmp_path: Path) -> None:
        """The implementation streams in 64 KiB chunks. Test with a
        file > one chunk to catch off-by-one errors at the boundary."""
        import hashlib

        # 200 KiB — definitely crosses the 64 KiB chunk boundary
        payload = b"X" * (200 * 1024)
        binary = tmp_path / "large.bin"
        binary.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert hash_native_binary(binary) == expected

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            hash_native_binary(tmp_path / "does_not_exist")

    def test_directory_raises_isadirectory(self, tmp_path: Path) -> None:
        with pytest.raises((IsADirectoryError, PermissionError)):
            hash_native_binary(tmp_path)


class TestFingerprintNativeBinary:
    """``fingerprint_native_binary`` is the gateway-boot convenience
    wrapper that pins a native executable into a ToolFingerprint."""

    def test_builds_binary_kind_with_explicit_version(self, tmp_path: Path) -> None:
        binary = tmp_path / "ollama"
        binary.write_bytes(b"\x7fELF...")
        fp = fingerprint_native_binary(
            name="ollama",
            path=binary,
            version="0.4.7",
            upstream_url="https://github.com/ollama/ollama",
            notes="captured at gateway boot",
        )
        assert fp.kind == BinaryKind.BINARY
        assert fp.name == "ollama"
        assert fp.version == "0.4.7"
        assert fp.upstream_url == "https://github.com/ollama/ollama"
        assert fp.notes == "captured at gateway boot"
        assert fp.source_path == str(binary)
        assert fp.content_hash == hash_native_binary(binary)

    def test_explicit_empty_version_skips_probe(self, tmp_path: Path) -> None:
        """``version=""`` (empty, not None) means "I checked, there is
        no version" — the probe should NOT run. Important for offline
        operators who want to keep the audit log deterministic."""
        binary = tmp_path / "no-version.bin"
        binary.write_bytes(b"opaque-bytes")
        fp = fingerprint_native_binary(name="opaque", path=binary, version="")
        assert fp.version == ""
        # Hash must still land
        assert fp.content_hash == hash_native_binary(binary)

    def test_version_none_triggers_probe_with_default_flag(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """``version=None`` (default) triggers ``<bin> --version`` probe."""
        binary = tmp_path / "fake.exe"
        binary.write_bytes(b"\x00\x01\x02")

        captured: dict[str, list[str]] = {"argv": []}

        class _FakeProc:
            def __init__(self) -> None:
                self.stdout = b"fake-tool 1.2.3 (build abc123)\n"
                self.stderr = b""
                self.returncode = 0

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            captured["argv"] = list(argv)
            return _FakeProc()

        import subprocess as sp_module

        monkeypatch.setattr(sp_module, "run", fake_run)

        fp = fingerprint_native_binary(name="fake", path=binary)
        # Probe was called with the default ``--version`` flag and the
        # binary path
        assert captured["argv"] == [str(binary), "--version"]
        # First non-empty stdout line landed verbatim in the version field
        assert fp.version == "fake-tool 1.2.3 (build abc123)"

    def test_custom_version_flag(self, tmp_path: Path, monkeypatch) -> None:
        """Some binaries use ``-v`` instead of ``--version``."""
        binary = tmp_path / "old-tool"
        binary.write_bytes(b"x")
        captured: dict[str, list[str]] = {"argv": []}

        class _FakeProc:
            stdout = b"old-tool v0.9\n"
            stderr = b""
            returncode = 0

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            captured["argv"] = list(argv)
            return _FakeProc()

        import subprocess as sp_module

        monkeypatch.setattr(sp_module, "run", fake_run)

        fp = fingerprint_native_binary(name="old-tool", path=binary, version_flag="-v")
        assert captured["argv"] == [str(binary), "-v"]
        assert fp.version == "old-tool v0.9"

    def test_probe_timeout_returns_empty_version(self, tmp_path: Path, monkeypatch) -> None:
        """A binary that hangs on stdout must not block boot — the
        probe times out and we record version=''."""
        binary = tmp_path / "hangs.bin"
        binary.write_bytes(b"x")

        import subprocess as sp_module

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            raise sp_module.TimeoutExpired(cmd=argv, timeout=5.0)

        monkeypatch.setattr(sp_module, "run", fake_run)

        fp = fingerprint_native_binary(name="hangs", path=binary)
        assert fp.version == ""
        # But the hash must still land
        assert fp.content_hash == hash_native_binary(binary)

    def test_probe_subprocess_error_returns_empty_version(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """OSError (e.g. PermissionError on exec) → empty version, not crash."""
        binary = tmp_path / "denied.bin"
        binary.write_bytes(b"x")

        import subprocess as sp_module

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            raise PermissionError("not executable")

        monkeypatch.setattr(sp_module, "run", fake_run)

        fp = fingerprint_native_binary(name="denied", path=binary)
        assert fp.version == ""

    def test_probe_empty_output_returns_empty_version(self, tmp_path: Path, monkeypatch) -> None:
        """A binary that exits 0 without writing anything → no version."""
        binary = tmp_path / "silent.bin"
        binary.write_bytes(b"x")

        class _FakeProc:
            stdout = b""
            stderr = b""
            returncode = 0

        import subprocess as sp_module

        monkeypatch.setattr(sp_module, "run", lambda *a, **kw: _FakeProc())

        fp = fingerprint_native_binary(name="silent", path=binary)
        assert fp.version == ""

    def test_probe_falls_through_to_stderr(self, tmp_path: Path, monkeypatch) -> None:
        """Some CLIs print the version banner to stderr (Java, gradle).
        The probe concatenates stdout+stderr so the version still lands."""
        binary = tmp_path / "stderr-version.bin"
        binary.write_bytes(b"x")

        class _FakeProc:
            stdout = b""
            stderr = b"banner-tool 2.0\n"
            returncode = 0

        import subprocess as sp_module

        monkeypatch.setattr(sp_module, "run", lambda *a, **kw: _FakeProc())

        fp = fingerprint_native_binary(name="banner-tool", path=binary)
        assert fp.version == "banner-tool 2.0"

    def test_probe_caps_long_lines(self, tmp_path: Path, monkeypatch) -> None:
        """A binary that prints a 1 KB single line gets truncated to
        200 chars to keep audit logs readable."""
        binary = tmp_path / "verbose.bin"
        binary.write_bytes(b"x")

        class _FakeProc:
            stdout = b"V" * 1024 + b"\n"
            stderr = b""
            returncode = 0

        import subprocess as sp_module

        monkeypatch.setattr(sp_module, "run", lambda *a, **kw: _FakeProc())

        fp = fingerprint_native_binary(name="verbose", path=binary)
        assert len(fp.version) <= 200
        assert fp.version == "V" * 200

    def test_probe_rejects_huge_output(self, tmp_path: Path, monkeypatch) -> None:
        """If a binary spews >4 KB to stdout, treat it as opaque (don't
        try to parse) — bounds the gateway-boot audit log."""
        binary = tmp_path / "spammy.bin"
        binary.write_bytes(b"x")

        class _FakeProc:
            stdout = b"S" * 10000  # 10 KB > 4 KB cap
            stderr = b""
            returncode = 0

        import subprocess as sp_module

        monkeypatch.setattr(sp_module, "run", lambda *a, **kw: _FakeProc())

        fp = fingerprint_native_binary(name="spammy", path=binary)
        assert fp.version == ""

    def test_fingerprint_round_trips_through_ledger(self, tmp_path: Path) -> None:
        """A native-binary fingerprint registers cleanly into the
        canonical FingerprintLedger and round-trips via content_hash."""
        binary = tmp_path / "tool"
        binary.write_bytes(b"\x7fELF...payload")
        fp = fingerprint_native_binary(name="tool", path=binary, version="1.0")

        ledger = FingerprintLedger()
        was_new = ledger.register(fp)
        assert was_new is True
        # Idempotent re-register
        assert ledger.register(fp) is False
        # Lookup by content_hash returns the same fingerprint
        assert ledger.get(fp.content_hash) is fp
        # Filter by kind picks up the new BINARY entry
        names_by_kind = {f.name for f in ledger.filter_by_kind(BinaryKind.BINARY)}
        assert names_by_kind == {"tool"}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestProcessLocalLedger:
    def test_default_ledger_is_a_fingerprint_ledger(self) -> None:
        assert isinstance(FINGERPRINT_LEDGER, FingerprintLedger)


class TestFingerprintLedgerSelfAuditMigration:
    """TRUST-10 self-audit: importing the fingerprint module records
    a v0→v1 migration step into the canonical MIGRATION_LEDGER.
    """

    def test_migration_step_landed(self) -> None:
        from cognithor.security.migration_ledger import (
            MIGRATION_LEDGER,
            MigrationDomain,
            MigrationStatus,
        )

        step = MIGRATION_LEDGER.get("fingerprint_ledger:v0-no-ledger:v1-dual-index-ledger")
        assert step is not None
        assert step.status == MigrationStatus.APPLIED
        assert step.domain == MigrationDomain.FINGERPRINT_LEDGER
        assert step.applied_by == "system"
        assert (
            MIGRATION_LEDGER.head_version(MigrationDomain.FINGERPRINT_LEDGER)
            == "v1-dual-index-ledger"
        )

    def test_repeated_calls_idempotent(self) -> None:
        from cognithor.security.fingerprint import (
            _record_fingerprint_ledger_migration,
        )

        _record_fingerprint_ledger_migration()
        _record_fingerprint_ledger_migration()
        _record_fingerprint_ledger_migration()
