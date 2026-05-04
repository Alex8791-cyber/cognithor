"""Tests for ``cognithor.cli.receipt_cmd`` (TRUST-1 receipt CLI)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from cognithor.audit import AuditLogger
from cognithor.cli.receipt_cmd import cmd_export_all, cmd_list, cmd_show, cmd_verify

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def receipt_logger() -> AuditLogger:
    """A fresh logger with two tagged entries for a single session."""
    logger = AuditLogger()
    logger.log_tool_call("read_file", session_id="run_42")
    logger.log_gatekeeper("ALLOW", "GREEN tool", tool_name="read_file", session_id="run_42")
    return logger


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


class TestCmdShow:
    def test_empty_session_id_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cmd_show(session_id="")
        assert rc == 2
        assert "session_id is required" in capsys.readouterr().err

    def test_show_writes_to_out(self, tmp_path: Path) -> None:
        out = tmp_path / "receipt.json"
        # No on-disk audit log + ghost session — receipt is empty but valid.
        rc = cmd_show(session_id="run_xyz", out=out)
        assert rc == 0
        assert out.exists()
        bundle = json.loads(out.read_text(encoding="utf-8"))
        assert bundle["session_id"] == "run_xyz"
        assert bundle["entry_count"] == 0
        # Default — no trust block when not requested.
        assert "trust" not in bundle

    def test_show_with_trust(self, tmp_path: Path) -> None:
        out = tmp_path / "receipt.json"
        rc = cmd_show(session_id="run_xyz", include_trust=True, out=out)
        assert rc == 0
        bundle = json.loads(out.read_text(encoding="utf-8"))
        assert "trust" in bundle
        trust = bundle["trust"]
        assert trust["run_id"] == "run_xyz"
        for key in (
            "permission_scopes",
            "cost",
            "fingerprints",
            "escalations",
            "provenance",
            "migrations",
        ):
            assert key in trust

    def test_show_with_signing_key(self, tmp_path: Path) -> None:
        out = tmp_path / "receipt.json"
        rc = cmd_show(
            session_id="run_xyz",
            signing_key="hunter2",
            include_trust=True,
            out=out,
        )
        assert rc == 0
        bundle = json.loads(out.read_text(encoding="utf-8"))
        assert bundle["signature"]
        # Round-trip via verify_receipt_signature.
        assert AuditLogger.verify_receipt_signature(bundle, "hunter2") is True

    def test_show_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cmd_show(session_id="run_xyz")
        assert rc == 0
        out = capsys.readouterr().out
        bundle = json.loads(out)
        assert bundle["session_id"] == "run_xyz"


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


class TestCmdVerify:
    def test_empty_key_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bundle_path = tmp_path / "b.json"
        bundle_path.write_text("{}", encoding="utf-8")
        rc = cmd_verify(bundle_path=bundle_path, signing_key="")
        assert rc == 2
        assert "key is required" in capsys.readouterr().err

    def test_missing_file_returns_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cmd_verify(bundle_path=tmp_path / "does_not_exist.json", signing_key="hunter2")
        assert rc == 2
        assert "cannot read" in capsys.readouterr().err

    def test_invalid_json_returns_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = tmp_path / "bad.json"
        bundle_path.write_text("not json {", encoding="utf-8")
        rc = cmd_verify(bundle_path=bundle_path, signing_key="hunter2")
        assert rc == 2
        assert "invalid JSON" in capsys.readouterr().err

    def test_non_object_bundle_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = tmp_path / "list.json"
        bundle_path.write_text("[1, 2, 3]", encoding="utf-8")
        rc = cmd_verify(bundle_path=bundle_path, signing_key="hunter2")
        assert rc == 2
        assert "not a JSON object" in capsys.readouterr().err

    def test_unsigned_bundle_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = tmp_path / "unsigned.json"
        bundle_path.write_text(json.dumps({"session_id": "x", "signature": ""}), encoding="utf-8")
        rc = cmd_verify(bundle_path=bundle_path, signing_key="hunter2")
        assert rc == 1
        assert "no signature" in capsys.readouterr().err

    def test_valid_signature_returns_0(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "signed.json"
        cmd_show(
            session_id="run_xyz",
            signing_key="hunter2",
            include_trust=True,
            out=out,
        )
        rc = cmd_verify(bundle_path=out, signing_key="hunter2")
        assert rc == 0
        assert "signature valid" in capsys.readouterr().out

    def test_wrong_key_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out = tmp_path / "signed.json"
        cmd_show(
            session_id="run_xyz",
            signing_key="hunter2",
            include_trust=True,
            out=out,
        )
        rc = cmd_verify(bundle_path=out, signing_key="WRONG")
        assert rc == 1
        assert "does not match" in capsys.readouterr().err

    def test_tampered_bundle_returns_1(self, tmp_path: Path) -> None:
        out = tmp_path / "signed.json"
        cmd_show(
            session_id="run_xyz",
            signing_key="hunter2",
            include_trust=True,
            out=out,
        )
        bundle = json.loads(out.read_text(encoding="utf-8"))
        # Tamper with the trust block.
        bundle["trust"]["run_id"] = "tampered"
        out.write_text(json.dumps(bundle), encoding="utf-8")
        rc = cmd_verify(bundle_path=out, signing_key="hunter2")
        assert rc == 1


def _write_audit_jsonl(
    log_dir: Path,
    *,
    name: str = "audit_2026-05-04.jsonl",
    rows: list[dict[str, str]],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / name).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestCmdList:
    def test_invalid_limit_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cmd_list(log_dir=tmp_path, limit=0)
        assert rc == 2
        assert "limit" in capsys.readouterr().err

    def test_no_log_dir_prints_no_entries(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cmd_list(log_dir=None)
        assert rc == 0
        assert "no audit entries found" in capsys.readouterr().out

    def test_missing_log_dir_prints_no_entries(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cmd_list(log_dir=tmp_path / "nope")
        assert rc == 0
        assert "no audit entries found" in capsys.readouterr().out

    def test_lists_sessions_with_counts_and_last_seen(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_dir = tmp_path / "audit"
        _write_audit_jsonl(
            log_dir,
            rows=[
                {"session_id": "run-A", "timestamp": "2026-05-04T10:00:00Z"},
                {"session_id": "run-A", "timestamp": "2026-05-04T10:00:05Z"},
                {"session_id": "run-B", "timestamp": "2026-05-04T11:00:00Z"},
            ],
        )
        rc = cmd_list(log_dir=log_dir)
        assert rc == 0
        out = capsys.readouterr().out
        # Newest-first ordering: run-B before run-A.
        idx_b = out.index("run-B")
        idx_a = out.index("run-A")
        assert idx_b < idx_a
        assert "2026-05-04T11:00:00Z" in out
        # Run-A has 2 entries.
        assert " 2  " in out or "      2  " in out

    def test_unscoped_entries_bucket_under_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_dir = tmp_path / "audit"
        _write_audit_jsonl(
            log_dir,
            rows=[
                {"timestamp": "2026-05-04T10:00:00Z"},
                {"session_id": "", "timestamp": "2026-05-04T10:01:00Z"},
                {"session_id": "run-A", "timestamp": "2026-05-04T11:00:00Z"},
            ],
        )
        rc = cmd_list(log_dir=log_dir)
        assert rc == 0
        out = capsys.readouterr().out
        assert "(unscoped)" in out
        assert "run-A" in out

    def test_limit_truncates_to_newest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_dir = tmp_path / "audit"
        _write_audit_jsonl(
            log_dir,
            rows=[
                {"session_id": "run-A", "timestamp": "2026-05-04T10:00:00Z"},
                {"session_id": "run-B", "timestamp": "2026-05-04T11:00:00Z"},
                {"session_id": "run-C", "timestamp": "2026-05-04T12:00:00Z"},
            ],
        )
        rc = cmd_list(log_dir=log_dir, limit=2)
        assert rc == 0
        out = capsys.readouterr().out
        assert "run-C" in out
        assert "run-B" in out
        assert "run-A" not in out  # truncated

    def test_corrupt_lines_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_dir = tmp_path / "audit"
        log_dir.mkdir()
        (log_dir / "audit_2026-05-04.jsonl").write_text(
            '{"session_id":"run-A","timestamp":"2026-05-04T10:00:00Z"}\n'
            "this is not json\n"
            '{"session_id":"run-B","timestamp":"2026-05-04T11:00:00Z"}\n',
            encoding="utf-8",
        )
        rc = cmd_list(log_dir=log_dir)
        assert rc == 0
        out = capsys.readouterr().out
        assert "run-A" in out
        assert "run-B" in out


class TestCmdExportAll:
    def test_missing_log_dir_returns_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cmd_export_all(
            log_dir=tmp_path / "no_such_dir",
            out_dir=tmp_path / "out",
        )
        assert rc == 2
        assert "log-dir" in capsys.readouterr().err

    def test_exports_one_file_per_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_dir = tmp_path / "audit"
        out_dir = tmp_path / "out"
        _write_audit_jsonl(
            log_dir,
            rows=[
                {"session_id": "run-A", "timestamp": "2026-05-04T10:00:00Z"},
                {"session_id": "run-B", "timestamp": "2026-05-04T11:00:00Z"},
                {"session_id": "run-A", "timestamp": "2026-05-04T10:00:05Z"},
            ],
        )
        rc = cmd_export_all(log_dir=log_dir, out_dir=out_dir)
        assert rc == 0
        # One JSON per session_id + manifest.json.
        files = sorted(p.name for p in out_dir.iterdir())
        assert "run-A.json" in files
        assert "run-B.json" in files
        assert "manifest.json" in files
        # Manifest lists both sessions.
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["count"] == 2
        sids = {entry["session_id"] for entry in manifest["sessions"]}
        assert sids == {"run-A", "run-B"}
        assert "exported 2 session" in capsys.readouterr().out

    def test_unscoped_bucketed_under_filename(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "audit"
        out_dir = tmp_path / "out"
        _write_audit_jsonl(
            log_dir,
            rows=[
                {"timestamp": "2026-05-04T10:00:00Z"},
                {"session_id": "run-A", "timestamp": "2026-05-04T11:00:00Z"},
            ],
        )
        rc = cmd_export_all(log_dir=log_dir, out_dir=out_dir)
        assert rc == 0
        assert (out_dir / "_unscoped.json").exists()
        assert (out_dir / "run-A.json").exists()

    def test_signing_key_signs_every_receipt(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "audit"
        out_dir = tmp_path / "out"
        _write_audit_jsonl(
            log_dir,
            rows=[
                {"session_id": "run-A", "timestamp": "2026-05-04T10:00:00Z"},
                {"session_id": "run-B", "timestamp": "2026-05-04T11:00:00Z"},
            ],
        )
        rc = cmd_export_all(log_dir=log_dir, out_dir=out_dir, signing_key="hunter2")
        assert rc == 0
        for sid in ("run-A", "run-B"):
            bundle = json.loads((out_dir / f"{sid}.json").read_text(encoding="utf-8"))
            assert bundle["signature"]
            assert AuditLogger.verify_receipt_signature(bundle, "hunter2") is True

    def test_include_trust_folds_bundle(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "audit"
        out_dir = tmp_path / "out"
        _write_audit_jsonl(
            log_dir,
            rows=[
                {"session_id": "run-A", "timestamp": "2026-05-04T10:00:00Z"},
            ],
        )
        rc = cmd_export_all(log_dir=log_dir, out_dir=out_dir, include_trust=True)
        assert rc == 0
        bundle = json.loads((out_dir / "run-A.json").read_text(encoding="utf-8"))
        assert "trust" in bundle

    def test_empty_audit_log_writes_zero_session_manifest(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "audit"
        log_dir.mkdir()
        out_dir = tmp_path / "out"
        rc = cmd_export_all(log_dir=log_dir, out_dir=out_dir)
        assert rc == 0
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["count"] == 0
        assert manifest["sessions"] == []
