"""Tests for ``cognithor.cli.receipt_cmd`` (TRUST-1 receipt CLI)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from cognithor.audit import AuditLogger
from cognithor.cli.receipt_cmd import cmd_show, cmd_verify

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
