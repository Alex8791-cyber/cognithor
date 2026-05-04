"""Tests for cognithor.packs.cli — the `cognithor pack` subcommand."""

from __future__ import annotations

import hashlib
import json
import zipfile
from typing import TYPE_CHECKING

from cognithor.packs.cli import main as pack_main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _build_pack_zip(tmp_path: Path, pack_id: str = "cli-test") -> Path:
    eula = "EULA"
    eula_hash = hashlib.sha256(eula.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "namespace": "cognithor-official",
        "pack_id": pack_id,
        "version": "1.0.0",
        "display_name": "CLI Test",
        "description": "t",
        "license": "apache-2.0",
        "min_cognithor_version": ">=0.1.0",
        "eula_sha256": eula_hash,
        "publisher": {"id": "x", "display_name": "x"},
    }
    zip_path = tmp_path / f"{pack_id}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("pack_manifest.json", json.dumps(manifest))
        zf.writestr("eula.md", eula)
        zf.writestr(
            "pack.py",
            "from cognithor.packs.interface import AgentPack\n\n"
            "class Pack(AgentPack):\n"
            "    def register(self, ctx): pass\n",
        )
    return zip_path


class TestCliInstall:
    def test_install_local_zip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        packs_root = tmp_path / "packs"
        zip_path = _build_pack_zip(tmp_path)
        monkeypatch.setenv("COGNITHOR_PACKS_DIR", str(packs_root))
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        rc = pack_main(["install", str(zip_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "installed" in out.lower() or "ok" in out.lower()
        assert (packs_root / "cognithor-official" / "cli-test" / "pack_manifest.json").exists()

    def test_install_missing_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        packs_root = tmp_path / "packs"
        monkeypatch.setenv("COGNITHOR_PACKS_DIR", str(packs_root))
        rc = pack_main(["install", str(tmp_path / "nope.zip")])
        assert rc != 0


class TestCliList:
    def test_list_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        packs_root = tmp_path / "packs"
        packs_root.mkdir()
        monkeypatch.setenv("COGNITHOR_PACKS_DIR", str(packs_root))

        rc = pack_main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no packs" in out.lower() or out.strip() == "(no packs installed)"

    def test_list_after_install(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        packs_root = tmp_path / "packs"
        zip_path = _build_pack_zip(tmp_path)
        monkeypatch.setenv("COGNITHOR_PACKS_DIR", str(packs_root))
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        pack_main(["install", str(zip_path)])
        capsys.readouterr()  # drain install output

        rc = pack_main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "cli-test" in out
        assert "1.0.0" in out


class TestCliUpdate:
    """``cognithor pack update`` is not yet automated.

    Until it is, the stub must (a) require a qualified_id and (b) exit
    with a non-zero code so wrapping scripts and CI pipelines do not
    treat the printed message as a successful update (WF-6).
    """

    def test_update_requires_qualified_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import pytest as _pytest

        monkeypatch.setenv("COGNITHOR_PACKS_DIR", str(tmp_path / "packs"))
        with _pytest.raises(SystemExit) as exc_info:
            pack_main(["update"])
        assert exc_info.value.code == 2

    def test_update_stub_exits_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("COGNITHOR_PACKS_DIR", str(tmp_path / "packs"))
        rc = pack_main(["update", "cognithor-official/some-pack"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "not yet automated" in captured.err
        assert "cognithor-official/some-pack" in captured.err
        assert "rollback" in captured.err
