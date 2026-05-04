"""Tests for cognithor.packs.installer."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from cognithor.packs.errors import PackInstallError
from cognithor.packs.installer import PackInstaller


def _build_pack_zip(
    tmp_path: Path,
    *,
    namespace: str = "cognithor-official",
    pack_id: str = "installed-test",
    version: str = "1.0.0",
    eula_text: str = "TEST EULA TEXT",
    license_: str = "apache-2.0",
) -> Path:
    """Build a valid pack zip in tmp_path and return its path."""
    eula_hash = hashlib.sha256(eula_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "namespace": namespace,
        "pack_id": pack_id,
        "version": version,
        "display_name": "Installed Test",
        "description": "test",
        "license": license_,
        "min_cognithor_version": ">=0.1.0",
        "eula_sha256": eula_hash,
        "publisher": {"id": "cognithor-official", "display_name": "Cognithor"},
    }
    if license_ == "proprietary":
        manifest["pricing"] = {
            "indie": {
                "list_price": 149,
                "launch_price": 79,
                "post_launch_price": 99,
                "launch_cap": 100,
                "currency": "USD",
            }
        }

    zip_path = tmp_path / f"{pack_id}-{version}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("pack_manifest.json", json.dumps(manifest))
        zf.writestr("eula.md", eula_text)
        zf.writestr(
            "pack.py",
            "from cognithor.packs.interface import AgentPack\n\n"
            "class Pack(AgentPack):\n"
            "    def register(self, ctx): pass\n",
        )
    return zip_path


@pytest.fixture
def packs_root(tmp_path: Path) -> Path:
    d = tmp_path / "packs"
    d.mkdir()
    return d


class TestInstallLocalZip:
    def test_install_happy_path(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = _build_pack_zip(tmp_path)
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        manifest = installer.install_from_path(zip_path)

        assert manifest.pack_id == "installed-test"
        pack_dir = packs_root / "cognithor-official" / "installed-test"
        assert pack_dir.exists()
        assert (pack_dir / "pack_manifest.json").exists()
        assert (pack_dir / "eula.md").exists()
        assert (pack_dir / ".eula_accepted").exists()

    def test_install_declined_eula_aborts(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = _build_pack_zip(tmp_path)
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

        with pytest.raises(PackInstallError, match="EULA"):
            installer.install_from_path(zip_path)

        pack_dir = packs_root / "cognithor-official" / "installed-test"
        assert not pack_dir.exists()

    def test_install_already_installed_same_version(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = _build_pack_zip(tmp_path)
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        installer.install_from_path(zip_path)

        with pytest.raises(PackInstallError, match="already installed"):
            installer.install_from_path(zip_path)

    def test_install_upgrade(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old_zip = _build_pack_zip(tmp_path, version="1.0.0")
        new_zip = _build_pack_zip(tmp_path, version="1.1.0")
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        installer.install_from_path(old_zip)
        manifest = installer.install_from_path(new_zip)
        assert manifest.version == "1.1.0"

    def test_install_rejects_missing_eula(self, tmp_path: Path, packs_root: Path) -> None:
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "pack_manifest.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "namespace": "cognithor-official",
                        "pack_id": "bad",
                        "version": "1.0.0",
                        "display_name": "Bad",
                        "description": "bad",
                        "license": "apache-2.0",
                        "min_cognithor_version": ">=0.1.0",
                        "eula_sha256": "a" * 64,
                        "publisher": {"id": "x", "display_name": "x"},
                    }
                ),
            )
            zf.writestr("pack.py", "class Pack:\n    pass\n")
            # No eula.md on purpose.

        installer = PackInstaller(packs_root=packs_root)
        with pytest.raises(PackInstallError, match="eula.md"):
            installer.install_from_path(zip_path)


class TestRemove:
    def test_remove_installed(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = _build_pack_zip(tmp_path)
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(zip_path)

        installer.remove("cognithor-official/installed-test")

        pack_dir = packs_root / "cognithor-official" / "installed-test"
        assert not pack_dir.exists()

    def test_remove_nonexistent_raises(self, packs_root: Path) -> None:
        installer = PackInstaller(packs_root=packs_root)
        with pytest.raises(PackInstallError, match="not installed"):
            installer.remove("cognithor-official/does-not-exist")


class TestList:
    def test_list_returns_installed_manifests(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        z1 = _build_pack_zip(tmp_path, pack_id="a")
        z2 = _build_pack_zip(tmp_path, pack_id="b")
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z1)
        installer.install_from_path(z2)

        installed = installer.list_installed()
        ids = {m.pack_id for m in installed}
        assert ids == {"a", "b"}


# ============================================================================
# TRUST-4 — Rollback (operational-trust audit, 2026-05-04)
# ============================================================================


class TestRollback:
    """``PackInstaller.rollback(qualified_id, to_version=...)`` restores
    a previously-snapshotted version. Snapshots are taken automatically
    on every upgrade (to ``<root>/.backups/<ns>/<id>/<old_version>/``).
    """

    def test_first_install_creates_no_backup(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        z = _build_pack_zip(tmp_path, version="1.0.0")
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z)

        assert installer.list_backups("cognithor-official/installed-test") == []

    def test_upgrade_creates_backup_of_previous_version(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        z1 = _build_pack_zip(tmp_path, version="1.0.0")
        z2_dir = tmp_path / "v2"
        z2_dir.mkdir()
        z2 = _build_pack_zip(z2_dir, version="2.0.0")

        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z1)
        installer.install_from_path(z2)

        backups = installer.list_backups("cognithor-official/installed-test")
        assert backups == ["1.0.0"]

    def test_rollback_restores_previous_version(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        z1 = _build_pack_zip(tmp_path, version="1.0.0")
        z2_dir = tmp_path / "v2"
        z2_dir.mkdir()
        z2 = _build_pack_zip(z2_dir, version="2.0.0")

        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z1)
        installer.install_from_path(z2)
        installed = installer.list_installed()
        assert next(m.version for m in installed if m.pack_id == "installed-test") == "2.0.0"

        manifest = installer.rollback("cognithor-official/installed-test")
        assert manifest.version == "1.0.0"
        installed_after = installer.list_installed()
        assert next(m.version for m in installed_after if m.pack_id == "installed-test") == "1.0.0"

    def test_rollback_is_reversible(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling rollback twice flips back to the newer version —
        the swap snapshots the current install before restoring.
        """
        z1 = _build_pack_zip(tmp_path, version="1.0.0")
        z2_dir = tmp_path / "v2"
        z2_dir.mkdir()
        z2 = _build_pack_zip(z2_dir, version="2.0.0")

        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z1)
        installer.install_from_path(z2)

        installer.rollback("cognithor-official/installed-test")  # 2.0.0 → 1.0.0
        manifest_back = installer.rollback("cognithor-official/installed-test", to_version="2.0.0")
        assert manifest_back.version == "2.0.0"

    def test_rollback_to_version_not_in_backups_raises(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        z1 = _build_pack_zip(tmp_path, version="1.0.0")
        z2_dir = tmp_path / "v2"
        z2_dir.mkdir()
        z2 = _build_pack_zip(z2_dir, version="2.0.0")

        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z1)
        installer.install_from_path(z2)

        with pytest.raises(PackInstallError, match="not available"):
            installer.rollback("cognithor-official/installed-test", to_version="9.9.9")

    def test_rollback_with_no_backups_raises(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        z = _build_pack_zip(tmp_path, version="1.0.0")
        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z)

        with pytest.raises(PackInstallError, match="No backups"):
            installer.rollback("cognithor-official/installed-test")

    def test_rollback_uninstalled_pack_raises(self, packs_root: Path) -> None:
        installer = PackInstaller(packs_root=packs_root)
        with pytest.raises(PackInstallError, match="not installed"):
            installer.rollback("cognithor-official/never-installed")

    def test_backup_directory_hidden_from_loader(
        self, tmp_path: Path, packs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``.backups/`` must NOT show up in ``list_installed`` —
        otherwise old versions would leak into runtime registration.
        """
        z1 = _build_pack_zip(tmp_path, version="1.0.0")
        z2_dir = tmp_path / "v2"
        z2_dir.mkdir()
        z2 = _build_pack_zip(z2_dir, version="2.0.0")

        installer = PackInstaller(packs_root=packs_root)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        installer.install_from_path(z1)
        installer.install_from_path(z2)

        assert (packs_root / ".backups").is_dir()
        installed = installer.list_installed()
        assert len(installed) == 1
        assert installed[0].version == "2.0.0"
