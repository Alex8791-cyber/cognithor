"""Pack installer — install from local zip or URL, EULA click-through, upgrade, remove.

On-disk layout after installation::

    <packs_root>/
        <namespace>/
            <pack_id>/
                pack_manifest.json
                eula.md
                .eula_accepted      <- JSON: timestamp, user, eula_sha256, version
                pack.py             <- (or whatever entrypoint declares)

EULA click-through is required for every new install.  The ``.eula_accepted``
file is written only after the user types ``y`` at the prompt.  Upgrades
re-prompt the EULA when the ``eula_sha256`` changes.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from cognithor.packs.errors import PackInstallError, PackValidationError
from cognithor.packs.interface import PackManifest
from cognithor.packs.loader import PackLoader
from cognithor.utils.logging import get_logger

_log = get_logger(__name__)

# Version used when listing installed packs — high enough to satisfy any
# min_cognithor_version constraint so the loader never skips a pack.
_LIST_VERSION = "999.0.0"


class PackInstaller:
    """Install, upgrade, remove, and list agent packs.

    Parameters
    ----------
    packs_root:
        Root directory that contains ``<namespace>/<pack_id>/`` sub-trees.
        Created automatically if it doesn't exist.
    installer_version:
        Version string written into ``.eula_accepted`` for auditing.
    """

    def __init__(
        self,
        *,
        packs_root: Path,
        installer_version: str = "0.92.0",
    ) -> None:
        self._root = Path(packs_root)
        self._installer_version = installer_version
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install_from_path(self, zip_path: Path) -> PackManifest:
        """Install a pack from a local zip file.

        Parameters
        ----------
        zip_path:
            Path to the ``.zip`` bundle.

        Returns
        -------
        PackManifest
            The validated manifest of the newly installed pack.

        Raises
        ------
        PackInstallError
            If the zip is invalid, EULA is declined, the same version is
            already installed, or any other install step fails.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise PackInstallError(f"Zip file not found: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            raise PackInstallError(f"Not a valid zip file: {zip_path}")

        return self._install(zip_path)

    def install_from_url(self, url: str) -> PackManifest:
        """Download a pack from *url* and install it.

        Requires ``httpx`` to be installed (``pip install httpx``).

        Parameters
        ----------
        url:
            HTTP(S) URL pointing to a ``.zip`` bundle.

        Returns
        -------
        PackManifest
            The validated manifest of the newly installed pack.

        Raises
        ------
        PackInstallError
            On network error, invalid zip, or any install failure.
        """
        try:
            import httpx
        except ImportError as exc:
            raise PackInstallError("httpx is required for URL installs: pip install httpx") from exc

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "pack.zip"
            try:
                with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                    with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        with dest.open("wb") as fh:
                            for chunk in resp.iter_bytes(chunk_size=65536):
                                fh.write(chunk)
            except Exception as exc:
                raise PackInstallError(f"Download failed for {url!r}: {exc}") from exc

            return self._install(dest)

    def remove(self, qualified_id: str) -> None:
        """Remove an installed pack.

        Parameters
        ----------
        qualified_id:
            ``namespace/pack_id`` string.

        Raises
        ------
        PackInstallError
            If the pack is not currently installed.
        """
        namespace, pack_id = self._split_qid(qualified_id)
        pack_dir = self._root / namespace / pack_id
        if not pack_dir.exists():
            raise PackInstallError(f"Pack {qualified_id!r} is not installed (directory not found).")
        shutil.rmtree(pack_dir)
        _log.info("pack.removed", qualified_id=qualified_id)

        # Remove namespace dir if now empty.
        ns_dir = self._root / namespace
        if ns_dir.exists() and not any(ns_dir.iterdir()):
            ns_dir.rmdir()

    def list_installed(self) -> list[PackManifest]:
        """Return manifests for every currently installed pack.

        Uses a high synthetic version so ``min_cognithor_version`` never
        filters out installed packs.
        """
        loader = PackLoader(packs_dir=self._root, cognithor_version=_LIST_VERSION)
        return loader.discover()

    # ── Rollback (TRUST-4) ───────────────────────────────────────────

    def _backups_root(self) -> Path:
        """Return ``<packs_root>/.backups``. Hidden from ``PackLoader``
        thanks to its dot-prefix skip in ``discover()``.
        """
        return self._root / ".backups"

    def list_backups(self, qualified_id: str) -> list[str]:
        """Return the list of versions available for rollback for
        *qualified_id*, sorted oldest → newest.

        Each version corresponds to a snapshot taken just before that
        version was overwritten by a newer install. The currently
        installed version is NOT included.

        TRUST-4 (operational-trust audit, 2026-05-04).
        """
        namespace, pack_id = self._split_qid(qualified_id)
        backup_dir = self._backups_root() / namespace / pack_id
        if not backup_dir.exists():
            return []
        versions = [p.name for p in backup_dir.iterdir() if p.is_dir()]

        # Best-effort version sort: split on dots, parse ints. Unknown
        # forms fall back to lexical via a synthetic ``(-1, ...)`` key
        # that always sorts before semver tuples.
        def _vkey(v: str) -> tuple[int, ...]:
            try:
                parts = v.split("-", 1)[0].split(".")
                return tuple(int(p) for p in parts)
            except (ValueError, AttributeError):
                # Sentinel: lexical hash collapsed into a leading
                # negative slot so non-semver versions sort first.
                return (-1, hash(v) & 0xFFFF)

        return sorted(versions, key=_vkey)

    def rollback(
        self,
        qualified_id: str,
        *,
        to_version: str | None = None,
    ) -> PackManifest:
        """Roll an installed pack back to a previously-snapshotted version.

        TRUST-4 (operational-trust audit, 2026-05-04). Reviewer asked
        for "rollback for plugin/pack updates" — until now ``cognithor
        pack update`` was a stub with no downgrade path.

        Parameters
        ----------
        qualified_id:
            ``namespace/pack_id`` of the installed pack.
        to_version:
            Specific version to restore. Defaults to the most-recent
            snapshot (latest version EXCLUDING the currently installed
            one). ``ValueError`` if no snapshot is available.

        Returns
        -------
        PackManifest
            The manifest of the now-restored pack.

        Raises
        ------
        PackInstallError
            If the pack is not installed, no backups exist, or the
            requested ``to_version`` is unavailable.

        Side effect
        -----------
        Before restoring, the *current* version is itself snapshotted
        (so ``rollback`` is reversible — calling it twice with no
        arguments alternates between the two latest versions).
        """
        namespace, pack_id = self._split_qid(qualified_id)
        target_dir = self._root / namespace / pack_id
        if not target_dir.exists():
            raise PackInstallError(
                f"Pack {qualified_id!r} is not installed — nothing to roll back."
            )

        backups = self.list_backups(qualified_id)
        if not backups:
            raise PackInstallError(
                f"No backups available for {qualified_id!r}. Backups are taken on "
                "every upgrade — first install creates no snapshot."
            )

        if to_version is None:
            chosen = backups[-1]  # most-recent backup
        elif to_version not in backups:
            raise PackInstallError(
                f"Version {to_version!r} not available for {qualified_id!r}. Available: {backups}"
            )
        else:
            chosen = to_version

        backup_dir = self._backups_root() / namespace / pack_id / chosen
        if not backup_dir.exists():
            # Race / concurrent removal. Defensive.
            raise PackInstallError(f"Backup directory disappeared: {backup_dir}")

        # Snapshot the CURRENT install before swapping, so rollback is
        # itself reversible.
        try:
            current = self._read_manifest(target_dir)
            if current.version != chosen:
                pre_swap = self._backups_root() / namespace / pack_id / current.version
                if pre_swap.exists():
                    shutil.rmtree(pre_swap)
                pre_swap.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(target_dir, pre_swap)
        except Exception as exc:
            _log.warning(
                "pack.rollback_pre_swap_backup_failed",
                qualified_id=qualified_id,
                error=str(exc)[:200],
            )

        # Swap.
        shutil.rmtree(target_dir)
        shutil.copytree(backup_dir, target_dir)
        manifest = self._read_manifest(target_dir)

        _log.info(
            "pack.rolled_back",
            qualified_id=qualified_id,
            to_version=manifest.version,
        )
        return manifest

    def accept_eula(self, qualified_id: str) -> None:
        """Re-prompt and (re-)write ``.eula_accepted`` for an installed pack.

        Useful when an upgrade changes the EULA text.

        Raises
        ------
        PackInstallError
            If the pack is not installed or the EULA is declined.
        """
        namespace, pack_id = self._split_qid(qualified_id)
        pack_dir = self._root / namespace / pack_id
        if not pack_dir.exists():
            raise PackInstallError(f"Pack {qualified_id!r} is not installed.")
        manifest = self._read_manifest(pack_dir)
        eula_path = pack_dir / "eula.md"
        if not eula_path.exists():
            raise PackInstallError(f"eula.md missing in {pack_dir} — pack may be corrupted.")
        eula_text = eula_path.read_text(encoding="utf-8")
        if not self._prompt_eula(manifest, eula_text):
            raise PackInstallError(
                "EULA declined — pack remains installed but EULA acceptance was not updated."
            )
        self._write_acceptance(pack_dir, manifest)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _install(self, zip_path: Path) -> PackManifest:
        """Core install logic: extract → validate → EULA → place."""
        with tempfile.TemporaryDirectory() as td:
            extract_root = Path(td) / "extracted"
            extract_root.mkdir()

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)

            # Pack files may be at the zip root or inside a single sub-dir.
            pack_root = self._find_pack_root(extract_root)

            # --- Validate manifest ---
            manifest_path = pack_root / "pack_manifest.json"
            if not manifest_path.exists():
                raise PackInstallError(
                    "pack_manifest.json not found in zip — not a valid pack bundle."
                )
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = PackManifest.model_validate(raw)
            except PackValidationError as exc:
                raise PackInstallError(f"Manifest validation failed: {exc}") from exc
            except Exception as exc:
                raise PackInstallError(f"pack_manifest.json is invalid: {exc}") from exc

            # --- Validate eula.md presence ---
            eula_path = pack_root / "eula.md"
            if not eula_path.exists():
                raise PackInstallError("eula.md not found in zip — every pack must ship an EULA.")

            # --- Validate EULA hash ---
            eula_bytes = eula_path.read_bytes()
            actual_hash = hashlib.sha256(eula_bytes).hexdigest()
            if actual_hash != manifest.eula_sha256:
                raise PackInstallError(
                    f"eula.md SHA-256 mismatch for {manifest.qualified_id!r}. "
                    f"Expected {manifest.eula_sha256}, got {actual_hash}."
                )

            # --- Check existing installation ---
            target_dir = self._root / manifest.namespace / manifest.pack_id
            is_upgrade = False
            if target_dir.exists():
                existing = self._read_manifest(target_dir)
                if existing.version == manifest.version:
                    raise PackInstallError(
                        f"Pack {manifest.qualified_id!r} version"
                        f" {manifest.version} is already installed."
                        " Use --force to reinstall or install a newer version."
                    )
                is_upgrade = True
                _log.info(
                    "pack.upgrading",
                    qualified_id=manifest.qualified_id,
                    from_version=existing.version,
                    to_version=manifest.version,
                )

            # --- EULA click-through ---
            eula_text = eula_bytes.decode("utf-8")
            if not self._prompt_eula(manifest, eula_text, is_upgrade=is_upgrade):
                raise PackInstallError(
                    f"EULA declined — pack {manifest.qualified_id!r} was not installed."
                )

            # --- Backup current install before overwrite (TRUST-4) ---
            # Operational-trust audit (2026-05-04) — reviewer asked for
            # "rollback for plugin/pack updates". Snapshot the existing
            # install into ``<root>/.backups/<ns>/<id>/<old_version>/``
            # so ``rollback()`` can restore it later. Skipped on first
            # install (target_dir doesn't exist).
            if target_dir.exists():
                try:
                    existing = self._read_manifest(target_dir)
                    backup_dir = (
                        self._backups_root()
                        / manifest.namespace
                        / manifest.pack_id
                        / existing.version
                    )
                    if backup_dir.exists():
                        # Same-version backup already exists (re-install
                        # of an older version after rollback). Replace
                        # the snapshot so the most recent state of that
                        # version wins.
                        shutil.rmtree(backup_dir)
                    backup_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(target_dir, backup_dir)
                    _log.info(
                        "pack.backed_up",
                        qualified_id=manifest.qualified_id,
                        version=existing.version,
                        backup_path=str(backup_dir),
                    )
                except Exception as exc:
                    # Backup failure must not block the install — log
                    # and continue. Rollback is a best-effort feature.
                    _log.warning(
                        "pack.backup_failed",
                        qualified_id=manifest.qualified_id,
                        error=str(exc)[:200],
                    )

            # --- Place files ---
            if target_dir.exists():
                shutil.rmtree(target_dir)

            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(pack_root, target_dir)

            # --- Write acceptance marker ---
            self._write_acceptance(target_dir, manifest)

            _log.info(
                "pack.installed",
                qualified_id=manifest.qualified_id,
                version=manifest.version,
                upgrade=is_upgrade,
            )
            return manifest

    def _find_pack_root(self, extract_root: Path) -> Path:
        """Return the directory containing ``pack_manifest.json``.

        Handles two layouts:
        - Files at zip root: ``extract_root/pack_manifest.json``
        - Files inside a single subdirectory: ``extract_root/<name>/pack_manifest.json``
        """
        if (extract_root / "pack_manifest.json").exists():
            return extract_root

        children = [p for p in extract_root.iterdir() if p.is_dir()]
        if len(children) == 1 and (children[0] / "pack_manifest.json").exists():
            return children[0]

        raise PackInstallError(
            "pack_manifest.json not found at zip root or in a single subdirectory."
        )

    def _prompt_eula(
        self,
        manifest: PackManifest,
        eula_text: str,
        *,
        is_upgrade: bool = False,
    ) -> bool:
        """Print the EULA and ask the user to accept.

        Returns ``True`` if the user accepted, ``False`` otherwise.
        Calls ``input()`` directly so tests can monkeypatch ``builtins.input``.
        """
        action = "upgrade" if is_upgrade else "install"
        print(
            f"\n{'=' * 70}\n"
            f"  EULA for {manifest.display_name} v{manifest.version}"
            f"  ({manifest.qualified_id})\n"
            f"{'=' * 70}\n"
        )
        print(eula_text)
        print(f"\n{'=' * 70}")
        print(f"You must accept the EULA above to {action} this pack.")
        answer = input("Accept? [y/N]: ").strip().lower()
        return answer == "y"

    def _write_acceptance(self, pack_dir: Path, manifest: PackManifest) -> None:
        """Write ``.eula_accepted`` JSON file to *pack_dir*."""
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"

        acceptance = {
            "accepted_at": datetime.now(tz=UTC).isoformat(),
            "user": user,
            "eula_sha256": manifest.eula_sha256,
            "pack_version": manifest.version,
            "installer_version": self._installer_version,
        }
        accepted_path = pack_dir / ".eula_accepted"
        accepted_path.write_text(json.dumps(acceptance, indent=2), encoding="utf-8")

    def _read_manifest(self, pack_dir: Path) -> PackManifest:
        """Read and validate ``pack_manifest.json`` from *pack_dir*.

        Raises
        ------
        PackInstallError
            If the file is missing or invalid.
        """
        manifest_path = pack_dir / "pack_manifest.json"
        if not manifest_path.exists():
            raise PackInstallError(f"pack_manifest.json not found in {pack_dir}.")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            return PackManifest.model_validate(raw)
        except Exception as exc:
            raise PackInstallError(f"Invalid pack_manifest.json in {pack_dir}: {exc}") from exc

    @staticmethod
    def _split_qid(qualified_id: str) -> tuple[str, str]:
        """Split ``namespace/pack_id`` into a ``(namespace, pack_id)`` tuple.

        Raises
        ------
        PackInstallError
            If *qualified_id* is not in the expected format.
        """
        parts = qualified_id.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise PackInstallError(
                f"qualified_id must be in 'namespace/pack_id' format, got {qualified_id!r}."
            )
        return parts[0], parts[1]
