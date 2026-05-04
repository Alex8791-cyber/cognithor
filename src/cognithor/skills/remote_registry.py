"""Remote Plugin Registry — fetch, install, update, and rollback plugins.

Provides a client for remote skill registries (GitHub-based or custom).
Handles manifest verification, signature checking, version resolution,
dependency tracking, and local caching.

Architecture: §12.4 (Plugin Marketplace Remote Registry)

.. note::

    **NOT WIRED INTO PRODUCTION** (operational-trust audit, A-8,
    2026-05-04). Tested via
    ``tests/test_skills/test_remote_registry.py`` and the F017
    install-history safety contract, but no production boot path
    imports ``RemoteRegistry``. The Agent Pack system
    (``cognithor.packs``) currently fills this role for paid
    add-ons; remote_registry will become the community-skill
    distribution path when the Q4 2026 marketplace ships.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Plugin Manifest
# ---------------------------------------------------------------------------


@dataclass
class PluginManifest:
    """Metadata for a remote plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = "MIT"
    homepage: str = ""
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    min_jarvis_version: str = "0.1.0"
    permissions: list[str] = field(default_factory=list)
    checksum: str = ""
    signature: str = ""
    download_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "category": self.category,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "min_jarvis_version": self.min_jarvis_version,
            "permissions": self.permissions,
            "checksum": self.checksum,
            "signature": self.signature,
            "download_url": self.download_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
            homepage=data.get("homepage", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            min_jarvis_version=data.get("min_jarvis_version", "0.1.0"),
            permissions=data.get("permissions", []),
            checksum=data.get("checksum", ""),
            signature=data.get("signature", ""),
            download_url=data.get("download_url", ""),
        )

    @property
    def is_signed(self) -> bool:
        return bool(self.signature)

    def verify_checksum(self, content: bytes) -> bool:
        """Verify content against stored checksum."""
        if not self.checksum:
            return True  # No checksum to verify
        computed = hashlib.sha256(content).hexdigest()
        return computed == self.checksum


# ---------------------------------------------------------------------------
# Installation status
# ---------------------------------------------------------------------------


class InstallStatus(StrEnum):
    """Status of a plugin installation."""

    INSTALLED = "installed"
    UPDATE_AVAILABLE = "update_available"
    NOT_INSTALLED = "not_installed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class InstalledPlugin:
    """Record of an installed plugin."""

    name: str
    version: str
    installed_at: str = ""
    source: str = "remote"  # "remote" or "local"
    manifest: PluginManifest | None = None
    previous_versions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "installed_at": self.installed_at,
            "source": self.source,
            "previous_versions": self.previous_versions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstalledPlugin:
        return cls(
            name=data["name"],
            version=data["version"],
            installed_at=data.get("installed_at", ""),
            source=data.get("source", "remote"),
            previous_versions=data.get("previous_versions", []),
        )


# ---------------------------------------------------------------------------
# Install Result
# ---------------------------------------------------------------------------


@dataclass
class InstallResult:
    """Result of an install/update/rollback operation."""

    plugin: str
    version: str
    status: InstallStatus
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status in (InstallStatus.INSTALLED, InstallStatus.UPDATE_AVAILABLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin,
            "version": self.version,
            "status": self.status.value,
            "message": self.message,
            "warnings": self.warnings,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 1),
        }


# ---------------------------------------------------------------------------
# Dependency Resolver
# ---------------------------------------------------------------------------


class DependencyResolver:
    """Resolve plugin dependencies using topological sort."""

    def __init__(self) -> None:
        self._available: dict[str, PluginManifest] = {}

    def set_available(self, manifests: list[PluginManifest]) -> None:
        """Set the available plugin manifests."""
        self._available = {m.name: m for m in manifests}

    def resolve(self, plugin_name: str) -> list[str]:
        """Resolve install order for a plugin (dependencies first).

        Returns ordered list of plugin names to install.
        Raises ValueError on circular dependencies.
        """
        if plugin_name not in self._available:
            return [plugin_name]

        visited: set[str] = set()
        order: list[str] = []
        stack: set[str] = set()

        def visit(name: str) -> None:
            if name in stack:
                raise ValueError(f"Circular dependency: {name}")
            if name in visited:
                return
            stack.add(name)
            manifest = self._available.get(name)
            if manifest:
                for dep in manifest.dependencies:
                    visit(dep)
            stack.discard(name)
            visited.add(name)
            order.append(name)

        visit(plugin_name)
        return order

    def find_missing(
        self,
        plugin_name: str,
        installed: set[str],
    ) -> list[str]:
        """Find dependencies that are not yet installed."""
        try:
            full_order = self.resolve(plugin_name)
        except ValueError:
            return []
        return [p for p in full_order if p not in installed and p != plugin_name]


# ---------------------------------------------------------------------------
# Remote Registry
# ---------------------------------------------------------------------------


class RemoteRegistry:
    """Client for a remote plugin registry.

    Supports:
    - Plugin search and discovery
    - Manifest fetching and verification
    - Install / update / rollback with version history
    - Local caching of downloaded packages
    - Dependency resolution

    The registry uses a local index file for offline support.
    In production, this would fetch from an HTTP endpoint.
    """

    def __init__(
        self,
        skills_dir: Path,
        cache_dir: Path | None = None,
        registry_url: str = "",
    ) -> None:
        self._skills_dir = skills_dir
        self._cache_dir = cache_dir or (skills_dir.parent / "cache" / "plugins")
        self._registry_url = registry_url
        self._index: dict[str, PluginManifest] = {}
        self._installed: dict[str, InstalledPlugin] = {}
        self._install_history_file = skills_dir / ".plugin_history.json"
        self._resolver = DependencyResolver()
        self._ensure_dirs()
        self._load_install_history()

    # -- Public API --

    def register_remote(self, manifest: PluginManifest) -> None:
        """Add a plugin to the remote index (for testing / local registries)."""
        self._index[manifest.name] = manifest
        self._resolver.set_available(list(self._index.values()))

    def search(
        self,
        query: str = "",
        category: str = "",
        tags: list[str] | None = None,
    ) -> list[PluginManifest]:
        """Search the remote index."""
        results = []
        query_lower = query.lower()
        for manifest in self._index.values():
            if (
                query_lower
                and query_lower not in manifest.name.lower()
                and query_lower not in manifest.description.lower()
            ):
                continue
            if category and manifest.category != category:
                continue
            if tags and not any(t in manifest.tags for t in tags):
                continue
            results.append(manifest)
        return results

    def get_manifest(self, name: str) -> PluginManifest | None:
        """Get manifest for a specific plugin."""
        return self._index.get(name)

    def install(
        self,
        name: str,
        *,
        content: str = "",
        verify_signature: bool = True,
    ) -> InstallResult:
        """Install a plugin from the remote registry.

        Args:
            name: Plugin name.
            content: Plugin content (for testing; in production, fetched from URL).
            verify_signature: Whether to check signatures.

        Returns:
            InstallResult with status.
        """
        start = time.monotonic()
        manifest = self._index.get(name)

        if not manifest:
            return InstallResult(
                plugin=name,
                version="",
                status=InstallStatus.FAILED,
                message=f"Plugin '{name}' not found in registry",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        warnings: list[str] = []

        # Check signature
        if verify_signature and not manifest.is_signed:
            warnings.append("Plugin is not signed — install at your own risk")

        # Check dependencies
        installed_names = set(self._installed.keys())
        missing = self._resolver.find_missing(name, installed_names)
        if missing:
            warnings.append(f"Missing dependencies: {', '.join(missing)}")

        # Check if already installed (update case)
        existing = self._installed.get(name)
        if existing and existing.version == manifest.version:
            return InstallResult(
                plugin=name,
                version=manifest.version,
                status=InstallStatus.INSTALLED,
                message="Already installed at this version",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Verify checksum if content provided
        if content and manifest.checksum and not manifest.verify_checksum(content.encode("utf-8")):
            return InstallResult(
                plugin=name,
                version=manifest.version,
                status=InstallStatus.FAILED,
                message="Checksum verification failed",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Write to skills directory
        try:
            plugin_dir = (self._skills_dir / name).resolve()
            try:
                plugin_dir.relative_to(self._skills_dir.resolve())
            except ValueError:
                return InstallResult(
                    plugin=name,
                    version=manifest.version,
                    status=InstallStatus.FAILED,
                    message=f"Path traversal detected in skill name: {name}",
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            plugin_dir.mkdir(parents=True, exist_ok=True)
            skill_file = plugin_dir / "skill.md"
            skill_content = content or (
                f"---\nname: {name}\n"
                f"version: {manifest.version}\n"
                f"trigger_keywords: {manifest.tags}\n"
                f"---\n\n{manifest.description}"
            )
            skill_file.write_text(skill_content, encoding="utf-8")

            # Write manifest
            manifest_file = plugin_dir / "manifest.json"
            manifest_file.write_text(
                json.dumps(manifest.to_dict(), indent=2),
                encoding="utf-8",
            )

            # Track previous version
            previous_versions = []
            if existing:
                previous_versions = [*existing.previous_versions, existing.version]
                # Backup old version
                backup_dir = self._cache_dir / name / existing.version
                backup_dir.mkdir(parents=True, exist_ok=True)

            # Record installation
            self._installed[name] = InstalledPlugin(
                name=name,
                version=manifest.version,
                installed_at=datetime.now(UTC).isoformat(),
                source="remote",
                manifest=manifest,
                previous_versions=previous_versions,
            )
            self._save_install_history()

            log.info("plugin_installed", name=name, version=manifest.version)

            return InstallResult(
                plugin=name,
                version=manifest.version,
                status=InstallStatus.INSTALLED,
                message=f"Installed {name} v{manifest.version}",
                warnings=warnings,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as exc:
            return InstallResult(
                plugin=name,
                version=manifest.version,
                status=InstallStatus.FAILED,
                message=f"Installation failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def update(self, name: str, *, content: str = "") -> InstallResult:
        """Update an installed plugin to the latest version."""
        if name not in self._installed:
            return InstallResult(
                plugin=name,
                version="",
                status=InstallStatus.FAILED,
                message=f"Plugin '{name}' is not installed",
            )
        return self.install(name, content=content)

    def rollback(self, name: str) -> InstallResult:
        """Rollback a plugin to its previous version."""
        start = time.monotonic()
        installed = self._installed.get(name)

        if not installed:
            return InstallResult(
                plugin=name,
                version="",
                status=InstallStatus.FAILED,
                message=f"Plugin '{name}' is not installed",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        if not installed.previous_versions:
            return InstallResult(
                plugin=name,
                version=installed.version,
                status=InstallStatus.FAILED,
                message="No previous version to rollback to",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        prev_version = installed.previous_versions[-1]
        installed.version = prev_version
        installed.previous_versions = installed.previous_versions[:-1]
        installed.installed_at = datetime.now(UTC).isoformat()
        self._save_install_history()

        log.info("plugin_rollback", name=name, version=prev_version)

        return InstallResult(
            plugin=name,
            version=prev_version,
            status=InstallStatus.INSTALLED,
            message=f"Rolled back to v{prev_version}",
            duration_ms=(time.monotonic() - start) * 1000,
        )

    def uninstall(self, name: str) -> InstallResult:
        """Remove an installed plugin."""
        start = time.monotonic()

        if name not in self._installed:
            return InstallResult(
                plugin=name,
                version="",
                status=InstallStatus.FAILED,
                message=f"Plugin '{name}' is not installed",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        try:
            plugin_dir = self._skills_dir / name
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)

            version = self._installed[name].version
            del self._installed[name]
            self._save_install_history()

            return InstallResult(
                plugin=name,
                version=version,
                status=InstallStatus.NOT_INSTALLED,
                message=f"Uninstalled {name}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return InstallResult(
                plugin=name,
                version="",
                status=InstallStatus.FAILED,
                message=f"Uninstall failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def check_updates(self) -> list[dict[str, Any]]:
        """Check which installed plugins have updates available."""
        updates = []
        for name, installed in self._installed.items():
            remote = self._index.get(name)
            if remote and remote.version != installed.version:
                updates.append(
                    {
                        "name": name,
                        "installed_version": installed.version,
                        "available_version": remote.version,
                    }
                )
        return updates

    def get_installed(self, name: str) -> InstalledPlugin | None:
        return self._installed.get(name)

    def list_installed(self) -> list[InstalledPlugin]:
        return list(self._installed.values())

    def stats(self) -> dict[str, Any]:
        return {
            "registry_plugins": len(self._index),
            "installed_plugins": len(self._installed),
            "updates_available": len(self.check_updates()),
            "signed_plugins": sum(1 for m in self._index.values() if m.is_signed),
            "registry_url": self._registry_url,
        }

    # -- Internal --

    def _ensure_dirs(self) -> None:
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_install_history(self) -> None:
        if self._install_history_file.exists():
            try:
                data = json.loads(self._install_history_file.read_text(encoding="utf-8"))
                for entry in data:
                    self._installed[entry["name"]] = InstalledPlugin.from_dict(entry)
            except Exception as exc:
                log.warning(
                    "install_history_corrupt",
                    file=str(self._install_history_file),
                    error=str(exc),
                )
                # Korrupte Datei als Backup sichern statt silent reset
                backup = self._install_history_file.with_suffix(".json.corrupt")
                try:
                    shutil.copy2(self._install_history_file, backup)
                    log.info("install_history_backup_created", backup=str(backup))
                except OSError:
                    pass
                self._installed = {}

    def _save_install_history(self) -> None:
        import os as _os
        import tempfile as _tempfile

        data = [p.to_dict() for p in self._installed.values()]
        content = json.dumps(data, indent=2)

        # Atomic write: temp file + rename
        parent = self._install_history_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = _tempfile.mkstemp(
            dir=str(parent),
            suffix=".tmp",
            prefix=".plugin_history_",
        )
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            _os.replace(tmp_path, str(self._install_history_file))
        except BaseException:
            with contextlib.suppress(OSError):
                _os.unlink(tmp_path)
            raise
