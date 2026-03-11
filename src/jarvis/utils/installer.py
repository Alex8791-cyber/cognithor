"""Package installer abstraction with uv fast-path.

Provides detection and command generation for both ``uv`` and ``pip``.
When ``uv`` is available it is preferred (10x faster installs).
Falls back transparently to ``pip`` when ``uv`` is not present.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class InstallerBackend(StrEnum):
    """Which package manager to use."""

    UV = "uv"
    PIP = "pip"


@dataclass(frozen=True)
class InstallerInfo:
    """Detected installer information."""

    backend: InstallerBackend
    path: str
    version: str

    @property
    def is_uv(self) -> bool:
        return self.backend == InstallerBackend.UV

    def to_dict(self) -> dict:
        return {
            "backend": self.backend.value,
            "path": self.path,
            "version": self.version,
        }


def detect_uv() -> InstallerInfo | None:
    """Detect ``uv`` installation and return info, or *None*."""
    uv_path = shutil.which("uv")
    if uv_path is None:
        return None
    try:
        proc = subprocess.run(
            [uv_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            # Output: "uv 0.6.x" or similar
            version = proc.stdout.strip().removeprefix("uv").strip()
            return InstallerInfo(
                backend=InstallerBackend.UV,
                path=uv_path,
                version=version or "unknown",
            )
    except Exception:
        pass  # Cleanup — uv detection failure is non-critical
    return None


def detect_pip() -> InstallerInfo | None:
    """Detect ``pip`` availability and return info, or *None*."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            # "pip 24.0 from /path ..."
            parts = proc.stdout.strip().split()
            version = parts[1] if len(parts) >= 2 else "unknown"
            return InstallerInfo(
                backend=InstallerBackend.PIP,
                path=f"{sys.executable} -m pip",
                version=version,
            )
    except Exception:
        pass  # Cleanup — pip detection failure is non-critical
    return None


def detect_installer(*, prefer_uv: bool = True) -> InstallerInfo | None:
    """Detect the best available installer.

    When *prefer_uv* is ``True`` (default) and ``uv`` is found, it is
    returned.  Otherwise falls back to ``pip``.
    """
    if prefer_uv:
        uv = detect_uv()
        if uv is not None:
            return uv
    return detect_pip()


# ── Command builders ─────────────────────────────────────────────────────


def build_install_command(
    info: InstallerInfo,
    project_dir: str | Path,
    extras: str = "all",
    *,
    editable: bool = True,
    quiet: bool = True,
) -> list[str]:
    """Build the install command list for the given backend.

    Returns a list suitable for ``subprocess.run()``.
    """
    project = str(project_dir)
    spec = f".[{extras}]" if extras else "."
    if not editable:
        spec = spec  # non-editable keeps the same spec with uv pip install

    if info.is_uv:
        cmd = [info.path, "pip", "install"]
        if editable:
            cmd.extend(["-e", spec])
        else:
            cmd.append(spec)
        if quiet:
            cmd.append("--quiet")
        # uv uses --python to target the right interpreter
        cmd.extend(["--python", sys.executable])
    else:
        cmd = [sys.executable, "-m", "pip", "install"]
        if editable:
            cmd.extend(["-e", spec])
        else:
            cmd.append(spec)
        if quiet:
            cmd.extend(["--quiet", "--disable-pip-version-check"])

    return cmd


def build_sync_command(
    info: InstallerInfo,
    project_dir: str | Path,
    extras: str = "all",
) -> list[str] | None:
    """Build a ``uv sync`` command if the backend is uv.

    Returns *None* for pip (pip has no equivalent sync command).
    """
    if not info.is_uv:
        return None
    cmd = [info.path, "sync"]
    if extras:
        for extra in extras.split(","):
            extra = extra.strip()
            if extra:
                cmd.extend(["--extra", extra])
    cmd.extend(["--python", sys.executable])
    return cmd


def run_install(
    project_dir: str | Path,
    extras: str = "all",
    *,
    prefer_uv: bool = True,
    editable: bool = True,
    quiet: bool = True,
    timeout: int = 600,
) -> tuple[bool, InstallerInfo | None, str]:
    """Run package installation with automatic backend selection.

    Returns ``(success, installer_info, output_or_error)``.
    """
    info = detect_installer(prefer_uv=prefer_uv)
    if info is None:
        return False, None, "Neither uv nor pip found"

    cmd = build_install_command(
        info,
        project_dir,
        extras,
        editable=editable,
        quiet=quiet,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_dir),
        )
        if proc.returncode == 0:
            return True, info, proc.stdout.strip()
        return False, info, proc.stderr.strip()[-500:] if proc.stderr else ""
    except subprocess.TimeoutExpired:
        return False, info, f"Installation timed out after {timeout}s"
    except Exception as e:
        return False, info, str(e)
