"""WebSocket bearer-token loading + creation (Sprint-27 H3).

Single source of truth for the PSK that ``cognithor agent ws``
requires from clients. The token lives at
``~/.cognithor/auth.token`` (override via the
``COGNITHOR_HOME`` environment variable, the same convention the
rest of the codebase uses). On first run the file is generated
with 32 bytes of cryptographically random hex (64 hex chars) and
mode ``0o600`` so other users on a shared workstation cannot
read it. Best-effort on Windows where ``chmod`` is a no-op for
the file-permission bit (the file still resides under the
user-profile, which is access-controlled by the OS).

The token is **the only auth surface for the local WS server**.
Lose it = compromise the surface. The companion
``docs/superpowers/plans/2026-05-04-sprint27-ide-integration-decisions.md``
H3 hardening explicitly forbids unauthenticated binds.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from pathlib import Path

# 32 bytes → 64 hex chars. Conservative entropy for a localhost
# PSK that lives in a file on the same machine the server runs
# on; rotation is a follow-up.
_TOKEN_BYTES = 32


def _home_dir() -> Path:
    """Resolve the cognithor home directory (matches config.py convention)."""

    override = os.environ.get("COGNITHOR_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cognithor"


def auth_token_path() -> Path:
    """Where the token lives on disk."""

    return _home_dir() / "auth.token"


def load_or_create_token(*, path: Path | None = None) -> str:
    """Return the bearer token, generating it on first call.

    Returns the existing token verbatim if the file exists and
    contains a non-empty single-line value (with whitespace
    stripped). Otherwise generates a new 64-char hex token,
    writes it to disk with mode 0o600, and returns it.
    """

    target = path or auth_token_path()
    if target.exists():
        existing = target.read_text(encoding="utf-8").strip()
        if existing:
            return existing

    target.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(_TOKEN_BYTES)
    target.write_text(token + "\n", encoding="utf-8")
    # Windows: chmod is a best-effort no-op for the user permission
    # bit; the file still lives under the user-profile so it inherits
    # NTFS ACLs anyway.
    with contextlib.suppress(OSError, NotImplementedError):
        target.chmod(0o600)
    return token
