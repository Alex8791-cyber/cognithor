"""Owner-token identification for Trace-UI gating.

The owner is identified by a string match between the bootstrap-token's
user identifier and either:
  - The `COGNITHOR_OWNER_USER_ID` environment variable (preferred), or
  - A fallback derived from `pyproject.toml [project] authors[0].name`, or
  - As a last resort, a hardcoded `"Alexander Söllner"` (the original repo
    author).

The two fallback layers exist so single-user dev installs work out of the box
without configuration. **Production deployments must set
`COGNITHOR_OWNER_USER_ID` explicitly** — `check_owner_security_posture()`
returns the current source and lets the caller fail or warn. Startup code
that wires the Trace-UI calls it in production mode and refuses to bind
public-facing routes when the source is `"hardcoded_fallback"`.
"""

from __future__ import annotations

import os
import tomllib
from enum import Enum
from functools import lru_cache
from pathlib import Path


class OwnerSource(str, Enum):
    """Where the configured owner identity came from, in security order."""

    ENV = "env"  # explicit COGNITHOR_OWNER_USER_ID — strongest
    PYPROJECT = "pyproject"  # parsed from pyproject.toml authors[0].name
    HARDCODED_FALLBACK = "hardcoded_fallback"  # last-resort literal


class OwnerRequiredError(Exception):
    """Raised when an action requires an owner token but the supplied token
    does not match. FastAPI handlers translate this to HTTP 403."""


_PYPROJECT_FALLBACK_DEFAULT = "Alexander Söllner"


@lru_cache(maxsize=1)
def _pyproject_owner_fallback() -> tuple[str, bool]:
    """Read pyproject.toml [project] authors[0].name once per process.

    Returns `(name, was_loaded_from_pyproject)`. The boolean lets callers
    distinguish between an actual pyproject hit and the literal fallback.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            try:
                with candidate.open("rb") as fp:
                    data = tomllib.load(fp)
                authors = data.get("project", {}).get("authors", [])
                if authors and isinstance(authors[0], dict):
                    name = authors[0].get("name")
                    if isinstance(name, str) and name:
                        return (name, True)
            except (OSError, tomllib.TOMLDecodeError):
                pass
            break
    return (_PYPROJECT_FALLBACK_DEFAULT, False)


def _expected_owner_with_source() -> tuple[str, OwnerSource]:
    env = os.environ.get("COGNITHOR_OWNER_USER_ID")
    if env:
        return (env, OwnerSource.ENV)
    name, from_pyproject = _pyproject_owner_fallback()
    return (name, OwnerSource.PYPROJECT if from_pyproject else OwnerSource.HARDCODED_FALLBACK)


def _expected_owner() -> str:
    return _expected_owner_with_source()[0]


def check_owner_security_posture() -> tuple[OwnerSource, str]:
    """Return the current owner-identity source plus a human-readable message.

    Use this at startup to decide whether the deployment is sufficiently
    configured. ENV is the only safe value for production; PYPROJECT is fine
    for dev installs that ship the source distribution; HARDCODED_FALLBACK
    means **no config + no pyproject + no env** — a public-facing deployment
    in this state would let anyone with the literal author-name string
    impersonate the owner.
    """
    owner, source = _expected_owner_with_source()
    if source is OwnerSource.ENV:
        msg = "Owner from COGNITHOR_OWNER_USER_ID env var (production-grade)."
    elif source is OwnerSource.PYPROJECT:
        msg = (
            f"Owner from pyproject.toml authors[0].name ({owner!r}). "
            "OK for dev; for production set COGNITHOR_OWNER_USER_ID explicitly."
        )
    else:
        msg = (
            f"Owner from hardcoded fallback ({owner!r}) — pyproject.toml not "
            "found and no COGNITHOR_OWNER_USER_ID env var set. **Insecure for "
            "any non-localhost deployment.**"
        )
    return (source, msg)


def is_owner_token(token_user_id: str | None) -> bool:
    """Return True if `token_user_id` matches the configured owner."""
    if not token_user_id:
        return False
    return token_user_id == _expected_owner()


def require_owner(token_user_id: str | None) -> None:
    """Raise OwnerRequiredError if the supplied token is not the owner."""
    if not is_owner_token(token_user_id):
        raise OwnerRequiredError(
            "owner_only: token does not match the configured owner "
            "(set COGNITHOR_OWNER_USER_ID env var to override)."
        )
