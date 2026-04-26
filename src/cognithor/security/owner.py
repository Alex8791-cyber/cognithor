"""Owner-token identification for Trace-UI gating.

The owner is identified by a string match between the bootstrap-token's
user identifier and either:
  - The `COGNITHOR_OWNER_USER_ID` environment variable (preferred), or
  - A fallback derived from `pyproject.toml [project] authors[0].name`.

The fallback exists so single-user dev installs work out of the box without
configuration. Production deployments should set the env var explicitly.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path


class OwnerRequiredError(Exception):
    """Raised when an action requires an owner token but the supplied token
    does not match. FastAPI handlers translate this to HTTP 403."""


_PYPROJECT_FALLBACK_DEFAULT = "Alexander Söllner"


@lru_cache(maxsize=1)
def _pyproject_owner_fallback() -> str:
    """Read pyproject.toml [project] authors[0].name once per process."""
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
                        return name
            except (OSError, tomllib.TOMLDecodeError):
                pass
            break
    return _PYPROJECT_FALLBACK_DEFAULT


def _expected_owner() -> str:
    env = os.environ.get("COGNITHOR_OWNER_USER_ID")
    if env:
        return env
    return _pyproject_owner_fallback()


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
