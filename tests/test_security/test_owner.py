"""require_owner — owner identification for Trace-UI access gating."""

from __future__ import annotations

import pytest

from cognithor.security.owner import OwnerRequiredError, is_owner_token, require_owner


def test_owner_default_reads_pyproject_authors_when_env_unset(monkeypatch) -> None:
    """When COGNITHOR_OWNER_USER_ID env unset, default reads pyproject.toml."""
    monkeypatch.delenv("COGNITHOR_OWNER_USER_ID", raising=False)
    # Inferred default == "Alexander Söllner" (from pyproject.toml authors[0].name).
    assert is_owner_token("Alexander Söllner") is True
    assert is_owner_token("Anonymous") is False


def test_owner_explicit_env_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "test-owner-42")
    assert is_owner_token("test-owner-42") is True
    assert is_owner_token("Alexander Söllner") is False


def test_require_owner_returns_silently_for_owner(monkeypatch) -> None:
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "test-owner")
    require_owner("test-owner")  # must not raise


def test_require_owner_raises_for_non_owner(monkeypatch) -> None:
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "real-owner")
    with pytest.raises(OwnerRequiredError) as exc:
        require_owner("guest")
    assert "owner_only" in str(exc.value)


def test_require_owner_raises_for_empty_token(monkeypatch) -> None:
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner")
    with pytest.raises(OwnerRequiredError):
        require_owner("")
    with pytest.raises(OwnerRequiredError):
        require_owner(None)  # type: ignore[arg-type]
