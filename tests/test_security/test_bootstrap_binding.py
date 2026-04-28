"""Regression tests for bootstrap token binding (GHSA-cognithor-001).

The old ``/api/v1/bootstrap`` endpoint leaked the API token without auth.
After the fix the token is injected via ``<meta name="cognithor-token">``
in index.html at startup.  These tests assert:

1. The bootstrap endpoint (if still present) is consumed-once / protected.
2. ``__main__.py`` uses meta-tag injection, not an HTTP endpoint.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "cognithor"


class TestBootstrapEndpointProtection:
    """No unprotected route may leak the token."""

    def test_token_not_in_unprotected_routes(self) -> None:
        # `config_routes` ist seit 2026-04-29 ein Paket; Helfer wandern
        # schrittweise in Sub-Module
        # (siehe `docs/superpowers/plans/2026-04-29-config-routes-split.md`).
        # Scan ueber alle `.py`-Dateien im Paket-Verzeichnis.
        pkg_dir = _SRC / "channels" / "config_routes"
        if pkg_dir.is_dir():
            source_files = sorted(pkg_dir.glob("*.py"))
        else:
            source_files = [pkg_dir.with_suffix(".py")]
        source = "\n".join(p.read_text(encoding="utf-8") for p in source_files)
        # "bootstrap" must NOT appear in config_routes at all.
        # If it ever does, it must be behind ``dependencies=deps`` or
        # return "consumed" (the legacy already-consumed guard).
        matches = [line for line in source.splitlines() if "bootstrap" in line.lower()]
        for line in matches:
            assert (
                "dependencies" in line
                or "consumed" in line.lower()
                or line.lstrip().startswith("#")
                or line.lstrip().startswith('"""')
            ), f"Unprotected 'bootstrap' reference in config_routes: {line!r}"


class TestMetaTagInjection:
    """__main__.py must inject the token via <meta> tag, not an endpoint."""

    def test_main_uses_meta_tag_injection(self) -> None:
        source = (_SRC / "__main__.py").read_text(encoding="utf-8")

        # The meta-tag name used by the Flutter frontend.
        assert "cognithor-token" in source, "__main__.py must reference 'cognithor-token' meta tag"

        # Token injection happens via string replacement in HTML.
        assert ".replace(" in source or "replace(" in source, (
            "__main__.py must inject the token via string replacement"
        )

        # The meta tag pattern must exist.
        assert re.search(
            r'<meta\s+name="cognithor-token"',
            source,
        ), '__main__.py must contain the <meta name="cognithor-token"> tag'

        # The old endpoint may still exist (consumed-once guard) but the
        # primary mechanism MUST be meta-tag injection, not an endpoint.
        # Verify the comment referencing GHSA-cognithor-001 is present.
        assert "GHSA-cognithor-001" in source, (
            "__main__.py must reference GHSA-cognithor-001 security advisory"
        )
