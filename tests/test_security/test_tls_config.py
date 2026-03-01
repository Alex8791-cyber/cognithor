"""Tests für TLS-Konfiguration und SSL-Context-Helper."""

from __future__ import annotations

import logging
import ssl
import subprocess
import tempfile
from pathlib import Path

import pytest

from jarvis.security.token_store import create_ssl_context


def _generate_self_signed_cert(cert_path: Path, key_path: Path) -> bool:
    """Generiert ein selbstsigniertes Zertifikat mit openssl (falls verfügbar)."""
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path),
                "-out", str(cert_path),
                "-days", "1",
                "-nodes",
                "-subj", "/CN=localhost",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


class TestCreateSSLContext:
    """Tests für create_ssl_context()."""

    def test_missing_files_returns_none(self, tmp_path: Path) -> None:
        """Fehlende Zertifikate → None."""
        result = create_ssl_context(
            str(tmp_path / "nonexistent.pem"),
            str(tmp_path / "nonexistent.key"),
        )
        assert result is None

    def test_empty_strings_returns_none(self) -> None:
        """Leere Strings → None."""
        result = create_ssl_context("", "")
        assert result is None

    def test_no_config_returns_none(self) -> None:
        """Keine Konfiguration → None."""
        result = create_ssl_context("", "")
        assert result is None

    def test_valid_certs_returns_ssl_context(self, tmp_path: Path) -> None:
        """Mit gültigem Zertifikat → SSLContext."""
        cert_path = tmp_path / "cert.pem"
        key_path = tmp_path / "key.pem"

        if not _generate_self_signed_cert(cert_path, key_path):
            pytest.skip("openssl nicht verfügbar")

        ctx = create_ssl_context(str(cert_path), str(key_path))
        assert ctx is not None
        assert isinstance(ctx, ssl.SSLContext)

    def test_ssl_context_has_minimum_tls_version(self, tmp_path: Path) -> None:
        """SSLContext erzwingt mindestens TLS 1.2."""
        cert_path = tmp_path / "cert.pem"
        key_path = tmp_path / "key.pem"

        if not _generate_self_signed_cert(cert_path, key_path):
            pytest.skip("openssl nicht verfügbar")

        ctx = create_ssl_context(str(cert_path), str(key_path))
        assert ctx is not None
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_only_certfile_returns_none(self, tmp_path: Path) -> None:
        """Nur Certfile ohne Keyfile → None."""
        cert_path = tmp_path / "cert.pem"
        cert_path.write_text("dummy")
        result = create_ssl_context(str(cert_path), "")
        assert result is None


class TestTLSSecurityConfig:
    """Tests für SecurityConfig TLS-Felder."""

    def test_security_config_has_ssl_fields(self) -> None:
        """SecurityConfig hat ssl_certfile und ssl_keyfile Felder."""
        from jarvis.config import SecurityConfig

        cfg = SecurityConfig()
        assert cfg.ssl_certfile == ""
        assert cfg.ssl_keyfile == ""

    def test_security_config_accepts_ssl_values(self) -> None:
        """SecurityConfig akzeptiert SSL-Pfade."""
        from jarvis.config import SecurityConfig

        cfg = SecurityConfig(
            ssl_certfile="/path/to/cert.pem",
            ssl_keyfile="/path/to/key.pem",
        )
        assert cfg.ssl_certfile == "/path/to/cert.pem"
        assert cfg.ssl_keyfile == "/path/to/key.pem"
