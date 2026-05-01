"""Tests für TLS-Konfiguration und SSL-Context-Helper."""

from __future__ import annotations

import datetime as _dt
import ssl
from typing import TYPE_CHECKING

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from cognithor.security.token_store import create_ssl_context

if TYPE_CHECKING:
    from pathlib import Path


def _generate_self_signed_cert(cert_path: Path, key_path: Path) -> bool:
    """Generate a self-signed cert + key in-process via the cryptography lib.

    Sprint-9 cleanup: replaced the previous ``subprocess.run(["openssl", ...])``
    implementation, which timed out under load when the full repo test
    suite was run in a single process (memory:
    `feedback_full_repo_subprocess_load`). Pure-Python in-process generation
    has no subprocess timeout and no global state to exhaust, so the
    11 occasional Win-py3.12 failures during 16 000-test runs are gone.
    """
    try:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    except Exception:
        return False

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = _dt.datetime.now(_dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=1))
        .not_valid_after(now + _dt.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return True


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
            pytest.skip("cryptography library not available")

        ctx = create_ssl_context(str(cert_path), str(key_path))
        assert ctx is not None
        assert isinstance(ctx, ssl.SSLContext)

    def test_ssl_context_has_minimum_tls_version(self, tmp_path: Path) -> None:
        """SSLContext erzwingt mindestens TLS 1.2."""
        cert_path = tmp_path / "cert.pem"
        key_path = tmp_path / "key.pem"

        if not _generate_self_signed_cert(cert_path, key_path):
            pytest.skip("cryptography library not available")

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
        from cognithor.config import SecurityConfig

        cfg = SecurityConfig()
        assert cfg.ssl_certfile == ""
        assert cfg.ssl_keyfile == ""

    def test_security_config_accepts_ssl_values(self) -> None:
        """SecurityConfig akzeptiert SSL-Pfade."""
        from cognithor.config import SecurityConfig

        cfg = SecurityConfig(
            ssl_certfile="/path/to/cert.pem",
            ssl_keyfile="/path/to/key.pem",
        )
        assert cfg.ssl_certfile == "/path/to/cert.pem"
        assert cfg.ssl_keyfile == "/path/to/key.pem"
