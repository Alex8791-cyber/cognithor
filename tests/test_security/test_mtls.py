"""Tests for mTLS certificate generation and management."""

from __future__ import annotations

import ssl
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from jarvis.security.mtls import (
    ensure_mtls_certs,
    generate_ca,
    generate_client_cert,
    generate_server_cert,
)


class TestCertificateGeneration:
    """Test CA, server, and client certificate generation."""

    @pytest.fixture()
    def certs_dir(self, tmp_path):
        d = tmp_path / "certs"
        d.mkdir()
        return d

    def test_generate_ca(self, certs_dir):
        ca_key, ca_cert = generate_ca(certs_dir)
        assert (certs_dir / "ca.pem").exists()
        assert (certs_dir / "ca-key.pem").exists()
        # CA cert should be self-signed
        assert ca_cert.issuer == ca_cert.subject
        # Should have CA basic constraint
        bc = ca_cert.extensions.get_extension_for_class(x509.BasicConstraints)
        assert bc.value.ca is True

    def test_generate_server_cert(self, certs_dir):
        ca_key, ca_cert = generate_ca(certs_dir)
        generate_server_cert(ca_key, ca_cert, certs_dir)
        assert (certs_dir / "server.pem").exists()
        assert (certs_dir / "server-key.pem").exists()
        # Server cert should be signed by CA
        server_pem = (certs_dir / "server.pem").read_bytes()
        server_cert = x509.load_pem_x509_certificate(server_pem)
        assert server_cert.issuer == ca_cert.subject
        # Should have SAN with localhost
        san = server_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        dns_names = san.value.get_values_for_type(x509.DNSName)
        assert "localhost" in dns_names

    def test_generate_client_cert(self, certs_dir):
        ca_key, ca_cert = generate_ca(certs_dir)
        generate_client_cert(ca_key, ca_cert, certs_dir)
        assert (certs_dir / "client.pem").exists()
        assert (certs_dir / "client-key.pem").exists()
        client_pem = (certs_dir / "client.pem").read_bytes()
        client_cert = x509.load_pem_x509_certificate(client_pem)
        assert client_cert.issuer == ca_cert.subject
        # Should NOT be a CA
        bc = client_cert.extensions.get_extension_for_class(x509.BasicConstraints)
        assert bc.value.ca is False

    def test_ssl_context_validates_client(self, certs_dir):
        """Server SSL context requires and validates client certs."""
        ca_key, ca_cert = generate_ca(certs_dir)
        generate_server_cert(ca_key, ca_cert, certs_dir)
        generate_client_cert(ca_key, ca_cert, certs_dir)

        # Create server-side SSL context
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(
            certfile=str(certs_dir / "server.pem"),
            keyfile=str(certs_dir / "server-key.pem"),
        )
        ctx.load_verify_locations(cafile=str(certs_dir / "ca.pem"))
        ctx.verify_mode = ssl.CERT_REQUIRED
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_keys_are_valid(self, certs_dir):
        """Generated keys can be loaded successfully."""
        ca_key, _ = generate_ca(certs_dir)
        key_pem = (certs_dir / "ca-key.pem").read_bytes()
        loaded = load_pem_private_key(key_pem, password=None)
        assert loaded.key_size == 2048


class TestEnsureMtlsCerts:
    """Test the ensure_mtls_certs orchestrator."""

    def test_disabled_returns_none(self):
        """Returns None when mTLS is disabled."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.security.mtls.enabled = False
        assert ensure_mtls_certs(config) is None

    def test_no_config_returns_none(self):
        """Returns None when no config provided."""
        assert ensure_mtls_certs(None) is None

    def test_generates_all_certs(self, tmp_path):
        """Generates all certificates when none exist."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.security.mtls.enabled = True
        config.security.mtls.certs_dir = str(tmp_path / "certs")
        config.security.mtls.auto_generate = True

        result = ensure_mtls_certs(config)
        assert result is not None
        assert (result / "ca.pem").exists()
        assert (result / "server.pem").exists()
        assert (result / "client.pem").exists()
        assert (result / "client-key.pem").exists()

    def test_idempotent(self, tmp_path):
        """Does not regenerate if certs already exist."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.security.mtls.enabled = True
        config.security.mtls.certs_dir = str(tmp_path / "certs")
        config.security.mtls.auto_generate = True

        result1 = ensure_mtls_certs(config)
        # Get modification times
        ca_mtime = (result1 / "ca.pem").stat().st_mtime
        result2 = ensure_mtls_certs(config)
        # Should not have regenerated
        assert (result2 / "ca.pem").stat().st_mtime == ca_mtime

    def test_auto_generate_false_warns(self, tmp_path):
        """Returns None when certs missing and auto_generate is False."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.security.mtls.enabled = True
        config.security.mtls.certs_dir = str(tmp_path / "empty_certs")
        config.security.mtls.auto_generate = False

        assert ensure_mtls_certs(config) is None
