"""Tests fuer F-014: Signal Webhook ohne Signatur-Verifizierung.

Prueft dass:
  - webhook_secret Parameter existiert
  - Bei gesetztem Secret: HMAC-SHA256 Signatur verifiziert wird
  - Bei gesetztem Secret: fehlende/falsche Signatur -> 403
  - Bei gesetztem Secret: korrekte Signatur -> 200
  - Ohne Secret: Requests weiterhin akzeptiert (Backward-Compatible)
  - Warnung geloggt wird wenn webhook_host != 127.0.0.1 und kein Secret
  - Source-Code HMAC-Verifizierung enthaelt
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.channels.signal import SignalChannel


class TestWebhookSecretParameter:
    """Prueft dass der webhook_secret Parameter existiert."""

    def test_constructor_accepts_webhook_secret(self) -> None:
        ch = SignalChannel(webhook_secret="my-secret")
        assert ch._webhook_secret == "my-secret"

    def test_default_webhook_secret_is_empty(self) -> None:
        ch = SignalChannel()
        assert ch._webhook_secret == ""

    def test_constructor_signature_has_webhook_secret(self) -> None:
        sig = inspect.signature(SignalChannel.__init__)
        assert "webhook_secret" in sig.parameters


class TestHMACVerification:
    """Prueft die HMAC-SHA256 Signatur-Verifizierung."""

    @pytest.mark.asyncio
    async def test_valid_signature_accepted(self) -> None:
        ch = SignalChannel(webhook_secret="test-secret-123")
        ch._running = True

        payload = {"envelope": {"sourceNumber": "+49123", "dataMessage": {"message": "hi"}}}
        raw_body = json.dumps(payload).encode()
        expected_sig = hmac.new(b"test-secret-123", raw_body, hashlib.sha256).hexdigest()

        request = MagicMock()
        request.read = AsyncMock(return_value=raw_body)
        request.headers = {"X-Webhook-Signature": expected_sig}

        with patch.object(ch, "_process_webhook_payload", new_callable=AsyncMock):
            response = await ch._handle_webhook(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected(self) -> None:
        ch = SignalChannel(webhook_secret="test-secret-123")

        payload = {"envelope": {"sourceNumber": "+49123"}}
        raw_body = json.dumps(payload).encode()

        request = MagicMock()
        request.read = AsyncMock(return_value=raw_body)
        request.headers = {"X-Webhook-Signature": "wrong-signature"}

        response = await ch._handle_webhook(request)
        assert response.status == 403

    @pytest.mark.asyncio
    async def test_missing_signature_rejected(self) -> None:
        ch = SignalChannel(webhook_secret="test-secret-123")

        request = MagicMock()
        request.read = AsyncMock(return_value=b'{"test": 1}')
        request.headers = {}

        response = await ch._handle_webhook(request)
        assert response.status == 403

    @pytest.mark.asyncio
    async def test_empty_signature_rejected(self) -> None:
        ch = SignalChannel(webhook_secret="test-secret-123")

        request = MagicMock()
        request.read = AsyncMock(return_value=b'{"test": 1}')
        request.headers = {"X-Webhook-Signature": ""}

        response = await ch._handle_webhook(request)
        assert response.status == 403


class TestWithoutSecret:
    """Prueft dass ohne Secret alles weiterhin funktioniert."""

    @pytest.mark.asyncio
    async def test_no_secret_accepts_any_request(self) -> None:
        ch = SignalChannel(webhook_secret="")
        ch._running = True

        payload = {"envelope": {"sourceNumber": "+49123", "dataMessage": {"message": "hi"}}}

        request = MagicMock()
        request.json = AsyncMock(return_value=payload)
        # Kein Signatur-Header noetig
        request.headers = {}

        with patch.object(ch, "_process_webhook_payload", new_callable=AsyncMock):
            response = await ch._handle_webhook(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_no_secret_invalid_json_returns_400(self) -> None:
        ch = SignalChannel(webhook_secret="")

        request = MagicMock()
        request.json = AsyncMock(side_effect=ValueError("bad json"))

        response = await ch._handle_webhook(request)
        assert response.status == 400


class TestSecretWithInvalidJSON:
    """Prueft dass bei Secret + valider Signatur aber invalidem JSON 400 kommt."""

    @pytest.mark.asyncio
    async def test_valid_sig_invalid_json(self) -> None:
        ch = SignalChannel(webhook_secret="secret")

        raw_body = b"not valid json {"
        sig = hmac.new(b"secret", raw_body, hashlib.sha256).hexdigest()

        request = MagicMock()
        request.read = AsyncMock(return_value=raw_body)
        request.headers = {"X-Webhook-Signature": sig}

        response = await ch._handle_webhook(request)
        assert response.status == 400


class TestExposedWebhookWarning:
    """Deep-PR3 update: non-localhost bind without webhook_secret is now
    a hard ``RuntimeError`` (fail-closed) rather than a passive WARNING.

    Previously the channel logged a warning and proceeded to bind anyway,
    accepting unauthenticated POSTs from any host. The new contract:
    refuse to start unless the operator either binds to 127.0.0.1 or sets
    a HMAC secret.
    """

    @pytest.mark.asyncio
    async def test_warning_on_exposed_without_secret(self) -> None:
        ch = SignalChannel(
            webhook_host="0.0.0.0",
            webhook_secret="",
        )
        try:
            from aiohttp import web  # noqa: F401
        except ImportError:
            pytest.skip("aiohttp nicht installiert")
        with pytest.raises(RuntimeError, match="webhook_secret"):
            await ch._setup_webhook()

    @pytest.mark.asyncio
    async def test_no_warning_on_localhost(self) -> None:
        ch = SignalChannel(
            webhook_host="127.0.0.1",
            webhook_secret="",
        )
        with patch("cognithor.channels.signal.logger") as mock_logger:
            try:
                from aiohttp import web  # noqa: F401
            except ImportError:
                pytest.skip("aiohttp nicht installiert")
            with patch("aiohttp.web.AppRunner") as mock_runner:
                mock_instance = AsyncMock()
                mock_runner.return_value = mock_instance
                with patch("aiohttp.web.TCPSite") as mock_site:
                    mock_site_instance = AsyncMock()
                    mock_site.return_value = mock_site_instance
                    ch._http = AsyncMock()
                    ch._http.post = AsyncMock()
                    await ch._setup_webhook()
            # Kein Warning ueber fehlenden Secret
            for call in mock_logger.warning.call_args_list:
                assert "webhook_secret" not in str(call)

    @pytest.mark.asyncio
    async def test_no_warning_with_secret(self) -> None:
        ch = SignalChannel(
            webhook_host="0.0.0.0",
            webhook_secret="my-secret",
        )
        with patch("cognithor.channels.signal.logger") as mock_logger:
            try:
                from aiohttp import web  # noqa: F401
            except ImportError:
                pytest.skip("aiohttp nicht installiert")
            with patch("aiohttp.web.AppRunner") as mock_runner:
                mock_instance = AsyncMock()
                mock_runner.return_value = mock_instance
                with patch("aiohttp.web.TCPSite") as mock_site:
                    mock_site_instance = AsyncMock()
                    mock_site.return_value = mock_site_instance
                    ch._http = AsyncMock()
                    ch._http.post = AsyncMock()
                    await ch._setup_webhook()
            for call in mock_logger.warning.call_args_list:
                assert "webhook_secret" not in str(call)


class TestSourceLevelChecks:
    """Prueft den Source-Code auf HMAC-Implementierung."""

    def test_handle_webhook_uses_hmac(self) -> None:
        source = inspect.getsource(SignalChannel._handle_webhook)
        assert "hmac" in source.lower()

    def test_handle_webhook_uses_sha256(self) -> None:
        source = inspect.getsource(SignalChannel._handle_webhook)
        assert "sha256" in source.lower()

    def test_handle_webhook_uses_compare_digest(self) -> None:
        source = inspect.getsource(SignalChannel._handle_webhook)
        assert "compare_digest" in source

    def test_handle_webhook_returns_403(self) -> None:
        source = inspect.getsource(SignalChannel._handle_webhook)
        assert "403" in source
