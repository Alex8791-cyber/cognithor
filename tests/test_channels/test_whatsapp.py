"""Tests fuer den WhatsApp Cloud API Channel.

Testet: __init__ mit Config, Webhook-Verifizierung, HMAC-Signatur-Validierung,
Nachrichten-Parsing, Hilfsmethoden.
"""
from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock httpx before importing WhatsAppChannel so that missing httpx does not
# cause an ImportError.
_httpx_mock = MagicMock()
with patch.dict("sys.modules", {"httpx": _httpx_mock}):
    from jarvis.channels.whatsapp import WhatsAppChannel, MAX_TEXT_LENGTH


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def wa() -> WhatsAppChannel:
    """WhatsAppChannel mit Dummy-Config."""
    with patch.dict("sys.modules", {"httpx": _httpx_mock}):
        ch = WhatsAppChannel(
            api_token="test-token-12345",
            phone_number_id="123456789",
            verify_token="my-verify-token",
            webhook_port=9999,
            allowed_numbers=["+491234567890"],
        )
    return ch


# ============================================================================
# 1. Init / Properties
# ============================================================================


class TestWhatsAppInit:
    def test_name(self, wa: WhatsAppChannel) -> None:
        assert wa.name == "whatsapp"

    def test_config_stored(self, wa: WhatsAppChannel) -> None:
        assert wa._api_token == "test-token-12345"
        assert wa._phone_number_id == "123456789"
        assert wa._verify_token == "my-verify-token"
        assert wa._webhook_port == 9999
        assert "+491234567890" in wa._allowed_numbers

    def test_default_verify_token_generated(self) -> None:
        with patch.dict("sys.modules", {"httpx": _httpx_mock}):
            ch = WhatsAppChannel(
                api_token="tok",
                phone_number_id="pid",
                verify_token="",
            )
        # Should have auto-generated a non-empty verify token
        assert ch._verify_token != ""


# ============================================================================
# 2. Webhook Verification (GET)
# ============================================================================


class TestWebhookVerification:
    @pytest.mark.asyncio
    async def test_verification_success(self, wa: WhatsAppChannel) -> None:
        """Korrekte Verifikationsanfrage gibt Challenge zurueck."""
        request = MagicMock()
        request.query = {
            "hub.mode": "subscribe",
            "hub.verify_token": "my-verify-token",
            "hub.challenge": "challenge_string_123",
        }

        with patch.dict("sys.modules", {"aiohttp": MagicMock(), "aiohttp.web": MagicMock()}):
            from aiohttp import web
            with patch.object(web, "Response") as MockResponse:
                MockResponse.return_value = MagicMock()
                result = await wa._handle_verification(request)
                MockResponse.assert_called_once()
                call_kwargs = MockResponse.call_args
                # Check the challenge is returned in the text parameter
                assert "challenge_string_123" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_verification_wrong_token(self, wa: WhatsAppChannel) -> None:
        """Falsche Verifikation gibt 403 zurueck."""
        request = MagicMock()
        request.query = {
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "challenge",
        }

        with patch.dict("sys.modules", {"aiohttp": MagicMock(), "aiohttp.web": MagicMock()}):
            from aiohttp import web
            with patch.object(web, "Response") as MockResponse:
                MockResponse.return_value = MagicMock()
                result = await wa._handle_verification(request)
                call_kwargs = MockResponse.call_args
                assert "403" in str(call_kwargs) or call_kwargs[1].get("status") == 403


# ============================================================================
# 3. HMAC Signature Validation
# ============================================================================


class TestHMACValidation:
    def test_valid_signature(self, wa: WhatsAppChannel) -> None:
        payload = b'{"test": "data"}'
        expected = hmac_mod.new(
            b"test-token-12345", payload, hashlib.sha256
        ).hexdigest()
        sig_header = f"sha256={expected}"

        assert wa._verify_signature(payload, sig_header) is True

    def test_invalid_signature(self, wa: WhatsAppChannel) -> None:
        payload = b'{"test": "data"}'
        assert wa._verify_signature(payload, "sha256=invalid_hex") is False

    def test_empty_signature_header(self, wa: WhatsAppChannel) -> None:
        payload = b'{"test": "data"}'
        assert wa._verify_signature(payload, "") is False

    def test_missing_sha256_prefix(self, wa: WhatsAppChannel) -> None:
        payload = b'{"test": "data"}'
        assert wa._verify_signature(payload, "md5=abc123") is False


# ============================================================================
# 4. Message Parsing
# ============================================================================


class TestMessageParsing:
    def test_split_message_short(self, wa: WhatsAppChannel) -> None:
        """Kurze Nachricht wird nicht gesplittet."""
        result = wa._split_message("Hello World")
        assert result == ["Hello World"]

    def test_split_message_long(self, wa: WhatsAppChannel) -> None:
        """Lange Nachricht wird gesplittet."""
        long_text = "A" * (MAX_TEXT_LENGTH + 100)
        result = wa._split_message(long_text)
        assert len(result) >= 2
        # All chunks should be <= MAX_TEXT_LENGTH
        for chunk in result:
            assert len(chunk) <= MAX_TEXT_LENGTH

    def test_split_message_at_newline(self, wa: WhatsAppChannel) -> None:
        """Splittet bevorzugt am Newline."""
        text = "A" * (MAX_TEXT_LENGTH - 10) + "\n" + "B" * 20
        result = wa._split_message(text)
        assert len(result) == 2
        assert result[0].endswith("A")

    def test_get_or_create_session(self, wa: WhatsAppChannel) -> None:
        """Session-Mapping erstellt neue Session fuer neue Nummer."""
        sid1 = wa._get_or_create_session("+491234567890")
        sid2 = wa._get_or_create_session("+491234567890")
        sid3 = wa._get_or_create_session("+490000000000")
        assert sid1 == sid2  # Same number, same session
        assert sid1 != sid3  # Different number, different session

    def test_phone_for_session(self, wa: WhatsAppChannel) -> None:
        """Findet Telefonnummer fuer Session-ID."""
        sid = wa._get_or_create_session("+491234567890")
        phone = wa._phone_for_session(sid)
        assert phone == "+491234567890"

    def test_phone_for_session_not_found(self, wa: WhatsAppChannel) -> None:
        """Gibt None zurueck wenn Session nicht gefunden."""
        assert wa._phone_for_session("nonexistent") is None


# ============================================================================
# 5. Lifecycle
# ============================================================================


class TestWhatsAppLifecycle:
    @pytest.mark.asyncio
    async def test_stop_cleans_up(self, wa: WhatsAppChannel) -> None:
        """stop() raeumt auf ohne Crash."""
        wa._running = True
        wa._webhook_runner = None
        wa._http = None
        await wa.stop()
        assert wa._running is False

    @pytest.mark.asyncio
    async def test_send_without_running(self, wa: WhatsAppChannel) -> None:
        """Senden ohne _running ist ein No-Op."""
        wa._running = False
        from jarvis.models import OutgoingMessage
        msg = OutgoingMessage(channel="whatsapp", text="test", session_id="s1")
        await wa.send(msg)  # Should not raise
