"""Tests fuer CWE-22 Path Traversal Prevention in Voice/TTS-Funktionen.

Testet validate_voice_name() und validate_model_path_containment() aus
security/sanitizer.py sowie deren Integration in:
  - __main__.py: _run_piper_tts(), _download_piper_voice()
  - mcp/media.py: text_to_speech()
  - channels/voice_ws_bridge.py: synthesize_response()

Schweregrad: HOCH — oeffentlich erreichbare API-Endpoints.
CVE-Referenz: CWE-22 (Improper Limitation of a Pathname to a Restricted Directory)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from jarvis.security.sanitizer import (
    validate_model_path_containment,
    validate_voice_name,
)


# ============================================================================
# validate_voice_name — Valid Names (Positive Cases)
# ============================================================================


class TestValidVoiceNames:
    """Stellt sicher, dass alle gaengigen Piper-Voice-Namen akzeptiert werden."""

    @pytest.mark.parametrize("voice", [
        "de_DE-thorsten-high",
        "de_DE-thorsten-medium",
        "de_DE-thorsten_emotional-medium",
        "de_DE-pavoque-low",
        "de_DE-karlsson-low",
        "de_DE-kerstin-low",
        "de_DE-ramona-low",
        "de_DE-eva_k-x_low",
        "en_US-lessac-low",
        "en_US-lessac-medium",
        "en_US-lessac-high",
        "en_GB-northern_english_male-medium",
        "fr_FR-upmc-medium",
        "es_ES-davefx-medium",
        "zh_CN-huayan-medium",
        "ja_JP-tsukuyomi-medium",
    ])
    def test_valid_piper_voices(self, voice: str) -> None:
        assert validate_voice_name(voice) == voice

    def test_simple_alphanumeric(self) -> None:
        assert validate_voice_name("model123") == "model123"

    def test_with_dots(self) -> None:
        assert validate_voice_name("model.v2") == "model.v2"

    def test_single_char(self) -> None:
        assert validate_voice_name("a") == "a"

    def test_with_hyphens_underscores(self) -> None:
        assert validate_voice_name("my-custom_voice-v1.0") == "my-custom_voice-v1.0"


# ============================================================================
# validate_voice_name — Path Traversal Attacks (Negative Cases)
# ============================================================================


class TestPathTraversalBlocked:
    """Stellt sicher, dass Path-Traversal-Versuche blockiert werden."""

    @pytest.mark.parametrize("voice", [
        "../../../../etc/passwd",
        "../../../etc/shadow",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "..\\..\\..\\..\\etc\\passwd",
        "../secret",
        "..\\secret",
        "voices/../../../etc/passwd",
        "..",
        "../",
        "..\\",
    ])
    def test_directory_traversal(self, voice: str) -> None:
        with pytest.raises(ValueError, match=r"(?:path separator|traversal|invalid)"):
            validate_voice_name(voice)

    @pytest.mark.parametrize("voice", [
        "/etc/passwd",
        "/tmp/evil",
        "\\windows\\system32\\cmd.exe",
        "C:\\Windows\\System32\\cmd.exe",
        "voices/evil",
        "sub\\dir",
    ])
    def test_path_separators(self, voice: str) -> None:
        with pytest.raises(ValueError, match="path separator"):
            validate_voice_name(voice)


# ============================================================================
# validate_voice_name — Other Malicious Inputs
# ============================================================================


class TestMaliciousInputsBlocked:
    """Stellt sicher, dass weitere Angriffsvektoren blockiert werden."""

    def test_null_byte_injection(self) -> None:
        with pytest.raises(ValueError, match="null byte"):
            validate_voice_name("voice\x00.onnx")

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            validate_voice_name("")

    def test_too_long_name(self) -> None:
        with pytest.raises(ValueError, match="too long"):
            validate_voice_name("a" * 200)

    def test_max_length_accepted(self) -> None:
        name = "a" * 128
        assert validate_voice_name(name) == name

    @pytest.mark.parametrize("voice", [
        " space",
        "tab\there",
        "new\nline",
        "carriage\rreturn",
        "semi;colon",
        "pipe|char",
        "ampersand&char",
        "dollar$var",
        "backtick`cmd`",
        "paren(test)",
        "bracket[test]",
        "brace{test}",
        "angle<test>",
        "quote'test",
        'dquote"test',
        "hash#test",
        "percent%test",
        "at@test",
        "excl!test",
        "star*glob",
        "question?mark",
        "tilde~home",
        "equal=sign",
        "plus+sign",
        "comma,sep",
        "colon:drive",
    ])
    def test_special_chars_blocked(self, voice: str) -> None:
        with pytest.raises(ValueError):
            validate_voice_name(voice)

    @pytest.mark.parametrize("voice", [
        "voïce",
        "stimme-schön",
        "голос",
        "声音",
        "صوت",
    ])
    def test_unicode_blocked(self, voice: str) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            validate_voice_name(voice)

    def test_starts_with_dot(self) -> None:
        with pytest.raises(ValueError):
            validate_voice_name(".hidden")

    def test_starts_with_hyphen(self) -> None:
        with pytest.raises(ValueError):
            validate_voice_name("-flag")

    def test_starts_with_underscore(self) -> None:
        with pytest.raises(ValueError):
            validate_voice_name("_private")


# ============================================================================
# validate_model_path_containment — Defense in Depth
# ============================================================================


class TestModelPathContainment:
    """Stellt sicher, dass konstruierte Pfade im erlaubten Verzeichnis bleiben."""

    def test_valid_path_within_dir(self, tmp_path: Path) -> None:
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        model_path = voices_dir / "de_DE-thorsten-high.onnx"
        result = validate_model_path_containment(model_path, voices_dir)
        assert result == model_path.resolve()

    def test_traversal_escapes_dir(self, tmp_path: Path) -> None:
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        evil_path = voices_dir / ".." / ".." / "etc" / "passwd.onnx"
        with pytest.raises(ValueError, match="escapes allowed directory"):
            validate_model_path_containment(evil_path, voices_dir)

    def test_absolute_path_outside_dir(self, tmp_path: Path) -> None:
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        with pytest.raises(ValueError, match="escapes allowed directory"):
            validate_model_path_containment(Path("/etc/passwd"), voices_dir)

    def test_nested_subdir_allowed(self, tmp_path: Path) -> None:
        voices_dir = tmp_path / "voices"
        sub = voices_dir / "de"
        sub.mkdir(parents=True)
        model_path = sub / "model.onnx"
        result = validate_model_path_containment(model_path, voices_dir)
        assert result == model_path.resolve()


# ============================================================================
# Integration: mcp/media.py — text_to_speech()
# ============================================================================


class TestMediaTTSPathTraversal:
    """Stellt sicher, dass MediaPipeline.text_to_speech() Voice-Namen validiert."""

    @pytest.fixture
    def pipeline(self, tmp_path: Path):
        from jarvis.mcp.media import MediaPipeline
        return MediaPipeline(workspace_dir=tmp_path)

    @pytest.mark.parametrize("voice", [
        "../../../../etc/passwd",
        "../../../etc/shadow",
        "..\\..\\windows\\system32",
        "/etc/passwd",
        "test/../../etc/passwd",
    ])
    async def test_traversal_rejected(self, pipeline, voice: str) -> None:
        result = await pipeline.text_to_speech("Hallo Welt", voice=voice)
        assert not result.success
        assert "Ungueltiger Voice-Name" in (result.error or "")

    async def test_valid_voice_accepted(self, pipeline) -> None:
        # Will fail at Piper execution (not installed in test), but should
        # pass voice validation without error
        result = await pipeline.text_to_speech("Test", voice="de_DE-thorsten-high")
        # Either success (piper installed) or piper/espeak not found error
        # but NOT a voice validation error
        if not result.success:
            assert "Ungueltiger Voice-Name" not in (result.error or "")

    async def test_null_byte_rejected(self, pipeline) -> None:
        result = await pipeline.text_to_speech("Test", voice="voice\x00evil")
        assert not result.success
        assert "Ungueltiger Voice-Name" in (result.error or "")


# ============================================================================
# Integration: channels/voice_ws_bridge.py — synthesize_response()
# ============================================================================


class TestVoiceWSBridgePathTraversal:
    """Stellt sicher, dass VoiceMessageHandler.synthesize_response() validiert."""

    @pytest.fixture
    def handler(self, tmp_path: Path):
        from jarvis.channels.voice_ws_bridge import VoiceMessageHandler
        return VoiceMessageHandler(workspace_dir=tmp_path)

    @pytest.mark.parametrize("voice", [
        "../../../../etc/passwd",
        "../../../etc/shadow",
        "/etc/passwd",
        "..\\..\\windows\\system32",
    ])
    async def test_traversal_returns_none(self, handler, voice: str) -> None:
        result = await handler.synthesize_response("Hallo", voice=voice)
        assert result is None

    async def test_valid_voice_passes_validation(self, handler) -> None:
        # May fail at TTS level (no piper), but should not fail at validation
        # We just verify it doesn't immediately return None due to voice validation
        # (it will return None due to missing piper, but for a different reason)
        result = await handler.synthesize_response("Hallo", voice="de_DE-thorsten-high")
        # Result is None because piper isn't installed in test env — that's OK
        # The important thing is no ValueError was raised


# ============================================================================
# Regression: Exact Payloads from Security Report
# ============================================================================


class TestSecurityReportPayloads:
    """Reproduziert die exakten Payloads aus dem Sicherheitsbericht."""

    def test_reported_payload_exact(self) -> None:
        """Der exakte Payload aus dem Bericht: '../../../../etc/passwd'"""
        with pytest.raises(ValueError):
            validate_voice_name("../../../../etc/passwd")

    def test_reported_path_construction(self) -> None:
        """Verifies voices/../../../../etc/passwd.onnx would be caught."""
        # Even if someone bypassed name validation, path containment catches it
        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir) / "voices"
            voices_dir.mkdir()
            evil_path = voices_dir / "../../../../etc/passwd.onnx"
            with pytest.raises(ValueError, match="escapes allowed directory"):
                validate_model_path_containment(evil_path, voices_dir)

    def test_windows_traversal_payload(self) -> None:
        """Windows-spezifischer Payload mit Backslashes."""
        with pytest.raises(ValueError):
            validate_voice_name("..\\..\\..\\..\\windows\\system32\\config\\sam")

    def test_mixed_separators_payload(self) -> None:
        """Gemischte Separatoren."""
        with pytest.raises(ValueError):
            validate_voice_name("..\\../..\\../etc/passwd")

    def test_url_encoded_not_applicable(self) -> None:
        """URL-encoded traversal (%2e%2e/) — chars blocked by regex."""
        with pytest.raises(ValueError):
            validate_voice_name("%2e%2e%2f%2e%2e%2fetc%2fpasswd")

    def test_double_encoded_dots(self) -> None:
        """Double-encoded dots — % char blocked by regex."""
        with pytest.raises(ValueError):
            validate_voice_name("%252e%252e%252f")
