"""Tests für core/vision.py — Backend-agnostische multimodale Messages."""

from __future__ import annotations

import pytest

from jarvis.core.vision import (
    ImageContent,
    ImageMediaType,
    MultimodalMessage,
    build_vision_message,
    format_for_backend,
    is_multimodal_message,
)


# ============================================================================
# ImageContent
# ============================================================================


class TestImageContent:
    def test_defaults(self) -> None:
        img = ImageContent(data_b64="abc123")
        assert img.data_b64 == "abc123"
        assert img.media_type == ImageMediaType.PNG
        assert img.alt_text == ""

    def test_custom_media_type(self) -> None:
        img = ImageContent(data_b64="xyz", media_type=ImageMediaType.JPEG, alt_text="photo")
        assert img.media_type == ImageMediaType.JPEG
        assert img.alt_text == "photo"

    def test_all_media_types(self) -> None:
        assert ImageMediaType.PNG == "image/png"
        assert ImageMediaType.JPEG == "image/jpeg"
        assert ImageMediaType.GIF == "image/gif"
        assert ImageMediaType.WEBP == "image/webp"


# ============================================================================
# MultimodalMessage
# ============================================================================


class TestMultimodalMessage:
    def test_defaults(self) -> None:
        msg = MultimodalMessage()
        assert msg.role == "user"
        assert msg.text == ""
        assert msg.images == []
        assert msg.has_images() is False

    def test_with_images(self) -> None:
        img = ImageContent(data_b64="data")
        msg = MultimodalMessage(role="user", text="Describe", images=[img])
        assert msg.has_images() is True
        assert len(msg.images) == 1

    def test_text_only(self) -> None:
        msg = MultimodalMessage(text="Hello")
        assert msg.has_images() is False
        assert msg.text == "Hello"


# ============================================================================
# build_vision_message
# ============================================================================


class TestBuildVisionMessage:
    def test_single_image(self) -> None:
        msg = build_vision_message("Analysiere", ["base64data"])
        assert msg.text == "Analysiere"
        assert msg.role == "user"
        assert len(msg.images) == 1
        assert msg.images[0].data_b64 == "base64data"
        assert msg.images[0].media_type == ImageMediaType.PNG

    def test_multiple_images(self) -> None:
        msg = build_vision_message("Vergleiche", ["img1", "img2", "img3"])
        assert len(msg.images) == 3

    def test_empty_images_filtered(self) -> None:
        msg = build_vision_message("Test", ["valid", "", "also_valid"])
        assert len(msg.images) == 2

    def test_no_images(self) -> None:
        msg = build_vision_message("Nur Text", [])
        assert msg.has_images() is False
        assert msg.text == "Nur Text"

    def test_custom_role(self) -> None:
        msg = build_vision_message("X", ["d"], role="assistant")
        assert msg.role == "assistant"

    def test_custom_media_type(self) -> None:
        msg = build_vision_message("X", ["d"], media_type=ImageMediaType.JPEG)
        assert msg.images[0].media_type == ImageMediaType.JPEG

    def test_custom_alt_text(self) -> None:
        msg = build_vision_message("X", ["d"], alt_text="Seitenscreenshot")
        assert msg.images[0].alt_text == "Seitenscreenshot"


# ============================================================================
# format_for_backend — Anthropic
# ============================================================================


class TestFormatForBackendAnthropic:
    def test_single_image(self) -> None:
        msg = build_vision_message("Beschreibe die Seite", ["aGVsbG8="])
        result = format_for_backend(msg, "anthropic")

        assert result["role"] == "user"
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # 1 image + 1 text

        img_block = content[0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == "aGVsbG8="

        text_block = content[1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Beschreibe die Seite"

    def test_multiple_images(self) -> None:
        msg = build_vision_message("Vergleiche", ["img1", "img2"])
        result = format_for_backend(msg, "anthropic")
        content = result["content"]
        assert len(content) == 3  # 2 images + 1 text
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "text"

    def test_no_text(self) -> None:
        msg = MultimodalMessage(
            images=[ImageContent(data_b64="x")],
        )
        result = format_for_backend(msg, "anthropic")
        content = result["content"]
        assert len(content) == 1  # nur image, kein text-Block
        assert content[0]["type"] == "image"


# ============================================================================
# format_for_backend — OpenAI
# ============================================================================


class TestFormatForBackendOpenAI:
    def test_single_image(self) -> None:
        msg = build_vision_message("Beschreibe", ["aGVsbG8="])
        result = format_for_backend(msg, "openai")

        assert result["role"] == "user"
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 2

        img_block = content[0]
        assert img_block["type"] == "image_url"
        assert img_block["image_url"]["url"] == "data:image/png;base64,aGVsbG8="

        text_block = content[1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Beschreibe"

    def test_jpeg_format(self) -> None:
        msg = build_vision_message("X", ["data"], media_type=ImageMediaType.JPEG)
        result = format_for_backend(msg, "openai")
        url = result["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")


# ============================================================================
# format_for_backend — Ollama
# ============================================================================


class TestFormatForBackendOllama:
    def test_single_image(self) -> None:
        msg = build_vision_message("Beschreibe", ["aGVsbG8="])
        result = format_for_backend(msg, "ollama")

        assert result["role"] == "user"
        assert result["content"] == "Beschreibe"
        assert result["images"] == ["aGVsbG8="]

    def test_multiple_images(self) -> None:
        msg = build_vision_message("X", ["a", "b"])
        result = format_for_backend(msg, "ollama")
        assert len(result["images"]) == 2


# ============================================================================
# format_for_backend — Text-Fallback
# ============================================================================


class TestFormatTextFallback:
    def test_unknown_backend(self) -> None:
        msg = build_vision_message("Analysiere", ["data"], alt_text="Screenshot")
        result = format_for_backend(msg, "some_unknown_backend")

        assert result["role"] == "user"
        assert isinstance(result["content"], str)
        assert "[Screenshot]" in result["content"]
        assert "Analysiere" in result["content"]

    def test_no_alt_text(self) -> None:
        msg = MultimodalMessage(
            text="X",
            images=[ImageContent(data_b64="d", alt_text="")],
        )
        result = format_for_backend(msg, "unknown")
        assert "[Bild]" in result["content"]

    def test_text_only_any_backend(self) -> None:
        msg = build_vision_message("Nur Text", [])
        for backend in ("anthropic", "openai", "ollama", "unknown"):
            result = format_for_backend(msg, backend)
            assert result["content"] == "Nur Text"
            assert result["role"] == "user"


# ============================================================================
# is_multimodal_message
# ============================================================================


class TestIsMultimodalMessage:
    def test_anthropic_format(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": "x"}},
                {"type": "text", "text": "Beschreibe"},
            ],
        }
        assert is_multimodal_message(msg) is True

    def test_openai_format(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                {"type": "text", "text": "Beschreibe"},
            ],
        }
        assert is_multimodal_message(msg) is True

    def test_ollama_format(self) -> None:
        msg = {"role": "user", "content": "Beschreibe", "images": ["base64data"]}
        assert is_multimodal_message(msg) is True

    def test_plain_text_message(self) -> None:
        msg = {"role": "user", "content": "Hallo"}
        assert is_multimodal_message(msg) is False

    def test_empty_message(self) -> None:
        assert is_multimodal_message({}) is False

    def test_not_a_dict(self) -> None:
        assert is_multimodal_message("not a dict") is False
        assert is_multimodal_message(42) is False
        assert is_multimodal_message(None) is False

    def test_content_list_without_images(self) -> None:
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "nur text"}],
        }
        assert is_multimodal_message(msg) is False

    def test_empty_images_list(self) -> None:
        msg = {"role": "user", "content": "X", "images": []}
        assert is_multimodal_message(msg) is False

    def test_roundtrip_anthropic(self) -> None:
        """build → format → is_multimodal erkennt es."""
        msg = build_vision_message("Test", ["data"])
        formatted = format_for_backend(msg, "anthropic")
        assert is_multimodal_message(formatted) is True

    def test_roundtrip_openai(self) -> None:
        msg = build_vision_message("Test", ["data"])
        formatted = format_for_backend(msg, "openai")
        assert is_multimodal_message(formatted) is True

    def test_roundtrip_ollama(self) -> None:
        msg = build_vision_message("Test", ["data"])
        formatted = format_for_backend(msg, "ollama")
        assert is_multimodal_message(formatted) is True
