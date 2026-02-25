"""Core Vision Messages — Backend-agnostische multimodale Message-Konstruktion.

Stellt Typen und Funktionen bereit, um Bilder (Screenshots) zusammen
mit Text an beliebige LLM-Backends zu senden.  Jedes Backend hat ein
eigenes Format:

  - Anthropic: content-Array mit type=image / type=text Blöcken
  - OpenAI:    content-Array mit type=image_url Blöcken
  - Ollama:    content=text + images=[base64, ...]

Die Funktionen hier abstrahieren diese Unterschiede.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ── Typen ─────────────────────────────────────────────────────────────


class ImageMediaType(StrEnum):
    """Unterstützte Bild-Formate."""

    PNG = "image/png"
    JPEG = "image/jpeg"
    GIF = "image/gif"
    WEBP = "image/webp"


@dataclass
class ImageContent:
    """Ein einzelnes Base64-kodiertes Bild."""

    data_b64: str
    media_type: ImageMediaType = ImageMediaType.PNG
    alt_text: str = ""


@dataclass
class MultimodalMessage:
    """Eine Nachricht die Text und optional Bilder enthält."""

    role: str = "user"
    text: str = ""
    images: list[ImageContent] = field(default_factory=list)

    def has_images(self) -> bool:
        return len(self.images) > 0


# ── Builder ───────────────────────────────────────────────────────────


def build_vision_message(
    text: str,
    images_b64: list[str],
    *,
    media_type: ImageMediaType = ImageMediaType.PNG,
    role: str = "user",
    alt_text: str = "Screenshot",
) -> MultimodalMessage:
    """Baut eine MultimodalMessage aus Text und Base64-Bildern.

    Args:
        text: Der Text-Inhalt der Nachricht.
        images_b64: Liste von Base64-kodierten Bilddaten.
        media_type: Bildformat (Standard: PNG).
        role: Rolle der Nachricht (Standard: user).
        alt_text: Alt-Text für die Bilder.

    Returns:
        MultimodalMessage mit Text und Bildern.
    """
    images = [
        ImageContent(data_b64=b64, media_type=media_type, alt_text=alt_text)
        for b64 in images_b64
        if b64
    ]
    return MultimodalMessage(role=role, text=text, images=images)


# ── Backend-spezifische Formatierung ─────────────────────────────────


def format_for_backend(message: MultimodalMessage, backend_type: str) -> dict[str, Any]:
    """Konvertiert eine MultimodalMessage ins Backend-spezifische Format.

    Args:
        message: Die zu konvertierende Nachricht.
        backend_type: "anthropic", "openai", "ollama" oder anderes.

    Returns:
        Dict im Backend-spezifischen Message-Format.
    """
    if not message.has_images():
        return {"role": message.role, "content": message.text}

    if backend_type == "anthropic":
        return _format_anthropic(message)
    elif backend_type == "openai":
        return _format_openai(message)
    elif backend_type == "ollama":
        return _format_ollama(message)
    else:
        return _format_text_fallback(message)


def _format_anthropic(message: MultimodalMessage) -> dict[str, Any]:
    """Anthropic-Format: content-Array mit image/text Blöcken."""
    content: list[dict[str, Any]] = []
    for img in message.images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.media_type.value,
                "data": img.data_b64,
            },
        })
    if message.text:
        content.append({"type": "text", "text": message.text})
    return {"role": message.role, "content": content}


def _format_openai(message: MultimodalMessage) -> dict[str, Any]:
    """OpenAI-Format: content-Array mit image_url Blöcken."""
    content: list[dict[str, Any]] = []
    for img in message.images:
        data_url = f"data:{img.media_type.value};base64,{img.data_b64}"
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
    if message.text:
        content.append({"type": "text", "text": message.text})
    return {"role": message.role, "content": content}


def _format_ollama(message: MultimodalMessage) -> dict[str, Any]:
    """Ollama-Format: content=text, images=[base64, ...]."""
    return {
        "role": message.role,
        "content": message.text,
        "images": [img.data_b64 for img in message.images],
    }


def _format_text_fallback(message: MultimodalMessage) -> dict[str, Any]:
    """Text-Fallback für unbekannte Backends: Bilder als Alt-Text."""
    parts = []
    for img in message.images:
        alt = img.alt_text or "Bild"
        parts.append(f"[{alt}]")
    if message.text:
        parts.append(message.text)
    return {"role": message.role, "content": "\n".join(parts)}


# ── Erkennung ─────────────────────────────────────────────────────────


def is_multimodal_message(msg: dict[str, Any]) -> bool:
    """Erkennt ob eine Message multimodalen Inhalt hat.

    Prüft alle 3 Backend-Formate:
      - Anthropic/OpenAI: content ist eine Liste mit image/image_url Blöcken
      - Ollama: images-Feld vorhanden

    Args:
        msg: Eine Message als Dict.

    Returns:
        True wenn die Message Bilder enthält.
    """
    if not isinstance(msg, dict):
        return False

    # Ollama-Format: images-Feld
    images = msg.get("images")
    if isinstance(images, list) and len(images) > 0:
        return True

    # Anthropic/OpenAI-Format: content ist Liste
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type in ("image", "image_url"):
                return True

    return False
