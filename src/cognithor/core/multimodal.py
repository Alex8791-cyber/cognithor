"""Sprint-27 VLM-3 — multimodal-message helpers for the LLMBackend layer.

The OpenAI / vLLM chat schema accepts a typed-content list per
message:

.. code-block:: json

    {"role": "user", "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url",
       "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "video_url",
       "video_url": {"url": "file:///path/to/video.mp4"}}
    ]}

This module is the central place every backend (OllamaBackend,
OpenAIBackend, VLLMBackend, ...) consults to:

* count multimodal parts per message — drives the
  :data:`CostKind.VISION_TOKENS` ledger entry shipped in HF-2's
  cost-ledger update;
* validate the part structure before forwarding so a malformed
  payload is rejected at the boundary rather than half-way
  through the engine; and
* expose typed accessors so type-checkers can audit every
  multimodal-handling code path.

All wire types stay JSON-serialisable so Gatekeeper, audit log,
and telemetry can ride the same shapes without custom encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class MultimodalCounts:
    """Per-request count of image + video parts across all messages."""

    image_parts: int = 0
    video_parts: int = 0
    text_parts: int = 0

    @property
    def has_vision(self) -> bool:
        return self.image_parts > 0 or self.video_parts > 0

    @property
    def total_vision_parts(self) -> int:
        return self.image_parts + self.video_parts


# Per the Qwen3-VL spike doc: 1 image ≈ 256 vision tokens at the
# default ViT patch size; 1 video frame ≈ 256 vision tokens, with
# the v0.92.7 ffprobe pipeline sampling fps=3 short / num_frames=32
# long. Without a frame-count handle from the backend, we estimate
# 32 frames per video to budget the cost ledger conservatively.
_TOKENS_PER_IMAGE: Final[int] = 256
_TOKENS_PER_VIDEO_DEFAULT_FRAMES: Final[int] = 32
_TOKENS_PER_FRAME: Final[int] = 256


def estimate_vision_tokens(counts: MultimodalCounts) -> int:
    """Estimate the vision-token cost for a single chat request.

    Coarse upper bound — the actual count depends on frame
    sampling (fps=3 vs num_frames=32) which the backend
    determines via ffprobe at request time. Operators can
    reconcile this estimate against the engine's own usage
    counter once vLLM exposes per-request vision-token
    breakdowns (open feature request as of vLLM 0.6).
    """

    return (
        counts.image_parts * _TOKENS_PER_IMAGE
        + counts.video_parts * _TOKENS_PER_VIDEO_DEFAULT_FRAMES * _TOKENS_PER_FRAME
    )


def count_multimodal_parts(messages: list[dict[str, Any]]) -> MultimodalCounts:
    """Walk a chat-messages list and tally every multimodal part.

    Accepts both shapes:

    * Plain string content: ``{"role": "user", "content": "hi"}``
      — counts as one text part, no vision.
    * Typed list content: see module docstring. Each part with
      ``type == "image_url"`` or ``type == "video_url"`` counts
      toward the corresponding tally; everything else counts
      as text. Unknown types are ignored (forward-compat).

    Tolerates malformed messages — never raises. Operators get
    the best-effort tally and the backend is the source of truth
    for hard validation.
    """

    images = 0
    videos = 0
    texts = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content:
                texts += 1
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image_url":
                if _is_valid_url_part(part.get("image_url")):
                    images += 1
            elif ptype == "video_url":
                if _is_valid_url_part(part.get("video_url")):
                    videos += 1
            elif ptype == "text" and isinstance(part.get("text"), str) and part["text"]:
                texts += 1
            # Other/unknown types: silently ignored.

    return MultimodalCounts(image_parts=images, video_parts=videos, text_parts=texts)


def _is_valid_url_part(value: Any) -> bool:
    """Return True for ``{"url": "..."}`` shapes used by image_url / video_url."""

    if isinstance(value, dict):
        url = value.get("url")
        return isinstance(url, str) and bool(url)
    if isinstance(value, str):
        # Some clients send the URL directly without nesting.
        return bool(value)
    return False


def is_multimodal_request(messages: list[dict[str, Any]]) -> bool:
    """Cheap existential check — saves a full count when unused."""

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "video_url"):
                return True
    return False
