"""Sprint-27 VLM-3 — multimodal-message helper tests."""

from __future__ import annotations

from cognithor.core.multimodal import (
    MultimodalCounts,
    count_multimodal_parts,
    estimate_vision_tokens,
    is_multimodal_request,
)
from cognithor.security.cost_ledger import CostKind

# ---------------------------------------------------------------------------
# CostKind extension
# ---------------------------------------------------------------------------


class TestCostKindHasVisionTokens:
    def test_vision_tokens_value_in_enum(self) -> None:
        # Spike-doc decision: VLM-3 adds VISION_TOKENS to the enum.
        assert CostKind.VISION_TOKENS.value == "vision_tokens"
        assert CostKind("vision_tokens") is CostKind.VISION_TOKENS


# ---------------------------------------------------------------------------
# count_multimodal_parts
# ---------------------------------------------------------------------------


class TestCountMultimodalParts:
    def test_plain_string_content(self) -> None:
        counts = count_multimodal_parts(
            [{"role": "user", "content": "hello"}],
        )
        assert counts == MultimodalCounts(text_parts=1)

    def test_typed_text_only(self) -> None:
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "describe"}],
                },
            ],
        )
        assert counts.text_parts == 1
        assert counts.image_parts == 0
        assert counts.video_parts == 0

    def test_image_part(self) -> None:
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,xyz"},
                        },
                    ],
                },
            ],
        )
        assert counts.image_parts == 1
        assert counts.text_parts == 1
        assert counts.has_vision is True

    def test_video_part(self) -> None:
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "file:///tmp/clip.mp4"},
                        },
                    ],
                },
            ],
        )
        assert counts.video_parts == 1
        assert counts.has_vision is True
        assert counts.total_vision_parts == 1

    def test_mixed_multi_message(self) -> None:
        counts = count_multimodal_parts(
            [
                {"role": "system", "content": "you are helpful"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "watch this"},
                        {
                            "type": "video_url",
                            "video_url": {"url": "file:///clip1.mp4"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "file:///shot.png"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "file:///shot2.png"},
                        },
                    ],
                },
            ],
        )
        assert counts.image_parts == 2
        assert counts.video_parts == 1
        assert counts.text_parts == 2  # system string + user text part

    def test_string_url_form_accepted(self) -> None:
        # Some clients send {"image_url": "..."} without the {"url": ...} nesting.
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": "file:///shot.png"},
                    ],
                },
            ],
        )
        assert counts.image_parts == 1

    def test_empty_url_dropped(self) -> None:
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": ""}},
                        {"type": "video_url", "video_url": {}},
                    ],
                },
            ],
        )
        assert counts.image_parts == 0
        assert counts.video_parts == 0

    def test_unknown_type_ignored(self) -> None:
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "x"}},
                        {"type": "text", "text": "hi"},
                    ],
                },
            ],
        )
        assert counts.text_parts == 1
        assert counts.image_parts == 0

    def test_malformed_messages_ignored(self) -> None:
        # Should not raise on garbage input.
        counts = count_multimodal_parts(
            [
                None,  # type: ignore[list-item]
                {"role": "user", "content": 42},  # type: ignore[dict-item]
                "not a message",  # type: ignore[list-item]
                {"role": "user", "content": [None, "string", 123]},  # type: ignore[list-item]
            ],
        )
        assert counts == MultimodalCounts()

    def test_empty_string_content_does_not_count_text(self) -> None:
        counts = count_multimodal_parts([{"role": "user", "content": ""}])
        assert counts.text_parts == 0


# ---------------------------------------------------------------------------
# estimate_vision_tokens
# ---------------------------------------------------------------------------


class TestEstimateVisionTokens:
    def test_zero_when_no_vision(self) -> None:
        assert estimate_vision_tokens(MultimodalCounts(text_parts=3)) == 0

    def test_image_uses_256_tokens(self) -> None:
        assert estimate_vision_tokens(MultimodalCounts(image_parts=1)) == 256

    def test_video_uses_32_frames_x_256_tokens(self) -> None:
        # Spike-doc default num_frames=32 for long clips.
        assert estimate_vision_tokens(MultimodalCounts(video_parts=1)) == 32 * 256

    def test_combined_estimate(self) -> None:
        # 2 images + 1 video = 2*256 + 32*256
        assert (
            estimate_vision_tokens(
                MultimodalCounts(image_parts=2, video_parts=1),
            )
            == 2 * 256 + 32 * 256
        )


# ---------------------------------------------------------------------------
# is_multimodal_request
# ---------------------------------------------------------------------------


class TestIsMultimodal:
    def test_plain_text_is_not_multimodal(self) -> None:
        assert is_multimodal_request([{"role": "user", "content": "hi"}]) is False

    def test_text_typed_only_is_not_multimodal(self) -> None:
        assert (
            is_multimodal_request(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                ],
            )
            is False
        )

    def test_image_makes_multimodal(self) -> None:
        assert (
            is_multimodal_request(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "x"},
                            },
                        ],
                    },
                ],
            )
            is True
        )

    def test_video_makes_multimodal(self) -> None:
        assert (
            is_multimodal_request(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": "x"},
                            },
                        ],
                    },
                ],
            )
            is True
        )
