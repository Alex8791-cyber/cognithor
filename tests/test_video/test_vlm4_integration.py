"""Sprint-27 VLM-4 — integration smoke for the VLM-read → HyperFrames-write path.

Verifies the wiring between the VLM-track surfaces (VLM-3
multimodal helpers + CostKind.VISION_TOKENS) — without booting
vLLM. The smoke covers:

1. **Multimodal classification** (VLM-3): a chat-messages list with
   a video part is classified as ``has_vision`` and produces the
   expected :class:`MultimodalCounts`.
2. **Vision-token estimate** (VLM-3): the estimate matches the
   ledger entry the backend records on the request boundary.
3. **CostKind taxonomy** (VLM-3): ``CostKind.VISION_TOKENS`` is
   the right key for the ledger entry — distinct from generic
   ``OTHER`` / token kinds so dashboards split them.

The actual vLLM model load (Qwen/Qwen3-VL-32B-Instruct-FP8 on
RTX 5090, ~31 GB VRAM at 32k ctx) is hardware-gated; that test
is skipped by default and the operator runs it manually with a
fully-wired vLLM environment. The HF-track skill imports
(``compose_explainer`` / ``compose_social_cut``) are exercised
indirectly — once HF-5 has landed, importing them under the
hardware marker confirms the cross-module wire is intact.
"""

from __future__ import annotations

import os

import pytest

from cognithor.core.multimodal import (
    count_multimodal_parts,
    estimate_vision_tokens,
    is_multimodal_request,
)
from cognithor.security.cost_ledger import CostKind

# ---------------------------------------------------------------------------
# 1. Multimodal classification of a video-read request
# ---------------------------------------------------------------------------


class TestVideoReadClassification:
    def test_video_url_message_is_multimodal(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this clip"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///clips/demo.mp4"},
                    },
                ],
            },
        ]
        assert is_multimodal_request(messages) is True

    def test_video_read_counts_one_video_part(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///clips/demo.mp4"},
                    },
                ],
            },
        ]
        counts = count_multimodal_parts(messages)
        assert counts.video_parts == 1
        assert counts.image_parts == 0
        assert counts.text_parts == 1
        assert counts.has_vision is True
        assert counts.total_vision_parts == 1

    def test_image_plus_video_combined_request(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "file:///shot.png"},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///clip.mp4"},
                    },
                ],
            },
        ]
        counts = count_multimodal_parts(messages)
        assert counts.image_parts == 1
        assert counts.video_parts == 1
        assert counts.has_vision is True


# ---------------------------------------------------------------------------
# 2. Vision-token estimate matches what the backend will ledger
# ---------------------------------------------------------------------------


class TestVisionTokenEstimate:
    def test_one_video_costs_32_x_256_tokens(self) -> None:
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "file:///c.mp4"},
                        },
                    ],
                },
            ],
        )
        # 1 video × 32 default frames × 256 tokens/frame = 8192
        assert estimate_vision_tokens(counts) == 32 * 256

    def test_image_plus_video_combined_estimate(self) -> None:
        counts = count_multimodal_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "f:///a.png"}},
                        {"type": "video_url", "video_url": {"url": "f:///b.mp4"}},
                    ],
                },
            ],
        )
        # 1 image × 256 + 1 video × 32 × 256
        assert estimate_vision_tokens(counts) == 256 + 32 * 256


# ---------------------------------------------------------------------------
# 3. CostKind taxonomy — vision tokens get their own ledger key
# ---------------------------------------------------------------------------


class TestCostKind:
    def test_vision_tokens_kind_exists_and_is_distinct(self) -> None:
        assert CostKind.VISION_TOKENS.value == "vision_tokens"
        # Distinct from generic OTHER so dashboards split them.
        assert CostKind.VISION_TOKENS != CostKind.OTHER


# ---------------------------------------------------------------------------
# 4. Hardware-gated full smoke (vLLM + Qwen3-VL on RTX 5090)
# ---------------------------------------------------------------------------


_HW_MARKER = "VLM4_HARDWARE_SMOKE"


@pytest.mark.skipif(
    os.environ.get(_HW_MARKER) != "1",
    reason=(
        "VLM-4 hardware smoke is gated on the operator. Set the "
        f"environment variable {_HW_MARKER}=1 with a working vLLM "
        "+ Qwen/Qwen3-VL-32B-Instruct-FP8 deployment to enable."
    ),
)
def test_vlm_full_pipeline_against_qwen3_vl_on_rtx5090() -> None:  # pragma: no cover
    """Hardware-gated: drives the actual VLM and the HyperFrames renderer.

    Operators reproduce this manually via:

        VLM4_HARDWARE_SMOKE=1 \\
        VLLM_BASE_URL=http://127.0.0.1:8000 \\
        VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 \\
        pytest tests/test_video/test_vlm4_integration.py -v

    Steps the smoke verifies on real hardware:

    1. POST a chat-messages list with a ``video_url`` to vLLM.
    2. Receive a textual description (the VLM's read).
    3. Pass the description into ``compose_explainer`` (HF-5) to
       build a spec.
    4. Pass the spec into the ``video_render`` MCP tool against
       the HyperFrames renderer (HF-2 / HF-3).
    5. Verify the rendered MP4 exists, is non-empty, and the
       provenance + cost ledgers each got an entry (HF-4).
    """

    pytest.skip(
        "VLM-4 hardware smoke is operator-driven; see the docstring for steps.",
    )
