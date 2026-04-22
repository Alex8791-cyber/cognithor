"""Tests for Planner.formulate_response() returning ResponseEnvelope."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.config import CognithorConfig
from cognithor.core.observer import ResponseEnvelope


@pytest.fixture
def planner_with_mocks():
    from cognithor.core.model_router import ModelRouter
    from cognithor.core.planner import Planner

    cfg = CognithorConfig()
    cfg.observer.enabled = False  # isolate this test from observer integration (Task 15)
    ollama = AsyncMock()
    ollama.chat = AsyncMock(return_value={"message": {"content": "hello"}})
    router = MagicMock(spec=ModelRouter)
    router.select_model = MagicMock(return_value="qwen3:8b")
    p = Planner(
        config=cfg,
        ollama=ollama,
        model_router=router,
    )
    return p


class TestFormulateResponseReturnsEnvelope:
    async def test_returns_response_envelope(self, planner_with_mocks):
        from cognithor.models import WorkingMemory

        wm = WorkingMemory(session_id="s1")
        envelope = await planner_with_mocks.formulate_response(
            user_message="hi",
            results=[],
            working_memory=wm,
        )
        assert isinstance(envelope, ResponseEnvelope)
        assert envelope.content == "hello"
        assert envelope.directive is None


class TestFormulateResponseWithObserver:
    async def test_hallucination_triggers_regen_with_feedback(self, planner_with_mocks):
        from cognithor.models import WorkingMemory

        cfg = planner_with_mocks._config
        cfg.observer.enabled = True
        cfg.observer.max_retries = 2

        # Sequence: draft1 (bad) → audit1 (fail) → draft2 (good) → audit2 (pass)
        # planner uses the same ollama mock for both its own chat and the observer's.
        _ok = '{"passed": true, "reason": "", "evidence": "", "fix_suggestion": ""}'
        hallucination_audit = (
            "{"
            '"hallucination": {"passed": false, "reason": "unsupported date",'
            ' "evidence": "2015", "fix_suggestion": "remove"},'
            f'"sycophancy": {_ok},'
            f'"laziness": {_ok},'
            f'"tool_ignorance": {_ok}'
            "}"
        )
        pass_audit = (
            "{"
            f'"hallucination": {_ok},'
            f'"sycophancy": {_ok},'
            f'"laziness": {_ok},'
            f'"tool_ignorance": {_ok}'
            "}"
        )

        planner_with_mocks._ollama.chat = AsyncMock(
            side_effect=[
                # Draft 1 (hallucinates)
                {"message": {"content": "TechCorp was founded in 2015 (MADE UP)."}},
                # Observer audit 1 (fails hallucination)
                {"message": {"content": hallucination_audit}},
                # Draft 2 (after regen, clean)
                {"message": {"content": "TechCorp's founding year is not in the search results."}},
                # Observer audit 2 (passes)
                {"message": {"content": pass_audit}},
            ]
        )
        # Observer also calls list_models before audit — stub it to say observer model is available
        planner_with_mocks._ollama.list_models = AsyncMock(return_value=["qwen3:32b"])

        wm = WorkingMemory(session_id="s1")
        envelope = await planner_with_mocks.formulate_response(
            user_message="When was TechCorp founded?",
            results=[],
            working_memory=wm,
        )
        assert envelope.content == "TechCorp's founding year is not in the search results."
        assert envelope.directive is None  # hallucination regen stays inside Planner


class TestVisionRouting:
    """When working_memory.image_attachments is non-empty, the Planner
    must route the LLM call through ``vision_model_detail`` and pass the
    image paths to ``OllamaClient.chat(images=...)`` — giving a VLM like
    qwen3.6:27b a chance to actually see the image."""

    async def test_image_attachment_selects_vision_model(self, planner_with_mocks, tmp_path):
        from cognithor.models import WorkingMemory

        # Make vision_model_detail distinct from the text router default.
        planner_with_mocks._config.vision_model_detail = "qwen3.6:27b"

        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        wm = WorkingMemory(session_id="s1")
        wm.image_attachments = [str(img)]

        await planner_with_mocks.formulate_response(
            user_message="what's in this?",
            results=[],
            working_memory=wm,
        )

        # chat() must have been called with the vision model AND images.
        call = planner_with_mocks._ollama.chat.call_args
        assert call.kwargs.get("model") == "qwen3.6:27b"
        assert call.kwargs.get("images") == [str(img)]
        # Sanity: router.select_model must NOT have been used when images present.
        planner_with_mocks._router.select_model.assert_not_called()

    async def test_no_attachment_uses_text_router(self, planner_with_mocks):
        from cognithor.models import WorkingMemory

        wm = WorkingMemory(session_id="s2")
        # No image_attachments set → router's text model is used, images=None.
        await planner_with_mocks.formulate_response(
            user_message="just text",
            results=[],
            working_memory=wm,
        )
        call = planner_with_mocks._ollama.chat.call_args
        assert call.kwargs.get("images") is None
        planner_with_mocks._router.select_model.assert_called()
