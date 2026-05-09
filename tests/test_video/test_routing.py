"""Unit tests for cognithor.video.routing.

Two surfaces under test:

* :func:`check_profile_alignment` — pure-function consistency check
  between a router decision and the live ``list_models()`` output.
* :func:`video_chat_routed` — async glue: router decision +
  alignment + chat() call. Mocks ``VLLMBackend`` so the test does not
  touch any network.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.core.vlm_router import VLM_PROFILES, VlmRouter
from cognithor.video.routing import (
    ProfileAlignment,
    ProfileMismatchError,
    check_profile_alignment,
    ensure_profile_loaded,
    video_chat_routed,
)

# ---------------------------------------------------------------------------
# check_profile_alignment — pure function
# ---------------------------------------------------------------------------


class TestProfileAlignment:
    def test_aligned_when_recommended_model_is_available(self) -> None:
        router = VlmRouter()
        decision = router.select_profile_explained("Describe this clip.")
        # decision.profile.name == "fast" — model_id is Qwen3-VL-8B-Instruct
        alignment = check_profile_alignment(
            decision,
            available_models=[decision.profile.model_id, "extra-model-x"],
        )
        assert alignment.aligned is True
        assert alignment.actual_model == decision.profile.model_id
        assert alignment.recommended_model == decision.profile.model_id
        assert "extra-model-x" in alignment.available_models

    def test_misaligned_falls_back_to_first_available(self) -> None:
        router = VlmRouter()
        decision = router.select_profile_explained(
            "Compare frame 1 to frame 5."  # routes to premium
        )
        alignment = check_profile_alignment(
            decision,
            available_models=["some-other-model"],
        )
        assert alignment.aligned is False
        assert alignment.actual_model == "some-other-model"
        assert alignment.recommended_model == decision.profile.model_id

    def test_misaligned_no_models_actual_is_none(self) -> None:
        router = VlmRouter()
        decision = router.select_profile_explained("Describe this.")
        alignment = check_profile_alignment(
            decision,
            available_models=[],
        )
        assert alignment.aligned is False
        assert alignment.actual_model is None

    def test_alignment_is_a_frozen_dataclass(self) -> None:
        # Frozen because the receipt builder serializes it; mutation
        # would break audit consistency.
        from dataclasses import FrozenInstanceError

        router = VlmRouter()
        decision = router.select_profile_explained("anything")
        alignment = check_profile_alignment(decision, available_models=["foo"])
        assert isinstance(alignment, ProfileAlignment)
        with pytest.raises(FrozenInstanceError):
            alignment.aligned = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# video_chat_routed — async glue with mocked backend
# ---------------------------------------------------------------------------


def _mock_backend(
    *,
    available_models: list[str],
    response_content: str = "ok",
) -> Any:
    """Build a MagicMock that mimics VLLMBackend's relevant surface."""
    mock = MagicMock()
    mock.list_models = AsyncMock(return_value=available_models)
    response = MagicMock()
    response.content = response_content
    response.model = available_models[0] if available_models else "unknown"
    response.usage = None
    response.tool_calls = None
    response.raw = {}
    mock.chat = AsyncMock(return_value=response)
    return mock


class TestVideoChatRouted:
    @pytest.mark.asyncio
    async def test_aligned_path_uses_recommended_model(self) -> None:
        router = VlmRouter()
        backend = _mock_backend(
            available_models=["Qwen/Qwen3-VL-8B-Instruct"],
            response_content="A short video.",
        )
        response, decision, alignment = await video_chat_routed(
            backend=backend,
            router=router,
            prompt="Describe this clip.",
            video_url="http://host.docker.internal:8765/clip.mp4",
        )
        assert alignment.aligned is True
        assert alignment.actual_model == "Qwen/Qwen3-VL-8B-Instruct"
        assert decision.profile.name == "fast"
        assert response.content == "A short video."

        # backend.chat called with the recommended model
        backend.chat.assert_called_once()
        call_kwargs = backend.chat.call_args.kwargs
        assert call_kwargs["model"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert call_kwargs["video"]["url"].startswith("http://host.docker.internal")
        # frame sampling defaults to profile.max_frames (32 for fast)
        assert call_kwargs["video"]["sampling"]["num_frames"] == 32

    @pytest.mark.asyncio
    async def test_misaligned_falls_back_and_logs_warning(self) -> None:
        router = VlmRouter()
        # Server is running 27B premium, but prompt routes to fast
        backend = _mock_backend(
            available_models=["mmangkad/Qwen3.6-27B-NVFP4"],
        )
        response, decision, alignment = await video_chat_routed(
            backend=backend,
            router=router,
            prompt="Describe this clip.",  # would heuristic to fast
            video_url="http://example/clip.mp4",
        )
        assert decision.profile.name == "fast"
        assert decision.profile.model_id == "Qwen/Qwen3-VL-8B-Instruct"
        assert alignment.aligned is False
        assert alignment.actual_model == "mmangkad/Qwen3.6-27B-NVFP4"
        # The chat call still went through, against the running model
        backend.chat.assert_called_once()
        call_kwargs = backend.chat.call_args.kwargs
        assert call_kwargs["model"] == "mmangkad/Qwen3.6-27B-NVFP4"

    @pytest.mark.asyncio
    async def test_empty_model_list_raises(self) -> None:
        router = VlmRouter()
        backend = _mock_backend(available_models=[])
        with pytest.raises(RuntimeError, match="empty model list"):
            await video_chat_routed(
                backend=backend,
                router=router,
                prompt="Describe this clip.",
                video_url="http://example/clip.mp4",
            )

    @pytest.mark.asyncio
    async def test_quality_scope_pin_propagates(self) -> None:
        # Route a "compare" prompt that would normally go premium —
        # but pin to fast via quality_scope. The chat call should
        # request the fast-tier model (assuming it is available).
        router = VlmRouter()
        backend = _mock_backend(
            available_models=[
                "Qwen/Qwen3-VL-8B-Instruct",
                "mmangkad/Qwen3.6-27B-NVFP4",
            ],
        )
        with router.quality_scope("fast"):
            _, decision, alignment = await video_chat_routed(
                backend=backend,
                router=router,
                prompt="Compare frame 1 to frame 5.",
                video_url="http://example/clip.mp4",
            )
        assert decision.profile.name == "fast"
        assert decision.rule_id == "vlm_override_contextvar"
        assert alignment.aligned is True
        assert alignment.actual_model == "Qwen/Qwen3-VL-8B-Instruct"

    @pytest.mark.asyncio
    async def test_video_seconds_drives_long_form_escalation(self) -> None:
        router = VlmRouter()
        backend = _mock_backend(
            available_models=["mmangkad/Qwen3.6-27B-NVFP4"],
        )
        _, decision, _ = await video_chat_routed(
            backend=backend,
            router=router,
            prompt="Describe this lecture.",  # short prompt
            video_url="http://example/lecture.mp4",
            video_seconds=180.0,  # > 60 → long_form
        )
        assert decision.task_class.value == "long_form"
        assert decision.profile.name == "premium"

    @pytest.mark.asyncio
    async def test_extra_messages_prepended_correctly(self) -> None:
        router = VlmRouter()
        backend = _mock_backend(available_models=["Qwen/Qwen3-VL-8B-Instruct"])
        await video_chat_routed(
            backend=backend,
            router=router,
            prompt="What do you see?",
            video_url="http://x/clip.mp4",
            extra_messages=[{"role": "system", "content": "Answer in one sentence."}],
        )
        call_kwargs = backend.chat.call_args.kwargs
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "What do you see?"

    @pytest.mark.asyncio
    async def test_num_frames_override_propagates(self) -> None:
        router = VlmRouter()
        backend = _mock_backend(available_models=["Qwen/Qwen3-VL-8B-Instruct"])
        await video_chat_routed(
            backend=backend,
            router=router,
            prompt="Describe.",
            video_url="http://x/clip.mp4",
            num_frames=8,
        )
        call_kwargs = backend.chat.call_args.kwargs
        assert call_kwargs["video"]["sampling"]["num_frames"] == 8


# ---------------------------------------------------------------------------
# Mismatch policy: restart / raise / fallback
# ---------------------------------------------------------------------------


def _mock_orchestrator(*, post_swap_model: str) -> Any:
    """Build a MagicMock that mimics VLLMOrchestrator's swap surface.

    After ``stop_container()`` + ``start_container_with_profile()`` are
    called, the helper that exercises this mock should arrange for the
    backend's ``list_models`` to return ``[post_swap_model]`` so the
    re-check inside ``ensure_profile_loaded`` sees alignment.
    """
    orch = MagicMock()
    orch.stop_container = MagicMock(return_value=None)
    info = MagicMock()
    info.container_id = "mock_container"
    info.port = 8000
    info.model = post_swap_model
    orch.start_container_with_profile = MagicMock(return_value=info)
    return orch


class TestMismatchPolicy:
    @pytest.mark.asyncio
    async def test_on_mismatch_raise_aborts_with_actionable_error(self) -> None:
        router = VlmRouter()
        backend = _mock_backend(available_models=["mmangkad/Qwen3.6-27B-NVFP4"])
        with pytest.raises(ProfileMismatchError) as excinfo:
            await video_chat_routed(
                backend=backend,
                router=router,
                prompt="Describe this clip.",  # → fast
                video_url="http://x/clip.mp4",
                on_mismatch="raise",
            )
        # Error carries enough info for the UI to render an action button
        assert excinfo.value.recommended_model == "Qwen/Qwen3-VL-8B-Instruct"
        assert "mmangkad/Qwen3.6-27B-NVFP4" in excinfo.value.running_models
        assert excinfo.value.rule_id  # non-empty
        # backend.chat must NOT have been called — raise aborts before chat
        backend.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_mismatch_restart_swaps_then_runs_chat(self) -> None:
        router = VlmRouter()
        # Backend's list_models is dynamic: returns the OLD model first, then
        # the NEW model after the orchestrator swap is "performed".
        old_model = "mmangkad/Qwen3.6-27B-NVFP4"
        new_model = "Qwen/Qwen3-VL-8B-Instruct"
        list_models_responses = [
            [old_model],  # 1: video_chat_routed initial check
            [old_model],  # 2: ensure_profile_loaded entry check (before lock)
            [old_model],  # 3: ensure_profile_loaded re-check (inside lock)
            [new_model],  # 4: ensure_profile_loaded post-swap verify
        ]
        backend = _mock_backend(available_models=[old_model])
        backend.list_models = AsyncMock(side_effect=list_models_responses)

        orchestrator = _mock_orchestrator(post_swap_model=new_model)

        _, decision, alignment = await video_chat_routed(
            backend=backend,
            router=router,
            prompt="Describe this clip.",  # heuristic → fast
            video_url="http://x/clip.mp4",
            orchestrator=orchestrator,
            on_mismatch="restart",
        )
        # Orchestrator was actually invoked
        orchestrator.stop_container.assert_called_once()
        orchestrator.start_container_with_profile.assert_called_once()
        # The profile passed to the orchestrator is the recommended one
        passed_profile = orchestrator.start_container_with_profile.call_args.args[0]
        assert passed_profile.name == "fast"
        assert passed_profile.model_id == new_model
        # Final alignment shows the swap succeeded
        assert alignment.aligned is True
        assert alignment.actual_model == new_model
        # Chat was issued AFTER the swap
        backend.chat.assert_called_once()
        chat_kwargs = backend.chat.call_args.kwargs
        assert chat_kwargs["model"] == new_model

    @pytest.mark.asyncio
    async def test_on_mismatch_restart_without_orchestrator_raises(self) -> None:
        router = VlmRouter()
        backend = _mock_backend(available_models=["mmangkad/Qwen3.6-27B-NVFP4"])
        with pytest.raises(RuntimeError, match="requires an orchestrator"):
            await video_chat_routed(
                backend=backend,
                router=router,
                prompt="Describe this clip.",
                video_url="http://x/clip.mp4",
                on_mismatch="restart",
                orchestrator=None,  # explicit
            )

    @pytest.mark.asyncio
    async def test_aligned_profile_skips_swap(self) -> None:
        # If the running model already matches, no docker stop/start.
        router = VlmRouter()
        backend = _mock_backend(available_models=["Qwen/Qwen3-VL-8B-Instruct"])
        orchestrator = _mock_orchestrator(post_swap_model="should-not-be-used")

        await video_chat_routed(
            backend=backend,
            router=router,
            prompt="Describe this clip.",
            video_url="http://x/clip.mp4",
            orchestrator=orchestrator,
            on_mismatch="restart",
        )
        # Orchestrator must NOT have been touched — no swap needed
        orchestrator.stop_container.assert_not_called()
        orchestrator.start_container_with_profile.assert_not_called()


class TestEnsureProfileLoaded:
    """Direct tests for the standalone ensure_profile_loaded helper."""

    @pytest.mark.asyncio
    async def test_already_aligned_is_noop(self) -> None:
        profile = VLM_PROFILES["fast"]
        backend = _mock_backend(available_models=[profile.model_id])
        orchestrator = _mock_orchestrator(post_swap_model=profile.model_id)

        alignment = await ensure_profile_loaded(
            profile=profile,
            backend=backend,
            orchestrator=orchestrator,
        )
        assert alignment.aligned is True
        # No docker calls made — already aligned
        orchestrator.stop_container.assert_not_called()
        orchestrator.start_container_with_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_swaps_serialise_via_lock(self) -> None:
        # Two concurrent calls for the same target profile must NOT
        # both stop+start. The second one re-checks inside the lock
        # and short-circuits.
        profile = VLM_PROFILES["fast"]
        old_model = "mmangkad/Qwen3.6-27B-NVFP4"

        # Sequence: A.entry=old, B.entry=old, A.in_lock=old, A.post_swap=new,
        # B.in_lock=new (sees alignment, short-circuits). Five calls total.
        responses = [
            [old_model],  # 1: A entry
            [old_model],  # 2: B entry
            [old_model],  # 3: A in-lock re-check (forces swap)
            [profile.model_id],  # 4: A post-swap verify
            [profile.model_id],  # 5: B in-lock re-check (already aligned)
        ]
        backend = _mock_backend(available_models=[old_model])
        backend.list_models = AsyncMock(side_effect=responses)

        orchestrator = _mock_orchestrator(post_swap_model=profile.model_id)

        results = await asyncio.gather(
            ensure_profile_loaded(profile=profile, backend=backend, orchestrator=orchestrator),
            ensure_profile_loaded(profile=profile, backend=backend, orchestrator=orchestrator),
        )
        assert all(r.aligned for r in results)
        # CRITICAL: orchestrator's swap was called exactly ONCE despite
        # two concurrent ensure_profile_loaded calls. Without the lock
        # we'd see 2 stop/start pairs and a GPU memory fight.
        assert orchestrator.stop_container.call_count == 1
        assert orchestrator.start_container_with_profile.call_count == 1
