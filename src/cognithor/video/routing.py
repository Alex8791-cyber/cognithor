"""Wire-up between :class:`VlmRouter` and :class:`VLLMBackend`.

This module is the *single* place that knows how to take a video chat
request and: (a) consult the router for the recommended profile,
(b) cross-check whether the running vLLM server is actually serving
that profile's model, (c) issue the chat call, and (d) hand back both
the response and the routing decision so the TRUST-1 receipt builder
can capture the full provenance trail.

Why a separate module rather than a method on ``VLLMBackend``? Two
reasons:

1. ``VLLMBackend`` is the *transport* layer — model-agnostic, used by
   every channel that needs to talk to vLLM. Routing is a domain
   concern (which model is best for *this* request). Mixing them
   couples transport changes to routing changes; keeping them apart
   means an A/B harness can swap the router without touching the
   transport.
2. The router's recommendation may not match what's running. A 32 GB
   GPU can host exactly one Qwen3.6-27B-NVFP4 instance — the router
   may say "premium" but the running container might be ``fast``.
   Resolving that mismatch (warning vs hard-fail vs auto-restart) is
   policy that belongs *above* the transport, not inside it.

Public surface:

* :func:`video_chat_routed` — the main entry point. Async, returns
  ``(ChatResponse, VlmRoutingDecision)``.
* :func:`check_profile_alignment` — pure-function consistency check
  used by the demo / smoke scripts; no side effects.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.core.llm_backend import ChatResponse
    from cognithor.core.vllm_backend import VLLMBackend
    from cognithor.core.vllm_orchestrator import VLLMOrchestrator
    from cognithor.core.vlm_router import VlmProfile, VlmRouter, VlmRoutingDecision

log = get_logger(__name__)


# Module-level lock so two concurrent routed-chat calls cannot both
# trigger a profile swap at the same time — the second one would race
# against the first one's docker stop/start and either (a) crash the
# first request or (b) fight over GPU memory.
_swap_lock = asyncio.Lock()


class ProfileMismatchError(RuntimeError):
    """Raised when on_mismatch="raise" and the running model does not match.

    Carries the recommended profile and the running model so the caller
    can render an actionable error (e.g. show the user a "Restart vLLM
    with profile X" button in the UI).
    """

    def __init__(
        self,
        recommended_model: str,
        running_models: tuple[str, ...],
        rule_id: str,
    ) -> None:
        self.recommended_model = recommended_model
        self.running_models = running_models
        self.rule_id = rule_id
        super().__init__(
            f"VLM-router recommended {recommended_model!r} but the vLLM server "
            f"is serving {list(running_models)!r}. on_mismatch='raise' policy "
            f"refuses to fall back. (rule_id={rule_id})"
        )


@dataclass(frozen=True)
class ProfileAlignment:
    """Outcome of comparing a routing decision to a live vLLM server.

    ``aligned == True`` means the running server is serving the model
    the router recommended. False means a mismatch — caller decides
    whether to proceed with the running model, abort, or trigger a
    profile-switch via the launch wizard.
    """

    aligned: bool
    recommended_model: str
    available_models: tuple[str, ...]
    actual_model: str | None
    """The model the call will actually use — None if no fallback exists."""


def check_profile_alignment(
    decision: VlmRoutingDecision,
    available_models: list[str] | tuple[str, ...],
) -> ProfileAlignment:
    """Compare a router recommendation against the live ``list_models()``.

    Pure function — no I/O. Use this from a demo/smoke script after
    a single ``backend.list_models()`` call so the alignment check is
    deterministic + cheap to test.

    Args:
        decision: The output of :meth:`VlmRouter.select_profile_explained`.
        available_models: Whatever ``backend.list_models()`` returned.

    Returns:
        :class:`ProfileAlignment` with ``aligned`` bit + best-effort
        fallback model. When the recommended model is not running and
        at least one other VLM-class model is available, ``actual_model``
        falls back to the first one. When no models are available,
        ``actual_model`` is None and the caller must abort.
    """
    available_tuple = tuple(available_models)
    recommended = decision.profile.model_id
    if recommended in available_tuple:
        return ProfileAlignment(
            aligned=True,
            recommended_model=recommended,
            available_models=available_tuple,
            actual_model=recommended,
        )
    fallback = available_tuple[0] if available_tuple else None
    return ProfileAlignment(
        aligned=False,
        recommended_model=recommended,
        available_models=available_tuple,
        actual_model=fallback,
    )


async def ensure_profile_loaded(
    *,
    profile: VlmProfile,
    backend: VLLMBackend,
    orchestrator: VLLMOrchestrator,
    health_timeout: int = 300,
) -> ProfileAlignment:
    """Ensure the running vLLM container is serving ``profile.model_id``.

    No-op if already aligned. Otherwise: stop the running container,
    start a fresh one with the profile's exact flags, and wait for
    ``/health``. Holds a module-level lock so concurrent routed chats
    cannot stomp on each other.

    Returns:
        :class:`ProfileAlignment` reflecting the *post-swap* state. If
        the swap succeeds, ``aligned`` is True. If something fails
        mid-swap, the lock is released and the exception propagates —
        caller decides whether to retry.
    """
    available = await backend.list_models()
    fake_decision = type(
        "_Stub",
        (),
        {"profile": profile, "rule_id": "ensure_profile_loaded"},
    )()
    alignment = check_profile_alignment(fake_decision, available)
    if alignment.aligned:
        return alignment

    async with _swap_lock:
        # Re-check inside the lock — another coroutine may have done the
        # swap between our first check and acquiring the lock.
        available = await backend.list_models()
        alignment = check_profile_alignment(fake_decision, available)
        if alignment.aligned:
            return alignment

        log.info(
            "vlm_profile_swap_initiated",
            target_profile=profile.name,
            target_model=profile.model_id,
            currently_running=list(available),
            health_timeout_s=health_timeout,
        )
        # subprocess calls in VLLMOrchestrator are sync — push them off
        # the event loop so we don't block other awaits.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, orchestrator.stop_container)
        await loop.run_in_executor(
            None,
            lambda: orchestrator.start_container_with_profile(
                profile, health_timeout=health_timeout
            ),
        )

        # Backend may cache the model list — force a fresh round-trip.
        available = await backend.list_models()
        alignment = check_profile_alignment(fake_decision, available)
        if not alignment.aligned:
            # Rare: container started but the model id we got back
            # differs from what the profile claims (e.g. served-model-name
            # rewrites). Surface clearly.
            log.error(
                "vlm_profile_swap_yielded_unexpected_model",
                expected=profile.model_id,
                got=list(available),
            )
        else:
            log.info(
                "vlm_profile_swap_complete",
                profile=profile.name,
                model=profile.model_id,
            )
        return alignment


async def video_chat_routed(
    *,
    backend: VLLMBackend,
    router: VlmRouter,
    prompt: str,
    video_url: str,
    video_seconds: float | None = None,
    explicit_class: str | None = None,
    num_frames: int | None = None,
    temperature: float = 0.2,
    extra_messages: list[dict[str, object]] | None = None,
    orchestrator: VLLMOrchestrator | None = None,
    on_mismatch: Literal["restart", "fallback", "raise"] = "fallback",
    swap_timeout: int = 300,
) -> tuple[ChatResponse, VlmRoutingDecision, ProfileAlignment]:
    """Issue a video-aware chat call with router-driven model selection.

    Flow:

    1. Router picks a profile from ``prompt`` + ``video_seconds``.
    2. ``backend.list_models()`` is consulted to confirm the running
       server is serving the recommended model.
    3. If the running server matches → use the recommended model.
       If not → log a structured ``vlm_profile_mismatch`` warning and
       fall back to whatever the running server has. The decision +
       alignment record are still returned so the receipt captures
       the original recommendation.
    4. Chat is issued with the chosen model + the video attachment in
       the OpenAI ``video_url`` content shape (handled by
       ``VLLMBackend._attach_video_to_last_user``).

    Args:
        backend: A live :class:`VLLMBackend` (caller-owned).
        router: A :class:`VlmRouter`. Use ``router.quality_scope(...)``
            around this call to force a tier.
        prompt: The user's question or instruction about the video.
        video_url: HTTP/HTTPS URL the vLLM container can fetch
            (``host.docker.internal:<port>/...`` for local upload).
        video_seconds: Clip duration. Drives long-form escalation.
        explicit_class: Optional caller-side override of the
            classifier. See :func:`classify_vlm_task`.
        num_frames: Override for the frame-sampling cap. Defaults to
            the profile's ``max_frames``.
        temperature: vLLM sampling temperature. Defaults to 0.2 because
            video tasks usually want low-variance answers.
        extra_messages: Prepend a system message or any extra context.
            The user message containing the video is appended last by
            this function.

    Returns:
        ``(response, decision, alignment)`` — response is the
        :class:`ChatResponse` from vLLM; decision is the router's
        record (TRUST-2-ready); alignment is the consistency check
        between the recommendation and the live server.
    """
    decision = router.select_profile_explained(
        prompt,
        video_seconds=video_seconds,
        explicit_class=explicit_class,
    )
    available = await backend.list_models()
    alignment = check_profile_alignment(decision, available)

    # ── Mismatch policy ────────────────────────────────────────────────
    # Three behaviours, picked by ``on_mismatch``:
    #   * "restart"  — orchestrator stops the running container and
    #                  starts a fresh one with the recommended profile's
    #                  flags. Synchronous; takes ~120 s on a warm cache.
    #                  Holds a module-level lock so concurrent calls
    #                  serialise, not race.
    #   * "fallback" — log a structured warning, run the chat against
    #                  whatever's running. Receipt still records the
    #                  original recommendation.
    #   * "raise"    — refuse to fall back; raise ProfileMismatchError.
    #                  UI uses this to render an actionable
    #                  "Switch to <profile>?" button.
    if not alignment.aligned and on_mismatch == "restart":
        if orchestrator is None:
            raise RuntimeError(
                "on_mismatch='restart' requires an orchestrator argument. "
                "Pass cognithor.core.vllm_orchestrator.VLLMOrchestrator or "
                "switch to on_mismatch='fallback'."
            )
        log.info(
            "vlm_swap_triggered_by_mismatch",
            recommended=decision.profile.model_id,
            running=list(available),
            rule_id=decision.rule_id,
        )
        alignment = await ensure_profile_loaded(
            profile=decision.profile,
            backend=backend,
            orchestrator=orchestrator,
            health_timeout=swap_timeout,
        )

    if alignment.actual_model is None:
        # No models on the server at all — surface a clear error.
        msg = (
            f"vlm_router recommended {decision.profile.model_id!r} but the "
            f"vLLM server returned an empty model list. Aborting."
        )
        log.error(
            "vlm_no_models_available",
            recommended=decision.profile.model_id,
            rule_id=decision.rule_id,
        )
        raise RuntimeError(msg)

    if not alignment.aligned:
        if on_mismatch == "raise":
            raise ProfileMismatchError(
                recommended_model=decision.profile.model_id,
                running_models=alignment.available_models,
                rule_id=decision.rule_id,
            )
        # on_mismatch == "fallback" (or "restart" that yielded a partial swap)
        log.warning(
            "vlm_profile_mismatch",
            recommended=decision.profile.model_id,
            available=list(available),
            actual=alignment.actual_model,
            rule_id=decision.rule_id,
            hint=(
                "The running vLLM container is serving a different model "
                "than the router recommended. Proceeding with the running "
                "model. To honour the recommendation, pass "
                "on_mismatch='restart' with an orchestrator argument, or "
                f"manually run: `{' '.join(decision.profile.vllm_serve_command())}`."
            ),
        )

    messages: list[dict[str, object]] = list(extra_messages or [])
    messages.append({"role": "user", "content": prompt})

    sampling_frames = num_frames if num_frames is not None else decision.profile.max_frames

    response = await backend.chat(
        model=alignment.actual_model,
        messages=messages,
        temperature=temperature,
        video={
            "url": video_url,
            "sampling": {"num_frames": sampling_frames},
        },
    )
    log.info(
        "vlm_routed_chat_complete",
        rule_id=decision.rule_id,
        profile=decision.profile.name,
        recommended=decision.profile.model_id,
        actual=alignment.actual_model,
        aligned=alignment.aligned,
    )
    return response, decision, alignment


__all__ = [
    "ProfileAlignment",
    "ProfileMismatchError",
    "check_profile_alignment",
    "ensure_profile_loaded",
    "video_chat_routed",
]
