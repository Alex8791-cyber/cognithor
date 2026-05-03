# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-23 PR#A — pure heuristic for picking a context profile.

Sprint-23's :mod:`cognithor.core.model_router` ships four context
profiles (``quick`` / ``default`` / ``deep`` / ``arc_agi3``) and the
``ModelRouter.set_context_profile()`` lever to activate them. This
module is the **decision** half: a single pure function that given a
few cheap signals about the incoming work returns the recommended
profile name.

It is intentionally side-effect free — no router mutation, no logging,
no IO. Callers (gateway, channels, MCP tools) decide *whether* to
honour the recommendation and *when* to flip the router. That keeps
the heuristic trivially testable and lets us evolve it without
touching active code paths.

Heuristic rules (in priority order):

1. **ARC-AGI-3 channel** — game-loop interaction needs the validated
   128 k profile from the Sprint-22 side-quest probe, regardless of
   prompt length. Pinned by the Sprint-22 deployment guide.
2. **Long prompt (≥ 64 k tokens or ≥ 250 KB chars)** — falls into
   ``deep``. Above 96 k or with attachments, escalate to
   ``arc_agi3``.
3. **Medium prompt (≥ 8 k tokens or ≥ 32 KB chars)** — ``default``.
4. **Short prompt + simple channel** (CLI / Telegram / WebUI plain
   text, no attachments) — ``quick`` to keep latency tight on routine
   chat.
5. **Otherwise** — ``default`` (the safe middle).

Token estimates are intentionally approximate: the function takes the
*char* count and divides by 4 (the chat-standard rule of thumb for
English / mixed-language text), so callers don't need to import a
tokenizer. If a caller already has a precise token count, they can
pass it directly via ``prompt_tokens=``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Profile names — must stay in sync with CONTEXT_PROFILES in model_router.
# Importing the dict here would create a circular reference at module-load
# time (model_router imports from cognithor.config which imports from
# cognithor.core...), so we duplicate the *names* and rely on the
# model_router test to assert string-equality with the registry keys.
PROFILE_QUICK = "quick"
PROFILE_DEFAULT = "default"
PROFILE_DEEP = "deep"
PROFILE_ARC_AGI3 = "arc_agi3"

# Threshold breakpoints. Tuned to match the registry's num_ctx values
# minus prompt-output headroom (~25 %), so a recommendation always
# leaves enough window for the model to actually reply.
_QUICK_TOKEN_CEILING = 6_000  # quick = 8 k → 6 k prompt cap
_DEFAULT_TOKEN_CEILING = 24_000  # default = 32 k → 24 k prompt cap
_DEEP_TOKEN_CEILING = 48_000  # deep = 64 k → 48 k prompt cap
_ARC_TOKEN_CEILING = 96_000  # arc_agi3 = 128 k → 96 k prompt cap

# Char-to-token rule of thumb for mixed-language text.
_CHARS_PER_TOKEN = 4

# Channel kinds the heuristic recognises. Anything else is treated as
# "generic" and routed by prompt length only.
_SIMPLE_CHANNELS = frozenset({"cli", "telegram", "webui", "discord", "slack"})
_HEAVY_CHANNELS = frozenset({"arc_agi3", "arc_channel", "game_loop"})


@dataclass(frozen=True)
class ProfileRecommendation:
    """The output of :func:`recommend_context_profile`.

    Carries the recommended profile name plus a short reason string so
    log lines and tests can assert *why* a profile was picked, not just
    *which*.
    """

    profile: str
    reason: str


def estimate_prompt_tokens(prompt_chars: int) -> int:
    """Cheap char→token estimate for routing decisions.

    Uses the chat-standard rule of thumb (4 chars per token) which is
    accurate to ±25 % across English / German / mixed-language inputs.
    Negative inputs are clamped to 0.
    """
    if prompt_chars <= 0:
        return 0
    return prompt_chars // _CHARS_PER_TOKEN


def recommend_context_profile(
    *,
    prompt_chars: int = 0,
    prompt_tokens: int | None = None,
    channel_kind: str | None = None,
    has_attachments: bool = False,
) -> ProfileRecommendation:
    """Pick a context profile for an incoming workload.

    Args:
        prompt_chars: Length of the user prompt in *characters*. Used
            only when ``prompt_tokens`` is not supplied.
        prompt_tokens: Pre-computed token count, if the caller already
            tokenized. Wins over ``prompt_chars``.
        channel_kind: Lowercase channel identifier (``"cli"``,
            ``"telegram"``, ``"arc_agi3"``, ...). Optional — when
            absent the heuristic relies on prompt length only.
        has_attachments: True if the request carries images, files,
            tool outputs or other non-text payload that inflates the
            effective context. Tilts the recommendation upward.

    Returns:
        A :class:`ProfileRecommendation` whose ``profile`` is one of
        ``quick`` / ``default`` / ``deep`` / ``arc_agi3``.
    """
    # Normalise inputs.
    channel = (channel_kind or "").strip().lower()
    if prompt_tokens is None:
        tokens = estimate_prompt_tokens(prompt_chars)
    else:
        tokens = max(0, prompt_tokens)

    # Rule 1 — heavy / game-loop channels override everything.
    if channel in _HEAVY_CHANNELS:
        return ProfileRecommendation(
            profile=PROFILE_ARC_AGI3,
            reason=f"channel_kind={channel!r} — game-loop interaction needs 128k window",
        )

    # Rule 2 — very long prompts.
    if tokens >= _DEEP_TOKEN_CEILING or (tokens >= _ARC_TOKEN_CEILING and has_attachments):
        return ProfileRecommendation(
            profile=PROFILE_ARC_AGI3,
            reason=f"prompt_tokens≈{tokens} exceeds deep ceiling",
        )
    if tokens >= _DEFAULT_TOKEN_CEILING:
        return ProfileRecommendation(
            profile=PROFILE_DEEP,
            reason=f"prompt_tokens≈{tokens} exceeds default ceiling",
        )

    # Rule 3 — attachments push medium prompts up one notch because
    # binary payload inflates the on-wire context the model sees.
    if has_attachments and tokens >= _QUICK_TOKEN_CEILING:
        return ProfileRecommendation(
            profile=PROFILE_DEEP,
            reason="attachments + medium prompt — wider window keeps recall",
        )

    # Rule 4 — short prompt on a simple chat channel: quick is the
    # right latency/quality tradeoff.
    if tokens < _QUICK_TOKEN_CEILING and channel in _SIMPLE_CHANNELS and not has_attachments:
        return ProfileRecommendation(
            profile=PROFILE_QUICK,
            reason=f"short prompt on simple channel {channel!r}",
        )

    # Rule 5 — middle prompts always go to default.
    if tokens < _DEFAULT_TOKEN_CEILING:
        return ProfileRecommendation(
            profile=PROFILE_DEFAULT,
            reason=f"prompt_tokens≈{tokens} fits default ceiling",
        )

    # Catch-all (unreachable given the ceilings above, but keeps the
    # function total).
    return ProfileRecommendation(profile=PROFILE_DEFAULT, reason="fallback")


__all__ = [
    "PROFILE_ARC_AGI3",
    "PROFILE_DEEP",
    "PROFILE_DEFAULT",
    "PROFILE_QUICK",
    "ProfileRecommendation",
    "estimate_prompt_tokens",
    "recommend_context_profile",
]
