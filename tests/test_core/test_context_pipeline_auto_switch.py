"""Sprint-24 — ContextPipeline auto-switches the model_router context profile.

The pure heuristic from ``cognithor.core.context_profile_selector`` lives
in its own module and is well-covered. This file verifies the
*production wiring* side:

* When ``ContextPipelineConfig.auto_switch_context_profile`` is True and
  a ModelRouter is wired in, ``enrich(...)`` calls
  ``recommend_context_profile`` and applies the picked profile via
  ``ModelRouter.set_context_profile``.
* The choice is reflected on the returned ``ContextResult`` (so callers
  / observers can log + assert *why* a profile was picked).
* The auto-switch is skipped when the config flag is False, when no
  router is wired, or when the recommendation cannot be applied.
* Smalltalk skips enrichment but still applies the profile (latency-
  tight smalltalk on a chat channel still benefits from ``quick``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cognithor.config import ContextPipelineConfig
from cognithor.core.context_pipeline import ContextPipeline
from cognithor.models import WorkingMemory


@pytest.fixture
def pipeline_config() -> ContextPipelineConfig:
    return ContextPipelineConfig()


def _make_router() -> MagicMock:
    """Mock ModelRouter that records context-profile activations."""
    router = MagicMock()
    router.set_context_profile = MagicMock()
    return router


@pytest.mark.asyncio
async def test_auto_switch_picks_quick_for_short_cli_prompt(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """Short prompt on a simple channel → quick profile."""
    pipeline = ContextPipeline(pipeline_config)
    router = _make_router()
    pipeline.set_model_router(router)

    wm = WorkingMemory()
    result = await pipeline.enrich("Was ist 2+2?", wm, channel_kind="cli")

    assert result.selected_profile == "quick"
    assert "simple channel" in result.profile_reason
    router.set_context_profile.assert_called_once_with("quick")


@pytest.mark.asyncio
async def test_auto_switch_picks_arc_agi3_on_game_channel(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """ARC-AGI-3 channel pins to the validated 128 k profile."""
    pipeline = ContextPipeline(pipeline_config)
    router = _make_router()
    pipeline.set_model_router(router)

    wm = WorkingMemory()
    result = await pipeline.enrich("step 1234", wm, channel_kind="arc_agi3")

    assert result.selected_profile == "arc_agi3"
    assert "channel_kind='arc_agi3'" in result.profile_reason
    router.set_context_profile.assert_called_once_with("arc_agi3")


@pytest.mark.asyncio
async def test_auto_switch_picks_default_for_medium_prompt(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """Prompt that exceeds the quick ceiling but not default → default."""
    pipeline = ContextPipeline(pipeline_config)
    router = _make_router()
    pipeline.set_model_router(router)

    # 32 KB prompt → ~8 k tokens (above quick ceiling at 6 k, below default at 24 k)
    medium_prompt = "x" * 32_000

    wm = WorkingMemory()
    result = await pipeline.enrich(medium_prompt, wm, channel_kind="cli")

    assert result.selected_profile == "default"
    router.set_context_profile.assert_called_once_with("default")


@pytest.mark.asyncio
async def test_auto_switch_attachments_push_medium_to_deep(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """Image attachment + medium prompt → deep (extra recall window)."""
    pipeline = ContextPipeline(pipeline_config)
    router = _make_router()
    pipeline.set_model_router(router)

    wm = WorkingMemory()
    wm.image_attachments = ["/tmp/screenshot.png"]
    medium_prompt = "x" * 30_000  # ~7.5 k tokens, > quick ceiling

    result = await pipeline.enrich(medium_prompt, wm, channel_kind="cli")

    assert result.selected_profile == "deep"
    router.set_context_profile.assert_called_once_with("deep")


@pytest.mark.asyncio
async def test_auto_switch_disabled_via_config_skips_router_call(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """Flag off → never touches the router, returns empty profile fields."""
    pipeline_config.auto_switch_context_profile = False
    pipeline = ContextPipeline(pipeline_config)
    router = _make_router()
    pipeline.set_model_router(router)

    wm = WorkingMemory()
    result = await pipeline.enrich("Was ist 2+2?", wm, channel_kind="cli")

    assert result.selected_profile == ""
    assert result.profile_reason == ""
    router.set_context_profile.assert_not_called()


@pytest.mark.asyncio
async def test_auto_switch_skipped_when_no_router_wired(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """Flag on but no router → no-op + diagnostic reason."""
    pipeline = ContextPipeline(pipeline_config)
    # Deliberately do NOT call set_model_router(...).

    wm = WorkingMemory()
    result = await pipeline.enrich("Was ist 2+2?", wm, channel_kind="cli")

    assert result.selected_profile == ""
    assert result.profile_reason == "no model_router wired"


@pytest.mark.asyncio
async def test_auto_switch_swallows_router_failure(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """If the router rejects the profile (e.g. ValueError), enrichment continues."""
    pipeline = ContextPipeline(pipeline_config)
    router = MagicMock()
    router.set_context_profile = MagicMock(side_effect=ValueError("nope"))
    pipeline.set_model_router(router)

    wm = WorkingMemory()
    result = await pipeline.enrich("Was ist 2+2?", wm, channel_kind="cli")

    # The auto-switch step recorded its failure but the pipeline still
    # produced a result (smalltalk-skip path here).
    assert result.profile_reason == "set failed"
    assert result.selected_profile == ""


@pytest.mark.asyncio
async def test_auto_switch_runs_even_for_smalltalk(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """Smalltalk skips enrichment but still flips the profile —
    latency-tight short replies should not run on the deep profile.
    """
    pipeline = ContextPipeline(pipeline_config)
    router = _make_router()
    pipeline.set_model_router(router)

    wm = WorkingMemory()
    result = await pipeline.enrich("hallo", wm, channel_kind="cli")

    assert result.skipped is True
    assert result.skip_reason == "smalltalk"
    # Profile was picked + applied before the smalltalk shortcut returned.
    assert result.selected_profile == "quick"
    router.set_context_profile.assert_called_once_with("quick")


@pytest.mark.asyncio
async def test_auto_switch_no_channel_kind_falls_back_to_length_only(
    pipeline_config: ContextPipelineConfig,
) -> None:
    """No channel_kind → heuristic uses prompt length only."""
    pipeline = ContextPipeline(pipeline_config)
    router = _make_router()
    pipeline.set_model_router(router)

    wm = WorkingMemory()
    # Short prompt, no channel: heuristic returns ``default`` (Rule 5
    # fallback) because the simple-channel rule needs an explicit kind.
    result = await pipeline.enrich("Was ist 2+2?", wm)

    assert result.selected_profile == "default"
    router.set_context_profile.assert_called_once_with("default")
