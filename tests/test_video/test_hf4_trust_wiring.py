"""Sprint-27 HF-4 — TRUST wiring tests for the video pipeline.

Verifies that a successful ``video_render`` call emits both a
:class:`ProvenanceTag` (TRUST-9) and a :class:`CostEntry`
(TRUST-9 cost-ledger) into the canonical ledgers, AND that the
emission is best-effort: if either ledger raises, the MCP tool
still returns the render result to the caller.

Tests use the canonical singletons (``PROVENANCE_LEDGER``,
``COST_LEDGER``) and clear them inside fixtures so they don't
contaminate other test files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cognithor.mcp.video_tools import _video_render_handler
from cognithor.memory.provenance import (
    PROVENANCE_LEDGER,
    SourceType,
)
from cognithor.security.cost_ledger import COST_LEDGER, CostKind
from cognithor.video.renderer_base import (
    OutputFormat,
    RendererABC,
    RenderResult,
)


@pytest.fixture(autouse=True)
def _isolate_ledgers() -> Any:
    """Clear provenance + cost ledgers around every test."""

    PROVENANCE_LEDGER.clear()
    COST_LEDGER.clear()
    yield
    PROVENANCE_LEDGER.clear()
    COST_LEDGER.clear()


class _FakeRenderer(RendererABC):
    """In-process renderer returning a synthetic RenderResult."""

    NAME = "fake"

    def __init__(self) -> None:
        # Compatible no-op: we override render() to skip network entirely.
        pass

    async def is_available(self) -> bool:
        return True

    def supported_adapters(self) -> Any:  # type: ignore[override]
        from cognithor.video.renderer_base import DEFAULT_ALLOWED_ADAPTERS

        return DEFAULT_ALLOWED_ADAPTERS

    async def render(self, request: Any) -> RenderResult:
        # Synthetic output path; the file does NOT need to exist
        # because the TRUST-wiring uses the path as a string identifier.
        out = Path("/tmp") / "fake-render-output.mp4"
        return RenderResult(
            run_id=request.run_id,
            output_path=out,
            output_format=OutputFormat.MP4,
            duration_ms=4321,
            width=request.width,
            height=request.height,
            fps=request.fps,
            bytes_written=98765,
        )


@pytest.fixture
def fake_renderer_factory() -> Any:
    """Patch the registry so video_render dispatches to _FakeRenderer."""

    from cognithor.mcp import video_tools

    fake_registry = {"fake": _FakeRenderer}
    with patch.dict(video_tools.renderer_registry, fake_registry, clear=False):
        yield


# ---------------------------------------------------------------------------
# Happy-path emission
# ---------------------------------------------------------------------------


class TestEmitProvenance:
    @pytest.mark.asyncio
    async def test_provenance_tag_recorded_for_rendered_mp4(
        self,
        fake_renderer_factory: Any,
    ) -> None:
        result = await _video_render_handler(
            run_id="run-prov-1",
            html_text="<div data-composition-id='x'></div>",
            renderer="fake",
        )
        assert result["ok"] is True
        # Output path is the ledger key.
        out_path = result["output_path"]
        tag = PROVENANCE_LEDGER.current(out_path)
        assert tag is not None
        assert tag.source_type == SourceType.TOOL_OUTPUT
        # source_id should hash-encode the html_text input.
        assert tag.source_id.startswith("html-text:")
        # Notes mention the output format + dimensions.
        assert "mp4" in tag.notes
        assert "run-prov-1" in tag.notes

    @pytest.mark.asyncio
    async def test_provenance_source_id_distinguishes_text_vs_path(
        self,
        tmp_path: Path,
        fake_renderer_factory: Any,
    ) -> None:
        # Path-based input — source_id should include the path.
        html_path = tmp_path / "comp.html"
        html_path.write_text("<div/>", encoding="utf-8")
        result = await _video_render_handler(
            run_id="run-prov-2",
            html_path=str(html_path),
            renderer="fake",
        )
        assert result["ok"] is True
        tag = PROVENANCE_LEDGER.current(result["output_path"])
        assert tag is not None
        assert tag.source_id.startswith("html-path:")
        assert str(html_path) in tag.source_id


class TestEmitCost:
    @pytest.mark.asyncio
    async def test_cost_entry_recorded_zero_micro_usd(
        self,
        fake_renderer_factory: Any,
    ) -> None:
        await _video_render_handler(
            run_id="run-cost-1",
            html_text="<div/>",
            renderer="fake",
        )
        # Filter by tool name to find our entry without depending on
        # ledger size (other tests may have written too).
        entries = [e for e in COST_LEDGER.entries() if e.tool == "video_render"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry.kind == CostKind.OTHER
        assert entry.cost_usd_micro == 0
        assert entry.run_id == "run-cost-1"
        # Subprocess runtime rides unit_count.
        assert entry.unit_count == 4321  # synthesised by _FakeRenderer
        assert "mp4" in entry.notes
        assert "98765" in entry.notes


# ---------------------------------------------------------------------------
# Best-effort: TRUST emission must NEVER fail the caller
# ---------------------------------------------------------------------------


class TestBestEffort:
    @pytest.mark.asyncio
    async def test_provenance_failure_does_not_break_render(
        self,
        fake_renderer_factory: Any,
    ) -> None:
        with patch.object(
            PROVENANCE_LEDGER,
            "tag",
            side_effect=RuntimeError("ledger fault"),
        ):
            result = await _video_render_handler(
                run_id="run-fault-1",
                html_text="<div/>",
                renderer="fake",
            )
        # Render succeeded despite ledger fault.
        assert result["ok"] is True
        assert result["run_id"] == "run-fault-1"

    @pytest.mark.asyncio
    async def test_cost_failure_does_not_break_render(
        self,
        fake_renderer_factory: Any,
    ) -> None:
        with patch.object(
            COST_LEDGER,
            "record",
            side_effect=RuntimeError("ledger fault"),
        ):
            result = await _video_render_handler(
                run_id="run-fault-2",
                html_text="<div/>",
                renderer="fake",
            )
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Failure path — no TRUST emission
# ---------------------------------------------------------------------------


class TestNoEmissionOnFailure:
    @pytest.mark.asyncio
    async def test_unknown_renderer_emits_no_tag(self) -> None:
        await _video_render_handler(
            run_id="run-fail-1",
            html_text="<div/>",
            renderer="does-not-exist",
        )
        # Empty ledger → no entry was emitted
        assert len(PROVENANCE_LEDGER) == 0
        assert len(COST_LEDGER.entries()) == 0
