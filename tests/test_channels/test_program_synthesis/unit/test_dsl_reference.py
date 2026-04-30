# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Auto-generated DSL reference tests (D9 piece)."""

from __future__ import annotations

import io
from pathlib import Path

from cognithor.channels.program_synthesis.cli import main
from cognithor.channels.program_synthesis.cli.dsl_reference import (
    render_dsl_reference,
    write_dsl_reference,
)
from cognithor.channels.program_synthesis.core.version import (
    DSL_VERSION,
    PSE_VERSION,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

# ---------------------------------------------------------------------------
# render_dsl_reference
# ---------------------------------------------------------------------------


class TestRenderDslReference:
    def test_includes_pse_and_dsl_versions(self) -> None:
        body = render_dsl_reference()
        assert PSE_VERSION in body
        assert DSL_VERSION in body

    def test_lists_canonical_primitives(self) -> None:
        body = render_dsl_reference()
        # Spot-check every group has at least one expected primitive.
        for primitive_name in (
            "rotate90",
            "recolor",
            "scale_up_2x",
            "gravity_down",
            "connected_components_4",
            "mask_eq",
            "stack_horizontal",
            "const_color_0",
            "map_objects",
            "filter_objects",
            "align_to",
            "sort_objects",
            "branch",
        ):
            assert f"`{primitive_name}`" in body, f"missing {primitive_name}"

    def test_predicate_constructors_listed(self) -> None:
        body = render_dsl_reference()
        for name in (
            "color_eq",
            "color_in",
            "size_gt",
            "is_rectangle",
            "is_largest_in",
            "touches_border",
            "and",
            "or",
            "not",
        ):
            assert f"`{name}`" in body

    def test_groups_by_output_type(self) -> None:
        body = render_dsl_reference()
        assert "Output type: `Grid`" in body
        assert "Output type: `Color`" in body
        assert "Output type: `ObjectSet`" in body

    def test_total_count_present(self) -> None:
        body = render_dsl_reference()
        assert f"**{len(REGISTRY)} primitives**" in body

    def test_deterministic_across_calls(self) -> None:
        # Same registry → same bytes.
        a = render_dsl_reference()
        b = render_dsl_reference()
        assert a == b

    def test_table_columns_present(self) -> None:
        body = render_dsl_reference()
        assert "| Name | Signature | Cost | Description |" in body


# ---------------------------------------------------------------------------
# write_dsl_reference (file path)
# ---------------------------------------------------------------------------


class TestWriteDslReference:
    def test_round_trip_via_file(self, tmp_path: Path) -> None:
        target = tmp_path / "ref.md"
        n = write_dsl_reference(str(target))
        body = target.read_text(encoding="utf-8")
        assert n == len(body.encode("utf-8"))
        assert body.startswith("# Cognithor PSE — ARC-DSL Reference")

    def test_lf_line_endings_only(self, tmp_path: Path) -> None:
        target = tmp_path / "ref.md"
        write_dsl_reference(str(target))
        raw = target.read_bytes()
        # Spec convention for the project: LF-only line endings, even
        # on Windows. open(..., newline="\n") enforces this.
        assert b"\r\n" not in raw


# ---------------------------------------------------------------------------
# CLI subcommand wiring
# ---------------------------------------------------------------------------


class TestCliSubcommand:
    def test_pse_dsl_reference_to_stdout(self) -> None:
        buf = io.StringIO()
        rc = main(["dsl", "reference"], stream=buf)
        assert rc == 0
        out = buf.getvalue()
        assert "# Cognithor PSE — ARC-DSL Reference" in out
        assert "rotate90" in out

    def test_pse_dsl_reference_to_file(self, tmp_path: Path) -> None:
        target = tmp_path / "from_cli.md"
        buf = io.StringIO()
        rc = main(["dsl", "reference", "--output", str(target)], stream=buf)
        assert rc == 0
        # Echo confirms the write.
        assert "wrote" in buf.getvalue()
        assert str(target) in buf.getvalue()
        # File contents match the in-memory render.
        assert target.read_text(encoding="utf-8") == render_dsl_reference()


# ---------------------------------------------------------------------------
# Drift check — committed reference must match what the renderer
# currently produces. This locks the docs against silent staleness.
# ---------------------------------------------------------------------------


class TestCommittedReferenceUpToDate:
    def test_dsl_reference_md_matches_renderer(self) -> None:
        committed_path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "channels"
            / "program_synthesis"
            / "dsl_reference.md"
        )
        if not committed_path.is_file():
            # If the doc isn't committed yet (first-run scaffold),
            # skip rather than fail. Once committed in CI the test is
            # a hard drift gate.
            import pytest

            pytest.skip(f"dsl_reference.md not committed at {committed_path}")
        committed = committed_path.read_text(encoding="utf-8")
        assert committed == render_dsl_reference(), (
            "docs/channels/program_synthesis/dsl_reference.md is stale — "
            "regenerate via `cognithor pse dsl reference --output "
            "docs/channels/program_synthesis/dsl_reference.md`"
        )
