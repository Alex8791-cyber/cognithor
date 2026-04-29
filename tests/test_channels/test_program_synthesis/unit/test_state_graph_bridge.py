# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""State-Graph-Navigator bridge tests (spec §15)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.core.types import TaskSpec
from cognithor.channels.program_synthesis.integration.state_graph_bridge import (
    NEUTRAL_MULTIPLIER,
    PROMOTED_MULTIPLIER,
    SUPPORTED_HINT_KEYS,
    StateGraphBridge,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _spec() -> TaskSpec:
    return TaskSpec(
        examples=((_g([[1, 2]]), _g([[2, 1]])),),
    )


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_empty_sgn_result_returns_spec_unchanged(self) -> None:
        spec = _spec()
        out = StateGraphBridge.annotate(spec, {})
        assert out is spec

    def test_supported_hint_lands_in_annotations(self) -> None:
        spec = _spec()
        out = StateGraphBridge.annotate(spec, {"mirror_horizontal": True})
        keys = {k for k, _ in out.annotations}
        assert "sgn:mirror_horizontal" in keys

    def test_unsupported_hint_dropped_silently(self) -> None:
        spec = _spec()
        out = StateGraphBridge.annotate(spec, {"unknown_field": "value"})
        assert out is spec

    def test_partial_supported_keeps_only_known(self) -> None:
        spec = _spec()
        out = StateGraphBridge.annotate(spec, {"rotate90": True, "noise": "drop"})
        keys = {k for k, _ in out.annotations}
        assert "sgn:rotate90" in keys
        assert "sgn:noise" not in keys

    def test_existing_annotations_preserved(self) -> None:
        spec = TaskSpec(
            examples=((_g([[1]]), _g([[1]])),),
            annotations=(("source", "test"),),
        )
        out = StateGraphBridge.annotate(spec, {"rotate180": True})
        keys = {k for k, _ in out.annotations}
        assert "source" in keys
        assert "sgn:rotate180" in keys


# ---------------------------------------------------------------------------
# cost_multipliers
# ---------------------------------------------------------------------------


class TestCostMultipliers:
    def test_promotes_truthy_hint(self) -> None:
        out = StateGraphBridge.cost_multipliers({"sgn:rotate90": True})
        assert out == {"rotate90": PROMOTED_MULTIPLIER}

    def test_demotes_falsy_hint(self) -> None:
        out = StateGraphBridge.cost_multipliers({"sgn:rotate90": False})
        assert out == {"rotate90": NEUTRAL_MULTIPLIER}

    def test_ignores_unprefixed_keys(self) -> None:
        out = StateGraphBridge.cost_multipliers({"rotate90": True})
        assert out == {}

    def test_ignores_unsupported_primitive(self) -> None:
        out = StateGraphBridge.cost_multipliers({"sgn:hallucinated_primitive": True})
        assert out == {}

    def test_multiple_hints_combined(self) -> None:
        out = StateGraphBridge.cost_multipliers(
            {
                "sgn:rotate90": True,
                "sgn:mirror_vertical": True,
            }
        )
        assert out == {
            "rotate90": PROMOTED_MULTIPLIER,
            "mirror_vertical": PROMOTED_MULTIPLIER,
        }

    def test_empty_returns_empty(self) -> None:
        assert StateGraphBridge.cost_multipliers({}) == {}


class TestSupportedHintKeys:
    def test_includes_phase_1_geometry_set(self) -> None:
        for k in (
            "mirror_horizontal",
            "mirror_vertical",
            "rotate90",
            "rotate180",
            "rotate270",
        ):
            assert k in SUPPORTED_HINT_KEYS

    def test_includes_scale_set(self) -> None:
        for k in ("scale_up_2x", "scale_up_3x", "scale_down_2x"):
            assert k in SUPPORTED_HINT_KEYS

    def test_recolor_only_hint_present(self) -> None:
        assert "recolor_only" in SUPPORTED_HINT_KEYS
