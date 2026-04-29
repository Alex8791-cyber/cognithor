# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PGE-Trinity adapter tests (spec §12)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.integration.numpy_solver_bridge import (
    NumpySolverBridge,
)
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
    is_synthesizable,
)
from cognithor.channels.program_synthesis.integration.tactical_memory import (
    PSECache,
)
from cognithor.channels.program_synthesis.sandbox.strategies import (
    LinuxSubprocessStrategy,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _rotate90_request(*, max_depth: int = 2) -> SynthesisRequest:
    spec = TaskSpec(
        examples=(
            (_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),
            (_g([[5, 6, 7]]), _g([[5], [6], [7]])),
        ),
    )
    return SynthesisRequest(
        spec=spec,
        budget=Budget(max_depth=max_depth, wall_clock_seconds=10.0),
    )


# ---------------------------------------------------------------------------
# is_synthesizable classifier
# ---------------------------------------------------------------------------


class TestIsSynthesizable:
    def test_two_demo_grid_task_is_synthesizable(self) -> None:
        task = {
            "examples": [
                {"input": [[1, 2]], "output": [[2, 1]]},
                {"input": [[3, 4]], "output": [[4, 3]]},
            ],
        }
        assert is_synthesizable(task)

    def test_single_demo_rejected(self) -> None:
        task = {"examples": [{"input": [[1, 2]], "output": [[2, 1]]}]}
        assert not is_synthesizable(task)

    def test_missing_examples_rejected(self) -> None:
        assert not is_synthesizable({})

    def test_missing_input_or_output_rejected(self) -> None:
        task = {
            "examples": [
                {"input": [[1, 2]]},  # no output
                {"input": [[3, 4]], "output": [[4, 3]]},
            ],
        }
        assert not is_synthesizable(task)

    def test_non_grid_input_rejected(self) -> None:
        task = {
            "examples": [
                {"input": "not a grid", "output": [[1]]},
                {"input": [[1]], "output": [[1]]},
            ],
        }
        assert not is_synthesizable(task)

    def test_empty_grid_rejected(self) -> None:
        task = {
            "examples": [
                {"input": [], "output": [[1]]},
                {"input": [[1]], "output": [[1]]},
            ],
        }
        assert not is_synthesizable(task)

    def test_non_dict_task_rejected(self) -> None:
        assert not is_synthesizable("nope")  # type: ignore[arg-type]
        assert not is_synthesizable(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProgramSynthesisChannel
# ---------------------------------------------------------------------------


class TestChannel:
    def test_search_path_solves_rotate90(self) -> None:
        channel = ProgramSynthesisChannel(sandbox_strategy=LinuxSubprocessStrategy())
        req = _rotate90_request()
        result = channel.synthesize(req)
        assert result.status == SynthesisStatus.SUCCESS
        assert result.cache_hit is False

    def test_second_call_hits_cache(self) -> None:
        # Single shared channel + identical request → second call should
        # land in the cache layer.
        channel = ProgramSynthesisChannel(sandbox_strategy=LinuxSubprocessStrategy())
        req = _rotate90_request()
        first = channel.synthesize(req)
        assert first.status == SynthesisStatus.SUCCESS
        second = channel.synthesize(req)
        assert second.status == SynthesisStatus.SUCCESS
        assert second.cache_hit is True
        # Cached entry exposes the program source via annotations.
        annotations = dict(second.annotations)
        assert "rotate90" in annotations.get("cached_program_source", "")

    def test_numpy_fast_path_short_circuits_search(self) -> None:
        # Fake numpy solver that "knows" rotate90 — the channel should
        # skip the enumerator entirely and return the fast-path result.
        def rot90(inp: np.ndarray, demos):
            return np.rot90(inp, k=-1).copy().astype(np.int8)

        # Use a fresh cache so the rerun-cache test doesn't interfere.
        channel = ProgramSynthesisChannel(
            cache=PSECache(),
            numpy_bridge=NumpySolverBridge(solver_fn=rot90),
            sandbox_strategy=LinuxSubprocessStrategy(),
        )
        result = channel.synthesize(_rotate90_request())
        assert result.status == SynthesisStatus.SUCCESS
        annotations = dict(result.annotations)
        assert annotations.get("source") == "numpy_fast_path"
        # Cost candidates should remain at the bridge's 0 — search
        # never ran.
        assert result.cost_candidates == 0

    def test_cache_lookup_disabled_via_budget(self) -> None:
        channel = ProgramSynthesisChannel(sandbox_strategy=LinuxSubprocessStrategy())
        req = _rotate90_request()
        first = channel.synthesize(req)
        assert first.status == SynthesisStatus.SUCCESS
        # Same spec, but disable cache lookup → should NOT report
        # cache_hit even though the entry is present.
        req_no_cache = SynthesisRequest(
            spec=req.spec,
            budget=Budget(
                max_depth=req.budget.max_depth,
                wall_clock_seconds=req.budget.wall_clock_seconds,
                cache_lookup=False,
            ),
        )
        second = channel.synthesize(req_no_cache)
        assert second.cache_hit is False

    def test_sgn_hints_threaded_through_to_spec(self) -> None:
        channel = ProgramSynthesisChannel(sandbox_strategy=LinuxSubprocessStrategy())
        req = SynthesisRequest(
            spec=_rotate90_request().spec,
            budget=Budget(max_depth=2, wall_clock_seconds=10.0),
            sgn_hints={"rotate90": True},
        )
        # The hint shouldn't change correctness — rotate90 is still the
        # right answer. But the call should succeed without error.
        result = channel.synthesize(req)
        assert result.status == SynthesisStatus.SUCCESS

    def test_fresh_cache_has_zero_entries(self) -> None:
        cache = PSECache()
        ProgramSynthesisChannel(cache=cache)
        # Channel construction must not write to the cache.
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# SynthesisRequest dataclass
# ---------------------------------------------------------------------------


class TestSynthesisRequest:
    def test_default_budget_used_when_omitted(self) -> None:
        spec = TaskSpec(examples=((_g([[1]]), _g([[1]])),))
        req = SynthesisRequest(spec=spec)
        assert req.budget.max_depth == Budget().max_depth

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        spec = TaskSpec(examples=((_g([[1]]), _g([[1]])),))
        req = SynthesisRequest(spec=spec)
        with pytest.raises(FrozenInstanceError):
            req.spec = spec  # type: ignore[misc]
