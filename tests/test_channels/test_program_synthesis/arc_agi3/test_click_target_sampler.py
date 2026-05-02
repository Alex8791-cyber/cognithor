# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — ClickTargetSampler tests."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.click_target_sampler import (
    ClickTargetSampler,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


class TestNextClick:
    def test_pure_background_returns_none(self) -> None:
        sampler = ClickTargetSampler()
        grid = np.zeros((4, 4), dtype=np.int32)
        assert sampler.next_click(grid) is None

    def test_single_target_yields_centroid(self) -> None:
        sampler = ClickTargetSampler()
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[1, 2] = 5
        click = sampler.next_click(grid)
        # SDK: (x=col, y=row).
        assert click == (2, 1)

    def test_smaller_clusters_first(self) -> None:
        # Big cluster of 1s + tiny cluster of 2s. The tiny one is picked
        # first because it ranks higher on salience.
        sampler = ClickTargetSampler()
        grid = np.zeros((6, 6), dtype=np.int32)
        grid[0:3, 0:3] = 1  # 9-pixel block
        grid[5, 5] = 2  # 1-pixel target
        click = sampler.next_click(grid)
        assert click == (5, 5)

    def test_visited_skipped_on_subsequent_calls(self) -> None:
        sampler = ClickTargetSampler()
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[0, 0] = 1
        grid[3, 3] = 2
        first = sampler.next_click(grid)
        second = sampler.next_click(grid)
        assert first != second
        assert first is not None
        assert second is not None

    def test_exhausted_queue_returns_none(self) -> None:
        sampler = ClickTargetSampler()
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[0, 0] = 1
        sampler.next_click(grid)  # picks (0, 0)
        # Same signature → no rebuild. Queue is empty.
        assert sampler.next_click(grid) is None


class TestSignatureRebuild:
    def test_new_colour_triggers_rebuild(self) -> None:
        sampler = ClickTargetSampler()
        grid_a = np.zeros((4, 4), dtype=np.int32)
        grid_a[0, 0] = 1
        sampler.next_click(grid_a)  # consumes (0, 0)

        # Same shape, same colours → no rebuild → returns None
        assert sampler.next_click(grid_a) is None

        # Add a new colour → signature changes → rebuild → fresh queue
        grid_b = grid_a.copy()
        grid_b[2, 2] = 3
        click = sampler.next_click(grid_b)
        # Visited (0,0) is still in the visited set, so the new sample
        # picks (2, 2) (the only unseen non-background pixel).
        assert click == (2, 2)


class TestResetForNewLevel:
    def test_clears_visited_and_queue(self) -> None:
        sampler = ClickTargetSampler()
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[0, 0] = 1
        sampler.next_click(grid)
        assert sampler.visited == frozenset({(0, 0)})

        sampler.reset_for_new_level()
        assert sampler.visited == frozenset()
        # After reset, same coordinate can be re-emitted.
        click = sampler.next_click(grid)
        assert click == (0, 0)
