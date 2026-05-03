# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-20 Hebel V — win-demo store tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import EpisodeMemory
from cognithor.channels.program_synthesis.arc_agi3.win_demos import (
    WinDemoStore,
    decode_grid_csv,
    encode_grid_csv,
    render_win_demo,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

if TYPE_CHECKING:
    from pathlib import Path


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


class TestEncodeDecodeRoundtrip:
    def test_small_grid(self) -> None:
        g = _g([[0, 1, 2], [3, 4, 5]])
        s = encode_grid_csv(g)
        assert s.startswith("2x3,")
        back = decode_grid_csv(s)
        assert np.array_equal(g, back)

    def test_64x64(self) -> None:
        rng = np.random.default_rng(42)
        g = rng.integers(0, 16, size=(64, 64), dtype=np.int8)
        back = decode_grid_csv(encode_grid_csv(g))
        assert np.array_equal(g, back)

    def test_decode_rejects_wrong_cell_count(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="expected"):
            decode_grid_csv("3x3,0,1,2")  # claims 9 cells, has 3


class TestWinDemoStore:
    def test_empty_store_returns_no_records(self, tmp_path: Path) -> None:
        store = WinDemoStore(tmp_path)
        assert store.load_recent("bp35") == []

    def test_record_level_up_appends_trajectory(self, tmp_path: Path) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=0)
        m.append(grid=_g([[2]]), action_name="ACTION6", levels_completed=1)

        store = WinDemoStore(tmp_path)
        store.record_level_up(
            game_id="bp35",
            from_level=0,
            to_level=1,
            memory=m,
        )
        records = store.load_recent("bp35")
        assert len(records) == 1
        rec = records[0]
        assert rec["game_id"] == "bp35"
        assert rec["from_level"] == 0
        assert rec["to_level"] == 1
        # trajectory has the 2 step pairs (3 frames → 2 transitions)
        assert len(rec["trajectory"]) == 2
        # in chronological order, last action is the level-up
        assert rec["trajectory"][-1]["action"] == "ACTION6"

    def test_record_level_up_skips_when_no_progress(self, tmp_path: Path) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=0)

        store = WinDemoStore(tmp_path)
        # to_level == from_level: not a level-up; ignore.
        store.record_level_up(
            game_id="bp35",
            from_level=0,
            to_level=0,
            memory=m,
        )
        assert store.load_recent("bp35") == []

    def test_record_level_up_skips_when_memory_too_short(self, tmp_path: Path) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)

        store = WinDemoStore(tmp_path)
        store.record_level_up(
            game_id="bp35",
            from_level=0,
            to_level=1,
            memory=m,
        )
        # Only one frame → no transition to capture.
        assert store.load_recent("bp35") == []

    def test_load_recent_returns_most_recent_first(self, tmp_path: Path) -> None:
        m1 = EpisodeMemory()
        m1.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m1.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=1)
        m2 = EpisodeMemory()
        m2.append(grid=_g([[5]]), action_name="ACTION4", levels_completed=0)
        m2.append(grid=_g([[6]]), action_name="ACTION6", levels_completed=1)

        store = WinDemoStore(tmp_path)
        store.record_level_up(game_id="bp35", from_level=0, to_level=1, memory=m1)
        store.record_level_up(game_id="bp35", from_level=0, to_level=1, memory=m2)

        records = store.load_recent("bp35", max_records=2)
        # Most recent first → m2's ACTION6 wins
        assert records[0]["trajectory"][-1]["action"] == "ACTION6"
        assert records[1]["trajectory"][-1]["action"] == "ACTION3"

    def test_per_game_isolation(self, tmp_path: Path) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=1)

        store = WinDemoStore(tmp_path)
        store.record_level_up(game_id="bp35", from_level=0, to_level=1, memory=m)
        # Different game gets its own file → empty.
        assert store.load_recent("ft09") == []
        assert len(store.load_recent("bp35")) == 1

    def test_jsonl_format_one_record_per_line(self, tmp_path: Path) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=1)

        store = WinDemoStore(tmp_path)
        store.record_level_up(game_id="bp35", from_level=0, to_level=1, memory=m)
        store.record_level_up(game_id="bp35", from_level=0, to_level=1, memory=m)
        path = tmp_path / "bp35.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # round-trips


class TestRenderWinDemo:
    def test_empty_store_returns_empty_string(self, tmp_path: Path) -> None:
        store = WinDemoStore(tmp_path)
        assert render_win_demo(store, "bp35") == ""

    def test_renders_trajectory_with_header(self, tmp_path: Path) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=0)
        m.append(grid=_g([[2]]), action_name="ACTION6", levels_completed=1)

        store = WinDemoStore(tmp_path)
        store.record_level_up(game_id="bp35", from_level=0, to_level=1, memory=m)
        out = render_win_demo(store, "bp35")
        assert "Past winning trajectory (level 0 → 1" in out
        assert "ACTION3" in out
        assert "ACTION6" in out
        # Final marker telling the LLM where the win happened
        assert "level-up here" in out

    def test_does_not_leak_grid_csv_into_prompt(self, tmp_path: Path) -> None:
        """The demo's value is the action sequence, NOT the historical
        pixels — the LLM already has the current grid as an image.
        Including raw cell-level CSV would bloat the prompt.
        """
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=1)

        store = WinDemoStore(tmp_path)
        store.record_level_up(game_id="bp35", from_level=0, to_level=1, memory=m)
        out = render_win_demo(store, "bp35")
        assert "grid_csv" not in out
        assert "1x1," not in out  # the encoded shape header

    def test_caps_steps_per_record(self, tmp_path: Path) -> None:
        m = EpisodeMemory()
        for i in range(20):
            m.append(grid=_g([[i % 4]]), action_name=f"ACTION{i % 4}", levels_completed=0)
        m.append(grid=_g([[7]]), action_name="ACTION6", levels_completed=1)

        store = WinDemoStore(tmp_path)
        store.record_level_up(
            game_id="bp35",
            from_level=0,
            to_level=1,
            memory=m,
            max_steps_back=20,
        )
        out = render_win_demo(store, "bp35", max_steps_per_record=4)
        # 4 step lines + header + footer = 6 lines
        assert sum(1 for line in out.splitlines() if line.strip().startswith("step ")) == 4
