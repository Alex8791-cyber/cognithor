# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-20 Hebel V — few-shot win-demo store + render.

Sprint-19 capped at score 0/9: Hebel P's persisted reasoning showed
the LLM understands every safety hebel but doesn't know what *wins*
bp35. Sprint-20's Track A teaches the win condition by demonstration:
when a prior episode achieved ``levels_completed > 0``, persist the
``(grid, action_name, levels_completed)`` sequence around the level
transition into a per-game JSONL. The vision prompt then injects a
short "this exact (state, action) sequence completed level N before"
block so the LLM has a positive example to anchor on.

Bootstrap concern: until at least one episode wins ANYTHING, the demo
store is empty and the prompt block stays absent — byte-identical
legacy prompt. The store fills naturally once any agent (Cognithor
or external) lands a level on a given game.

Storage format (one JSON object per line)::

    {"game_id": "bp35", "from_level": 0, "to_level": 1,
     "captured_at": "2026-05-04T14:30:00Z",
     "trajectory": [
       {"action": "ACTION3", "grid_csv": "...", "pix_delta": 12},
       {"action": "ACTION6", "grid_csv": "...", "pix_delta": 24,
        "data": {"x": 12, "y": 8}},
       ...
     ]}

``grid_csv`` is the post-action grid serialised as a comma-separated
string of 64×64 int8 values (compact + grep-friendly + fast to parse;
binary alternatives like .npy create per-step files which are harder
to manage at scale).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
    )

    _Grid = NDArray[np.int8]


__all__ = [
    "WinDemoStore",
    "decode_grid_csv",
    "encode_grid_csv",
    "render_win_demo",
]


def encode_grid_csv(grid: _Grid) -> str:
    """Serialise an ``int8`` grid to a comma-separated string.

    Format: ``"<rows>x<cols>,<v0,0>,<v0,1>,...,<vR-1,C-1>"``. Round-trips
    through :func:`decode_grid_csv`.
    """
    rows, cols = grid.shape
    cells = ",".join(str(int(c)) for c in grid.flatten())
    return f"{rows}x{cols},{cells}"


def decode_grid_csv(s: str) -> _Grid:
    """Inverse of :func:`encode_grid_csv`."""
    import numpy as np

    shape_part, _, cells_part = s.partition(",")
    rows_s, _, cols_s = shape_part.partition("x")
    rows, cols = int(rows_s), int(cols_s)
    flat = [int(v) for v in cells_part.split(",")]
    if len(flat) != rows * cols:
        raise ValueError(f"decode_grid_csv: expected {rows * cols} cells, got {len(flat)}")
    arr: _Grid = np.array(flat, dtype=np.int8).reshape(rows, cols)
    return arr


class WinDemoStore:
    """Per-game JSONL store of winning trajectories.

    Each :meth:`record_level_up` call appends a single trajectory
    record to ``<root>/<game_id>.jsonl``. :meth:`load_recent` returns
    the most-recent ``max_records`` trajectories for a game.

    Empty / missing files are not errors — :meth:`load_recent` returns
    ``[]`` so the prompt builder simply renders no demo block.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, game_id: str) -> Path:
        # Hebel V: keep filenames safe — only the prefix matters for
        # game-family lookup, but persist the full id so ad-hoc tooling
        # can correlate against scorecards.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in game_id)
        return self.root / f"{safe}.jsonl"

    def record_level_up(
        self,
        *,
        game_id: str,
        from_level: int,
        to_level: int,
        memory: EpisodeMemory,
        max_steps_back: int = 8,
    ) -> None:
        """Append the trajectory leading to a level-up.

        The most-recent ``max_steps_back`` memory entries are captured
        with their grids + action names + per-step pixΔ. Caller should
        invoke this from the agent's level-transition handler.
        """
        import numpy as np

        if to_level <= from_level:
            return  # not a level-up; ignore so callers can be lazy
        window = memory.window(max_steps_back + 1)  # +1 to compute pixΔ for oldest
        if len(window) < 2:
            return
        # window is most-recent first; flip to chronological for the
        # demo (LLMs read trajectories left-to-right)
        chrono = list(reversed(window))
        traj: list[dict[str, Any]] = []
        for i in range(1, len(chrono)):
            after = chrono[i]
            before = chrono[i - 1]
            pix = (
                int(np.sum(before.grid != after.grid))
                if before.grid.shape == after.grid.shape
                else -1
            )
            traj.append(
                {
                    "action": after.action_name,
                    "grid_csv": encode_grid_csv(after.grid),
                    "pix_delta": pix,
                }
            )
        record = {
            "game_id": game_id,
            "from_level": from_level,
            "to_level": to_level,
            "captured_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "trajectory": traj,
        }
        with self._path(game_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def load_recent(
        self,
        game_id: str,
        *,
        max_records: int = 1,
    ) -> list[dict[str, Any]]:
        """Return up to ``max_records`` most-recent winning trajectories.

        Empty / missing files return ``[]``. Records are returned in
        most-recent-first order.
        """
        path = self._path(game_id)
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            return []
        records: list[dict[str, Any]] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= max_records:
                break
        return records


def render_win_demo(
    store: WinDemoStore,
    game_id: str,
    *,
    max_records: int = 1,
    max_steps_per_record: int = 6,
) -> str:
    """Render the most-recent winning trajectory for a game as prompt text.

    Format::

        Past winning trajectory (level 0 → 1, captured 2026-05-04):
          step 1: ACTION3 (pixΔ=12)
          step 2: ACTION6 (pixΔ=24)
          step 3: ACTION3 (pixΔ=8)
          ... (level-up here)

    Empty store → ``""`` (caller can use ``if win_demo:`` to gate the
    block in the template). The exact grid contents are intentionally
    NOT included — the LLM has the *current* grid as an image; the
    demo's value is the action *sequence*, not the historical pixels.
    """
    records = store.load_recent(game_id, max_records=max_records)
    if not records:
        return ""
    parts: list[str] = []
    for rec in records:
        from_level = rec.get("from_level", 0)
        to_level = rec.get("to_level", 1)
        captured = rec.get("captured_at", "")
        captured_short = captured.split("T", 1)[0] if "T" in captured else captured
        traj = rec.get("trajectory", [])[-max_steps_per_record:]
        header = (
            f"Past winning trajectory (level {from_level} → {to_level}"
            + (f", captured {captured_short}" if captured_short else "")
            + "):"
        )
        lines = [header]
        for i, step in enumerate(traj, start=1):
            act = step.get("action", "?")
            pix = step.get("pix_delta", "?")
            lines.append(f"  step {i}: {act} (pixΔ={pix})")
        lines.append("  ... (level-up here)")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
