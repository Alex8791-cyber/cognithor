# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-19 — Structured state rendering for the LLM prompt.

The single biggest gap in Sprint-15..18: the LLM gets the raw 64×64
ASCII grid + an action history with pixel-counts, but it has NO way
to know:

* Which pixels changed since the last action (only the count)
* What the cluster structure of the current grid is
* Where the cursor is (for click-actions)
* Whether the latest delta moved the agent closer to or further from
  a goal (it can't see "vorher vs jetzt" side-by-side)

ASCII-grid alone is essentially noise to a 27B LLM that hasn't been
trained on this representation. This module gives the LLM three
structured signals:

1. ``render_cluster_summary(grid)`` — one short line per non-background
   colour with cluster count, total pixel count, and centroid of the
   largest cluster ("color 4: 3 clusters totalling 187 cells, biggest
   at (32, 14) size 89").

2. ``render_delta_summary(prev_grid, curr_grid)`` — what the latest
   action *did* in semantic terms ("color 4 grew by 24 cells,
   color 7 shrank by 19 cells, cursor moved from (12, 8) to (12, 9)").

3. ``render_state_changes_in_window(window)`` — per-step delta-summaries
   over the last N memory entries, so the LLM sees the *trajectory*
   of structural change, not just action names + raw pixel counts.

These are deterministic Python — no LLM-side compute, no training. The
LLM still does the strategic reasoning, but on inputs it can actually
parse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
    )

    _Grid = NDArray[np.int8]


# Colours treated as "background" — not reported in cluster summaries.
# 0 (empty/bg) is universally background; 8 in many ARC games is also
# background-walkable (e.g. LockSmith floor). Make it a tuple so the
# caller can override per-game later if needed.
_DEFAULT_BACKGROUND_COLORS: tuple[int, ...] = (0,)


def render_cluster_summary(
    grid: _Grid,
    *,
    background_colors: tuple[int, ...] = _DEFAULT_BACKGROUND_COLORS,
    max_lines: int = 8,
) -> str:
    """One line per non-background colour with cluster topology.

    Format::

        color 4: 3 clusters, 187 cells, biggest 89@(32,14)
        color 7: 1 cluster,   12 cells, biggest 12@(8,8)
        color 11: 5 clusters, 24 cells, biggest 8@(45,30)

    Sorted by total cells descending so the LLM sees the
    structurally-important colours first. Truncated to ``max_lines``.
    Empty grids → ``"(empty grid)"``.
    """
    import numpy as np

    from cognithor.channels.program_synthesis.arc_agi3.fast_grid_planner import (
        find_clusters,
    )

    if grid.size == 0:
        return "(empty grid)"
    unique = np.unique(grid)
    rows: list[tuple[int, str]] = []
    for color in unique.tolist():
        if color in background_colors:
            continue
        clusters = find_clusters(grid, int(color))
        if not clusters:
            continue
        total = sum(c.size for c in clusters)
        biggest = max(clusters, key=lambda c: c.size)
        cy, cx = biggest.centroid
        rows.append(
            (
                total,
                f"color {color}: {len(clusters)} cluster"
                f"{'s' if len(clusters) != 1 else ''}, "
                f"{total} cell{'s' if total != 1 else ''}, "
                f"biggest {biggest.size}@({cy},{cx})",
            )
        )
    if not rows:
        return "(no non-background pixels)"
    rows.sort(reverse=True)
    lines = [r[1] for r in rows[:max_lines]]
    return "\n".join(lines)


def render_delta_summary(
    prev_grid: _Grid,
    curr_grid: _Grid,
    *,
    background_colors: tuple[int, ...] = _DEFAULT_BACKGROUND_COLORS,
) -> str:
    """One-line summary of *what changed* between two frames.

    Format::

        color 4: +24 cells (15 added, 9 lost), color 7: -19 cells

    Reports per-colour net change. Cells that flipped from one colour
    to another are accounted on both sides (so totals balance).
    Returns ``"(no change)"`` if the grids are identical, or
    ``"(grids have different shapes)"`` if shapes differ.
    """
    import numpy as np

    if prev_grid.shape != curr_grid.shape:
        return "(grids have different shapes)"
    if np.array_equal(prev_grid, curr_grid):
        return "(no change)"

    changed_mask = prev_grid != curr_grid
    prev_changed = prev_grid[changed_mask]
    curr_changed = curr_grid[changed_mask]
    rows: list[tuple[int, str]] = []
    seen: set[int] = set()
    for color in np.unique(np.concatenate([prev_changed, curr_changed])).tolist():
        if color in background_colors:
            continue
        if color in seen:
            continue
        seen.add(color)
        added = int(np.sum(curr_changed == color))
        lost = int(np.sum(prev_changed == color))
        net = added - lost
        if added == 0 and lost == 0:
            continue
        if net == 0:
            piece = f"color {color}: balanced ({added} added, {lost} lost)"
        else:
            sign = "+" if net > 0 else ""
            piece = f"color {color}: {sign}{net} cells ({added} added, {lost} lost)"
        rows.append((abs(net) if net != 0 else added + lost, piece))
    if not rows:
        return "(only background-color cells changed)"
    rows.sort(reverse=True)
    return ", ".join(r[1] for r in rows)


def render_state_changes_in_window(
    memory: EpisodeMemory,
    *,
    max_steps: int = 5,
    background_colors: tuple[int, ...] = _DEFAULT_BACKGROUND_COLORS,
) -> str:
    """Per-step delta-summary over the last ``max_steps`` memory entries.

    Format (most-recent first)::

        step -1 (ACTION3) pixΔ=24: color 4: +24 cells (15 added, 9 lost)
        step -2 (ACTION6) pixΔ=0: (no change)
        step -3 (ACTION4) pixΔ=24: color 7: -12 cells, color 11: +12 cells

    Empty memory → ``"(no actions yet)"``.

    This is the trajectory-of-structural-change signal. With it the
    LLM can answer "did the last few actions move things in the same
    direction, or am I oscillating?".

    Sprint-19 Hebel M: each line is prefixed with the absolute number
    of changed cells (``pixΔ=N``). Run #26c showed the agent kept
    triggering GAME_OVER with pixΔ=525/639 at the final steps despite
    the abstract GAME_OVER_AVOIDANCE_HINT — surfacing the concrete
    numeric trajectory in the prompt lets the LLM react to "my last 3
    moves changed 500+ cells each" directly instead of inferring it
    from per-colour deltas.
    """
    import numpy as np

    if max_steps < 1:
        raise ValueError(f"render_state_changes_in_window: max_steps must be >= 1, got {max_steps}")
    window = memory.window(max_steps + 1)  # +1 because we diff pairs
    if not window:
        return "(no actions yet)"
    parts: list[str] = []
    # window is most-recent first; pairs are (window[i], window[i+1])
    # where window[i] is the AFTER-state and window[i+1] is the BEFORE.
    for i in range(min(max_steps, len(window) - 1)):
        after = window[i]
        before = window[i + 1]
        idx = -(i + 1)
        if before.grid.shape == after.grid.shape:
            pix_delta = int(np.sum(before.grid != after.grid))
        else:
            pix_delta = -1  # shape change — treat as unknown rather than crashing
        delta = render_delta_summary(
            before.grid,
            after.grid,
            background_colors=background_colors,
        )
        pix_label = "pixΔ=?" if pix_delta < 0 else f"pixΔ={pix_delta}"
        parts.append(f"step {idx} ({after.action_name}) {pix_label}: {delta}")
    if not parts:
        return "(no completed transitions yet)"
    return "\n".join(parts)


__all__ = [
    "render_cluster_summary",
    "render_delta_summary",
    "render_state_changes_in_window",
]
