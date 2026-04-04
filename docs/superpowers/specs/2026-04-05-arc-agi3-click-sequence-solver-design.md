# ARC-AGI-3 ClickSequenceSolver Design

**Date:** 2026-04-05
**Status:** Approved
**Scope:** New strategy `sequence_click` in existing `per_game_solver.py`, minor changes to `game_profile.py` and `game_analyzer.py`

## Problem

The existing `cluster_click` strategy solves FT09 (toggle-subset mechanic) but fails on all 7 other click-tagged games. These games use a different mechanic: clicking specific positions in specific sequences causes state transitions (water level routing, object manipulation, etc.). The click positions and their effects change within a level (sub-levels). No toggles are detected.

Example: VC33 is a water-level-routing puzzle. Clicking "valve" positions (color 9 in the 64x64 grid, rendered as blue in the UI) pumps water between containers. After pumping, the grid changes massively (sub-level transition) and new valve positions appear elsewhere. Level 1 requires 8 clicks across 2 sub-levels.

## Verified Observations (from VC33 analysis)

- `env.reset()` costs 0.5ms, `env.step()` costs 0.3ms — fast enough for BFS
- Most click positions produce `puzzle_diff == 0` (only orange bar changes). Only specific "valve" positions produce `puzzle_diff > 0` (actual grid changes)
- Effective positions can be identified by scanning a 2px grid (1024 tests, ~1s total)
- After a massive grid change (>500px), effective positions shift to new locations
- VC33 Level 0: 3x click at (63,35) solves it (1 valve, 3 clicks)
- VC33 Level 1: 1 click at (63,35) triggers sub-level, then 2x (1,25) + 5x (1,45) solves it (2 valves, 7 clicks after sub-level)
- BFS with 4 effective positions found the L1 solution in 150 states, ~2 seconds
- The color used for valves in the SDK grid (color 9, "brown" in palette) is rendered as blue in the actual game UI
- Baseline actions per VC33 level: [6, 13, 31, 59, 92, 24, 82] — later levels are much harder

## Architecture

No new files. Three changes to existing modules:

1. **`per_game_solver.py`** — Add `_scan_effective_positions()`, `_execute_sequence_click()`, and route `sequence_click` in `_execute_strategy()`
2. **`game_profile.py`** — Add `has_toggles: bool` field
3. **`game_analyzer.py`** — Set `has_toggles` from sacrifice level data

## Component 1: Effective Position Scanner

### Method: `_scan_effective_positions(env, replay_sequence) -> list[tuple[int, int]]`

Scans a 2px grid (32x32 = 1024 positions) to find which click positions actually change the puzzle grid (not just the orange bar).

**Algorithm:**
1. For each position (x, y) in range(0, 64, 2) x range(0, 64, 2):
   - `env.reset()`
   - Replay all clicks in `replay_sequence`
   - Capture `grid_before` (rows 1-63, excluding bar at row 0)
   - `env.step(6, data={"x": x, "y": y})`
   - Capture `grid_after` (rows 1-63)
   - `puzzle_diff = np.sum(grid_before != grid_after)`
   - If `puzzle_diff > 0`: record (x, y, puzzle_diff)
2. Group positions by `puzzle_diff` value (positions with the same diff likely control the same valve)
3. For each group: keep one representative (the one closest to the group centroid)
4. Sort groups by `puzzle_diff` descending (biggest effect first)
5. Keep at most 6 groups

**Performance:** 1024 positions x (0.5ms reset + replay_len * 0.3ms + 0.3ms step) = ~1s for empty replay, ~3s for replay of 10 steps.

**Edge cases:**
- If 0 effective positions found → return empty list, solver gives up
- If a click produces `levels_completed > current` → that's an instant win, return it immediately
- If a click produces GAME_OVER → skip that position

### Grouping logic

Two positions belong to the same group if:
- Their `puzzle_diff` values are within 10% of each other, AND
- They are spatially close (Manhattan distance < 8 pixels)

This handles valves that span multiple pixels (e.g., a 4x4 valve block produces the same diff from any pixel within it).

## Component 2: BFS Click Sequence Search

### Method: `_execute_sequence_click(max_actions) -> StrategyOutcome`

Uses BFS to find a click sequence that advances `levels_completed`.

**Algorithm:**

```
1. Create one env via arcade.make() — reuse for all tests via env.reset()
2. replay_prefix = [] (clicks to reach current level from reset)
3. scan effective positions → action_set
4. if action_set empty → return StrategyOutcome(won=False)
5. Initialize BFS:
   - queue = deque([ [] ])  # empty sequence
   - visited = { hash(initial_grid) }
6. While queue not empty AND time < timeout:
   a. seq = queue.popleft()
   b. if len(seq) >= max_depth (12): skip
   c. For each (cx, cy) in action_set:
      - new_seq = seq + [(cx, cy)]
      - env.reset()
      - Replay replay_prefix + new_seq
      - If GAME_OVER during replay: skip
      - If levels_completed increased: SOLVED → return
      - grid = extract frame (rows 1-63)
      - state_hash = hash(grid.tobytes())
      - If state_hash not in visited:
        - Check for sub-level: if puzzle_diff from last click > 500:
          - Re-scan effective positions from current state
          - Update action_set
          - Reset BFS with current seq as new replay_prefix
        - Else: add (new_seq, state_hash) to queue and visited
7. If queue exhausted: return StrategyOutcome(won=False)
```

**Sub-level detection:**
When a click produces `puzzle_diff > 500` (massive grid restructuring), it indicates a sub-level transition. At this point:
- The current click sequence (replay_prefix + seq + current_click) becomes the new `replay_prefix`
- Effective positions are re-scanned from the new grid state
- BFS restarts with depth 0 but keeping the accumulated replay_prefix
- Maximum 5 sub-level transitions per level

**State hashing:**
- Hash the grid bytes excluding row 0 (orange bar) to avoid counting bar-only changes as new states
- Use `hash(grid[1:].tobytes())` — Python's built-in hash is fast enough

**BFS vs DFS decision:**
BFS finds the shortest solution first, which is important because:
- Shorter solutions leave more bar budget for later levels
- The baseline counts suggest optimal solutions exist
- With 4-6 actions and depth 12, the search space is manageable: 4^12 = 16M max, but state hashing prunes heavily (VC33 L1 needed only 150 states)

## Component 3: Solver Ladder & Profile Routing

### GameProfile change

Add one field to `GameProfile` dataclass:
```python
has_toggles: bool = False
```

### GameAnalyzer change

In `analyze()`, after building the profile, set:
```python
profile.has_toggles = len(report.toggle_pairs) > 0
```

### Serialization

Add `has_toggles` to `to_dict()` and `from_dict()` in GameProfile. Default `False` for backward compatibility with existing cached profiles.

### Solver Ladder in PerGameSolver

In `_allocate_budget()`, change the click defaults based on `has_toggles`:

```
has_toggles == True:
  cluster_click: 60%
  sequence_click: 30%
  targeted_click: 10%

has_toggles == False:
  sequence_click: 60%
  cluster_click: 30%
  targeted_click: 10%
```

The `default_strategies()` method on GameProfile must be updated to check `has_toggles`.

## Component 4: Strategy Routing in _execute_strategy

Add routing for `sequence_click` in `_execute_strategy()`:

```python
if strategy == "sequence_click":
    return self._execute_sequence_click(max_actions)
```

The method creates its own env via `arcade.make()`, does not use the passed env (same pattern as `cluster_click`).

## Component 5: Multi-Level Loop

The `_execute_sequence_click` method solves levels iteratively (like `_execute_cluster_click`):

```
1. prev_clicks = []  # accumulated click history across levels
2. For level in range(max_levels):
   a. replay_prefix = flatten(prev_clicks)
   b. Run BFS from replay_prefix
   c. If BFS returns solution:
      - Append solution to prev_clicks
      - outcome.levels_solved += 1
      - Continue to next level
   d. Else: break
3. Return outcome
```

Each level's solution is appended to the replay prefix for the next level. This means Level 3's BFS starts by replaying all of Levels 0-2's clicks, then searches from there.

**Replay cost scaling:** Level N requires replaying sum(clicks for levels 0..N-1) steps before each BFS test. For VC33:
- L0: 3 replay steps → 1ms/test
- L1: 3+8=11 replay steps → 4ms/test
- L2: 11+? replay steps → grows linearly

At 4ms/test and 200 states: 0.8s per BFS. Still well within 120s timeout even for later levels.

## Error Handling

**GAME_OVER during position scan:**
- Skip that position, don't count it as effective

**GAME_OVER during BFS replay:**
- Discard entire path (the accumulated sequence is invalid)
- This can happen if a sub-level transition changes the game state such that earlier clicks no longer work — in practice this shouldn't happen since we replay from env.reset()

**Timeout (120s per level):**
- BFS checks `time.monotonic()` before each expansion
- Returns `StrategyOutcome(won=False)` with `steps` count so far

**No effective positions after sub-level:**
- If re-scan finds 0 effective positions → sub-level is a dead end
- Break out of BFS, return what we have

**Too many states (memory):**
- If `len(visited) > 50_000` → abort BFS (grid is 4KB, 50K states = 200MB)
- In practice with 4-6 actions and heavy dedup, this limit is rarely hit

## Performance Budget

| Operation | Cost | Count (per level) | Total |
|-----------|------|-------------------|-------|
| Position scan | ~1s | 1-5 (per sub-level) | ~5s |
| BFS node test | ~1-4ms | 50-500 states | ~2s |
| Sub-level re-scan | ~1s | 0-5 | ~5s |
| **Total per level** | | | **~12s typical** |

120s timeout gives ~10 level attempts. VC33 has 7 levels. Budget is comfortable.

## Testing

- `test_per_game_solver.py` — tests for `_scan_effective_positions` and `_execute_sequence_click`
- Mock env that simulates sub-level transitions (grid changes massively after specific clicks)
- Mock env where specific sequences of clicks advance `levels_completed`
- Verify state hashing excludes row 0 (bar)
- Verify grouping logic deduplicates same-valve positions
- Verify sub-level detection triggers re-scan
- Verify BFS finds shortest solution

## Files

### Modified
- `src/jarvis/arc/per_game_solver.py` — add `_scan_effective_positions()`, `_execute_sequence_click()`, route in `_execute_strategy()`, update `default_strategies()` fallback
- `src/jarvis/arc/game_profile.py` — add `has_toggles` field, update `to_dict()`, `from_dict()`, `default_strategies()`
- `src/jarvis/arc/game_analyzer.py` — set `has_toggles` from `toggle_pairs`
- `tests/test_arc/test_per_game_solver.py` — add tests for sequence_click
- `tests/test_arc/test_game_profile.py` — add test for has_toggles serialization

### Not modified
- `src/jarvis/arc/game_analyzer.py` sacrifice level logic — no changes needed, it already detects toggle_pairs
- `src/jarvis/arc/__main__.py` — no changes, existing `--mode analyzer` works
- `src/jarvis/arc/cluster_solver.py` — untouched, still used for FT09-style games
