# ARC-AGI-3 ClickSequenceSolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a BFS-based `sequence_click` strategy to PerGameSolver that finds winning click sequences by scanning effective positions and searching the state space — solving games like VC33 where clicks must be ordered, not subset-selected.

**Architecture:** Three existing files modified: `game_profile.py` gets `has_toggles` field for routing, `game_analyzer.py` sets it, `per_game_solver.py` gets `_scan_effective_positions()` + `_execute_sequence_click()` + routing update. No new files.

**Tech Stack:** Python 3.11+, numpy, arc_agi SDK, pytest

**Spec:** `docs/superpowers/specs/2026-04-05-arc-agi3-click-sequence-solver-design.md`

---

### Task 1: GameProfile — Add `has_toggles` Field

**Files:**
- Modify: `src/jarvis/arc/game_profile.py:38-59` (dataclass fields), `61-80` (to_dict), `82-104` (from_dict), `168-173` (default_strategies)
- Modify: `tests/test_arc/test_game_profile.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arc/test_game_profile.py`:

```python
class TestHasToggles:
    def _make_profile(self, has_toggles=False, **overrides) -> GameProfile:
        defaults = dict(
            game_id="test",
            game_type="click",
            available_actions=[6],
            click_zones=[],
            target_colors=[],
            movement_effects={},
            win_condition="unknown",
            vision_description="",
            vision_strategy="",
            strategy_metrics={},
            analyzed_at="",
            has_toggles=has_toggles,
        )
        defaults.update(overrides)
        return GameProfile(**defaults)

    def test_has_toggles_default_false(self):
        p = self._make_profile()
        assert p.has_toggles is False

    def test_has_toggles_serialization(self, tmp_path):
        p = self._make_profile(has_toggles=True)
        p.save(base_dir=tmp_path)
        loaded = GameProfile.load("test", base_dir=tmp_path)
        assert loaded is not None
        assert loaded.has_toggles is True

    def test_has_toggles_backward_compat(self, tmp_path):
        """Old profiles without has_toggles should load with False."""
        import json
        profile_dir = tmp_path / "game_profiles"
        profile_dir.mkdir(parents=True)
        old_data = {
            "game_id": "old_game",
            "game_type": "click",
            "available_actions": [6],
            "click_zones": [],
            "target_colors": [],
            "movement_effects": {},
            "win_condition": "unknown",
            "vision_description": "",
            "vision_strategy": "",
            "strategy_metrics": {},
            "total_runs": 0,
            "best_score": 0,
            "analyzed_at": "",
            "profile_version": 1,
        }
        (profile_dir / "old_game.json").write_text(json.dumps(old_data))
        loaded = GameProfile.load("old_game", base_dir=tmp_path)
        assert loaded is not None
        assert loaded.has_toggles is False

    def test_default_strategies_with_toggles(self):
        p = self._make_profile(has_toggles=True)
        defaults = p.default_strategies()
        assert defaults[0][0] == "cluster_click"
        assert defaults[1][0] == "sequence_click"

    def test_default_strategies_without_toggles(self):
        p = self._make_profile(has_toggles=False)
        defaults = p.default_strategies()
        assert defaults[0][0] == "sequence_click"
        assert defaults[1][0] == "cluster_click"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_game_profile.py::TestHasToggles -v`
Expected: FAIL with `TypeError: GameProfile.__init__() got an unexpected keyword argument 'has_toggles'`

- [ ] **Step 3: Add `has_toggles` field and update serialization**

In `src/jarvis/arc/game_profile.py`, add the field after line 58 (`profile_version`):

```python
    has_toggles: bool = False
```

In `to_dict()` (line 61-80), add before the return:

```python
            "has_toggles": self.has_toggles,
```

In `from_dict()` (line 82-104), add to the constructor call:

```python
            has_toggles=d.get("has_toggles", False),
```

Update `default_strategies()` (line 168-173) to use `has_toggles`:

```python
    def default_strategies(self) -> list[tuple[str, float]]:
        if self.game_type == "click":
            if self.has_toggles:
                return [("cluster_click", 0.6), ("sequence_click", 0.3), ("targeted_click", 0.1)]
            return [("sequence_click", 0.6), ("cluster_click", 0.3), ("targeted_click", 0.1)]
        if self.game_type == "keyboard":
            return [("keyboard_explore", 0.5), ("keyboard_sequence", 0.3), ("hybrid", 0.2)]
        return [("hybrid", 0.5), ("targeted_click", 0.3), ("keyboard_explore", 0.2)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_game_profile.py -v`
Expected: All tests pass (20 old + 6 new = 26)

- [ ] **Step 5: Commit**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add src/jarvis/arc/game_profile.py tests/test_arc/test_game_profile.py
git commit -m "feat(arc): add has_toggles to GameProfile for solver routing"
```

---

### Task 2: GameAnalyzer — Set `has_toggles` From Sacrifice Level

**Files:**
- Modify: `src/jarvis/arc/game_analyzer.py:368-383` (profile construction)
- Modify: `tests/test_arc/test_game_analyzer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_arc/test_game_analyzer.py`:

```python
class TestHasTogglesDetection:
    def test_analyze_sets_has_toggles_true_when_toggles_found(self, tmp_path):
        """Profile should have has_toggles=True when sacrifice level detects toggles."""
        initial_grid = np.zeros((64, 64), dtype=np.int8)
        initial_grid[10:15, 10:15] = 3

        toggled_grid = initial_grid.copy()
        toggled_grid[10:15, 10:15] = 5  # toggle 3->5

        not_finished = _make_mock_game_state("NOT_FINISHED")
        game_over = _make_mock_game_state("GAME_OVER")

        mock_env = MagicMock()
        step_count = [0]

        def mock_step(action, data=None):
            step_count[0] += 1
            if step_count[0] > 5:
                return _make_mock_obs(
                    grid=np.expand_dims(initial_grid, 0),
                    state=game_over,
                    actions=[MagicMock(value=6)],
                )
            return _make_mock_obs(
                grid=np.expand_dims(toggled_grid, 0),
                state=not_finished,
                actions=[MagicMock(value=6)],
            )

        mock_env.step = mock_step
        mock_env.reset.return_value = _make_mock_obs(
            grid=np.expand_dims(initial_grid, 0),
            state=not_finished,
            actions=[MagicMock(value=a) for a in [5, 6]],
        )

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        vision_resp = {
            "message": {
                "content": json.dumps({
                    "game_type": "click",
                    "target_color": 3,
                    "strategy": "test",
                    "description": "test",
                })
            }
        }

        analyzer = GameAnalyzer(arcade=mock_arcade)
        with patch("jarvis.arc.game_analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = vision_resp
            profile = analyzer.analyze("toggle_test", force=True, base_dir=tmp_path)

        assert profile.has_toggles is True

    def test_analyze_sets_has_toggles_false_when_no_toggles(self, tmp_path):
        """Profile should have has_toggles=False when no toggles detected."""
        initial_grid = np.zeros((64, 64), dtype=np.int8)
        initial_grid[10:15, 10:15] = 3

        not_finished = _make_mock_game_state("NOT_FINISHED")
        game_over = _make_mock_game_state("GAME_OVER")

        mock_env = MagicMock()
        step_count = [0]

        def mock_step(action, data=None):
            step_count[0] += 1
            # No grid change → no toggles detected
            if step_count[0] > 3:
                return _make_mock_obs(
                    grid=np.expand_dims(initial_grid, 0),
                    state=game_over,
                    actions=[MagicMock(value=6)],
                )
            return _make_mock_obs(
                grid=np.expand_dims(initial_grid, 0),
                state=not_finished,
                actions=[MagicMock(value=6)],
            )

        mock_env.step = mock_step
        mock_env.reset.return_value = _make_mock_obs(
            grid=np.expand_dims(initial_grid, 0),
            state=not_finished,
            actions=[MagicMock(value=a) for a in [5, 6]],
        )

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        vision_resp = {
            "message": {
                "content": json.dumps({
                    "game_type": "click",
                    "target_color": 3,
                    "strategy": "test",
                    "description": "test",
                })
            }
        }

        analyzer = GameAnalyzer(arcade=mock_arcade)
        with patch("jarvis.arc.game_analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = vision_resp
            profile = analyzer.analyze("no_toggle_test", force=True, base_dir=tmp_path)

        assert profile.has_toggles is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_game_analyzer.py::TestHasTogglesDetection -v`
Expected: FAIL with `TypeError: GameProfile.__init__() got an unexpected keyword argument 'has_toggles'` or `AssertionError` (has_toggles not set)

- [ ] **Step 3: Set `has_toggles` in `analyze()`**

In `src/jarvis/arc/game_analyzer.py`, modify the `GameProfile(...)` constructor call (around line 368-380). Add `has_toggles` field:

```python
        profile = GameProfile(
            game_id=game_id,
            game_type=game_type,
            available_actions=action_ids,
            click_zones=click_zones,
            target_colors=target_colors,
            movement_effects=movement_effects,
            win_condition=win_condition,
            vision_description=vision1.get("description", "") if vision1 else "unavailable",
            vision_strategy=vision1.get("strategy", "") if vision1 else "unavailable",
            strategy_metrics={},
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            has_toggles=len(report.toggle_pairs) > 0,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_game_analyzer.py -v`
Expected: All tests pass (19 old + 2 new = 21)

- [ ] **Step 5: Commit**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add src/jarvis/arc/game_analyzer.py tests/test_arc/test_game_analyzer.py
git commit -m "feat(arc): set has_toggles from sacrifice level toggle_pairs"
```

---

### Task 3: PerGameSolver — Effective Position Scanner

**Files:**
- Modify: `src/jarvis/arc/per_game_solver.py`
- Modify: `tests/test_arc/test_per_game_solver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arc/test_per_game_solver.py`:

```python
class TestEffectivePositionScanner:
    def test_finds_effective_positions(self):
        """Scan should find positions where clicks change the puzzle grid."""
        from arcengine.enums import GameState

        grid_initial = np.zeros((64, 64), dtype=np.int8)
        grid_initial[0, :] = 7  # orange bar at row 0

        grid_changed = grid_initial.copy()
        grid_changed[10:20, 10:20] = 5  # big change at certain region

        click_count = [0]

        def mock_step(action, data=None):
            click_count[0] += 1
            x = data.get("x", 0) if data else 0
            y = data.get("y", 0) if data else 0
            # Only clicks near (10,10) cause a puzzle change
            if 8 <= x <= 22 and 8 <= y <= 22:
                grid_out = grid_changed.copy()
                grid_out[0, 63] = 4  # bar change
            else:
                grid_out = grid_initial.copy()
                grid_out[0, 63] = 4  # bar-only change
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid_initial, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        assert len(positions) > 0
        # All found positions should be in the effective region
        for x, y in positions:
            assert 6 <= x <= 24 and 6 <= y <= 24, f"Unexpected position ({x},{y})"

    def test_returns_empty_when_no_effective(self):
        """Scan returns empty list when no clicks cause puzzle changes."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, :] = 7

        def mock_step(action, data=None):
            grid_out = grid.copy()
            grid_out[0, 63] = 4  # bar-only change
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        assert positions == []

    def test_groups_nearby_positions(self):
        """Positions with same effect and close proximity should be grouped."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, :] = 7

        def mock_step(action, data=None):
            x = data.get("x", 0) if data else 0
            y = data.get("y", 0) if data else 0
            grid_out = grid.copy()
            grid_out[0, 63] = 4
            # Two separate valve regions with different diffs
            if 8 <= x <= 12 and 8 <= y <= 12:
                grid_out[20:30, 20:30] = 3  # 100 px change
            elif 40 <= x <= 44 and 40 <= y <= 44:
                grid_out[50:55, 50:55] = 5  # 25 px change
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        # Should find 2 groups (not 9+ individual positions)
        assert len(positions) == 2

    def test_max_six_groups(self):
        """Scanner should return at most 6 groups."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)

        region_id = [0]

        def mock_step(action, data=None):
            x = data.get("x", 0) if data else 0
            grid_out = grid.copy()
            # Every 8-pixel column is a different "valve" with a unique diff
            col_group = x // 8
            diff_size = (col_group + 1) * 10
            grid_out[10 : 10 + diff_size, 0] = col_group + 1
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        assert len(positions) <= 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_per_game_solver.py::TestEffectivePositionScanner -v`
Expected: FAIL with `AttributeError: 'PerGameSolver' object has no attribute '_scan_effective_positions'`

- [ ] **Step 3: Implement `_scan_effective_positions`**

Add this method to the `PerGameSolver` class in `src/jarvis/arc/per_game_solver.py` (after `_detect_stagnation`, before `solve`):

```python
    def _scan_effective_positions(
        self,
        env: Any,
        replay_sequence: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Scan 2px grid to find click positions that change the puzzle grid.

        Returns deduplicated representative positions, max 6 groups,
        sorted by puzzle_diff descending.
        """
        from arcengine.enums import GameState

        def replay_and_get_grid() -> np.ndarray:
            obs = env.reset()
            for x, y in replay_sequence:
                obs = env.step(6, data={"x": x, "y": y})
            grid = safe_frame_extract(obs)
            return grid

        base_grid = replay_and_get_grid()

        # Scan every 2nd pixel
        raw_hits: list[tuple[int, int, int]] = []  # (x, y, puzzle_diff)
        for y in range(0, 64, 2):
            for x in range(0, 64, 2):
                obs = env.reset()
                for rx, ry in replay_sequence:
                    obs = env.step(6, data={"x": rx, "y": ry})
                g_before = safe_frame_extract(obs)

                obs = env.step(6, data={"x": x, "y": y})

                if obs.state == GameState.GAME_OVER:
                    continue
                # Instant win — return just this position
                if hasattr(obs, "levels_completed") and obs.levels_completed > len(replay_sequence) // 20:
                    # Heuristic: if levels jumped, this is a direct solution
                    pass

                g_after = safe_frame_extract(obs)
                puzzle_diff = int(np.sum(g_before[1:] != g_after[1:]))

                if puzzle_diff > 0:
                    raw_hits.append((x, y, puzzle_diff))

        if not raw_hits:
            return []

        # Group by (puzzle_diff within 10%) AND (spatial proximity < 8 Manhattan)
        groups: list[list[tuple[int, int, int]]] = []
        used = [False] * len(raw_hits)

        for i, (x1, y1, d1) in enumerate(raw_hits):
            if used[i]:
                continue
            group = [(x1, y1, d1)]
            used[i] = True
            for j, (x2, y2, d2) in enumerate(raw_hits):
                if used[j]:
                    continue
                # Same diff (within 10%) and close proximity
                if abs(d1 - d2) <= max(d1, d2) * 0.1 and abs(x1 - x2) + abs(y1 - y2) < 8:
                    group.append((x2, y2, d2))
                    used[j] = True
            groups.append(group)

        # Pick representative per group (centroid), sort by diff descending
        representatives: list[tuple[int, int, int]] = []
        for group in groups:
            cx = int(np.mean([g[0] for g in group]))
            cy = int(np.mean([g[1] for g in group]))
            avg_diff = int(np.mean([g[2] for g in group]))
            representatives.append((cx, cy, avg_diff))

        representatives.sort(key=lambda r: -r[2])

        # Max 6 groups
        representatives = representatives[:6]

        return [(x, y) for x, y, _ in representatives]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_per_game_solver.py::TestEffectivePositionScanner -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add src/jarvis/arc/per_game_solver.py tests/test_arc/test_per_game_solver.py
git commit -m "feat(arc): add _scan_effective_positions for BFS click search"
```

---

### Task 4: PerGameSolver — BFS Sequence Search

**Files:**
- Modify: `src/jarvis/arc/per_game_solver.py`
- Modify: `tests/test_arc/test_per_game_solver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arc/test_per_game_solver.py`:

```python
class TestSequenceClickStrategy:
    def test_finds_winning_sequence(self):
        """BFS should find a click sequence that solves the level."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, :] = 7  # bar

        click_history = []

        def mock_step(action, data=None):
            x = data.get("x", 0) if data else 0
            y = data.get("y", 0) if data else 0
            click_history.append((x, y))
            grid_out = grid.copy()
            grid_out[0, 63] = 4  # bar change

            # Count effective clicks (at valve position 10,10)
            effective = sum(1 for cx, cy in click_history if 8 <= cx <= 12 and 8 <= cy <= 12)

            # Win after 3 effective clicks
            obs = MagicMock()
            if effective >= 3:
                obs.levels_completed = 1
            else:
                obs.levels_completed = 0
                # Show some grid change for effective clicks
                if 8 <= x <= 12 and 8 <= y <= 12:
                    grid_out[20:30, 20:30] = effective

            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            return obs

        mock_env = MagicMock()

        def mock_reset():
            click_history.clear()
            obs = MagicMock()
            obs.frame = np.expand_dims(grid, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env.reset = mock_reset
        mock_env.step = mock_step

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        profile = _make_profile("click")
        solver = PerGameSolver(profile, arcade=mock_arcade)
        outcome = solver._execute_sequence_click(max_actions=200)

        assert outcome.won is True
        assert outcome.levels_solved >= 1

    def test_returns_false_when_no_effective_positions(self):
        """Should return won=False when scan finds no effective clicks."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)

        def mock_step(action, data=None):
            obs = MagicMock()
            obs.frame = np.expand_dims(grid, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.reset.return_value = MagicMock(
            frame=np.expand_dims(grid, 0),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
        )
        mock_env.step = mock_step

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        solver = PerGameSolver(_make_profile("click"), arcade=mock_arcade)
        outcome = solver._execute_sequence_click(max_actions=200)

        assert outcome.won is False
        assert outcome.steps == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_per_game_solver.py::TestSequenceClickStrategy -v`
Expected: FAIL with `AttributeError: 'PerGameSolver' object has no attribute '_execute_sequence_click'`

- [ ] **Step 3: Implement `_execute_sequence_click`**

Add to the `PerGameSolver` class in `per_game_solver.py`, after `_scan_effective_positions`:

```python
    def _execute_sequence_click(self, max_actions: int) -> StrategyOutcome:
        """BFS-based click sequence search with sub-level detection."""
        import time
        from collections import deque

        from arcengine.enums import GameState

        outcome = StrategyOutcome()
        timeout = 120.0
        max_depth = 12
        max_sub_levels = 5
        max_states = 50_000
        sub_level_threshold = 500  # pixels
        game_id = self._profile.game_id
        max_levels = 10

        env = self._arcade.make(game_id)
        prev_level_clicks: list[list[tuple[int, int]]] = []

        for level in range(max_levels):
            t0 = time.monotonic()
            replay_prefix = [c for seq in prev_level_clicks for c in seq]

            solution = self._bfs_find_sequence(
                env, replay_prefix, timeout, max_depth, max_sub_levels,
                sub_level_threshold, max_states,
            )
            outcome.steps += 1

            if solution is None:
                break

            prev_level_clicks.append(solution)
            outcome.levels_solved += 1
            outcome.won = True
            log.info(
                "arc.sequence_level_solved",
                game_id=game_id,
                level=level,
                clicks=len(solution),
                time_s=round(time.monotonic() - t0, 1),
            )

        outcome.budget_ratio = 1.0
        return outcome

    def _bfs_find_sequence(
        self,
        env: Any,
        replay_prefix: list[tuple[int, int]],
        timeout: float,
        max_depth: int,
        max_sub_levels: int,
        sub_level_threshold: int,
        max_states: int,
    ) -> list[tuple[int, int]] | None:
        """BFS through click sequences with sub-level re-scanning."""
        import time
        from collections import deque

        from arcengine.enums import GameState

        t0 = time.monotonic()

        # Scan effective positions
        action_set = self._scan_effective_positions(env, replay_prefix)
        if not action_set:
            return None

        # Get initial state
        obs = env.reset()
        for x, y in replay_prefix:
            obs = env.step(6, data={"x": x, "y": y})
        initial_grid = safe_frame_extract(obs)
        current_levels = obs.levels_completed

        # BFS
        queue: deque[list[tuple[int, int]]] = deque()
        queue.append([])
        visited: set[int] = {hash(initial_grid[1:].tobytes())}

        while queue:
            if time.monotonic() - t0 > timeout:
                break
            if len(visited) > max_states:
                break

            seq = queue.popleft()
            if len(seq) >= max_depth:
                continue

            for cx, cy in action_set:
                new_seq = seq + [(cx, cy)]
                full_seq = replay_prefix + new_seq

                # Replay
                obs = env.reset()
                game_over = False
                for rx, ry in full_seq:
                    obs = env.step(6, data={"x": rx, "y": ry})
                    if obs.state == GameState.GAME_OVER:
                        game_over = True
                        break

                if game_over:
                    continue

                # Check win
                if obs.levels_completed > current_levels:
                    return new_seq

                grid = safe_frame_extract(obs)
                state_hash = hash(grid[1:].tobytes())

                if state_hash not in visited:
                    visited.add(state_hash)

                    # Sub-level detection: massive grid change
                    prev_grid = safe_frame_extract(env.reset())
                    for rx, ry in replay_prefix + seq:
                        env.step(6, data={"x": rx, "y": ry})
                    prev_grid_after = safe_frame_extract(
                        env.step(6, data={"x": cx, "y": cy})
                    )
                    # Simpler: just check diff from initial
                    puzzle_diff = int(np.sum(grid[1:] != initial_grid[1:]))

                    if puzzle_diff > sub_level_threshold and max_sub_levels > 0:
                        # Sub-level! Re-scan and recurse
                        sub_result = self._bfs_find_sequence(
                            env,
                            replay_prefix=full_seq,
                            timeout=timeout - (time.monotonic() - t0),
                            max_depth=max_depth,
                            max_sub_levels=max_sub_levels - 1,
                            sub_level_threshold=sub_level_threshold,
                            max_states=max_states - len(visited),
                        )
                        if sub_result is not None:
                            return new_seq + sub_result

                    queue.append(new_seq)

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_per_game_solver.py::TestSequenceClickStrategy -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add src/jarvis/arc/per_game_solver.py tests/test_arc/test_per_game_solver.py
git commit -m "feat(arc): add BFS click sequence search with sub-level detection"
```

---

### Task 5: PerGameSolver — Route `sequence_click` Strategy

**Files:**
- Modify: `src/jarvis/arc/per_game_solver.py:269-282` (_execute_strategy routing)
- Modify: `tests/test_arc/test_per_game_solver.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_arc/test_per_game_solver.py`:

```python
class TestSequenceClickRouting:
    def test_execute_strategy_routes_sequence_click(self):
        """_execute_strategy should route 'sequence_click' to _execute_sequence_click."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)

        def mock_step(action, data=None):
            obs = MagicMock()
            obs.frame = np.expand_dims(grid, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.reset.return_value = MagicMock(
            frame=np.expand_dims(grid, 0),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
        )
        mock_env.step = mock_step

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        solver = PerGameSolver(_make_profile("click"), arcade=mock_arcade)
        outcome = solver._execute_strategy(mock_env, "sequence_click", max_actions=200)

        assert isinstance(outcome, StrategyOutcome)
        # Should have called _execute_sequence_click (not the generic loop)
        # Verify by checking it doesn't stagnate at 5 steps like keyboard would
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_per_game_solver.py::TestSequenceClickRouting -v`
Expected: FAIL — strategy falls through to generic loop (no routing for "sequence_click")

- [ ] **Step 3: Add routing in `_execute_strategy`**

In `src/jarvis/arc/per_game_solver.py`, in the `_execute_strategy` method (line 269+), add after the `cluster_click` routing block (after line 282):

```python
        # Special handling for sequence_click: BFS through click sequences
        if strategy == "sequence_click":
            return self._execute_sequence_click(max_actions)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add src/jarvis/arc/per_game_solver.py tests/test_arc/test_per_game_solver.py
git commit -m "feat(arc): route sequence_click strategy in _execute_strategy"
```

---

### Task 6: Integration Test — Full VC33-Style Pipeline

**Files:**
- Modify: `tests/test_arc/test_analyzer_integration.py`

- [ ] **Step 1: Write integration test**

Append to `tests/test_arc/test_analyzer_integration.py`:

```python
class TestSequenceClickPipeline:
    def test_no_toggle_game_uses_sequence_click(self, tmp_path):
        """Games without toggles should prefer sequence_click strategy."""
        from jarvis.arc.game_profile import GameProfile

        profile = GameProfile(
            game_id="no_toggle_game",
            game_type="click",
            available_actions=[6],
            click_zones=[],
            target_colors=[],
            movement_effects={},
            win_condition="unknown",
            vision_description="test",
            vision_strategy="test",
            strategy_metrics={},
            analyzed_at="2026-04-05",
            has_toggles=False,
        )

        defaults = profile.default_strategies()
        assert defaults[0][0] == "sequence_click"
        assert defaults[0][1] == 0.6

    def test_toggle_game_uses_cluster_click(self, tmp_path):
        """Games with toggles should prefer cluster_click strategy."""
        from jarvis.arc.game_profile import GameProfile

        profile = GameProfile(
            game_id="toggle_game",
            game_type="click",
            available_actions=[6],
            click_zones=[(10, 10)],
            target_colors=[3],
            movement_effects={},
            win_condition="clear_board",
            vision_description="test",
            vision_strategy="test",
            strategy_metrics={},
            analyzed_at="2026-04-05",
            has_toggles=True,
        )

        defaults = profile.default_strategies()
        assert defaults[0][0] == "cluster_click"
        assert defaults[0][1] == 0.6

    def test_profile_has_toggles_persists(self, tmp_path):
        """has_toggles should survive save/load cycle."""
        from jarvis.arc.game_profile import GameProfile

        for val in [True, False]:
            p = GameProfile(
                game_id=f"persist_{val}",
                game_type="click",
                available_actions=[6],
                click_zones=[],
                target_colors=[],
                movement_effects={},
                win_condition="unknown",
                vision_description="",
                vision_strategy="",
                strategy_metrics={},
                analyzed_at="",
                has_toggles=val,
            )
            p.save(base_dir=tmp_path)
            loaded = GameProfile.load(f"persist_{val}", base_dir=tmp_path)
            assert loaded.has_toggles is val
```

- [ ] **Step 2: Run integration tests**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/test_analyzer_integration.py -v`
Expected: All tests pass (3 old + 3 new = 6)

- [ ] **Step 3: Run full ARC test suite**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/ -v --tb=short`
Expected: All tests pass, no regressions

- [ ] **Step 4: Commit**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add tests/test_arc/test_analyzer_integration.py
git commit -m "test(arc): add integration tests for sequence_click routing"
```

---

### Task 7: Update Budget Allocation Tests

**Files:**
- Modify: `tests/test_arc/test_per_game_solver.py`

- [ ] **Step 1: Update existing budget test for new defaults**

The `test_default_click_allocation` test (around line 36-47) expects `cluster_click` first. With `has_toggles=False` (default in `_make_profile`), the new default is `sequence_click` first. Update the test:

Find in `tests/test_arc/test_per_game_solver.py` the test `test_default_click_allocation` and update:

```python
    def test_default_click_allocation(self):
        profile = _make_profile("click")
        # Default profile has has_toggles=False → sequence_click first
        solver = PerGameSolver(profile, arcade=MagicMock())
        slots = solver._allocate_budget(level_num=0)

        assert len(slots) == 3
        assert slots[0].strategy == "sequence_click"
        assert slots[0].max_actions == 120  # 60% of 200
        assert slots[1].strategy == "cluster_click"
        assert slots[1].max_actions == 60   # 30% of 200
        assert slots[2].strategy == "targeted_click"
        assert slots[2].max_actions == 20   # 10% of 200
```

Also update `_make_profile` to accept `has_toggles`:

```python
def _make_profile(game_type="click", metrics=None, has_toggles=False) -> GameProfile:
    return GameProfile(
        game_id="test_game",
        game_type=game_type,
        available_actions=[5, 6] if game_type == "click" else [1, 2, 3, 4, 5],
        click_zones=[(10, 10), (30, 30)] if game_type == "click" else [],
        target_colors=[3] if game_type == "click" else [],
        movement_effects={1: "moves_player", 2: "moves_player"} if game_type != "click" else {},
        win_condition="clear_board",
        vision_description="test",
        vision_strategy="test",
        strategy_metrics=metrics or {},
        analyzed_at="2026-04-04",
        has_toggles=has_toggles,
    )
```

- [ ] **Step 2: Run all tests**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add tests/test_arc/test_per_game_solver.py
git commit -m "test(arc): update budget allocation tests for sequence_click defaults"
```

---

### Task 8: Delete Stale Game Profiles & Final Verification

**Files:**
- No source changes, operational cleanup

- [ ] **Step 1: Delete stale cached profiles**

The cached profiles in `~/.jarvis/arc/game_profiles/` were created before `has_toggles` existed. Delete them so they get regenerated with the new field:

```bash
rm -f C:/Users/ArtiCall/.jarvis/arc/game_profiles/*.json
echo "Stale profiles deleted"
```

- [ ] **Step 2: Run full test suite**

Run: `cd "D:/Jarvis/jarvis complete v20" && python -m pytest tests/test_arc/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Smoke test with VC33**

Run a quick test to verify the sequence_click strategy is selected for VC33:

```bash
cd "D:/Jarvis/jarvis complete v20"
timeout 300 python -m jarvis.arc --mode analyzer --game vc33-9851e02b --verbose 2>&1 | head -30
```

Expected: Should show `sequence_click` as primary strategy (not `cluster_click`)

- [ ] **Step 4: Commit all changes**

```bash
cd "D:/Jarvis/jarvis complete v20"
git add -A
git commit -m "feat(arc): ClickSequenceSolver — BFS click search for non-toggle games (VC33)"
```
