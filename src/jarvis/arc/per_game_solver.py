"""ARC-AGI-3 PerGameSolver -- budget-based strategy execution per game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jarvis.arc.error_handler import safe_frame_extract
from jarvis.arc.game_profile import GameProfile
from jarvis.utils.logging import get_logger

__all__ = ["BudgetSlot", "PerGameSolver", "SolveResult", "StrategyOutcome"]

log = get_logger(__name__)

# Default total budget per game type
_BUDGET_BY_TYPE = {"click": 200, "keyboard": 200, "mixed": 100}

_STAGNATION_WINDOW = 5
_STAGNATION_THRESHOLD = 10  # pixels


@dataclass
class BudgetSlot:
    """One strategy with its allocated action budget."""

    strategy: str
    max_actions: int
    priority: int


@dataclass
class StrategyOutcome:
    """Result of executing one strategy on one level."""

    won: bool = False
    game_over: bool = False
    stagnated: bool = False
    steps: int = 0
    levels_solved: int = 0
    budget_ratio: float = 0.0
    winning_clicks: list[tuple[int, int]] | None = None


@dataclass
class SolveResult:
    """Outcome of solving a game."""

    game_id: str
    levels_completed: int
    total_steps: int
    strategy_log: list[dict]
    score: float


class PerGameSolver:
    """Budget-based solver that combines strategies from a GameProfile."""

    def __init__(self, profile: GameProfile, arcade: Any):
        self._profile = profile
        self._arcade = arcade

    def _allocate_budget(self, level_num: int) -> list[BudgetSlot]:
        """Allocate action budget across strategies."""
        total = _BUDGET_BY_TYPE.get(self._profile.game_type, 100)

        ranked = self._profile.ranked_strategies()
        if ranked:
            # Use learned ranking: top 3 with 50/30/20 split
            top3 = ranked[:3]
            ratios = [0.5, 0.3, 0.2]
        else:
            # Use defaults for this game type
            defaults = self._profile.default_strategies()
            top3 = [name for name, _ in defaults]
            ratios = [ratio for _, ratio in defaults]

        slots = []
        for i, strategy in enumerate(top3):
            ratio = ratios[i] if i < len(ratios) else 0.1
            slots.append(BudgetSlot(
                strategy=strategy,
                max_actions=int(total * ratio),
                priority=i,
            ))

        return slots

    def _execute_cluster_click(
        self,
        initial_grid: np.ndarray,
        target_color: int | None,
        max_actions: int,
    ) -> StrategyOutcome:
        """Smart cluster-click: elimination-based search, then brute-force fallback."""
        import time

        from arcengine.enums import GameState

        from jarvis.arc.cluster_solver import ClusterSolver

        outcome = StrategyOutcome()
        timeout = 120.0  # per level timeout (seconds)

        if target_color is None:
            unique_colors = [int(c) for c in np.unique(initial_grid) if c != 0]
            if not unique_colors:
                return outcome
            best_color = max(
                unique_colors,
                key=lambda c: len(ClusterSolver(target_color=c, max_skip=0).find_clusters(initial_grid)),
            )
            target_color = best_color

        game_id = self._profile.game_id
        prev_solutions: list[list[tuple[int, int]]] = []
        max_levels = 10

        for level in range(max_levels):
            t0 = time.monotonic()
            solution = self._smart_find_solution(
                game_id, target_color, prev_solutions, level, timeout,
            )
            outcome.steps += 1

            if solution is None:
                break

            prev_solutions.append(solution)
            outcome.levels_solved += 1
            outcome.won = True
            outcome.winning_clicks = solution
            log.info(
                "arc.analyzer_level_solved",
                game_id=game_id,
                level=level,
                clicks=len(solution),
                time_s=round(time.monotonic() - t0, 1),
            )

        outcome.budget_ratio = 1.0
        return outcome

    def _smart_find_solution(
        self,
        game_id: str,
        target_color: int,
        prev_solutions: list[list[tuple[int, int]]],
        target_level: int,
        timeout: float,
    ) -> list[tuple[int, int]] | None:
        """Smart elimination search using env.reset() (0.5ms) instead of arcade.make() (380ms).

        Strategy:
          1. Click all → win? Done.
          2. Find poison clusters (cause GAME_OVER) → remove iteratively.
          3. Single elimination (skip 1 at a time).
          4. Progressive elimination (skip 2, 3, ... at a time).
        """
        import itertools
        import time

        from arcengine.enums import GameState

        from jarvis.arc.cluster_solver import ClusterSolver

        t0 = time.monotonic()
        solver = ClusterSolver(target_color=target_color, max_skip=0)

        # Create ONE env, reuse via reset() — 760x faster than arcade.make()
        env = self._arcade.make(game_id)

        def replay_to_level() -> Any:
            """Reset env and replay previous solutions to reach target level."""
            obs = env.reset()
            for sol in prev_solutions:
                for cx, cy in sol:
                    obs = env.step(6, data={"x": cx, "y": cy})
            return obs

        # Get current level grid and cluster positions
        obs = replay_to_level()
        grid = np.array(obs.frame)
        if grid.ndim == 3:
            grid = grid[0]
        centers = solver.find_clusters(grid)
        n = len(centers)

        if n == 0:
            return None

        current_levels = obs.levels_completed
        combos_tested = 0

        def test_combo(click_indices: list[int]) -> bool:
            """Test a click combo using env.reset() + replay. ~1ms per test."""
            nonlocal combos_tested
            combos_tested += 1
            obs2 = replay_to_level()
            for idx in click_indices:
                cx, cy = centers[idx]
                obs2 = env.step(6, data={"x": cx, "y": cy})
                if obs2.state == GameState.GAME_OVER:
                    return False
            return obs2.levels_completed > current_levels

        def find_poison_clusters(indices: list[int]) -> set[int]:
            """Click clusters in order, find which one triggers GAME_OVER."""
            obs2 = replay_to_level()
            for idx in indices:
                cx, cy = centers[idx]
                obs2 = env.step(6, data={"x": cx, "y": cy})
                if obs2.state == GameState.GAME_OVER:
                    return {idx}
            return set()

        # Phase 1: Try clicking ALL clusters
        all_idx = list(range(n))
        if test_combo(all_idx):
            log.info("arc.smart_solve", phase=1, level=target_level, n=n, combos=combos_tested)
            return [centers[i] for i in all_idx]

        # Phase 2: Iteratively remove poison clusters
        poison: set[int] = set()
        safe_idx = list(all_idx)
        for _ in range(min(n, 8)):
            if time.monotonic() - t0 > timeout:
                break
            found = find_poison_clusters(safe_idx)
            if not found:
                break
            poison |= found
            safe_idx = [i for i in all_idx if i not in poison]
            if test_combo(safe_idx):
                log.info("arc.smart_solve", phase=2, level=target_level, n=n, poison=len(poison), combos=combos_tested)
                return [centers[i] for i in safe_idx]

        # Phase 3: Single elimination — skip each cluster one at a time
        for skip_idx in range(n):
            if time.monotonic() - t0 > timeout:
                break
            combo = [i for i in range(n) if i != skip_idx]
            if test_combo(combo):
                log.info("arc.smart_solve", phase=3, level=target_level, n=n, skipped=1, combos=combos_tested)
                return [centers[i] for i in combo]

        # Phase 4: Progressive elimination — skip 2, 3, ..., up to max_skip
        max_skip = min(n - 1, 6)
        for skip_count in range(2, max_skip + 1):
            if time.monotonic() - t0 > timeout:
                break
            for skip_combo in itertools.combinations(range(n), skip_count):
                if time.monotonic() - t0 > timeout:
                    break
                combo = [i for i in range(n) if i not in skip_combo]
                if test_combo(combo):
                    log.info(
                        "arc.smart_solve", phase=4, level=target_level, n=n,
                        skipped=skip_count, combos=combos_tested,
                    )
                    return [centers[i] for i in combo]

        log.info("arc.smart_solve_failed", level=target_level, n=n, combos=combos_tested,
                 time_s=round(time.monotonic() - t0, 1))
        return None

    def _execute_strategy(
        self, env: Any, strategy: str, max_actions: int
    ) -> StrategyOutcome:
        """Execute a single strategy with a given action budget."""
        from arcengine.enums import GameState

        # Special handling for cluster_click: delegates to ClusterSolver.find_solution()
        if strategy == "cluster_click":
            target_color = self._profile.target_colors[0] if self._profile.target_colors else None
            # Get initial grid for auto-detect fallback
            peek_env = self._arcade.make(self._profile.game_id)
            obs_peek = peek_env.reset()
            last_grid = safe_frame_extract(obs_peek)
            return self._execute_cluster_click(last_grid, target_color, max_actions)

        # Special handling for sequence_click: BFS through click sequences
        if strategy == "sequence_click":
            return self._execute_sequence_click(max_actions)

        outcome = StrategyOutcome()
        frame_history: list[np.ndarray] = []
        initial_levels = None

        for step in range(max_actions):
            action_id, data = self._pick_action(strategy, frame_history)
            obs = env.step(action_id, data=data)
            grid = safe_frame_extract(obs)
            frame_history.append(grid)
            outcome.steps += 1

            if initial_levels is None and hasattr(obs, "levels_completed"):
                initial_levels = obs.levels_completed

            # Check terminal states
            if obs.state == GameState.WIN:
                outcome.won = True
                outcome.levels_solved = (
                    getattr(obs, "levels_completed", 0) - (initial_levels or 0) + 1
                )
                outcome.budget_ratio = outcome.steps / max_actions
                return outcome

            if obs.state == GameState.GAME_OVER:
                outcome.game_over = True
                outcome.budget_ratio = outcome.steps / max_actions
                return outcome

            # Check stagnation
            if self._detect_stagnation(frame_history):
                outcome.stagnated = True
                outcome.budget_ratio = outcome.steps / max_actions
                return outcome

        outcome.budget_ratio = 1.0
        return outcome

    def _pick_action(
        self, strategy: str, frame_history: list[np.ndarray]
    ) -> tuple[int, dict | None]:
        """Pick next action based on strategy."""
        profile = self._profile

        if strategy == "cluster_click" or strategy == "targeted_click":
            # Click on known zones from profile
            if profile.click_zones:
                idx = len(frame_history) % len(profile.click_zones)
                x, y = profile.click_zones[idx]
                return 6, {"x": x, "y": y}
            # Fallback: center position
            return 6, {"x": 32, "y": 32}

        if strategy == "keyboard_explore":
            # Cycle through directions
            directions = [a for a in profile.available_actions if a in (1, 2, 3, 4)]
            if directions:
                idx = len(frame_history) % len(directions)
                return directions[idx], None
            return 1, None

        if strategy == "keyboard_sequence":
            # Structured sequences: repeat each direction 4 times before switching
            directions = [a for a in profile.available_actions if a in (1, 2, 3, 4)]
            if directions:
                repeat = 4
                dir_idx = (len(frame_history) // repeat) % len(directions)
                return directions[dir_idx], None
            return 5, None  # interact as fallback

        if strategy == "hybrid":
            # Alternate: keyboard for first 3/4, click for 1/4
            if profile.click_zones and len(frame_history) % 4 == 3:
                idx = len(frame_history) % len(profile.click_zones)
                x, y = profile.click_zones[idx]
                return 6, {"x": x, "y": y}
            directions = [a for a in profile.available_actions if a in (1, 2, 3, 4)]
            if directions:
                idx = len(frame_history) % len(directions)
                return directions[idx], None
            return 5, None

        # Unknown strategy -> interact
        return 5, None

    def solve(
        self,
        max_levels: int = 10,
        timeout_s: float = 1200.0,
        base_dir: Any | None = None,
    ) -> SolveResult:
        """Solve the game level by level with budget-based strategy mix."""
        import time

        from arcengine.enums import GameState

        env = self._arcade.make(self._profile.game_id)
        obs = env.reset()

        result = SolveResult(
            game_id=self._profile.game_id,
            levels_completed=0,
            total_steps=0,
            strategy_log=[],
            score=0.0,
        )

        start_time = time.monotonic()
        max_resets = 3

        for level_num in range(max_levels):
            if time.monotonic() - start_time > timeout_s:
                log.info("arc.solver_timeout", game_id=self._profile.game_id)
                break

            level_result = self._solve_level(env, level_num, max_resets, start_time, timeout_s)
            result.total_steps += level_result["steps"]
            result.strategy_log.append(level_result)

            if level_result["won"]:
                result.levels_completed += 1
                # Update profile metrics for the winning strategy
                self._profile.update_metrics(
                    level_result["strategy"],
                    won=True,
                    levels_solved=1,
                    steps=level_result["steps"],
                    budget_ratio=level_result.get("budget_ratio", 1.0),
                )
            else:
                # Update metrics for failed strategies
                for failed in level_result.get("tried", []):
                    self._profile.update_metrics(
                        failed,
                        won=False,
                        levels_solved=0,
                        steps=level_result["steps"],
                        budget_ratio=1.0,
                    )

        self._profile.update_run(score=result.levels_completed)
        self._profile.save(base_dir=base_dir)
        result.score = float(result.levels_completed)
        return result

    def _solve_level(
        self,
        env: Any,
        level_num: int,
        max_resets: int,
        start_time: float,
        timeout_s: float,
    ) -> dict:
        """Try all budget slots on one level."""
        import time

        from arcengine.enums import GameState

        slots = self._allocate_budget(level_num)
        total_steps = 0
        tried: list[str] = []
        resets_used = 0

        for slot in slots:
            if time.monotonic() - start_time > timeout_s:
                break

            tried.append(slot.strategy)
            outcome = self._execute_strategy(env, slot.strategy, slot.max_actions)
            total_steps += outcome.steps

            if outcome.won:
                return {
                    "level": level_num,
                    "strategy": slot.strategy,
                    "won": True,
                    "steps": total_steps,
                    "budget_ratio": outcome.budget_ratio,
                    "tried": tried,
                }

            if outcome.game_over:
                resets_used += 1
                if resets_used >= max_resets:
                    break
                # Reset level
                try:
                    env.reset()
                except Exception:
                    break

        return {
            "level": level_num,
            "strategy": tried[-1] if tried else "none",
            "won": False,
            "steps": total_steps,
            "tried": tried,
        }

    def _detect_stagnation(self, frame_history: list[np.ndarray]) -> bool:
        """Check if recent frames show no meaningful change."""
        if len(frame_history) < _STAGNATION_WINDOW:
            return False

        window = frame_history[-_STAGNATION_WINDOW:]
        for i in range(1, len(window)):
            diff = int(np.sum(window[i] != window[i - 1]))
            if diff >= _STAGNATION_THRESHOLD:
                return False

        return True

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
            return safe_frame_extract(obs)

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

                g_after = safe_frame_extract(obs)
                puzzle_diff = int(np.sum(g_before[1:] != g_after[1:]))

                if puzzle_diff > 0:
                    raw_hits.append((x, y, puzzle_diff))

        if not raw_hits:
            return []

        # Group by exact puzzle_diff AND spatial proximity (Manhattan < 6)
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
                # Same diff AND directly adjacent (same valve)
                if d1 == d2 and abs(x1 - x2) + abs(y1 - y2) <= 4:
                    group.append((x2, y2, d2))
                    used[j] = True
            groups.append(group)

        # Pick representative per group (pixel with max diff)
        representatives: list[tuple[int, int, int]] = []
        for group in groups:
            best = max(group, key=lambda g: g[2])
            representatives.append(best)

        representatives.sort(key=lambda r: -r[2])
        representatives = representatives[:6]

        return [(x, y) for x, y, _ in representatives]

    def _execute_sequence_click(self, max_actions: int) -> StrategyOutcome:
        """BFS-based click sequence search with sub-level detection."""
        import time

        outcome = StrategyOutcome()
        game_id = self._profile.game_id
        max_levels = 10
        timeout = 120.0

        env = self._arcade.make(game_id)
        prev_level_clicks: list[list[tuple[int, int]]] = []

        for level in range(max_levels):
            t0 = time.monotonic()
            replay_prefix = [c for seq in prev_level_clicks for c in seq]

            solution = self._bfs_find_sequence(
                env, replay_prefix, timeout,
                max_depth=12, max_sub_levels=5,
                sub_level_threshold=500, max_states=50_000,
            )

            if solution is None:
                break

            outcome.steps += 1
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

                    # Sub-level detection: massive grid change from initial
                    puzzle_diff = int(np.sum(grid[1:] != initial_grid[1:]))

                    if puzzle_diff > sub_level_threshold and max_sub_levels > 0:
                        # Sub-level transition — re-scan and recurse
                        remaining_timeout = timeout - (time.monotonic() - t0)
                        if remaining_timeout <= 0:
                            break
                        sub_result = self._bfs_find_sequence(
                            env,
                            replay_prefix=full_seq,
                            timeout=remaining_timeout,
                            max_depth=max_depth,
                            max_sub_levels=max_sub_levels - 1,
                            sub_level_threshold=sub_level_threshold,
                            max_states=max_states - len(visited),
                        )
                        if sub_result is not None:
                            return new_seq + sub_result

                    queue.append(new_seq)

        return None
