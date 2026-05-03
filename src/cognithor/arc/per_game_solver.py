"""ARC-AGI-3 PerGameSolver -- budget-based strategy execution per game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.arc.error_handler import safe_frame_extract
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.arc.game_profile import GameProfile

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
    strategy_log: list[dict[str, Any]]
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
            slots.append(
                BudgetSlot(
                    strategy=strategy,
                    max_actions=int(total * ratio),
                    priority=i,
                )
            )

        return slots

    def _execute_cluster_click(
        self,
        initial_grid: np.ndarray[Any, Any],
        target_color: int | None,
        max_actions: int,
    ) -> StrategyOutcome:
        """Smart cluster-click: elimination-based search, then brute-force fallback."""
        import time

        from cognithor.arc.cluster_solver import ClusterSolver

        outcome = StrategyOutcome()
        timeout = 120.0  # per level timeout (seconds)

        if target_color is None:
            unique_colors = [int(c) for c in np.unique(initial_grid) if c != 0]
            if not unique_colors:
                return outcome
            best_color = max(
                unique_colors,
                key=lambda c: len(
                    ClusterSolver(target_color=c, max_skip=0).find_clusters(initial_grid)
                ),
            )
            target_color = best_color

        game_id = self._profile.game_id
        prev_solutions: list[list[tuple[int, int]]] = []
        max_levels = 10

        for level in range(max_levels):
            t0 = time.monotonic()
            solution = self._smart_find_solution(
                game_id,
                target_color,
                prev_solutions,
                level,
                timeout,
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

        from cognithor.arc.cluster_solver import ClusterSolver

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
                log.info(
                    "arc.smart_solve",
                    phase=2,
                    level=target_level,
                    n=n,
                    poison=len(poison),
                    combos=combos_tested,
                )
                return [centers[i] for i in safe_idx]

        # Phase 3: Single elimination — skip each cluster one at a time
        for skip_idx in range(n):
            if time.monotonic() - t0 > timeout:
                break
            combo = [i for i in range(n) if i != skip_idx]
            if test_combo(combo):
                log.info(
                    "arc.smart_solve",
                    phase=3,
                    level=target_level,
                    n=n,
                    skipped=1,
                    combos=combos_tested,
                )
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
                        "arc.smart_solve",
                        phase=4,
                        level=target_level,
                        n=n,
                        skipped=skip_count,
                        combos=combos_tested,
                    )
                    return [centers[i] for i in combo]

        log.info(
            "arc.smart_solve_failed",
            level=target_level,
            n=n,
            combos=combos_tested,
            time_s=round(time.monotonic() - t0, 1),
        )
        return None

    def _execute_strategy(self, env: Any, strategy: str, max_actions: int) -> StrategyOutcome:
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

        # Special handling for keyboard strategies: incremental DFS
        if strategy in ("keyboard_explore", "keyboard_sequence"):
            # Try Smart Explorer first (systematic graph exploration)
            smart_result = self._execute_smart_explore(max_actions)
            if smart_result.won:
                return smart_result
            # Fall back to keyboard DFS
            return self._execute_keyboard(max_actions)

        outcome = StrategyOutcome()
        frame_history: list[np.ndarray[Any, Any]] = []
        initial_levels = None

        for _step in range(max_actions):
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
        self, strategy: str, frame_history: list[np.ndarray[Any, Any]]
    ) -> tuple[int, dict[str, Any] | None]:
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

        env = self._arcade.make(self._profile.game_id)
        env.reset()

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
    ) -> dict[str, Any]:
        """Try all budget slots on one level."""
        import time

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

    def _detect_stagnation(self, frame_history: list[np.ndarray[Any, Any]]) -> bool:
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

        def replay_and_get_grid() -> np.ndarray[Any, Any]:
            obs = env.reset()
            for x, y in replay_sequence:
                obs = env.step(6, data={"x": x, "y": y})
            return safe_frame_extract(obs)

        replay_and_get_grid()

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

                if puzzle_diff > 10:  # ignore tiny diffs (timer ticks, cosmetic)
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
                if abs(d1 - d2) <= max(d1, d2) * 0.15 and abs(x1 - x2) + abs(y1 - y2) <= 8:
                    group.append((x2, y2, d2))
                    used[j] = True
            groups.append(group)

        # Pick representative per group (pixel with max diff)
        representatives: list[tuple[int, int, int]] = []
        for group in groups:
            best = max(group, key=lambda g: g[2])
            representatives.append(best)

        representatives.sort(key=lambda r: -r[2])
        representatives = representatives[:4]

        return [(x, y) for x, y, _ in representatives]

    def _execute_sequence_click(self, max_actions: int) -> StrategyOutcome:
        """BFS-based click sequence search with sub-level detection."""
        import time

        from arcengine.enums import GameState

        outcome = StrategyOutcome()
        game_id = self._profile.game_id
        max_levels = 10
        timeout = 300.0  # 5 min per level (higher levels need more search depth)

        env = self._arcade.make(game_id)
        prev_level_clicks: list[list[tuple[int, int]]] = []

        for level in range(max_levels):
            t0 = time.monotonic()
            replay_prefix = [c for seq in prev_level_clicks for c in seq]

            solution = self._bfs_find_sequence(
                env,
                replay_prefix,
                timeout,
                max_depth=15,
                max_sub_levels=5,
                sub_level_threshold=500,
                max_states=50_000,
            )

            if solution is None:
                break

            # Verify the solution actually advances levels_completed
            full_verify = [c for seq in prev_level_clicks for c in seq] + solution
            obs_verify = env.reset()
            for rx, ry in full_verify:
                obs_verify = env.step(6, data={"x": rx, "y": ry})
            if obs_verify.levels_completed <= level:
                # False positive — solution doesn't actually solve the level
                log.info("arc.sequence_false_positive", level=level, clicks=len(solution))
                break

            # Shorten solution: remove redundant clicks
            original_len = len(solution)
            replay_prefix = [c for seq in prev_level_clicks for c in seq]
            i = 0
            while i < len(solution):
                candidate = solution[:i] + solution[i + 1 :]
                full = replay_prefix + candidate
                obs_s = env.reset()
                ok = True
                for rx, ry in full:
                    obs_s = env.step(6, data={"x": rx, "y": ry})
                    if obs_s.state == GameState.GAME_OVER:
                        ok = False
                        break
                if ok and obs_s.levels_completed > level:
                    solution = candidate
                else:
                    i += 1
            if len(solution) < original_len:
                log.info("arc.click_path_shortened", original=original_len, shortened=len(solution))

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

    def _execute_smart_explore(self, max_actions: int) -> StrategyOutcome:
        """Systematic state-action graph exploration."""
        from cognithor.arc.smart_explorer import SmartExplorer

        se = SmartExplorer(
            self._arcade,
            self._profile.game_id,
            self._profile.available_actions,
        )
        result = se.solve(max_levels=10, timeout_s=300.0)
        return StrategyOutcome(
            won=result.levels_completed > 0,
            levels_solved=result.levels_completed,
            steps=result.total_steps,
        )

    def _execute_keyboard(self, max_actions: int) -> StrategyOutcome:
        """Delegate to KeyboardSolver for incremental DFS on keyboard games."""
        from cognithor.arc.keyboard_solver import KeyboardSolver

        actions = [a for a in self._profile.available_actions if a in (1, 2, 3, 4, 5, 7)]
        if not actions:
            return StrategyOutcome()

        # For mixed games: scan effective click positions
        click_positions: list[tuple[int, int]] = []
        if 6 in self._profile.available_actions:
            env_scan = self._arcade.make(self._profile.game_id)
            click_positions = self._scan_effective_positions(env_scan, [])

        ks = KeyboardSolver(
            self._arcade,
            self._profile.game_id,
            actions,
            click_positions=click_positions or None,
        )
        result = ks.solve(max_levels=10, timeout_s=300.0)

        return StrategyOutcome(
            won=result.levels_completed > 0,
            levels_solved=result.levels_completed,
            steps=result.total_steps,
        )

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

        from arcengine.enums import GameState

        t0 = time.monotonic()

        # Scan effective positions and classify into valves vs triggers
        all_positions = self._scan_effective_positions(env, replay_prefix)
        if not all_positions:
            return None

        # Classify: measure each position's effect to separate triggers from valves
        obs = env.reset()
        for x, y in replay_prefix:
            obs = env.step(6, data={"x": x, "y": y})
        initial_grid = safe_frame_extract(obs)
        current_levels = obs.levels_completed

        valves: list[tuple[int, int]] = []
        triggers: list[tuple[int, int]] = []

        for cx, cy in all_positions:
            obs = env.reset()
            for rx, ry in replay_prefix:
                obs = env.step(6, data={"x": rx, "y": ry})
            g_before = safe_frame_extract(obs)
            obs = env.step(6, data={"x": cx, "y": cy})
            if obs.state == GameState.GAME_OVER:
                continue
            if obs.levels_completed > current_levels:
                return [(cx, cy)]  # instant win
            g_after = safe_frame_extract(obs)
            diff = int(np.sum(g_before[1:] != g_after[1:]))
            if diff > sub_level_threshold:
                triggers.append((cx, cy))
            else:
                valves.append((cx, cy))

        log.info("arc.bfs_classified", valves=len(valves), triggers=len(triggers))

        # Phase 1: BFS with valves only (no triggers) — "pump first"
        result = self._bfs_valves_only(
            env,
            replay_prefix,
            valves,
            initial_grid,
            current_levels,
            t0,
            timeout,
            max_depth,
            max_states,
        )
        if result is not None:
            return result

        # Phase 1.5: Incremental Click-DFS (for deep solutions BFS can't reach)
        if valves and not triggers:
            remaining_time = timeout - (time.monotonic() - t0)
            if remaining_time > 10:
                result = self._incremental_click_dfs(
                    env,
                    replay_prefix,
                    valves,
                    current_levels,
                    remaining_time,
                    max_depth=60,
                )
                if result is not None:
                    return result

        # Phase 2: For each trigger, try pre-pumping then triggering
        if triggers and max_sub_levels > 0:
            result = self._pump_then_trigger(
                env,
                replay_prefix,
                valves,
                triggers,
                initial_grid,
                current_levels,
                t0,
                timeout,
                max_depth,
                max_sub_levels,
                sub_level_threshold,
                max_states,
            )
            if result is not None:
                return result

        # Phase 3: Greedy fallback
        action_set = valves + triggers
        return self._greedy_effect_solve(
            env, replay_prefix, action_set, current_levels, timeout - (time.monotonic() - t0)
        )

    def _incremental_click_dfs(
        self,
        env: Any,
        replay_prefix: list[tuple[int, int]],
        valves: list[tuple[int, int]],
        current_levels: int,
        timeout: float,
        max_depth: int = 60,
    ) -> list[tuple[int, int]] | None:
        """Incremental DFS for click games: step forward, reset only on backtrack."""
        import time

        from arcengine.enums import GameState

        t0 = time.monotonic()
        max_states = 200_000

        obs = env.reset()
        for x, y in replay_prefix:
            obs = env.step(6, data={"x": x, "y": y})
        initial_grid = safe_frame_extract(obs)

        path: list[int] = []  # indices into valves
        visited: set[int] = {hash(initial_grid[1:].tobytes())}
        stack: list[list[int]] = [list(range(len(valves)))]

        while stack:
            if time.monotonic() - t0 > timeout:
                break
            if len(visited) > max_states:
                break

            remaining = stack[-1]
            if not remaining:
                stack.pop()
                if path:
                    path.pop()
                obs = env.reset()
                for x, y in replay_prefix:
                    obs = env.step(6, data={"x": x, "y": y})
                for idx in path:
                    obs = env.step(6, data={"x": valves[idx][0], "y": valves[idx][1]})
                continue

            idx = remaining.pop()
            if len(path) >= max_depth:
                continue

            cx, cy = valves[idx]
            obs = env.step(6, data={"x": cx, "y": cy})

            if obs.levels_completed > current_levels:
                path.append(idx)
                solution = [valves[i] for i in path]
                log.info(
                    "arc.click_dfs_solved",
                    clicks=len(solution),
                    states=len(visited),
                    time_s=round(time.monotonic() - t0, 1),
                )
                return solution

            if obs.state == GameState.GAME_OVER:
                obs = env.reset()
                for x, y in replay_prefix:
                    obs = env.step(6, data={"x": x, "y": y})
                for i in path:
                    obs = env.step(6, data={"x": valves[i][0], "y": valves[i][1]})
                continue

            grid = safe_frame_extract(obs)
            h = hash(grid[1:].tobytes())

            if h in visited:
                # Double-click attempt
                obs = env.step(6, data={"x": cx, "y": cy})
                if obs.levels_completed > current_levels:
                    path.extend([idx, idx])
                    solution = [valves[i] for i in path]
                    log.info("arc.click_dfs_solved", clicks=len(solution), states=len(visited))
                    return solution
                g2 = safe_frame_extract(obs)
                h2 = hash(g2[1:].tobytes())
                if h2 not in visited:
                    visited.add(h2)
                    path.extend([idx, idx])
                    stack.append(list(range(len(valves))))
                    continue
                # Reset
                obs = env.reset()
                for x, y in replay_prefix:
                    obs = env.step(6, data={"x": x, "y": y})
                for i in path:
                    obs = env.step(6, data={"x": valves[i][0], "y": valves[i][1]})
                continue

            visited.add(h)
            path.append(idx)
            stack.append(list(range(len(valves))))

        log.info(
            "arc.click_dfs_exhausted",
            states=len(visited),
            depth=len(path),
            time_s=round(time.monotonic() - t0, 1),
        )
        return None

    def _bfs_valves_only(
        self,
        env: Any,
        replay_prefix: list[tuple[int, int]],
        valves: list[tuple[int, int]],
        initial_grid: np.ndarray[Any, Any],
        current_levels: int,
        t0: float,
        timeout: float,
        max_depth: int,
        max_states: int,
    ) -> list[tuple[int, int]] | None:
        """BFS using only valve clicks (no sub-level triggers)."""
        import time
        from collections import deque

        from arcengine.enums import GameState

        if not valves:
            return None

        queue: deque[list[tuple[int, int]]] = deque()
        queue.append([])
        visited: set[int] = {hash(initial_grid[1:].tobytes())}

        while queue:
            if time.monotonic() - t0 > timeout * 0.8:  # use 80% of timeout for BFS
                break
            if len(visited) > max_states // 2:
                break

            seq = queue.popleft()
            if len(seq) >= max_depth:
                continue

            for cx, cy in valves:
                new_seq = [*seq, (cx, cy)]
                full_seq = replay_prefix + new_seq

                obs = env.reset()
                game_over = False
                for rx, ry in full_seq:
                    obs = env.step(6, data={"x": rx, "y": ry})
                    if obs.state == GameState.GAME_OVER:
                        game_over = True
                        break

                if game_over:
                    continue

                if obs.levels_completed > current_levels:
                    return new_seq

                grid = safe_frame_extract(obs)
                state_hash = hash(grid[1:].tobytes())

                if state_hash not in visited:
                    visited.add(state_hash)
                    queue.append(new_seq)

        return None

    def _pump_then_trigger(
        self,
        env: Any,
        replay_prefix: list[tuple[int, int]],
        valves: list[tuple[int, int]],
        triggers: list[tuple[int, int]],
        initial_grid: np.ndarray[Any, Any],
        current_levels: int,
        t0: float,
        timeout: float,
        max_depth: int,
        max_sub_levels: int,
        sub_level_threshold: int,
        max_states: int,
    ) -> list[tuple[int, int]] | None:
        """Try various amounts of pre-pumping before each trigger.

        For each trigger, try: 0 pre-pumps, then 1, then 2, ... up to max_depth.
        After triggering, recursively solve the post-trigger state.
        """
        import time
        from collections import deque

        from arcengine.enums import GameState

        if not triggers:
            return None

        # Collect all reachable pre-pump states via BFS on valves
        # Each state is (click_sequence, grid_hash)
        pre_pump_seqs: list[list[tuple[int, int]]] = [[]]  # start with 0 pumps
        visited: set[int] = {hash(initial_grid[1:].tobytes())}

        if valves:
            queue: deque[list[tuple[int, int]]] = deque()
            queue.append([])

            while queue:
                if time.monotonic() - t0 > timeout * 0.6:
                    break
                if len(visited) > max_states // 3:
                    break

                seq = queue.popleft()
                if len(seq) >= max_depth // 2:  # limit pre-pump depth
                    continue

                for cx, cy in valves:
                    new_seq = [*seq, (cx, cy)]
                    full_seq = replay_prefix + new_seq

                    obs = env.reset()
                    game_over = False
                    for rx, ry in full_seq:
                        obs = env.step(6, data={"x": rx, "y": ry})
                        if obs.state == GameState.GAME_OVER:
                            game_over = True
                            break

                    if game_over:
                        continue

                    if obs.levels_completed > current_levels:
                        return new_seq  # solved without trigger!

                    grid = safe_frame_extract(obs)
                    state_hash = hash(grid[1:].tobytes())

                    if state_hash not in visited:
                        visited.add(state_hash)
                        pre_pump_seqs.append(new_seq)
                        queue.append(new_seq)

        log.info(
            "arc.pump_then_trigger", pre_pump_states=len(pre_pump_seqs), triggers=len(triggers)
        )

        # For each pre-pump state, try each trigger, then recurse
        for pre_seq in pre_pump_seqs:
            if time.monotonic() - t0 > timeout * 0.8:
                break

            for tx, ty in triggers:
                trigger_seq = [*pre_seq, (tx, ty)]
                full_trigger_seq = replay_prefix + trigger_seq

                # Execute trigger
                obs = env.reset()
                game_over = False
                for rx, ry in full_trigger_seq:
                    obs = env.step(6, data={"x": rx, "y": ry})
                    if obs.state == GameState.GAME_OVER:
                        game_over = True
                        break

                if game_over:
                    continue

                if obs.levels_completed > current_levels:
                    return trigger_seq

                # After trigger: try height-space planner
                remaining = timeout - (time.monotonic() - t0)
                if remaining <= 0:
                    break

                action_set_post = self._scan_effective_positions(env, full_trigger_seq)
                if not action_set_post:
                    continue

                # Try height-space A* on post-trigger state
                height_result = self._height_space_solve(
                    env,
                    full_trigger_seq,
                    action_set_post,
                    current_levels,
                    remaining * 0.5,
                )
                if height_result is not None:
                    return trigger_seq + height_result

                # Try BFS on post-trigger state
                obs2 = env.reset()
                for rx, ry in full_trigger_seq:
                    obs2 = env.step(6, data={"x": rx, "y": ry})
                post_grid = safe_frame_extract(obs2)

                post_result = self._bfs_valves_only(
                    env,
                    full_trigger_seq,
                    action_set_post,
                    post_grid,
                    current_levels,
                    t0,
                    timeout,
                    max_depth,
                    max_states // 3,
                )
                if post_result is not None:
                    return trigger_seq + post_result

        return None

    def _greedy_effect_solve(
        self,
        env: Any,
        replay_prefix: list[tuple[int, int]],
        action_set: list[tuple[int, int]],
        current_levels: int,
        timeout: float,
    ) -> list[tuple[int, int]] | None:
        """Greedy fallback: repeatedly click the valve with the largest puzzle effect.

        Learns an effect matrix (which click produces how much change) and
        iteratively applies the best action. Re-scans after sub-level transitions.
        """
        import time

        from arcengine.enums import GameState

        if timeout <= 0 or not action_set:
            return None

        t0 = time.monotonic()
        max_clicks = 100  # safety limit
        solution: list[tuple[int, int]] = []

        # Build initial effect matrix: click each valve once, measure diff
        effects: dict[tuple[int, int], int] = {}
        for cx, cy in action_set:
            obs = env.reset()
            for rx, ry in replay_prefix:
                obs = env.step(6, data={"x": rx, "y": ry})
            g_before = safe_frame_extract(obs)
            obs = env.step(6, data={"x": cx, "y": cy})
            if obs.state == GameState.GAME_OVER:
                effects[(cx, cy)] = -1  # poison
                continue
            g_after = safe_frame_extract(obs)
            effects[(cx, cy)] = int(np.sum(g_before[1:] != g_after[1:]))

        log.info("arc.effect_matrix", effects={f"{x},{y}": d for (x, y), d in effects.items()})

        # Try height-space planning first (much smarter than greedy)
        height_result = self._height_space_solve(
            env,
            replay_prefix,
            action_set,
            current_levels,
            timeout - (time.monotonic() - t0),
        )
        if height_result is not None:
            return height_result

        # Round-robin index for cycling through valves
        valve_cycle_idx = 0
        stagnation_count = 0

        for _click_num in range(max_clicks):
            if time.monotonic() - t0 > timeout:
                break

            # Sort valves by effect, pick next in round-robin
            active_valves = sorted(
                [c for c in action_set if effects.get(c, 0) > 0],
                key=lambda c: -effects[c],
            )
            if not active_valves:
                break

            # Cycle through valves: after 3 clicks on same valve, try next
            best_click = active_valves[valve_cycle_idx % len(active_valves)]
            if stagnation_count >= 3:
                valve_cycle_idx += 1
                stagnation_count = 0
                best_click = active_valves[valve_cycle_idx % len(active_valves)]

            solution.append(best_click)
            full_seq = replay_prefix + solution

            # Execute and check
            obs = env.reset()
            for rx, ry in full_seq:
                obs = env.step(6, data={"x": rx, "y": ry})
                if obs.state == GameState.GAME_OVER:
                    # Last click was bad — remove it and blacklist
                    solution.pop()
                    effects[best_click] = -1
                    break
            else:
                if obs.levels_completed > current_levels:
                    log.info("arc.greedy_solved", clicks=len(solution))
                    return solution

                stagnation_count += 1

                # Re-measure effects from current state (they may change)
                safe_frame_extract(obs)
                new_effects: dict[tuple[int, int], int] = {}
                for cx, cy in action_set:
                    if effects.get((cx, cy), 0) < 0:
                        new_effects[(cx, cy)] = -1
                        continue
                    obs2 = env.reset()
                    for rx, ry in full_seq:
                        obs2 = env.step(6, data={"x": rx, "y": ry})
                    g_b = safe_frame_extract(obs2)
                    obs2 = env.step(6, data={"x": cx, "y": cy})
                    if obs2.state == GameState.GAME_OVER:
                        new_effects[(cx, cy)] = -1
                        continue
                    g_a = safe_frame_extract(obs2)
                    new_effects[(cx, cy)] = int(np.sum(g_b[1:] != g_a[1:]))

                    # Check if this click would solve it
                    if obs2.levels_completed > current_levels:
                        solution.append((cx, cy))
                        log.info("arc.greedy_solved", clicks=len(solution))
                        return solution

                # Check for sub-level (new valves may appear)
                if any(
                    new_effects[c] != effects.get(c, 0)
                    for c in action_set
                    if new_effects.get(c, 0) > 0
                ):
                    # Effects changed — re-scan positions
                    new_positions = self._scan_effective_positions(env, full_seq)
                    if new_positions:
                        for p in new_positions:
                            if p not in action_set:
                                action_set.append(p)
                                # Measure new valve
                                obs3 = env.reset()
                                for rx, ry in full_seq:
                                    obs3 = env.step(6, data={"x": rx, "y": ry})
                                g_b3 = safe_frame_extract(obs3)
                                obs3 = env.step(6, data={"x": p[0], "y": p[1]})
                                if obs3.state != GameState.GAME_OVER:
                                    g_a3 = safe_frame_extract(obs3)
                                    new_effects[p] = int(np.sum(g_b3[1:] != g_a3[1:]))

                effects = new_effects

        return None

    def _height_space_solve(
        self,
        env: Any,
        replay_prefix: list[tuple[int, int]],
        action_set: list[tuple[int, int]],
        current_levels: int,
        timeout: float,
    ) -> list[tuple[int, int]] | None:
        """Simulation-A*: real env.step() calls with height-based state dedup.

        Unlike pure A* with constant deltas, this measures REAL heights after
        each click. Handles state-dependent effects (valves have less effect
        when source container is nearly empty).
        """
        import heapq
        import time

        from arcengine.enums import GameState

        if timeout <= 0:
            return None

        t0 = time.monotonic()

        # --- Identify containers from grid ---
        obs = env.reset()
        for rx, ry in replay_prefix:
            obs = env.step(6, data={"x": rx, "y": ry})
        grid = safe_frame_extract(obs)

        col_white = np.array([int(np.sum(grid[1:57, x] == 0)) for x in range(64)])
        containers: list[tuple[int, int]] = []
        in_container = False
        start = 0
        for x in range(64):
            if col_white[x] > 0 and not in_container:
                start = x
                in_container = True
            elif col_white[x] == 0 and in_container:
                containers.append((start, x - 1))
                in_container = False
        if in_container:
            containers.append((start, 63))

        if not containers:
            return None

        def measure_heights(g: np.ndarray[Any, Any]) -> tuple[int, ...]:
            return tuple(int(np.sum(g[1:57, cs : ce + 1] == 0)) for cs, ce in containers)

        current_heights = measure_heights(grid)

        # --- Find target heights from marker colors ---
        # Try teal(14), violet(15), green(4), gray(11) as possible markers
        marker_ys, marker_xs = np.array([], dtype=int), np.array([], dtype=int)
        for marker_color in [14, 15, 10, 13]:
            ys_m, xs_m = np.where(grid == marker_color)
            if len(ys_m) > 0 and len(ys_m) <= 30:  # markers are small
                marker_ys = np.concatenate([marker_ys, ys_m])
                marker_xs = np.concatenate([marker_xs, xs_m])
        # Also try green(4) and gray(11) if they're small enough to be markers
        for marker_color in [4, 11]:
            ys_m, xs_m = np.where(grid == marker_color)
            if 0 < len(ys_m) <= 30:
                marker_ys = np.concatenate([marker_ys, ys_m])
                marker_xs = np.concatenate([marker_xs, xs_m])

        teal_ys, teal_xs = marker_ys, marker_xs

        target_list: list[int] = []
        for idx, (cs, ce) in enumerate(containers):
            cc = (cs + ce) // 2
            cw = ce - cs + 1
            best_y = -1
            best_dist = 999
            for ty, tx in zip(teal_ys.tolist(), teal_xs.tolist(), strict=False):
                d = abs(tx - cc)
                if d < best_dist and d < cw + 4:
                    best_dist = d
                    best_y = ty
            if best_y >= 0:
                th = 0
                for r in range(56, best_y - 1, -1):
                    for c in range(cs, ce + 1):
                        if grid[r, c] in (0, 3):
                            th += 1
                target_list.append(th)
            else:
                target_list.append(current_heights[idx])

        target = tuple(target_list)

        if current_heights == target:
            return []

        log.info(
            "arc.sim_astar_start",
            containers=len(containers),
            valves=len(action_set),
            current=current_heights,
            target=target,
        )

        # --- Simulation BFS: real clicks, height-based dedup ---
        # Use BFS (not A*) — target heights are unreliable, rely only on levels_completed
        def heuristic(h: tuple[int, ...]) -> int:
            return 0  # pure BFS — levels_completed is the only reliable goal signal

        # pq: (priority, depth, heights, click_path)
        pq: list[tuple[int, int, tuple[int, ...], list[tuple[int, int]]]] = [
            (heuristic(current_heights), 0, current_heights, [])
        ]
        visited: dict[tuple[int, ...], int] = {current_heights: 0}
        max_depth = 100  # baseline max is 92 actions for VC33 L4
        max_states = 500_000

        while pq:
            if time.monotonic() - t0 > timeout:
                break
            if len(visited) > max_states:
                break

            _, depth, heights, path = heapq.heappop(pq)

            if depth >= max_depth:
                continue

            for vx, vy in action_set:
                # Real simulation: replay prefix + path + this click
                full_seq = replay_prefix + path + [(vx, vy)]

                obs = env.reset()
                game_over = False
                for rx, ry in full_seq:
                    obs = env.step(6, data={"x": rx, "y": ry})
                    if obs.state == GameState.GAME_OVER:
                        game_over = True
                        break

                if game_over:
                    continue

                if obs.levels_completed > current_levels:
                    solution = [*path, (vx, vy)]
                    log.info("arc.sim_astar_solved", clicks=len(solution), states=len(visited))
                    return solution

                g = safe_frame_extract(obs)
                new_heights = measure_heights(g)
                new_depth = depth + 1

                # Skip if we've seen this height state at equal or lower depth
                if visited.get(new_heights, 999) <= new_depth:
                    continue
                visited[new_heights] = new_depth

                h = heuristic(new_heights)
                # Prune: if heuristic increased compared to parent, lower priority
                new_path = [*path, (vx, vy)]
                heapq.heappush(pq, (new_depth + h, new_depth, new_heights, new_path))

        # Log reachable state range for debugging
        if visited:
            all_states = list(visited.keys())
            for dim in range(len(containers)):
                vals = sorted(set(s[dim] for s in all_states))
                log.info(
                    "arc.sim_astar_dim_range", dim=dim, min=vals[0], max=vals[-1], unique=len(vals)
                )

        log.info(
            "arc.sim_astar_failed", states=len(visited), time_s=round(time.monotonic() - t0, 1)
        )
        return None
