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
        """Smart elimination search: O(n) for common cases, brute-force fallback for small n."""
        import itertools
        import time

        from arcengine.enums import GameState

        from jarvis.arc.cluster_solver import ClusterSolver

        t0 = time.monotonic()
        solver = ClusterSolver(target_color=target_color, max_skip=0)

        def make_env_at_level():
            """Create env and replay to target level."""
            env = self._arcade.make(game_id)
            obs = env.reset()
            for sol in prev_solutions:
                for cx, cy in sol:
                    obs = env.step(6, data={"x": cx, "y": cy})
            return env, obs

        # Get current level grid
        env, obs = make_env_at_level()
        grid = np.array(obs.frame)
        if grid.ndim == 3:
            grid = grid[0]
        centers = solver.find_clusters(grid)
        n = len(centers)

        if n == 0:
            return None

        current_levels = obs.levels_completed

        def test_combo(click_indices: list[int]) -> bool:
            """Test a click combo. Returns True if level advances."""
            env2, obs2 = make_env_at_level()
            for idx in click_indices:
                cx, cy = centers[idx]
                obs2 = env2.step(6, data={"x": cx, "y": cy})
                if obs2.state == GameState.GAME_OVER:
                    return False
            return obs2.levels_completed > current_levels

        def find_poison_clusters(all_indices: list[int]) -> set[int]:
            """Click all clusters and identify which click triggers GAME_OVER."""
            env2, obs2 = make_env_at_level()
            for idx in all_indices:
                cx, cy = centers[idx]
                obs2 = env2.step(6, data={"x": cx, "y": cy})
                if obs2.state == GameState.GAME_OVER:
                    return {idx}  # This cluster is poison
            return set()

        # Phase 1: Try clicking ALL clusters
        all_idx = list(range(n))
        if test_combo(all_idx):
            return [centers[i] for i in all_idx]

        # Phase 2: Identify poison clusters — which clicks cause GAME_OVER?
        poison = find_poison_clusters(all_idx)
        if poison:
            # Remove poison clusters and try again
            safe_idx = [i for i in all_idx if i not in poison]
            if test_combo(safe_idx):
                return [centers[i] for i in safe_idx]

            # Maybe there are more poison clusters after the first
            # Iteratively remove poison until we win or run out
            for _ in range(min(n, 5)):
                if time.monotonic() - t0 > timeout:
                    break
                more_poison = find_poison_clusters(safe_idx)
                if not more_poison:
                    break
                safe_idx = [i for i in safe_idx if i not in more_poison]
                if test_combo(safe_idx):
                    return [centers[i] for i in safe_idx]

        # Phase 3: Single elimination — remove each cluster one at a time
        for skip_idx in range(n):
            if time.monotonic() - t0 > timeout:
                break
            combo = [i for i in range(n) if i != skip_idx]
            if test_combo(combo):
                return [centers[i] for i in combo]

        # Phase 4: Progressive elimination — try removing pairs, triples
        for skip_count in range(2, min(n, 7)):
            if time.monotonic() - t0 > timeout:
                break
            for skip_combo in itertools.combinations(range(n), skip_count):
                if time.monotonic() - t0 > timeout:
                    break
                combo = [i for i in range(n) if i not in skip_combo]
                if test_combo(combo):
                    return [centers[i] for i in combo]

        # Phase 5: For small n (≤10), full brute-force via find_solution
        if n <= 10 and time.monotonic() - t0 < timeout:
            full_solver = ClusterSolver(target_color=target_color, max_skip=6)
            return full_solver.find_solution(
                env_factory=lambda: self._arcade.make(game_id),
                action_id=6,
                prev_solutions=prev_solutions,
                target_level=target_level,
            )

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
