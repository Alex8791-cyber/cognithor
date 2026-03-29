"""ARC-AGI-3 State Graph Navigator: directed graph of observed state transitions.

Maps the state space as a graph and finds shortest paths to WIN states.
This graph-based approach is key to achieving strong performance on ARC-AGI-3
puzzles (analogous to the RL+graph approaches that significantly outperform
pure LLM methods).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

__all__ = [
    "StateGraphNavigator",
    "StateNode",
    "StateEdge",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StateNode:
    """A node in the state graph representing a unique observed grid state."""

    state_hash: str
    visit_count: int = 0
    is_win: bool = False
    is_game_over: bool = False
    level: int = 0


@dataclass
class StateEdge:
    """A directed edge in the state graph representing an observed transition."""

    action: str
    action_data: dict | None = None
    pixels_changed: int = 0
    traversal_count: int = 0


# ---------------------------------------------------------------------------
# StateGraphNavigator
# ---------------------------------------------------------------------------


class StateGraphNavigator:
    """Directed graph of observed state transitions for ARC-AGI-3 puzzles.

    Maintains a graph where nodes are unique grid states (identified by hash)
    and edges are actions that caused transitions between states.  BFS over
    the graph finds the shortest path to any known WIN state, enabling
    deterministic replay of winning action sequences.

    Args:
        max_states: Hard cap on the number of stored nodes to prevent
            unbounded memory growth during long episodes.
    """

    def __init__(self, max_states: int = 200_000) -> None:
        self.nodes: dict[str, StateNode] = {}
        # edges[from_hash][edge_key] = (to_hash, StateEdge)
        # edge_key is "{action}:{action_data_repr}" for deduplication
        self.edges: dict[str, dict[str, tuple[str, StateEdge]]] = defaultdict(dict)
        self.win_states: set[str] = set()
        self.game_over_states: set[str] = set()
        self._cached_win_path: list[tuple[str, dict | None, str]] | None = None
        self._cache_valid_from: str | None = None
        self.max_states = max_states
        self._hash_cache: dict[tuple, str] = {}  # shape-aware like episode_memory
        self.total_edges: int = 0
        self.action_patterns_from_previous: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def hash_grid(self, grid: np.ndarray) -> str:
        """Return a 16-character hex MD5 hash of *grid*, cached by content.

        The cache key includes shape and dtype so that arrays with identical
        raw bytes but different shapes or dtypes do not collide.  Same
        pattern as :meth:`EpisodeMemory.hash_grid`.
        """
        raw = grid.tobytes()
        key = (grid.shape, grid.dtype.str, raw)
        cached = self._hash_cache.get(key)
        if cached is not None:
            return cached
        shape_prefix = np.array(grid.shape, dtype=np.int64).tobytes()
        digest = hashlib.md5(shape_prefix + raw, usedforsecurity=False).hexdigest()[:16]  # noqa: S324
        self._hash_cache[key] = digest
        return digest

    # ------------------------------------------------------------------
    # State detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_win(game_state_str: str) -> bool:
        """Return True if *game_state_str* represents a WIN outcome."""
        return game_state_str in ("WIN", "GameState.WIN")

    @staticmethod
    def _is_game_over(game_state_str: str) -> bool:
        """Return True if *game_state_str* represents a GAME_OVER outcome."""
        return game_state_str in ("GAME_OVER", "GameState.GAME_OVER")

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def add_transition(
        self,
        from_grid: np.ndarray,
        action_str: str,
        action_data: dict | None,
        to_grid: np.ndarray,
        pixels_changed: int,
        game_state: str,
        level: int = 0,
    ) -> tuple[str, str]:
        """Record a state transition as a directed edge in the graph.

        Creates :class:`StateNode` objects for both states if they are new
        (subject to *max_states* cap).  Updates visit counts and traversal
        counters, marks WIN / GAME_OVER states, and invalidates the cached
        win path whenever a new edge is added.

        Args:
            from_grid: Grid before the action.
            action_str: Human-readable action identifier.
            action_data: Optional structured action parameters.
            to_grid: Grid after the action.
            pixels_changed: Number of pixels that changed.
            game_state: String representation of the resulting game state.
            level: Current puzzle level.

        Returns:
            ``(from_hash, to_hash)`` pair of 16-char hex strings.
        """
        from_hash = self.hash_grid(from_grid)
        to_hash = self.hash_grid(to_grid)

        is_win = self._is_win(game_state)
        is_go = self._is_game_over(game_state)

        # Create / update nodes (respect max_states cap)
        if from_hash not in self.nodes and len(self.nodes) < self.max_states:
            self.nodes[from_hash] = StateNode(state_hash=from_hash, level=level)
        if from_hash in self.nodes:
            self.nodes[from_hash].visit_count += 1

        if to_hash not in self.nodes and len(self.nodes) < self.max_states:
            self.nodes[to_hash] = StateNode(
                state_hash=to_hash,
                is_win=is_win,
                is_game_over=is_go,
                level=level,
            )
        if to_hash in self.nodes:
            node = self.nodes[to_hash]
            if is_win:
                node.is_win = True
            if is_go:
                node.is_game_over = True

        # Track win / game-over sets (independent of node cap)
        if is_win:
            self.win_states.add(to_hash)
        if is_go:
            self.game_over_states.add(to_hash)

        # Edge key: unique per (source, action, action_data)
        ad_repr = repr(action_data) if action_data is not None else ""
        edge_key = f"{action_str}:{ad_repr}"

        from_edges = self.edges[from_hash]
        if edge_key in from_edges:
            # Edge already known — increment traversal count only
            _, existing_edge = from_edges[edge_key]
            existing_edge.traversal_count += 1
        else:
            # New edge
            new_edge = StateEdge(
                action=action_str,
                action_data=action_data,
                pixels_changed=pixels_changed,
                traversal_count=1,
            )
            from_edges[edge_key] = (to_hash, new_edge)
            self.total_edges += 1
            # Invalidate cached win path on structural change
            self._cached_win_path = None
            self._cache_valid_from = None

        return from_hash, to_hash

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def find_win_path(
        self,
        from_hash: str,
    ) -> list[tuple[str, dict | None, str]] | None:
        """BFS from *from_hash* to the nearest known WIN state.

        Skips GAME_OVER states during traversal.  Returns a cached result
        when the graph has not changed since the last call from the same
        starting node.

        Args:
            from_hash: Hash of the starting state.

        Returns:
            A list of ``(action_str, action_data, next_state_hash)`` tuples
            representing the shortest path, an empty list if *from_hash* is
            already a WIN state, or ``None`` if no win path exists.
        """
        if not self.win_states:
            return None

        # Already at a win state
        if from_hash in self.win_states:
            return []

        # Return cached result if still valid
        if self._cached_win_path is not None and self._cache_valid_from == from_hash:
            return self._cached_win_path

        # BFS
        queue: deque[tuple[str, list[tuple[str, dict | None, str]]]] = deque()
        queue.append((from_hash, []))
        visited: set[str] = {from_hash}

        while queue:
            current, path = queue.popleft()

            for _edge_key, (next_hash, edge) in self.edges.get(current, {}).items():
                if next_hash in visited:
                    continue
                # Skip game-over states
                if next_hash in self.game_over_states:
                    continue

                new_path = path + [(edge.action, edge.action_data, next_hash)]

                if next_hash in self.win_states:
                    self._cached_win_path = new_path
                    self._cache_valid_from = from_hash
                    return new_path

                visited.add(next_hash)
                queue.append((next_hash, new_path))

        return None

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def get_best_exploration_action(
        self,
        current_hash: str,
        available_actions: list[str],
    ) -> tuple[str, dict | None] | None:
        """Suggest the best action to take for exploration from *current_hash*.

        Priority 1 — Untested actions: actions in *available_actions* that
        have not yet been tried from the current state.  Ranked by
        ``action_patterns_from_previous`` score (descending) to prefer
        actions that worked in past levels.

        Priority 2 — Least-visited neighbor: among already-tested actions,
        prefer the one leading to the state with the lowest visit count
        (exploration value = 1 / (visit_count + 1)).  Game-over states are
        skipped.

        Args:
            current_hash: Hash of the current state.
            available_actions: All actions the agent may take.

        Returns:
            ``(action_str, action_data)`` pair, or ``None`` if *available_actions*
            is empty.
        """
        if not available_actions:
            return None

        tried_edge_keys = set(self.edges.get(current_hash, {}).keys())
        tried_actions: set[str] = set()
        for ek in tried_edge_keys:
            action_part = ek.split(":", 1)[0]
            tried_actions.add(action_part)

        # Priority 1: untested actions
        untested = [a for a in available_actions if a not in tried_actions]
        if untested:
            # Rank by previous-level pattern score (higher = better)
            untested.sort(
                key=lambda a: self.action_patterns_from_previous.get(a, 0.0),
                reverse=True,
            )
            return untested[0], None

        # Priority 2: already-tested action leading to least-visited neighbor
        best_action: str | None = None
        best_ad: dict | None = None
        best_value: float = -1.0

        for _edge_key, (next_hash, edge) in self.edges.get(current_hash, {}).items():
            if edge.action not in available_actions:
                continue
            if next_hash in self.game_over_states:
                continue
            node = self.nodes.get(next_hash)
            visit = node.visit_count if node is not None else 0
            value = 1.0 / (visit + 1)
            if value > best_value:
                best_value = value
                best_action = edge.action
                best_ad = edge.action_data

        if best_action is not None:
            return best_action, best_ad

        # Fallback: return first available action
        return available_actions[0], None

    # ------------------------------------------------------------------
    # Navigation readiness
    # ------------------------------------------------------------------

    def should_navigate(self) -> bool:
        """Return True if at least one WIN state has been discovered."""
        return bool(self.win_states)

    # ------------------------------------------------------------------
    # Level management
    # ------------------------------------------------------------------

    def prepare_for_new_level(self) -> None:
        """Extract cross-level action patterns, then reset all graph state.

        Patterns are derived from the win-rate of each action in the
        outgoing edges of win-adjacent nodes.  The score for an action is
        the ratio of edges leading to WIN states over total edges using
        that action, providing a useful prior for the next level.
        """
        # Compute action patterns from current graph
        action_win_count: dict[str, int] = defaultdict(int)
        action_total_count: dict[str, int] = defaultdict(int)

        for from_hash, from_edges in self.edges.items():
            for _ek, (to_hash, edge) in from_edges.items():
                action_total_count[edge.action] += edge.traversal_count
                if to_hash in self.win_states:
                    action_win_count[edge.action] += edge.traversal_count

        self.action_patterns_from_previous = {
            action: action_win_count.get(action, 0) / max(total, 1)
            for action, total in action_total_count.items()
        }

        # Reset graph structures
        self.nodes.clear()
        self.edges.clear()
        self.win_states.clear()
        self.game_over_states.clear()
        self._cached_win_path = None
        self._cache_valid_from = None
        self._hash_cache.clear()
        self.total_edges = 0

    # ------------------------------------------------------------------
    # Statistics / summaries
    # ------------------------------------------------------------------

    def get_exploration_coverage(self) -> dict:
        """Return a dictionary of graph statistics for monitoring.

        Keys:
            states (int): Total node count.
            edges (int): Total edge count.
            win_states (int): Number of known WIN states.
            game_over_states (int): Number of known GAME_OVER states.
            coverage (float): Ratio of visited states to *max_states*.
            has_win_path (bool): Whether any WIN state is reachable.
        """
        return {
            "states": len(self.nodes),
            "edges": self.total_edges,
            "win_states": len(self.win_states),
            "game_over_states": len(self.game_over_states),
            "coverage": len(self.nodes) / self.max_states,
            "has_win_path": bool(self.win_states),
        }

    def get_summary_for_llm(self) -> str:
        """Return a compact, human-readable summary for injection into LLM context."""
        cov = self.get_exploration_coverage()
        lines: list[str] = [
            f"StateGraph: {cov['states']} Zustaende, {cov['edges']} Kanten",
            f"Win-Zustaende: {cov['win_states']}, GameOver-Zustaende: {cov['game_over_states']}",
            f"Abdeckung: {cov['coverage']:.1%} von max {self.max_states}",
        ]
        if self.win_states:
            lines.append("WIN-Pfad verfuegbar: ja")
        if self.action_patterns_from_previous:
            top = sorted(
                self.action_patterns_from_previous.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]
            lines.append(
                "Beste Aktionen (vorheriges Level): " + ", ".join(f"{a}({s:.0%})" for a, s in top)
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _compute_histogram(self, grid: np.ndarray) -> tuple:
        """Return a histogram tuple of pixel value counts for *grid*.

        Produces a 13-bin histogram covering values 0-12 (inclusive),
        suitable for use as lightweight node metadata or a feature vector.
        """
        counts = np.zeros(13, dtype=np.int64)
        flat = grid.ravel()
        for val in range(13):
            counts[val] = int(np.sum(flat == val))
        return tuple(counts.tolist())
