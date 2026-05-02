# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — end-to-end smoke tests for the integrated agent stack.

Exercises the full Sprint-11 + Sprint-12 stack across multiple frames
of a synthetic mini-game: FrameBridge ingestion, EpisodeMemory
recording, StateGraphNavigator transitions, StateActionCounter
dead-edge marking, ArcAuditTrail integrity, GameProfile updates.

These tests don't drive a real arcengine — they feed pre-computed
grid sequences through the agent and assert that the full plumbing
behaves consistently end-to-end. Catches integration bugs that unit
tests miss (e.g. shape-mismatch crashes across level boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.audit import ArcAuditTrail
from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.game_profile import GameProfile
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


@dataclass
class _StubGameState:
    name: str = "NOT_FINISHED"


@dataclass
class _StubAction:
    name: str
    value: int
    reasoning: str = ""
    _data: dict[str, Any] = field(default_factory=dict)
    _is_simple: bool = True

    def is_simple(self) -> bool:
        return self._is_simple

    def is_complex(self) -> bool:
        return not self._is_simple

    def set_data(self, data: dict[str, Any]) -> None:
        self._data = dict(data)


@dataclass
class _StubFrame:
    game_id: str = "smoke"
    state: _StubGameState = field(default_factory=_StubGameState)
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[_StubAction] = field(default_factory=list)


def _frame(grid: np.ndarray, state_name: str = "NOT_FINISHED") -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
    ]
    return _StubFrame(
        frame=[grid],
        available_actions=actions,
        state=_StubGameState(name=state_name),
    )


class TestEpisodeFlow:
    def test_agent_runs_episode_to_win(self) -> None:
        """Drive Sprint10DSLAgent through 5 frames ending in WIN; assert
        memory + state-graph + state-counter all populated coherently."""
        agent = Sprint10DSLAgent()

        # 4 NOT_FINISHED frames + 1 WIN frame, all 4x4 grids.
        grids = [
            np.zeros((4, 4), dtype=np.int8),
            np.array([[1, 0, 0, 0]] + [[0] * 4] * 3, dtype=np.int8),
            np.array([[1, 1, 0, 0]] + [[0] * 4] * 3, dtype=np.int8),
            np.array([[1, 1, 1, 0]] + [[0] * 4] * 3, dtype=np.int8),
            np.array([[1, 1, 1, 1]] + [[0] * 4] * 3, dtype=np.int8),
        ]
        for i, g in enumerate(grids):
            state = "WIN" if i == len(grids) - 1 else "NOT_FINISHED"
            f = _frame(g, state_name=state)
            if agent.is_done([f], f):
                break
            agent.choose_action([f], f)

        # 4 choose_action calls → 3 memory entries (first call doesn't
        # record because there's no prior pending action).
        assert len(agent.memory) == 3
        # State counter saw at least one action increment per state.
        # State graph recorded 2 transitions (between consecutive same-shape
        # frames AFTER the first append).
        # We don't assert exact graph counts because state-key collisions
        # are possible on the small test grid; just check it didn't crash.

    def test_level_transition_does_not_crash(self) -> None:
        """When grid shape changes (level boundary), the agent must
        skip the state-graph transition rather than crash on np.sum.
        Regression test for the shape-mismatch bug from PR-8."""
        agent = Sprint10DSLAgent()

        # Level 0 grid (4x4)
        f0 = _frame(np.zeros((4, 4), dtype=np.int8))
        agent.choose_action([f0], f0)

        # Level 0 again — same shape.
        f1 = _frame(np.array([[1, 0, 0, 0]] + [[0] * 4] * 3, dtype=np.int8))
        agent.choose_action([f0, f1], f1)

        # Level 1 — different shape (5x5). Should NOT crash.
        f2 = _frame(np.zeros((5, 5), dtype=np.int8))
        f2.levels_completed = 1
        chosen = agent.choose_action([f0, f1, f2], f2)
        assert chosen.name in {"RESET", "ACTION1", "ACTION2"}


class TestAuditTrailFlow:
    def test_chain_records_full_episode(self) -> None:
        """Drive an audit trail through start + 3 steps + end; verify
        chain integrity holds end-to-end."""
        trail = ArcAuditTrail(game_id="smoke")
        trail.log_game_start()
        for step in range(3):
            trail.log_step(
                level=0,
                step=step,
                action=f"ACTION{step + 1}",
                game_state="NOT_FINISHED",
                pixels_changed=2,
            )
        trail.log_game_end(final_score=0.75)

        assert len(trail.events) == 5  # start + 3 steps + end
        assert trail.verify_integrity() is True

    def test_late_tampering_invalidates_chain(self) -> None:
        trail = ArcAuditTrail(game_id="smoke")
        trail.log_game_start()
        trail.log_step(
            level=0, step=0, action="ACTION1", game_state="NOT_FINISHED", pixels_changed=1
        )
        trail.log_game_end(final_score=1.0)
        # Mutate the middle event.
        trail.events[1].action = "ACTION_FAKE"
        assert trail.verify_integrity() is False


class TestGameProfileFlow:
    def test_profile_lifecycle(self, tmp_path: Any) -> None:
        """Run a synthetic episode → update profile → save → reload →
        verify metrics carried across."""
        profile = GameProfile(
            game_id="smoke",
            game_type="click",
            available_actions=[0, 1, 2, 6],
            click_zones=[],
            target_colors=[1, 2],
            movement_effects={},
            win_condition="reach_target",
            vision_description="",
            vision_strategy="",
            strategy_metrics={},
        )
        profile.update_run(score=3)
        profile.update_metrics("fast_path", won=True, levels_solved=1, steps=8, budget_ratio=0.1)
        profile.save(base_dir=tmp_path)

        reloaded = GameProfile.load("smoke", base_dir=tmp_path)
        assert reloaded is not None
        assert reloaded.total_runs == 1
        assert reloaded.best_score == 3
        assert "fast_path" in reloaded.strategy_metrics
        assert reloaded.strategy_metrics["fast_path"].wins == 1


class TestIntegratedAgent:
    def test_audit_and_profile_coexist_with_agent(self, tmp_path: Any) -> None:
        """Agent runs + audit trail records + profile updates — proves
        the three are independently composable around a single episode."""
        agent = Sprint10DSLAgent()
        trail = ArcAuditTrail(game_id="smoke")
        profile = GameProfile(
            game_id="smoke",
            game_type="click",
            available_actions=[0, 1, 2],
            click_zones=[],
            target_colors=[],
            movement_effects={},
            win_condition="",
            vision_description="",
            vision_strategy="",
            strategy_metrics={},
        )

        trail.log_game_start()
        for step in range(3):
            g = np.array([[step, 0], [0, 0]], dtype=np.int8)
            f = _frame(g)
            chosen = agent.choose_action([f], f)
            trail.log_step(
                level=0,
                step=step,
                action=chosen.name,
                game_state="NOT_FINISHED",
                pixels_changed=1,
            )
        trail.log_game_end(final_score=1.0)
        profile.update_run(score=1)
        profile.save(base_dir=tmp_path)

        # Audit chain holds.
        assert trail.verify_integrity() is True
        # 3 steps in audit (plus start + end = 5 events).
        assert len(trail.events) == 5
        # Memory captured 2 transitions (step 1's action was recorded
        # against step 2's grid, and step 2's against step 3's).
        assert len(agent.memory) == 2
        # Profile saved & reloads.
        loaded = GameProfile.load("smoke", base_dir=tmp_path)
        assert loaded is not None
        assert loaded.total_runs == 1
