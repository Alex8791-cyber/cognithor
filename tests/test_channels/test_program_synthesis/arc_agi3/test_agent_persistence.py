# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 PR-6+7 — Sprint10DSLAgent persistence wiring tests.

Verifies that ``Sprint10DSLAgent`` correctly drives an injected
``ArcAuditTrail`` (PR-7) and ``GameProfile`` (PR-6) through the
episode lifecycle: ``log_game_start`` on the first ``choose_action``,
one ``log_step`` per call, ``log_game_end`` + profile update on
``finalize_episode``.
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


def _frame(grid: np.ndarray) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
    ]
    return _StubFrame(frame=[grid], available_actions=actions)


def _make_profile() -> GameProfile:
    return GameProfile(
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


class TestAuditTrailWiring:
    def test_no_trail_no_logging(self) -> None:
        agent = Sprint10DSLAgent()  # no trail
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        # Nothing crashes; nothing logged anywhere.

    def test_game_start_logged_on_first_call(self) -> None:
        trail = ArcAuditTrail(game_id="smoke")
        agent = Sprint10DSLAgent(audit_trail=trail)
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        assert len(trail.events) == 2  # game_start + step
        assert trail.events[0].event_type == "game_start"
        assert trail.events[1].event_type == "step"

    def test_step_per_choose_action(self) -> None:
        trail = ArcAuditTrail(game_id="smoke")
        agent = Sprint10DSLAgent(audit_trail=trail)
        for i in range(3):
            f = _frame(np.array([[i, 0]], dtype=np.int8))
            agent.choose_action([f], f)
        # 1 game_start + 3 steps.
        assert len(trail.events) == 4
        step_events = [e for e in trail.events if e.event_type == "step"]
        assert len(step_events) == 3

    def test_finalize_logs_game_end(self) -> None:
        trail = ArcAuditTrail(game_id="smoke")
        agent = Sprint10DSLAgent(audit_trail=trail)
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        agent.finalize_episode(score=2, won=True, levels_solved=2)
        # 1 start + 1 step + 1 end.
        assert len(trail.events) == 3
        assert trail.events[-1].event_type == "game_end"
        assert trail.events[-1].score == 2.0

    def test_step_picks_up_telemetry_and_mtp_kwargs(self) -> None:
        # Sprint-15 wiring: when a Telemetry + MTPStats aggregator is
        # passed in and has at least one entry by the time the agent
        # logs its step, the audit event carries the new fields on the
        # same hash chain.
        from cognithor.channels.program_synthesis.arc_agi3.llm_telemetry import (
            LLMCallRecord,
            LLMTelemetry,
        )
        from cognithor.channels.program_synthesis.arc_agi3.mtp_stats import (
            MTPSnapshot,
            MTPStats,
        )

        trail = ArcAuditTrail(game_id="smoke")
        tele = LLMTelemetry()
        # Simulate a choice-fn already having pushed one record.
        tele.records.append(
            LLMCallRecord(
                call_index=0,
                input_tokens=512,
                output_tokens=200,
                think_tokens=140,
                finish_reason="stop",
                wall_clock_s=12.5,
            )
        )
        mtp = MTPStats()
        mtp.snapshots.append(MTPSnapshot(100, 70, 120, 3))

        agent = Sprint10DSLAgent(audit_trail=trail, telemetry=tele, mtp_stats=mtp)
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)

        step_events = [e for e in trail.events if e.event_type == "step"]
        assert len(step_events) == 1
        ev = step_events[0]
        assert ev.llm_input_tokens == 512
        assert ev.llm_output_tokens == 200
        assert ev.llm_think_tokens == 140
        assert ev.llm_finish_reason == "stop"
        assert ev.llm_wall_clock_s == 12.5
        assert ev.mtp_drafts_proposed == 100
        assert ev.mtp_drafts_accepted == 70
        assert ev.mtp_acceptance_rate == 0.7
        # Hash chain still verifies with the extra fields included.
        assert trail.verify_integrity() is True

    def test_finalize_idempotent(self) -> None:
        trail = ArcAuditTrail(game_id="smoke")
        agent = Sprint10DSLAgent(audit_trail=trail)
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        agent.finalize_episode(score=1, won=False, levels_solved=0)
        agent.finalize_episode(score=1, won=False, levels_solved=0)  # second call
        # Only ONE game_end recorded.
        end_events = [e for e in trail.events if e.event_type == "game_end"]
        assert len(end_events) == 1

    def test_chain_integrity_holds(self) -> None:
        trail = ArcAuditTrail(game_id="smoke")
        agent = Sprint10DSLAgent(audit_trail=trail)
        for i in range(3):
            f = _frame(np.array([[i, 0]], dtype=np.int8))
            agent.choose_action([f], f)
        agent.finalize_episode(score=3, won=True, levels_solved=2)
        assert trail.verify_integrity() is True


class TestGameProfileWiring:
    def test_no_profile_no_update(self) -> None:
        agent = Sprint10DSLAgent()  # no profile
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        agent.finalize_episode(score=1, won=True, levels_solved=1)
        # Nothing crashes; no profile to update.

    def test_finalize_updates_profile(self) -> None:
        profile = _make_profile()
        agent = Sprint10DSLAgent(game_profile=profile)
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        agent.finalize_episode(score=5, won=True, levels_solved=3, budget_ratio=0.4)

        assert profile.total_runs == 1
        assert profile.best_score == 5
        assert "sprint10_dsl" in profile.strategy_metrics
        m = profile.strategy_metrics["sprint10_dsl"]
        assert m.attempts == 1
        assert m.wins == 1
        assert m.total_levels_solved == 3

    def test_custom_strategy_name(self) -> None:
        profile = _make_profile()
        agent = Sprint10DSLAgent(game_profile=profile, strategy_name="experiment_xyz")
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        agent.finalize_episode(score=0, won=False, levels_solved=0)
        assert "experiment_xyz" in profile.strategy_metrics

    def test_loss_recorded(self) -> None:
        profile = _make_profile()
        agent = Sprint10DSLAgent(game_profile=profile)
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        agent.choose_action([f], f)
        agent.finalize_episode(score=0, won=False, levels_solved=0)
        m = profile.strategy_metrics["sprint10_dsl"]
        assert m.attempts == 1
        assert m.wins == 0


class TestCombinedWiring:
    def test_audit_and_profile_together(self) -> None:
        trail = ArcAuditTrail(game_id="smoke")
        profile = _make_profile()
        agent = Sprint10DSLAgent(audit_trail=trail, game_profile=profile)
        for i in range(3):
            f = _frame(np.array([[i, 0]], dtype=np.int8))
            agent.choose_action([f], f)
        agent.finalize_episode(score=2, won=True, levels_solved=2, budget_ratio=0.5)

        # Audit: 1 start + 3 steps + 1 end = 5 events.
        assert len(trail.events) == 5
        assert trail.verify_integrity() is True
        # Profile updated.
        assert profile.total_runs == 1
        assert profile.strategy_metrics["sprint10_dsl"].wins == 1
