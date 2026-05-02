#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 Phase-A live A/B validation driver.

Runs an A/B comparison between four Cognithor PSE agents against three
ARC-AGI-3 games using the in-process arc_agi SDK + EpisodeRunner.

Agents:
  random_baseline  - RandomActionAgent
  dsl_baseline     - Sprint10DSLAgent (no Sprint-12 wiring)
  dsl_full         - Sprint10DSLAgent with all Sprint-12 wirings
                     (fast-path + frame_analyzer + click_target_sampler
                     + audit + profile)
  llm_full         - LLMReasoningAgent over in-process vLLM
                     (qwen3.6:27b NVFP4) with the same persistence

Games: bp35 (click-target), ft09 (click+movement), lp85 (toggle).

Run from inside an env that has:
  * arc_agi (the official SDK) installed
  * cognithor (this repo) installed editable
  * vllm 0.20.0 + sakamakismile/Qwen3.6-27B-NVFP4 (only needed for llm_full)

Usage::

    cd ~/ARC-AGI-3-Agents
    uv run python /mnt/d/Jarvis/jarvis\\ complete\\ v20/scripts/sprint12_phase_a_validation.py
    # or with a subset:
    uv run python sprint12_phase_a_validation.py --games bp35 --agents random,dsl_baseline

Results land in ``cognithor_bench/results/sprint12_phase_a/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sprint-12 imports — fail fast if the new stack isn't installed.
# ---------------------------------------------------------------------------

try:
    from cognithor.channels.program_synthesis.arc_agi3 import (
        ArcAuditTrail,
        ClickTargetSampler,
        EpisodeRunner,
        FrameAnalyzer,
        GameProfile,
        LLMReasoningAgent,
        LLMTelemetry,
        MTPStats,
        RandomActionAgent,
        Sprint10DSLAgent,
        build_inprocess_vllm_choice_fn,
    )
except ImportError as exc:
    print(f"FATAL: cognithor.channels.program_synthesis.arc_agi3 not importable: {exc}")
    print("Install via: uv pip install -e /path/to/cognithor")
    sys.exit(1)


DEFAULT_GAMES = ["bp35", "ft09", "lp85"]
DEFAULT_AGENTS = ["random_baseline", "dsl_baseline", "dsl_full", "llm_full"]


@dataclass
class RunResult:
    """One row in the A/B matrix."""

    agent: str
    game_id: str
    levels_completed: int
    win_levels: int
    total_steps: int
    final_state: str
    won: bool
    score: float
    wall_clock_s: float
    audit_path: str | None
    error: str | None


def _make_random_agent(game_id: str, results_dir: Path) -> tuple[Any, str | None]:
    return RandomActionAgent(), None


def _make_dsl_baseline(game_id: str, results_dir: Path) -> tuple[Any, str | None]:
    # Plain Sprint10DSLAgent — no Sprint-12 wirings.
    return Sprint10DSLAgent(), None


def _make_dsl_full(game_id: str, results_dir: Path) -> tuple[Any, str | None]:
    audit_path = results_dir / f"{game_id}_dsl_full.jsonl"
    trail = ArcAuditTrail(game_id=game_id)
    profile = GameProfile.load(game_id) or _empty_profile(game_id)
    agent = Sprint10DSLAgent(
        audit_trail=trail,
        game_profile=profile,
        strategy_name="dsl_full",
        frame_analyzer=FrameAnalyzer(),
        click_target_sampler=ClickTargetSampler(),
        fast_path_enabled=True,
    )
    # Stash the trail + path so we can export it post-run.
    agent.__dict__["_phase_a_trail"] = trail
    agent.__dict__["_phase_a_audit_path"] = audit_path
    agent.__dict__["_phase_a_profile"] = profile
    return agent, str(audit_path)


def _make_llm_full(game_id: str, results_dir: Path) -> tuple[Any, str | None]:
    audit_path = results_dir / f"{game_id}_llm_full.jsonl"
    trail = ArcAuditTrail(game_id=game_id)
    profile = GameProfile.load(game_id) or _empty_profile(game_id)
    # Sprint-15 telemetry aggregators. Both threaded into the choice_fn
    # (so each llm.chat() pushes a record/snapshot) AND into the agent
    # (so the audit trail picks them up at log_step time). Without
    # this dual-wiring the audit fields stay None despite correct
    # code on main — the silent-None failure mode the reviewer flagged.
    mtp_stats = MTPStats()
    telemetry = LLMTelemetry()
    try:
        # MTP speculative decoding for ~1.9× decode lift. The MTP-aware
        # checkpoint and the speculative_config are matched: both must
        # use the same num_speculative_tokens. The Text-NVFP4-MTP
        # variant pairs with the base NVFP4 model — note the explicit
        # `-Text-` infix (not the bare `-NVFP4-MTP` we initially
        # assumed; that path 401s on HF as of 2026-05-02).
        choice_fn = build_inprocess_vllm_choice_fn(
            speculative_config={
                "model": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
                "num_speculative_tokens": 3,
            },
            kv_cache_dtype="fp8",
            mtp_stats=mtp_stats,
            telemetry=telemetry,
        )
    except RuntimeError as exc:
        print(f"  [llm_full] vLLM init failed ({exc}); falling back to dsl_full")
        return _make_dsl_full(game_id, results_dir)
    # LLM agent intentionally has NO ClickTargetSampler — the LLM is
    # the click strategy, and a sampler would short-circuit every
    # ACTION6 emission before the LLM is queried. fast_path_enabled
    # stays True (toggle fast-path is cheaper than LLM and only fires
    # when a clean toggle pair is observed).
    agent = LLMReasoningAgent(
        choice_fn=choice_fn,
        audit_trail=trail,
        game_profile=profile,
        strategy_name="llm_full",
        frame_analyzer=FrameAnalyzer(),
        fast_path_enabled=True,
        telemetry=telemetry,
        mtp_stats=mtp_stats,
    )
    agent.__dict__["_phase_a_trail"] = trail
    agent.__dict__["_phase_a_audit_path"] = audit_path
    agent.__dict__["_phase_a_profile"] = profile
    agent.__dict__["_phase_a_mtp_stats"] = mtp_stats
    agent.__dict__["_phase_a_telemetry"] = telemetry
    return agent, str(audit_path)


def assert_telemetry_active(agent: Any, *, strict: bool = True) -> None:
    """Sanity-check after the first episode that the telemetry wiring
    actually catches data.

    Silent-None failure mode: if the choice-fn factory's kwargs aren't
    threaded through correctly, every code path runs green but the
    aggregators stay empty. This assertion fails loudly after the
    first run rather than 20 episodes later when the JSONL is
    inspected.

    Set ``strict=False`` for debugging when you genuinely expect zero
    LLM calls (e.g. a pure-DSL fallback episode).
    """
    mtp = agent.__dict__.get("_phase_a_mtp_stats")
    tele = agent.__dict__.get("_phase_a_telemetry")
    if mtp is None or tele is None:
        return  # not an llm_full agent
    issues: list[str] = []
    if len(tele.records) == 0:
        issues.append(
            "LLMTelemetry.records is empty — choice_fn never fired or telemetry kwarg lost"
        )
    if len(mtp.snapshots) == 0:
        issues.append(
            "MTPStats.snapshots is empty — either MTP off, or mtp_stats kwarg "
            "lost (check build_inprocess_vllm_choice_fn signature)"
        )
    # TTFT coverage check — flags the disable_log_stats trap.
    if tele.records:
        with_ttft = sum(1 for r in tele.records if r.ttft_s is not None)
        coverage = with_ttft / len(tele.records)
        if coverage < 0.8:
            issues.append(
                f"TTFT coverage = {coverage:.0%} — vLLM RequestOutput.metrics not "
                "populated. Check engine_args.disable_log_stats=False (or the "
                "vLLM-version-specific equivalent)."
            )
    if issues:
        msg = "; ".join(issues)
        if strict:
            raise RuntimeError(f"Sprint-15 telemetry sanity-check failed: {msg}")
        print(f"  [warn] telemetry sanity-check: {msg}")


def _empty_profile(game_id: str) -> GameProfile:
    return GameProfile(
        game_id=game_id,
        game_type="mixed",
        available_actions=[],
        click_zones=[],
        target_colors=[],
        movement_effects={},
        win_condition="",
        vision_description="",
        vision_strategy="",
        strategy_metrics={},
    )


_AGENT_FACTORIES = {
    "random_baseline": _make_random_agent,
    "dsl_baseline": _make_dsl_baseline,
    "dsl_full": _make_dsl_full,
    "llm_full": _make_llm_full,
}


def run_one(agent_label: str, game_id: str, max_steps: int, results_dir: Path) -> RunResult:
    factory = _AGENT_FACTORIES[agent_label]
    agent, audit_path = factory(game_id, results_dir)

    print(f"  {agent_label:18s} {game_id:5s}  ", end="", flush=True)
    t0 = time.monotonic()
    runner = EpisodeRunner(agent=agent, game_id=game_id, max_steps=max_steps)
    result = runner.run()
    wall_clock = time.monotonic() - t0

    # Sprint-15: sanity-check telemetry after the first episode of any
    # llm_full run so silent-None failures (kwargs not threaded, vLLM
    # disable_log_stats trap) surface immediately. Non-strict so a
    # pure-DSL-fallback episode doesn't tank the whole driver run.
    try:
        assert_telemetry_active(agent, strict=False)
    except Exception as exc:
        print(f"\n    telemetry sanity-check error: {exc}")

    # Export audit + persist profile if wired.
    trail = agent.__dict__.get("_phase_a_trail") if hasattr(agent, "__dict__") else None
    if trail is not None and audit_path is not None:
        try:
            trail.export_jsonl(audit_path)
        except Exception as exc:
            print(f"\n    audit export failed: {exc}")
    profile = agent.__dict__.get("_phase_a_profile") if hasattr(agent, "__dict__") else None
    if profile is not None:
        try:
            profile.save()
        except Exception as exc:
            print(f"\n    profile save failed: {exc}")

    print(
        f"levels={result.levels_completed}/{result.win_levels} "
        f"steps={result.total_steps:3d} "
        f"score={result.score:.3f} "
        f"state={result.final_state:12s} "
        f"t={wall_clock:6.1f}s"
    )

    return RunResult(
        agent=agent_label,
        game_id=game_id,
        levels_completed=result.levels_completed,
        win_levels=result.win_levels,
        total_steps=result.total_steps,
        final_state=result.final_state,
        won=result.won,
        score=result.score,
        wall_clock_s=round(wall_clock, 2),
        audit_path=audit_path,
        error=result.error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint-12 Phase-A validation driver")
    parser.add_argument(
        "--games",
        type=lambda s: s.split(","),
        default=DEFAULT_GAMES,
        help=f"Comma-separated game IDs (default: {','.join(DEFAULT_GAMES)})",
    )
    parser.add_argument(
        "--agents",
        type=lambda s: s.split(","),
        default=DEFAULT_AGENTS,
        help=f"Comma-separated agent labels (default: {','.join(DEFAULT_AGENTS)})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="Per-episode action cap (matches ARC-AGI-3 MAX_ACTIONS=80)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override results directory (default: cognithor_bench/results/sprint12_phase_a/<ts>/)",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    results_dir = args.results_dir or (
        Path(__file__).resolve().parent.parent
        / "cognithor_bench"
        / "results"
        / "sprint12_phase_a"
        / timestamp
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sprint-12 Phase-A validation — results: {results_dir}")
    print(f"Games:  {args.games}")
    print(f"Agents: {args.agents}")
    print(f"Max steps per episode: {args.max_steps}")
    print()

    rows: list[RunResult] = []
    for agent_label in args.agents:
        if agent_label not in _AGENT_FACTORIES:
            print(f"WARN: unknown agent label '{agent_label}', skipping")
            continue
        for game_id in args.games:
            try:
                rows.append(run_one(agent_label, game_id, args.max_steps, results_dir))
            except Exception as exc:
                print(f"  {agent_label:18s} {game_id:5s}  CRASHED: {exc}")
                rows.append(
                    RunResult(
                        agent=agent_label,
                        game_id=game_id,
                        levels_completed=0,
                        win_levels=0,
                        total_steps=0,
                        final_state="CRASH",
                        won=False,
                        score=0.0,
                        wall_clock_s=0.0,
                        audit_path=None,
                        error=str(exc),
                    )
                )

    out = results_dir / "results.json"
    out.write_text(
        json.dumps([asdict(r) for r in rows], indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {len(rows)} rows → {out}")

    print("\n=== Summary ===")
    print(f"{'agent':<18} {'game':<6} {'lvls':>4} {'steps':>5} {'score':>6} {'state':<12}")
    for r in rows:
        print(
            f"{r.agent:<18} {r.game_id:<6} "
            f"{r.levels_completed:>4d} {r.total_steps:>5d} "
            f"{r.score:>6.3f} {r.final_state:<12}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
