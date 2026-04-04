"""ARC-AGI-3 CLI entry point for Cognithor.

Usage:
    python -m jarvis.arc --game <game_id> [options]
    python -m jarvis.arc --list-games
    python -m jarvis.arc --mode benchmark
    python -m jarvis.arc --mode swarm --parallel 4

Modes:
    single      Run a single game (default)
    benchmark   Run all known games sequentially
    swarm       Run games in parallel via ArcSwarmOrchestrator
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis.arc",
        description="Cognithor ARC-AGI-3 Benchmark Agent",
    )
    parser.add_argument(
        "--game",
        metavar="GAME_ID",
        default="",
        help="Game/environment ID to play (required for single mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "benchmark", "swarm", "analyzer"],
        default="single",
        help="Run mode: single (default), benchmark, swarm (parallel), analyzer (game analysis)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Disable LLM planner (algorithmic-only mode)",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        default=False,
        help="Enable CNN visual encoder",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        metavar="N",
        help="Max parallel workers for swarm mode (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output",
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        default=False,
        help="List available game IDs and exit",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="",
        help="Path to Jarvis config.yaml (optional)",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        default=False,
        help="Force re-analysis of games (ignore cached profiles, analyzer mode only)",
    )
    return parser


def _load_config(config_path: str) -> Any:
    """Load Jarvis config, optionally from a custom path."""
    try:
        from jarvis.config import load_config

        if config_path:
            from pathlib import Path

            return load_config(Path(config_path))
        return load_config()
    except Exception as exc:
        print(f"[WARN] Could not load Jarvis config: {exc}", file=sys.stderr)
        return None


def _run_single(game_id: str, use_llm: bool, verbose: bool, config: Any) -> int:
    """Run the agent on a single game. Returns exit code."""
    try:
        from jarvis.arc.agent import CognithorArcAgent
    except ImportError as exc:
        print(
            f"[FAIL] Could not import CognithorArcAgent: {exc}\n"
            "Make sure jarvis[arc] dependencies are installed.",
            file=sys.stderr,
        )
        return 1

    llm_call_interval = 10
    max_steps = 1000
    max_resets = 5

    if config is not None:
        arc_cfg = getattr(config, "arc", None)
        if arc_cfg is not None:
            llm_call_interval = arc_cfg.llm_call_interval
            max_steps = arc_cfg.max_steps_per_level
            max_resets = arc_cfg.max_resets_per_level

    if verbose:
        print(f"[INFO] Starting single game: {game_id}")
        print(f"[INFO] LLM planner: {'enabled' if use_llm else 'disabled'}")

    try:
        agent = CognithorArcAgent(
            game_id=game_id,
            use_llm_planner=use_llm,
            llm_call_interval=llm_call_interval,
            max_steps_per_level=max_steps,
            max_resets_per_level=max_resets,
        )
        result = agent.run()
    except Exception as exc:
        print(f"[FAIL] Agent run failed: {exc}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1

    print("[RESULT]")
    print(f"  game_id         : {result.get('game_id', game_id)}")
    print(f"  levels_completed: {result.get('levels_completed', 0)}")
    print(f"  total_steps     : {result.get('total_steps', 0)}")
    print(f"  score           : {result.get('score', 0.0):.4f}")
    return 0


def _run_benchmark(use_llm: bool, verbose: bool, config: Any) -> int:
    """Run all known games sequentially (swarm with max_parallel=1)."""
    return _run_swarm(use_llm=use_llm, parallel=1, verbose=verbose, config=config)


def _run_swarm(use_llm: bool, parallel: int, verbose: bool, config: Any) -> int:
    """Run games in parallel via ArcSwarmOrchestrator. Returns exit code."""
    try:
        from jarvis.arc.swarm import ArcSwarmOrchestrator
    except ImportError as exc:
        print(
            f"[FAIL] Could not import ArcSwarmOrchestrator: {exc}",
            file=sys.stderr,
        )
        return 1

    # Determine game list from config or fall back to empty discovery
    game_ids: list[str] = []
    if config is not None:
        arc_cfg = getattr(config, "arc", None)
        if arc_cfg is not None and arc_cfg.swarm_max_parallel and parallel == 4:
            # Use config parallelism unless overridden via CLI (4 is default)
            parallel = arc_cfg.swarm_max_parallel

    if not game_ids:
        # Try to discover available games from the adapter
        try:
            from jarvis.arc.adapter import ArcEnvironmentAdapter

            game_ids = ArcEnvironmentAdapter.list_games()
        except Exception:
            game_ids = []

    if not game_ids:
        print(
            "[WARN] No game IDs found. Provide a --game argument or ensure ARC SDK is installed.",
            file=sys.stderr,
        )
        return 1

    if verbose:
        print(f"[INFO] Swarm mode: {len(game_ids)} game(s), max_parallel={parallel}")

    import asyncio

    orchestrator = ArcSwarmOrchestrator(
        max_parallel=parallel,
        use_llm=use_llm,
        config=config,
    )

    try:
        asyncio.run(orchestrator.run_swarm(game_ids))
    except Exception as exc:
        print(f"[FAIL] Swarm run failed: {exc}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1

    print(orchestrator.get_summary())
    return 0


def _run_analyzer(game_id: str, reanalyze: bool, verbose: bool, config: Any) -> int:
    """Run GameAnalyzer + PerGameSolver. Returns exit code."""
    try:
        from jarvis.arc.game_analyzer import GameAnalyzer
        from jarvis.arc.per_game_solver import PerGameSolver
    except ImportError as exc:
        print(f"[FAIL] Could not import GameAnalyzer: {exc}", file=sys.stderr)
        return 1

    try:
        import arc_agi

        arcade = arc_agi.Arcade()
    except Exception as exc:
        print(f"[FAIL] Could not create Arcade: {exc}", file=sys.stderr)
        return 1

    # Determine games to analyze
    if game_id:
        game_ids = [game_id]
    else:
        try:
            from jarvis.arc.adapter import ArcEnvironmentAdapter

            game_ids = ArcEnvironmentAdapter.list_games()
        except Exception:
            game_ids = []

    if not game_ids:
        print("[FAIL] No games found.", file=sys.stderr)
        return 1

    if verbose:
        print(f"[INFO] Analyzer mode: {len(game_ids)} game(s), reanalyze={reanalyze}")

    analyzer = GameAnalyzer(arcade=arcade)
    total_levels = 0
    total_score = 0.0

    for gid in game_ids:
        if verbose:
            print(f"\n[INFO] Analyzing {gid}...")

        try:
            profile = analyzer.analyze(gid, force=reanalyze)
            if verbose:
                print(f"  type={profile.game_type}, actions={profile.available_actions}")
                print(f"  click_zones={len(profile.click_zones)}, win={profile.win_condition}")

            solver = PerGameSolver(profile, arcade=arcade)
            result = solver.solve()

            total_levels += result.levels_completed
            total_score += result.score

            print(f"[RESULT] {gid}: {result.levels_completed} levels, {result.total_steps} steps")
            for entry in result.strategy_log:
                status = "WIN" if entry["won"] else "FAIL"
                print(f"  Level {entry['level']}: {status} via {entry['strategy']} ({entry['steps']} steps)")

        except Exception as exc:
            print(f"[FAIL] {gid}: {exc}", file=sys.stderr)
            if verbose:
                import traceback

                traceback.print_exc()

    print(f"\n[SUMMARY] Total levels: {total_levels}, Total score: {total_score:.1f}")

    try:
        scorecard = arcade.get_scorecard()
        print(f"[SCORECARD] {scorecard.score}")
    except Exception:
        pass

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --list-games
    if args.list_games:
        try:
            from jarvis.arc.adapter import ArcEnvironmentAdapter

            games = ArcEnvironmentAdapter.list_games()
        except ImportError as exc:
            print(f"[FAIL] ARC adapter not available: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[FAIL] Could not list games: {exc}", file=sys.stderr)
            return 1

        if games:
            print("Available games:")
            for g in games:
                print(f"  {g}")
        else:
            print("No games found (ARC SDK may not be installed).")
        return 0

    config = _load_config(args.config)
    use_llm = not args.no_llm

    if args.mode == "single":
        if not args.game:
            print("[FAIL] --game is required for single mode.", file=sys.stderr)
            parser.print_help()
            return 1
        return _run_single(
            game_id=args.game,
            use_llm=use_llm,
            verbose=args.verbose,
            config=config,
        )

    if args.mode == "benchmark":
        return _run_benchmark(use_llm=use_llm, verbose=args.verbose, config=config)

    if args.mode == "swarm":
        return _run_swarm(
            use_llm=use_llm,
            parallel=args.parallel,
            verbose=args.verbose,
            config=config,
        )

    if args.mode == "analyzer":
        return _run_analyzer(
            game_id=args.game,
            reanalyze=args.reanalyze,
            verbose=args.verbose,
            config=config,
        )

    print(f"[FAIL] Unknown mode: {args.mode}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
