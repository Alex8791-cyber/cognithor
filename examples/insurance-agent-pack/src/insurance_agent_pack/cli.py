"""CLI for the insurance-agent-pack reference example."""

from __future__ import annotations

import argparse
import asyncio
import sys

from insurance_agent_pack.crew import build_team


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="insurance-agent-pack",
        description="Reference Cognithor pack: §34d-NEUTRAL DACH insurance pre-advisory.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the pre-advisory crew.")
    run.add_argument("--interview", action="store_true", help="Interactive interview mode.")
    run.add_argument("--model", default="ollama/qwen3:8b")
    return p


def _interview_inputs() -> dict[str, str]:
    print("=== Versicherungs-Pre-Beratung ===")
    print("Alle Eingaben sind synthetisch. Diese Software ist keine §34d-Beratung.")
    name = input("Vorname (frei wählbar, kein Klarname nötig): ").strip()
    age = input("Alter: ").strip()
    role = input("Berufsstatus (GGF/selbständig/angestellt/freiberufler): ").strip()
    existing = input("Bestehende Policen (kurz, kommasepariert oder 'keine'): ").strip()
    return {
        "name": name or "Anon",
        "age": age,
        "berufsstatus": role,
        "bestehende_policen": existing,
    }


def _cmd_run(args: argparse.Namespace) -> int:
    crew = build_team(model=args.model)
    if args.interview:
        inputs = _interview_inputs()
    else:
        inputs = {}
    output = asyncio.run(crew.kickoff_async(inputs))
    print()
    print(getattr(output, "raw", "") or "")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
