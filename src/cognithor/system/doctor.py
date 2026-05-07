"""`cognithor doctor` — Hardware-aware diagnostic + (re)configuration CLI.

Subcommands:
  cognithor doctor                       — Detection + capabilities + recommendations (no write)
  cognithor doctor --reconfigure         — Run wizard, confirm, write config.yaml + sidecar
  cognithor doctor --apply-recommendation — Non-interactive: apply top recommendation
  cognithor doctor --refresh-manifest    — Force online refresh of manifest cache
  cognithor doctor --export-profile PATH — Anonymized profile dump for bug reports
  cognithor doctor --rollback            — Restore most recent config backup

Wired into the CLI entry point via `cognithor.cli.doctor_cmd:main`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cognithor.system.apply_engine import apply_solution, list_backups, rollback_last
from cognithor.system.capabilities import map_to_capabilities
from cognithor.system.detector import SystemDetector
from cognithor.system.manifest_loader import ManifestLoader, ManifestRecalledError
from cognithor.system.sanity import validate
from cognithor.system.solver import OBJECTIVE_PRESETS, solve
from cognithor.system.wizard.cli import (
    WizardCancelled,
    render_capabilities_summary,
    render_profile_summary,
    render_solution_card,
    run_wizard,
)
from cognithor.utils.logging import get_logger

log = get_logger(__name__)


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _diagnostic_only() -> int:
    """`cognithor doctor` — pure diagnostic, no writes."""
    detector = SystemDetector()
    profile = detector.run_full_scan()
    print(render_profile_summary(profile))

    warnings = validate(profile)
    if warnings:
        _print_section("Sanity-Warnings")
        for w in warnings:
            sym = "[!]" if w.severity != "error" else "[X]"
            print(f"  {sym} {w.rule_id}: {w.message}")

    caps = map_to_capabilities(profile)
    _print_section("Capabilities")
    print(render_capabilities_summary(caps))

    loader = ManifestLoader()
    try:
        manifest, source = loader.load(prefer_online=False)
    except ManifestRecalledError as exc:
        print(f"\n[ERROR] {exc}")
        return 2
    pricing = loader.load_pricing()
    print(
        f"\nManifest: {source.manifest_version} "
        f"(origin={source.origin}, signed={source.signature_verified})"
    )

    _print_section("Top Recommendations (preset=balanced)")
    sols = solve(manifest, caps, OBJECTIVE_PRESETS["balanced"], pricing=pricing, max_solutions=4)
    for idx, sol in enumerate(sols, start=1):
        print()
        print(render_solution_card(idx, sol, manifest, recommended=(idx == 1)))

    _print_section("Next Steps")
    print(
        "  cognithor doctor --reconfigure          interaktiver Wizard, schreibt config\n"
        "  cognithor doctor --apply-recommendation non-interactiv top-Empfehlung\n"
        "  cognithor doctor --refresh-manifest     online-refresh des Tier-Manifests\n"
        "  cognithor doctor --rollback             letztes Backup wiederherstellen"
    )
    return 0


def _reconfigure(*, interactive: bool = True) -> int:
    """Run wizard end-to-end with confirmation + apply."""
    try:
        result = run_wizard(interactive=interactive)
    except WizardCancelled:
        print("Abgebrochen — keine Änderung.")
        return 1

    if not result.confirmed or result.selected_solution is None:
        print("Keine Tier-Auswahl getroffen — keine Änderung.")
        return 0

    try:
        apply = apply_solution(
            solution=result.selected_solution,
            manifest=ManifestLoader().load()[0],
            capabilities=result.capabilities,
            objective=result.objective,
            user_confirmed=True,
        )
    except Exception as exc:
        print(f"[ERROR] Apply fehlgeschlagen: {exc}")
        return 2

    print(f"\n[OK] Tier '{apply.selected_tier_id}' angewendet.")
    print(f"     config.yaml: {apply.config_path}")
    if apply.backup_path:
        print(f"     backup:      {apply.backup_path}")
    print(f"     sidecar:     {apply.config_path.parent}/.hardware_aware.json")
    print(f"     marker:      {apply.initialized_marker_path}")
    return 0


def _apply_recommendation_noninteractive() -> int:
    """Run wizard, automatically pick top-1 runnable solution. No prompts."""
    detector = SystemDetector()
    profile = detector.run_full_scan()
    caps = map_to_capabilities(profile)
    loader = ManifestLoader()
    manifest, _ = loader.load()
    pricing = loader.load_pricing()
    sols = solve(manifest, caps, OBJECTIVE_PRESETS["balanced"], pricing=pricing, max_solutions=5)

    runnable = [s for s in sols if s.is_immediately_runnable]
    if not runnable:
        print("[ERROR] Keine sofort-startbare Lösung — Hardware unter allen Schwellen.")
        # Write recommendation.json for later
        rec_path = Path.home() / ".cognithor" / "recommendation.json"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(
            json.dumps(
                [{"tier_id": s.tier_id, "blockers": list(s.blockers)} for s in sols], indent=2
            ),
            encoding="utf-8",
        )
        print(f"   Empfehlungen geschrieben nach: {rec_path}")
        return 3

    chosen = runnable[0]
    print(f"Applying top recommendation: {chosen.tier_id} (score={chosen.score:.3f})")
    try:
        result = apply_solution(
            solution=chosen,
            manifest=manifest,
            capabilities=caps,
            objective=OBJECTIVE_PRESETS["balanced"],
            user_confirmed=True,
        )
    except Exception as exc:
        print(f"[ERROR] Apply fehlgeschlagen: {exc}")
        return 2
    print(f"[OK] Tier '{result.selected_tier_id}' angewendet.")
    return 0


def _refresh_manifest() -> int:
    loader = ManifestLoader()
    try:
        manifest, source = loader.load(prefer_online=True, force_refresh=True)
    except ManifestRecalledError as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(f"[OK] Manifest refreshed: {source.manifest_version} (origin={source.origin})")
    print(f"     {len(manifest.tiers)} tiers, {len(manifest.models)} models")
    print(f"     signed: {source.signature_verified}")
    return 0


def _export_profile(path: str) -> int:
    detector = SystemDetector()
    profile = detector.run_full_scan()
    Path(path).write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    print(f"[OK] Profile exported to: {path}")
    return 0


def _rollback() -> int:
    backups = list_backups()
    if not backups:
        print("[ERROR] No backups available.")
        return 2
    restored = rollback_last()
    print(f"[OK] Restored from: {restored}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cognithor doctor",
        description="Hardware-aware diagnostic and (re)configuration tool.",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Interactive wizard with confirmation + write",
    )
    parser.add_argument(
        "--apply-recommendation",
        action="store_true",
        help="Non-interactive: apply top runnable recommendation",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Force online refresh of the manifest cache",
    )
    parser.add_argument(
        "--export-profile",
        metavar="PATH",
        help="Write anonymized hardware profile JSON for bug reports",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Restore the most recent config.yaml backup",
    )
    args = parser.parse_args(argv)

    # Mutually exclusive — pick first match
    if args.rollback:
        return _rollback()
    if args.export_profile:
        return _export_profile(args.export_profile)
    if args.refresh_manifest:
        return _refresh_manifest()
    if args.apply_recommendation:
        return _apply_recommendation_noninteractive()
    if args.reconfigure:
        return _reconfigure(interactive=True)
    return _diagnostic_only()


if __name__ == "__main__":
    sys.exit(main())
