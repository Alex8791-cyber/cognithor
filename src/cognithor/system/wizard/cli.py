"""Layer 5 — CLI First-Run Wizard.

Pipeline:
  1. Run Detection (≤12s budget)
  2. Sanity-validate
  3. Map to Capabilities
  4. Load Manifest
  5. Ask user for Objective
  6. Solve (Pareto)
  7. Show solutions, ask for choice
  8. Confirm → call Apply-Engine (L6)

The wizard is **idempotent** and **resumable** — cancelling at any
prompt leaves no state behind. The `.cognithor_initialized` marker is
ONLY written after a successful `apply()` in L6.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from cognithor.system.capabilities import Capabilities, map_to_capabilities
from cognithor.system.detector import SystemDetector, SystemProfile
from cognithor.system.manifest_loader import (
    ManifestLoader,
    ManifestRecalledError,
    ManifestSource,
)
from cognithor.system.manifest_models import Manifest
from cognithor.system.sanity import SanityWarning, validate
from cognithor.system.solver import (
    OBJECTIVE_PRESETS,
    Solution,
    UserObjective,
    solve,
)
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["WizardResult", "render_profile_summary", "run_wizard"]


# ── ANSI helpers ────────────────────────────────────────────────────────────

_C_RESET = "\033[0m"
_C_BOLD = "\033[1m"
_C_DIM = "\033[2m"
_C_OK = "\033[32m"
_C_WARN = "\033[33m"
_C_FAIL = "\033[31m"
_C_ACCENT = "\033[36m"
_C_BLOCK = "\033[35m"


def _ansi_enabled() -> bool:
    if not sys.stdout.isatty():
        return False
    if "NO_COLOR" in __import__("os").environ:
        return False
    return True


def _utf8_safe() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


def _c(s: str, color: str) -> str:
    if not _ansi_enabled():
        return s
    return f"{color}{s}{_C_RESET}"


# Glyph fallbacks for non-UTF-8 terminals (Win-cp1252)
_GLYPH = {
    "ok": "✓" if _utf8_safe() else "+",
    "warn": "⚠" if _utf8_safe() else "!",
    "fail": "✗" if _utf8_safe() else "x",
    "star": "★" if _utf8_safe() else "*",
    "le": "≤" if _utf8_safe() else "<=",
    "bullet": "·" if _utf8_safe() else ".",
    "block": "█" if _utf8_safe() else "#",
    "dot": "·" if _utf8_safe() else "-",
    "tl": "┌" if _utf8_safe() else "+",
    "tr": "┐" if _utf8_safe() else "+",
    "bl": "└" if _utf8_safe() else "+",
    "br": "┘" if _utf8_safe() else "+",
    "h": "─" if _utf8_safe() else "-",
    "ellipsis": "…" if _utf8_safe() else "...",
}


@dataclass(frozen=True)
class WizardResult:
    confirmed: bool
    selected_solution: Solution | None
    profile: SystemProfile
    capabilities: Capabilities
    manifest_source: ManifestSource
    objective: UserObjective
    sanity_warnings: tuple[SanityWarning, ...]


# ── Renderers ───────────────────────────────────────────────────────────────


def render_profile_summary(profile: SystemProfile) -> str:
    lines: list[str] = []
    h = _GLYPH["h"]
    title = "Hardware-Profil"
    box_top = f"{_GLYPH['tl']}{h * 2} {title} {h * (60 - len(title) - 4)}{_GLYPH['tr']}"
    box_bot = f"{_GLYPH['bl']}{h * 60}{_GLYPH['br']}"
    lines.append(_c(box_top, _C_DIM))
    for key in (
        "os",
        "cpu",
        "ram",
        "gpu",
        "docker",
        "wsl2",
        "container",
        "disk",
        "network",
        "vllm",
        "huggingface",
    ):
        if key not in profile.results:
            continue
        r = profile.results[key]
        marker = (
            _c(_GLYPH["ok"], _C_OK)
            if r.status == "ok"
            else _c(_GLYPH["warn"], _C_WARN)
            if r.status == "warn"
            else _c(_GLYPH["fail"], _C_FAIL)
        )
        lines.append(f" {marker} {key:<13} {r.value}")
    lines.append(_c(box_bot, _C_DIM))
    return "\n".join(lines)


def render_capabilities_summary(caps: Capabilities) -> str:
    flags = [
        ("NVFP4", caps.can_run_nvfp4),
        ("FP8", caps.can_run_fp8_marlin),
        ("GGUF-CUDA", caps.can_run_gguf_cuda),
        ("GGUF-Metal", caps.can_run_gguf_metal),
        ("vLLM-Container", caps.can_run_vllm_container),
        ("vLLM-Inproc", caps.can_run_vllm_inprocess),
        ("Ollama", caps.can_run_ollama_native),
        ("Multi-GPU", caps.has_multi_gpu_homogeneous),
        ("Internet", caps.has_internet),
    ]
    parts = [f"{_c('✓', _C_OK)} {n}" if v else f"{_c('✗', _C_FAIL)} {n}" for n, v in flags]
    return (
        f"Capabilities: {' · '.join(parts)}\n"
        f"  vram_class={caps.vram_class}  ram_class={caps.ram_class}  disk_class={caps.disk_class}"
    )


def render_solution_card(
    idx: int, sol: Solution, manifest: Manifest, *, recommended: bool = False
) -> str:
    tier = next((t for t in manifest.tiers if t.id == sol.tier_id), None)
    if tier is None:
        return f"  {idx}. {sol.tier_id} (tier-data missing)"

    bd = sol.score_breakdown
    star = _c(_GLYPH["star"], _C_ACCENT) if recommended else " "
    head_color = _C_BOLD if sol.is_immediately_runnable else _C_BLOCK
    head = _c(f"{star} #{idx} · {tier.display_name}", head_color)
    bar = (
        f"   Q {_progress(bd.get('quality', 0))}  S {_progress(bd.get('speed', 0))}  "
        f"C {_progress(bd.get('cost', 0))}  P {_progress(bd.get('privacy', 0))}"
    )
    cost_str = (
        "€0/Monat lokal"
        if sol.estimated_cost_eur_per_month == 0
        else f"~€{sol.estimated_cost_eur_per_month:.0f}/Monat (Cloud)"
    )
    setup = (
        f"Setup ~{sol.estimated_setup_minutes}min · "
        f"Disk {sol.estimated_disk_gb:.0f} GB · {cost_str}"
    )
    rationale = tier.rationale_de or tier.rationale_en
    rationale_short = rationale.replace("\n", " ").strip()[:90]

    out: list[str] = [head, bar, _c(f"   {setup}", _C_DIM), _c(f"   {rationale_short}", _C_DIM)]

    out.append(
        "   Modelle: "
        f"planner={tier.model_set.planner} · "
        f"executor={tier.model_set.executor} · "
        f"coder={tier.model_set.coder}"
    )
    if sol.blockers:
        out.append(_c(f"   ⚠ blocked by: {', '.join(sol.blockers)}", _C_WARN))
    if sol.warnings:
        out.append(_c(f"   ⚠ warnings: {', '.join(sol.warnings)}", _C_WARN))
    return "\n".join(out)


def _progress(v: float, *, width: int = 10) -> str:
    filled = round(v * width)
    return (
        _c(_GLYPH["block"] * filled + _GLYPH["dot"] * (width - filled), _C_ACCENT)
        + f" {int(v * 100)}"
    )


# ── Prompt helpers ──────────────────────────────────────────────────────────


def _prompt(msg: str, *, default: str | None = None, valid: tuple[str, ...] = ()) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            ans = input(f"{msg}{suffix} > ").strip()
        except (EOFError, KeyboardInterrupt):
            raise WizardCancelled from None
        if not ans and default is not None:
            return default
        if not valid or ans in valid:
            return ans
        print(_c(f"Bitte einen der Werte eingeben: {valid}", _C_WARN))


class WizardCancelled(Exception):
    pass


# ── Main wizard ─────────────────────────────────────────────────────────────


def run_wizard(
    *,
    interactive: bool | None = None,
    pre_selected_objective: UserObjective | None = None,
    detector: SystemDetector | None = None,
    loader: ManifestLoader | None = None,
) -> WizardResult:
    """Run the wizard. Returns WizardResult — does NOT write config.

    `interactive=False` returns the recommendation without prompting (for
    headless / CI use). `interactive=None` auto-detects TTY.
    """
    if interactive is None:
        interactive = sys.stdout.isatty() and sys.stdin.isatty()

    print(_c("Cognithor First-Run — Hardware-Aware Setup", _C_BOLD))
    print(_c(f"Detection laeuft ({_GLYPH['le']}12s){_GLYPH['ellipsis']}", _C_DIM))

    detector = detector or SystemDetector()
    profile = detector.run_full_scan()
    print(render_profile_summary(profile))

    sanity = validate(profile)
    for w in sanity:
        sym = _c(_GLYPH["warn"], _C_WARN) if w.severity != "error" else _c(_GLYPH["fail"], _C_FAIL)
        print(f"  {sym} {w.message}")

    caps = map_to_capabilities(profile)
    print()
    print(render_capabilities_summary(caps))

    loader = loader or ManifestLoader()
    try:
        manifest, source = loader.load(prefer_online=True)
    except ManifestRecalledError as exc:
        print(_c(f"\n✗ {exc}", _C_FAIL))
        raise
    print()
    print(
        _c(
            f"Manifest: {source.manifest_version} (origin={source.origin}, "
            f"signed={source.signature_verified})",
            _C_DIM,
        )
    )
    pricing = loader.load_pricing()

    # Objective selection
    if pre_selected_objective:
        objective = pre_selected_objective
        objective_name = "custom"
    elif interactive:
        print()
        print(_c("Was ist dir wichtig?", _C_BOLD))
        print("  [1] Ausgewogen (Standard)")
        print("  [2] Beste Qualität")
        print("  [3] Schnellste Antworten")
        print("  [4] Maximale Privacy (offline-only)")
        print("  [5] Geringste Kosten")
        choice = _prompt("Wahl", default="1", valid=("1", "2", "3", "4", "5"))
        objective_name = {
            "1": "balanced",
            "2": "quality",
            "3": "speed",
            "4": "privacy",
            "5": "cost",
        }[choice]
        objective = OBJECTIVE_PRESETS[objective_name]
    else:
        objective_name = "balanced"
        objective = OBJECTIVE_PRESETS["balanced"]

    solutions = solve(manifest, caps, objective, pricing=pricing, max_solutions=4)
    if not solutions:
        print(_c("\n✗ Keine Lösung gefunden — Hardware unter allen Mindestschwellen.", _C_FAIL))
        raise SystemExit(2)

    print()
    print(_c(f"Pareto-optimal für '{objective_name}':", _C_BOLD))
    print()
    for idx, sol in enumerate(solutions, start=1):
        print(render_solution_card(idx, sol, manifest, recommended=(idx == 1)))
        print()

    if not interactive:
        # Non-interactive: write recommendation.json, do NOT apply
        return WizardResult(
            confirmed=False,
            selected_solution=solutions[0],
            profile=profile,
            capabilities=caps,
            manifest_source=source,
            objective=objective,
            sanity_warnings=sanity,
        )

    valid = tuple(str(i) for i in range(1, len(solutions) + 1))
    valid_with_extras = (*valid, "m", "a")
    print(f"  [1-{len(solutions)}] Tier wählen   [m] Manuell konfigurieren   [a] Abbrechen")
    choice = _prompt("Wahl", default="1", valid=valid_with_extras)
    if choice == "a":
        raise WizardCancelled
    if choice == "m":
        print(
            _c(
                "Manuell-Modus — Cognithor startet mit Default-Config. "
                "Editiere ~/.cognithor/config.yaml",
                _C_DIM,
            )
        )
        return WizardResult(
            confirmed=False,
            selected_solution=None,
            profile=profile,
            capabilities=caps,
            manifest_source=source,
            objective=objective,
            sanity_warnings=sanity,
        )

    selected = solutions[int(choice) - 1]
    if selected.blockers:
        print(_c(f"\n⚠ Diese Wahl hat Blocker: {', '.join(selected.blockers)}", _C_WARN))
        print(
            _c(
                "  Cognithor kann starten, aber dieser Tier ist erst nach "
                "Beheben der Blocker nutzbar.",
                _C_WARN,
            )
        )
        ack = _prompt("Trotzdem auswählen? (y/n)", default="n", valid=("y", "n"))
        if ack != "y":
            raise WizardCancelled

    return WizardResult(
        confirmed=True,
        selected_solution=selected,
        profile=profile,
        capabilities=caps,
        manifest_source=source,
        objective=objective,
        sanity_warnings=sanity,
    )
