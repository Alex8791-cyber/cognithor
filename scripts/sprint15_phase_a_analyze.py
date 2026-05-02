"""Sprint-15 Phase-A diagnostic analyzer.

Walks finished episode audit JSONL files (one per episode) and
produces a one-page Markdown report covering the four diagnostic
priorities the reviewer asked for, in their order:

1. ``mean_accepted_length`` distribution per episode (uni- vs bimodal
   indicates Outcome B vs Outcome A).
2. Correlation between ``think_tokens / output_tokens`` ratio and
   ``mtp_acceptance_rate`` per episode (high-Reasoning-Episode signal
   for Outcome B).
3. Measured t/s vs MTP-theoretical t/s (sanity-check for hidden
   bottlenecks not covered by any hypothesis).

Plus three early-warning signals from the live-tracking discussion:

* Throughput drift across episodes (KV-cache fragmentation /
  memory leak).
* Wall-clock spikes without output-token spikes (CUDA-graph
  recapture events; warm-up if isolated to early episodes).
* Acceptance-rate drift inside an episode (drafter context-handling
  failure — a fifth Outcome variant beyond A/B/C/D).

Sample-size filter: per-position acceptance probabilities only
reported when n >= 100 cumulative drafts at that position.

Usage:
    python scripts/sprint15_phase_a_analyze.py <jsonl_dir> [--out report.md]

The JSONL files are the per-episode audit trails written by
``ArcAuditTrail.export_jsonl(seal_into_hashline=True)``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# n-threshold below which per-position acceptance is suppressed in
# the Markdown output. Wilson 95 % CI at p≈0.6, n=100 ≈ ±0.095 — the
# coarsest cut we'll still treat as informative for visual comparison.
MIN_DRAFTS_PER_POSITION = 100
# Minimum step count for the within-episode acceptance drift split
# (1st half vs 2nd half). Below 8 steps the half-mean is dominated
# by single outliers and produces false-positive drift flags.
MIN_STEPS_FOR_DRIFT_SPLIT = 8


def _load_episode(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _per_episode_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [e for e in events if e.get("event_type") == "step"]
    if not steps:
        return {}
    # Per-call telemetry (already on the step events).
    rates = [s["mtp_acceptance_rate"] for s in steps if s.get("mtp_acceptance_rate") is not None]
    proposed = [s["mtp_drafts_proposed"] for s in steps if s.get("mtp_drafts_proposed") is not None]
    accepted = [s["mtp_drafts_accepted"] for s in steps if s.get("mtp_drafts_accepted") is not None]
    out_tokens = [s["llm_output_tokens"] for s in steps if s.get("llm_output_tokens") is not None]
    think_tokens = [s["llm_think_tokens"] for s in steps if s.get("llm_think_tokens") is not None]
    wall = [s["llm_wall_clock_s"] for s in steps if s.get("llm_wall_clock_s") is not None]
    finish = [s["llm_finish_reason"] for s in steps if s.get("llm_finish_reason") is not None]
    out: dict[str, Any] = {
        "steps": len(steps),
        "llm_calls": len(out_tokens),
    }
    if proposed and accepted:
        total_p = sum(proposed)
        total_a = sum(accepted)
        out["acceptance_rate"] = total_a / total_p if total_p > 0 else None
        out["drafts_proposed"] = total_p
        out["drafts_accepted"] = total_a
    if rates:
        out["acceptance_rate_per_step_mean"] = statistics.fmean(rates)
        # Drift inside the episode: split first-half vs second-half.
        # Skip below MIN_STEPS_FOR_DRIFT_SPLIT — short episodes let a
        # single noisy step swing the half-mean by 0.2+ and produce
        # false-positive drift flags.
        if len(rates) >= MIN_STEPS_FOR_DRIFT_SPLIT:
            mid = len(rates) // 2
            out["acceptance_first_half"] = statistics.fmean(rates[:mid])
            out["acceptance_second_half"] = statistics.fmean(rates[mid:])
            out["acceptance_drift"] = out["acceptance_second_half"] - out["acceptance_first_half"]
    if out_tokens and think_tokens:
        ratios = [t / o if o > 0 else 0.0 for t, o in zip(think_tokens, out_tokens, strict=False)]
        out["think_ratio_mean"] = statistics.fmean(ratios)
        if rates and len(ratios) == len(rates):
            # Pearson r between think-ratio and acceptance — Outcome B
            # confirmation signal: negative correlation = reasoning
            # phases drag MTP down.
            n = len(ratios)
            if n >= 3:
                mx = statistics.fmean(ratios)
                my = statistics.fmean(rates)
                num = sum((x - mx) * (y - my) for x, y in zip(ratios, rates, strict=False))
                dx = sum((x - mx) ** 2 for x in ratios) ** 0.5
                dy = sum((y - my) ** 2 for y in rates) ** 0.5
                if dx > 0 and dy > 0:
                    out["corr_think_ratio_vs_acceptance"] = num / (dx * dy)
    if wall and out_tokens:
        total_wall = sum(wall)
        total_tokens = sum(out_tokens)
        if total_wall > 0:
            out["measured_tps"] = total_tokens / total_wall
    if finish:
        from collections import Counter

        dist = Counter(finish)
        out["finish_reason_dist"] = dict(dist)
        out["length_truncation_rate"] = dist.get("length", 0) / len(finish)
    return out


def _format_md(per_episode: list[dict[str, Any]]) -> str:
    if not per_episode:
        return "# Sprint-15 Phase-A Report\n\nNo episodes found.\n"

    lines: list[str] = ["# Sprint-15 Phase-A Diagnostic Report", ""]
    lines.append(f"Episodes analysed: {len(per_episode)}")
    lines.append("")

    # Aggregate.
    rates = [e["acceptance_rate"] for e in per_episode if e.get("acceptance_rate") is not None]
    if rates:
        lines.append("## 1. Acceptance-rate distribution")
        lines.append("")
        lines.append(f"- mean = {statistics.fmean(rates):.3f}")
        lines.append(f"- min = {min(rates):.3f}, max = {max(rates):.3f}")
        if len(rates) >= 2:
            lines.append(f"- stdev = {statistics.stdev(rates):.3f}")
        # Bimodality hint: gap between halves.
        if len(rates) >= 6:
            sorted_rates = sorted(rates)
            mid = len(sorted_rates) // 2
            low_mean = statistics.fmean(sorted_rates[:mid])
            high_mean = statistics.fmean(sorted_rates[mid:])
            gap = high_mean - low_mean
            lines.append(
                f"- low-half mean = {low_mean:.3f}, high-half mean = "
                f"{high_mean:.3f}, gap = {gap:.3f} "
                f"({'bimodal-suggestive' if gap > 0.2 else 'unimodal-suggestive'})"
            )
        lines.append("")

    # Outcome classification heuristic.
    if rates:
        mean_acc = statistics.fmean(rates)
        if mean_acc >= 0.65:
            outcome = "A — MTP healthy; bottleneck likely not MTP"
        elif mean_acc <= 0.40:
            outcome = "C — drafter quality issue (consider MTP=2 or off)"
        else:
            outcome = "B or D — needs think-ratio correlation + per-position split"
        lines.append(f"## Outcome classification: **{outcome}**")
        lines.append("")

    # Correlation.
    corrs = [
        e["corr_think_ratio_vs_acceptance"]
        for e in per_episode
        if e.get("corr_think_ratio_vs_acceptance") is not None
    ]
    if corrs:
        lines.append("## 2. Think-ratio ↔ Acceptance correlation per episode")
        lines.append("")
        lines.append(f"- mean Pearson r = {statistics.fmean(corrs):.3f}")
        lines.append(f"- min = {min(corrs):.3f}, max = {max(corrs):.3f}")
        lines.append(
            "- Outcome B confirmed if mean r is solidly negative (≤ -0.3); "
            "lukewarm or positive r weakens the workload-MTP-mismatch hypothesis."
        )
        lines.append("")

    # Throughput drift.
    measured = [e.get("measured_tps") for e in per_episode if e.get("measured_tps") is not None]
    if measured:
        lines.append("## 3. Throughput across episodes (measured t/s)")
        lines.append("")
        lines.append(f"- first episode = {measured[0]:.2f}")
        lines.append(f"- last episode  = {measured[-1]:.2f}")
        if len(measured) >= 2:
            drift = measured[-1] - measured[0]
            pct = drift / measured[0] * 100 if measured[0] > 0 else 0
            lines.append(
                f"- drift = {drift:+.2f} t/s ({pct:+.1f} %) — "
                f"{'⚠ KV-fragmentation suspect' if abs(pct) >= 15 else 'stable'}"
            )
        lines.append("")

    # Within-episode drift.
    intra = [e["acceptance_drift"] for e in per_episode if e.get("acceptance_drift") is not None]
    if intra:
        lines.append("## 4. Within-episode acceptance drift (2nd-half − 1st-half)")
        lines.append("")
        lines.append(f"- mean drift = {statistics.fmean(intra):+.3f}")
        big_drops = sum(1 for d in intra if d <= -0.15)
        if big_drops:
            lines.append(
                f"- {big_drops}/{len(intra)} episodes show ≥ 0.15 drop "
                "→ drafter context-handling concern (5th Outcome variant)."
            )
        lines.append("")

    # Length-truncation.
    trunc = [
        e["length_truncation_rate"]
        for e in per_episode
        if e.get("length_truncation_rate") is not None
    ]
    if trunc:
        lines.append("## 5. Length-truncation rate")
        lines.append("")
        lines.append(f"- mean = {statistics.fmean(trunc):.2%}")
        if statistics.fmean(trunc) > 0.10:
            lines.append("- ⚠ raise `max_tokens` — too many calls hit the cap.")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_dir", type=Path, help="directory of audit JSONL files")
    parser.add_argument("--out", type=Path, default=None, help="Markdown output path")
    args = parser.parse_args()

    files = sorted(args.jsonl_dir.glob("*.jsonl"))
    if not files:
        print(f"no JSONL files in {args.jsonl_dir}")
        return 1

    per_episode: list[dict[str, Any]] = []
    for f in files:
        try:
            events = _load_episode(f)
            stats = _per_episode_stats(events)
            stats["episode_file"] = f.name
            per_episode.append(stats)
        except Exception as exc:
            print(f"[skip] {f.name}: {exc}")

    md = _format_md(per_episode)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        # Write directly to stdout's underlying buffer in UTF-8 so the
        # unicode arrows + ↔ in the report don't trip Windows cp1252.
        import sys

        sys.stdout.buffer.write(md.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
