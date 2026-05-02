"""Sprint-16 prep: dump exactly what the LLM did in run #7.

Read the per-step audit JSONL, print one line per step (action +
state delta + token telemetry) so the actual failure mode is
visible at a glance. Sprint-16's intervention should target what
this dump surfaces, not what we'd guess.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sprint16_failure_mode_dump.py <jsonl>")
        return 2
    path = Path(sys.argv[1])
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [e for e in events if e["event_type"] == "step"]
    print(f"=== {path.name} — {len(steps)} step events ===\n")

    header = f"{'idx':>3} {'lvl':>3} {'action':<12} {'state':<13} {'pixΔ':>6} {'in':>5} {'out':>5} {'think':>5} {'wall':>5}"
    print(header)
    print("-" * len(header))
    for i, s in enumerate(steps):
        action = s["action"]
        state = s["game_state"]
        pixd = s.get("pixels_changed", 0) or 0
        in_t = s.get("llm_input_tokens", 0) or 0
        out_t = s.get("llm_output_tokens", 0) or 0
        think = s.get("llm_think_tokens", 0) or 0
        wall = s.get("llm_wall_clock_s", 0.0) or 0.0
        print(f"{i:>3} {s['level']:>3} {action:<12} {state:<13} {pixd:>6} {in_t:>5} {out_t:>5} {think:>5} {wall:>5.1f}")

    print()
    acts = Counter(s["action"] for s in steps)
    print("=== action distribution ===")
    for a, n in acts.most_common():
        print(f"  {a:<12} {n:>3} ({n / len(steps) * 100:>4.0f}%)")

    no_op = sum(1 for s in steps if (s.get("pixels_changed") or 0) == 0)
    print(f"\n=== {len(steps) - no_op}/{len(steps)} moved pixels; {no_op} no-ops ({no_op / len(steps) * 100:.0f}%) ===")

    # Repetition probe: longest run of the same action
    cur, longest = 1, 1
    for a, b in zip(steps, steps[1:], strict=False):
        if a["action"] == b["action"]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    print(f"=== longest same-action streak: {longest} ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
