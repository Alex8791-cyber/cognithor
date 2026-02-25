#!/usr/bin/env python3
"""
Record demo.py as an asciicast v2 file (.cast) on Windows.

Produces a file compatible with asciinema-player, agg (GIF), svg-term, etc.

Usage:  python record_demo.py           -> demo.cast
        python record_demo.py out.cast  -> out.cast
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

COLS = 120
ROWS = 42
TITLE = "Cognithor \u00b7 Agent OS \u2014 Demo"
TARGET_DURATION = 50.0  # seconds for the final recording


def main() -> None:
    outfile = sys.argv[1] if len(sys.argv) > 1 else "demo.cast"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["COLUMNS"] = str(COLS)
    env["LINES"] = str(ROWS)
    env["FORCE_COLOR"] = "1"
    env["TERM"] = "xterm-256color"

    header = {
        "version": 2,
        "width": COLS,
        "height": ROWS,
        "timestamp": int(time.time()),
        "title": TITLE,
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
    }

    print(f"Recording demo.py --fast -> {outfile}")
    print(f"Terminal size: {COLS}x{ROWS}")

    proc = subprocess.Popen(
        [sys.executable, "demo.py", "--fast"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    start = time.monotonic()
    chunks: list[tuple[float, str]] = []

    assert proc.stdout is not None
    while True:
        data = proc.stdout.read(128)
        if not data:
            break
        elapsed = time.monotonic() - start
        text = data.decode("utf-8", errors="replace")
        chunks.append((elapsed, text))

    proc.wait()

    if not chunks:
        print("No output captured!")
        return

    # ── Post-process timing ────────────────────────────────────────
    # The --fast run completes in ~1-2s. We redistribute timestamps
    # to create a cinematic ~50s playback with natural pacing:
    # - Small delays between chunks (simulates rendering)
    # - Larger pauses at scene boundaries (detected by clear-screen or rules)
    events: list[tuple[float, str]] = []
    cursor = 0.5  # start at 0.5s

    for i, (_raw_ts, text) in enumerate(chunks):
        events.append((round(cursor, 4), text))

        # Detect scene boundaries for longer pauses
        is_scene_break = (
            "\x1b[2J" in text        # clear screen (boot)
            or "\u2550" * 4 in text   # ════ rule characters
            or "System Online" in text
            or "Live Session" in text
            or "PGE Trinity" in text
            or "Security Demonstration" in text
            or "Multi-Channel" in text
            or "Reflection" in text
            or "System Overview" in text
        )

        if is_scene_break:
            cursor += 1.5  # dramatic pause between scenes
        elif "\n" in text and len(text) > 100:
            cursor += 0.4  # medium pause for large output blocks
        else:
            cursor += 0.15  # fast tick for streaming/typing effects

    # Scale to target duration if needed
    actual_duration = events[-1][0]
    if actual_duration > 0:
        scale = TARGET_DURATION / actual_duration
        events = [(round(ts * scale, 4), text) for ts, text in events]

    # Write cast file
    with open(outfile, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header) + "\n")
        for ts, text in events:
            f.write(json.dumps([ts, "o", text]) + "\n")

    n_events = len(events)
    final_ts = events[-1][0]
    esc_count = sum(text.count("\x1b") for _, text in events)
    fsize = os.path.getsize(outfile)

    print(f"Recorded {n_events} events, ~{final_ts:.0f}s duration")
    print(f"ANSI color sequences: {esc_count}")
    print(f"File size: {fsize // 1024}KB")
    print(f"Output: {outfile}")
    print()
    print("Next steps:")
    print(f"  1. Upload:  asciinema upload {outfile}")
    print(f"  2. GIF:     agg {outfile} demo.gif --cols {COLS} --rows {ROWS}")
    print(f"  3. SVG:     svg-term --in {outfile} --out demo.svg")
    print(f"  4. Player:  https://github.com/asciinema/asciinema-player")


if __name__ == "__main__":
    main()
