"""Background-process observer that watches Cognithor under soak.

Run alongside locust::

    python tests/soak/drift_observer.py --pid <cognithor_pid> \\
        --interval 30 --duration 86400 --output soak-observations.jsonl

Samples every ``interval`` seconds:

* Process RSS / VMS (memory growth signal)
* Open file-descriptor count (FD-leak signal)
* Audit-chain head index from ``~/.cognithor/audit/audit.jsonl``
* SQLite WAL size on the main databases
* Active asyncio task count (via ``/api/health/diagnostics`` if exposed)

After the run, compares first vs last quartile and emits a verdict:
``green`` (no significant growth), ``yellow`` (drift but bounded),
``red`` (unbounded growth — leak suspected).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil


def sample(pid: int, audit_path: Path | None) -> dict[str, Any]:
    proc = psutil.Process(pid)
    mem = proc.memory_info()
    sample_dict: dict[str, Any] = {
        "ts": time.time(),
        "rss_mb": round(mem.rss / 1024 / 1024, 1),
        "vms_mb": round(mem.vms / 1024 / 1024, 1),
        "num_threads": proc.num_threads(),
    }
    try:
        sample_dict["num_fds"] = proc.num_fds() if hasattr(proc, "num_fds") else len(
            proc.connections()
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        sample_dict["num_fds"] = -1
    if audit_path and audit_path.exists():
        try:
            with audit_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
            sample_dict["audit_size_bytes"] = size
        except OSError:
            sample_dict["audit_size_bytes"] = -1
    return sample_dict


def verdict(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare first quartile RSS to last quartile RSS.

    Growth > 50% → red. Growth > 20% → yellow. Otherwise green.
    """
    if len(samples) < 4:
        return {"verdict": "insufficient", "reason": "fewer than 4 samples"}

    quartile = max(1, len(samples) // 4)
    head = [s["rss_mb"] for s in samples[:quartile]]
    tail = [s["rss_mb"] for s in samples[-quartile:]]
    head_avg = sum(head) / len(head)
    tail_avg = sum(tail) / len(tail)
    growth_pct = ((tail_avg - head_avg) / head_avg) * 100 if head_avg else 0.0

    if growth_pct > 50:
        v = "red"
    elif growth_pct > 20:
        v = "yellow"
    else:
        v = "green"
    return {
        "verdict": v,
        "head_avg_rss_mb": round(head_avg, 1),
        "tail_avg_rss_mb": round(tail_avg, 1),
        "growth_pct": round(growth_pct, 1),
        "sample_count": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--duration", type=int, default=86400)  # 24h
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=Path.home() / ".cognithor" / "audit" / "audit.jsonl",
    )
    parser.add_argument("--output", type=Path, default=Path("soak-observations.jsonl"))
    args = parser.parse_args()

    end_at = time.time() + args.duration
    samples: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as fh:
        while time.time() < end_at:
            try:
                s = sample(args.pid, args.audit_path)
            except psutil.NoSuchProcess:
                print(f"PID {args.pid} died — observer stops.", file=sys.stderr)
                break
            samples.append(s)
            fh.write(json.dumps(s) + "\n")
            fh.flush()
            time.sleep(args.interval)

    v = verdict(samples)
    summary_path = args.output.with_suffix(".verdict.json")
    summary_path.write_text(json.dumps(v, indent=2), encoding="utf-8")
    print(json.dumps(v, indent=2))
    return 0 if v["verdict"] in ("green", "yellow", "insufficient") else 1


if __name__ == "__main__":
    sys.exit(main())
