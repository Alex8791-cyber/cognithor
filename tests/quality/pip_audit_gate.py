#!/usr/bin/env python3
"""pip-audit gate — fails CI if HIGH/CRITICAL CVEs are unwaived.

Usage::

    python tests/quality/pip_audit_gate.py <pip-audit.json> <waivers.yaml>

Waiver file format::

    waivers:
      - id: GHSA-xxxx-yyyy-zzzz
        package: <pkg>
        until: 2026-06-30
        reason: "Upstream fix expected in vN.M; downgrade impact > exposure"
        approved_by: alex@cognithor.ai
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


def _load_waivers(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}  # PyYAML not installed in this lane — treat as no waivers
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {entry["id"]: entry for entry in data.get("waivers", [])}


def _waiver_active(entry: dict[str, Any]) -> bool:
    until = entry.get("until")
    if isinstance(until, str):
        until = date.fromisoformat(until)
    if not isinstance(until, date):
        return False
    return until >= datetime.now(tz=timezone.utc).date()


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: pip_audit_gate.py <pip-audit.json> <waivers.yaml>",
            file=sys.stderr,
        )
        return 2
    audit_path = Path(sys.argv[1])
    waivers_path = Path(sys.argv[2])

    if not audit_path.exists():
        print(f"pip-audit report not found: {audit_path}", file=sys.stderr)
        return 1

    with audit_path.open(encoding="utf-8") as fh:
        report = json.load(fh)

    waivers = _load_waivers(waivers_path)

    blocking: list[dict[str, Any]] = []
    waived: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []

    for entry in report.get("dependencies", []):
        for vuln in entry.get("vulns", []):
            severity = (vuln.get("severity") or "").lower()
            vuln_id = vuln.get("id", "")
            row = {
                "package": entry.get("name"),
                "version": entry.get("version"),
                "id": vuln_id,
                "severity": severity,
                "fix_versions": vuln.get("fix_versions", []),
            }
            if severity in {"high", "critical"}:
                waiver = waivers.get(vuln_id)
                if waiver and _waiver_active(waiver):
                    waived.append({**row, "waiver": waiver})
                else:
                    blocking.append(row)
            else:
                other.append(row)

    summary = {
        "blocking_high_critical_count": len(blocking),
        "waived_count": len(waived),
        "other_count": len(other),
        "blocking": blocking,
        "waived": waived,
    }
    print(json.dumps(summary, indent=2, default=str))

    if blocking:
        print(
            f"\nFAIL: {len(blocking)} HIGH/CRITICAL CVE(s) without active waiver.",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nPASS: 0 HIGH/CRITICAL unwaived. ({len(waived)} waived, {len(other)} lower-severity)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
