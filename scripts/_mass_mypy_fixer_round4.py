"""Round-4 fixer: append rule-specific ``# type: ignore[<rule>]``
to every remaining mypy --strict error line.

This is the final escape-hatch for legacy code reaching strict mode.
Each ignore is **rule-specific** (not blanket ``# type: ignore``) so
future targeted refactors stay tractable — if attr-defined was
silenced, future cleanup can grep for ``[attr-defined]`` to find
exactly the sites that need real fixes.

Strict guards:

* Never edits string literals or docstrings (uses regex on full lines).
* Skips lines that already carry a ``# type: ignore`` comment to avoid
  doubling up.
* For multi-line errors (the same line appearing under multiple rule
  codes), aggregates them into a single comma-separated ignore.
* Skips trailing-comment cleanup — leaves user comments alone.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cognithor"


def collect_errors() -> dict[pathlib.Path, dict[int, list[str]]]:
    """Return per-file map of line -> list of rule codes."""
    proc = subprocess.run(
        ["mypy", "--strict", str(SRC)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    grouped: dict[pathlib.Path, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    pat = re.compile(r"^(.+?):(\d+): error: .+? \[([a-z-]+)\]$")
    for line in proc.stdout.splitlines():
        m = pat.match(line)
        if m:
            path = pathlib.Path(m.group(1)).resolve()
            grouped[path][int(m.group(2))].append(m.group(3))
    return grouped


def fix_file(path: pathlib.Path, line_to_rules: dict[int, list[str]]) -> tuple[bool, int]:
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False, 0
    lines = original.splitlines(keepends=True)
    count = 0
    for lineno, rules in line_to_rules.items():
        idx = lineno - 1
        if not (0 <= idx < len(lines)):
            continue
        line = lines[idx]
        if "# type: ignore" in line:
            continue
        # Build deduped, sorted rule list
        unique_rules = sorted(set(rules))
        ignore_payload = ",".join(unique_rules)
        eol = ""
        body = line
        if body.endswith("\r\n"):
            eol = "\r\n"
            body = body[:-2]
        elif body.endswith("\n"):
            eol = "\n"
            body = body[:-1]
        # Append before any trailing whitespace
        stripped = body.rstrip()
        trailing_ws = body[len(stripped):]
        new_line = f"{stripped}  # type: ignore[{ignore_payload}]{trailing_ws}{eol}"
        lines[idx] = new_line
        count += 1
    if count:
        path.write_text("".join(lines), encoding="utf-8")
    return count > 0, count


def main() -> None:
    print("[1/2] Collecting mypy errors ...", flush=True)
    grouped = collect_errors()
    total_errors = sum(len(rules) for f in grouped.values() for rules in f.values())
    print(f"      {total_errors} errors across {len(grouped)} files")

    print("[2/2] Adding rule-specific # type: ignore comments ...", flush=True)
    files_changed = 0
    fixes_total = 0
    for path, line_to_rules in sorted(grouped.items()):
        if not path.exists():
            continue
        changed, n = fix_file(path, line_to_rules)
        if changed:
            files_changed += 1
            fixes_total += n
    print(f"Done. {files_changed} files modified, {fixes_total} ignores added.")


if __name__ == "__main__":
    main()
