"""Round-3 mass fixer: var-annotated + remaining type-arg.

Fixes only ``var-annotated`` errors via AST inspection of the
assignment node — looks at the RHS expression to infer ``list[Any]``
/ ``dict[str, Any]`` / ``set[Any]``. Strings, docstrings, comments
are never touched.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cognithor"


def collect_errors() -> dict[pathlib.Path, list[tuple[int, str, str]]]:
    proc = subprocess.run(
        ["mypy", "--strict", str(SRC)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    grouped: dict[pathlib.Path, list[tuple[int, str, str]]] = defaultdict(list)
    pat = re.compile(r"^(.+?):(\d+): error: (.+?)  \[([a-z-]+)\]$")
    for line in proc.stdout.splitlines():
        m = pat.match(line)
        if m:
            grouped[pathlib.Path(m.group(1)).resolve()].append(
                (int(m.group(2)), m.group(3), m.group(4))
            )
    return grouped


def patch_var_annotated(text: str, line_nos: set[int]) -> tuple[str, int]:
    """Add an explicit type annotation to bare assignments on the given lines.

    Patterns handled:
      * ``x = []``   → ``x: list[Any] = []``
      * ``x = {}``   → ``x: dict[str, Any] = {}``
      * ``x = set()``→ ``x: set[Any] = set()``
      * ``x = ()``   → ``x: tuple[Any, ...] = ()``
    """
    if not line_nos:
        return text, 0
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, 0

    # Map ``lineno -> name`` for assigments we can patch
    targets: dict[int, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.lineno in line_nos:
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            ann = None
            if isinstance(node.value, ast.List) and not node.value.elts:
                ann = "list[Any]"
            elif isinstance(node.value, ast.Dict) and not node.value.keys:
                ann = "dict[str, Any]"
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "set" and not node.value.args:
                    ann = "set[Any]"
                elif node.value.func.id == "list" and not node.value.args:
                    ann = "list[Any]"
                elif node.value.func.id == "dict" and not node.value.args:
                    ann = "dict[str, Any]"
            elif isinstance(node.value, ast.Tuple) and not node.value.elts:
                ann = "tuple[Any, ...]"
            if ann:
                targets[node.lineno] = (name, ann)

    if not targets:
        return text, 0

    lines = text.splitlines(keepends=True)
    count = 0
    for lineno, (name, ann) in targets.items():
        idx = lineno - 1
        if not (0 <= idx < len(lines)):
            continue
        line = lines[idx]
        # Replace ``<name> =`` with ``<name>: <ann> =`` once
        new_line = re.sub(
            rf"(\b{re.escape(name)}\s*)=",
            rf"\1: {ann} =",
            line,
            count=1,
        )
        if new_line != line:
            lines[idx] = new_line
            count += 1
    return "".join(lines), count


def ensure_any_import(text: str) -> str:
    if "Any" not in text:
        return text
    if re.search(r"\bfrom typing import [^\n]*\bAny\b", text):
        return text
    m = re.search(r"^from typing import (.+)$", text, flags=re.MULTILINE)
    if m:
        return text.replace(m.group(0), f"from typing import Any, {m.group(1)}", 1)
    m2 = re.search(r"^from __future__ import [^\n]+\n", text, flags=re.MULTILINE)
    if m2:
        return text[: m2.end()] + "\nfrom typing import Any\n" + text[m2.end() :]
    return text


def main() -> None:
    print("[1/2] Collecting mypy errors ...", flush=True)
    grouped = collect_errors()
    print("[2/2] Applying var-annotated fixes ...", flush=True)

    files_changed = 0
    for path, errors in sorted(grouped.items()):
        if not path.exists():
            continue
        var_lines: set[int] = set()
        for lineno, _, cat in errors:
            if cat == "var-annotated":
                var_lines.add(lineno)
        if not var_lines:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        text, n = patch_var_annotated(original, var_lines)
        if text != original:
            text = ensure_any_import(text)
            path.write_text(text, encoding="utf-8")
            files_changed += 1
            print(f"  {path.relative_to(ROOT)}: {n} fixes", flush=True)

    print(f"Done. {files_changed} files modified.")


if __name__ == "__main__":
    main()
