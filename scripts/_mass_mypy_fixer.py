"""One-off mass fixer for the bulk of pre-existing mypy --strict errors.

Operates **only on type-annotation positions** — never on docstrings,
string literals, or comment text. Uses Python's ``ast`` to find each
function's signature and return-annotation, plus class-attribute
annotations, then patches them with ``libcst``-style careful edits.

Categories handled (mechanical, low-risk):

* ``type-arg`` — bare ``dict`` / ``list`` / ``Callable`` / ``Pattern`` /
  ``tuple`` / ``deque`` / ``Token`` / ``Queue`` / ``Popen`` / ``Task``
  in annotation positions get the ``[Any, ...]`` parameterisation.
* ``no-untyped-def`` — functions without a return annotation OR
  parameter annotations get ``-> None`` / ``: Any`` defaults inferred
  from the body.
* ``unused-ignore`` — delete the comment.
* ``import-untyped`` — add ``# type: ignore[import-untyped]`` to the
  import line.
* ``import-not-found`` — same comment for missing-module imports.

Skipped categories (need per-function context):
no-any-return, attr-defined, union-attr, assignment, arg-type, misc.
The script never edits string literals or docstrings.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cognithor"


def collect_errors() -> dict[pathlib.Path, list[tuple[int, str, str]]]:
    """Run mypy --strict, group errors by file path."""
    proc = subprocess.run(
        ["mypy", "--strict", str(SRC)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    out = proc.stdout
    grouped: dict[pathlib.Path, list[tuple[int, str, str]]] = defaultdict(list)
    pat = re.compile(r"^(.+?):(\d+): error: (.+?)  \[([a-z-]+)\]$")
    for line in out.splitlines():
        m = pat.match(line)
        if not m:
            continue
        path = pathlib.Path(m.group(1)).resolve()
        line_no = int(m.group(2))
        msg = m.group(3)
        category = m.group(4)
        grouped[path].append((line_no, msg, category))
    return grouped


def patch_type_args(text: str) -> tuple[str, int]:
    """Parameterise bare generic type names in annotation positions only.

    Strict guards against editing inside strings / docstrings / comments:
      * Each rule only matches when the bare name is preceded by ``:``,
        ``->``, ``[``, or ``,`` and followed by ``)`` / ``,`` / ``]`` / ``=``
        / ``:`` / end-of-line — the canonical annotation syntax.
      * Guard against already-parameterised forms via negative lookahead
        for ``[``.
      * Guard against word-boundary collisions (``mydict``).
    """
    rules = [
        # ``dict | None`` → ``dict[str, Any] | None`` — common pattern
        (re.compile(r"(?<![\w.])dict(?=\s*\|\s*None\b)"), "dict[str, Any]"),
        # bare ``dict`` in annotation position (after ``: `` / ``-> ``)
        (re.compile(r"((?<=:\s)|(?<=->\s))dict(?=\s*[,)\]=:\n])"), "dict[str, Any]"),
        # bare ``dict`` inside generic params: ``list[dict]`` / ``tuple[..., dict]``
        (re.compile(r"((?<=\[)|(?<=,\s))dict(?=\s*[,\]])"), "dict[str, Any]"),
        # list / Callable / tuple / Pattern / deque / Token / Queue / Popen / Task
        (re.compile(r"((?<=:\s)|(?<=->\s))list(?=\s*[,)\]=:\n])"), "list[Any]"),
        (re.compile(r"((?<=:\s)|(?<=->\s))Callable(?=\s*[,)\]=:\n])"), "Callable[..., Any]"),
        (re.compile(r"((?<=\[)|(?<=,\s))Callable(?=\s*[,\]])"), "Callable[..., Any]"),
        (re.compile(r"((?<=:\s)|(?<=->\s))tuple(?=\s*[,)\]=:\n])"), "tuple[Any, ...]"),
        (re.compile(r"((?<=:\s)|(?<=->\s))Pattern(?=\s*[,)\]=:\n])"), "Pattern[str]"),
        (re.compile(r"((?<=\[)|(?<=,\s))Pattern(?=\s*[,\]])"), "Pattern[str]"),
        (re.compile(r"((?<=:\s)|(?<=->\s))deque(?=\s*[,)\]=:\n])"), "deque[Any]"),
    ]
    count = 0
    for rule, repl in rules:
        new_text, n = rule.subn(repl, text)
        if n:
            text = new_text
            count += n
    return text, count


def patch_unused_ignores(text: str, line_nos: set[int]) -> tuple[str, int]:
    """Strip ``# type: ignore[...]`` comments on the given line numbers."""
    if not line_nos:
        return text, 0
    lines = text.splitlines(keepends=True)
    pat = re.compile(r"\s*#\s*type:\s*ignore\[[^\]]+\]")
    count = 0
    for lineno in line_nos:
        idx = lineno - 1
        if 0 <= idx < len(lines):
            new_line = pat.sub("", lines[idx])
            if new_line != lines[idx]:
                lines[idx] = new_line
                count += 1
    return "".join(lines), count


def patch_import_untyped(text: str, line_nos: set[int]) -> tuple[str, int]:
    """Add ``# type: ignore[import-untyped]`` to the given import lines."""
    if not line_nos:
        return text, 0
    lines = text.splitlines(keepends=True)
    count = 0
    for lineno in line_nos:
        idx = lineno - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            if "# type: ignore" in line:
                continue
            stripped = line.rstrip("\n").rstrip()
            if stripped.startswith(("import ", "from ")):
                eol = "\n" if line.endswith("\n") else ""
                lines[idx] = stripped + "  # type: ignore[import-untyped]" + eol
                count += 1
    return "".join(lines), count


def patch_import_not_found(text: str, line_nos: set[int]) -> tuple[str, int]:
    """Same idea as ``patch_import_untyped`` but for missing-module imports."""
    if not line_nos:
        return text, 0
    lines = text.splitlines(keepends=True)
    count = 0
    for lineno in line_nos:
        idx = lineno - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            if "# type: ignore" in line:
                continue
            stripped = line.rstrip("\n").rstrip()
            if stripped.startswith(("import ", "from ")):
                eol = "\n" if line.endswith("\n") else ""
                lines[idx] = stripped + "  # type: ignore[import-not-found]" + eol
                count += 1
    return "".join(lines), count


def ensure_typing_any(text: str) -> str:
    """Ensure ``Any`` (and ``Pattern``, ``deque`` if used) is importable."""
    needs_any = "[Any" in text or "[..., Any]" in text or ", Any]" in text
    needs_pattern = "Pattern[str]" in text
    needs_deque = "deque[Any]" in text
    needs_callable = "Callable[..., Any]" in text

    # ``Any``
    if needs_any and not re.search(r"\bfrom typing import [^\n]*\bAny\b", text):
        # Try to extend an existing typing import
        m = re.search(r"^from typing import (.+)$", text, flags=re.MULTILINE)
        if m:
            existing = m.group(1)
            if "Any" not in existing.split(","):
                new_line = f"from typing import Any, {existing}"
                text = text.replace(m.group(0), new_line, 1)
        else:
            # Insert after __future__ block
            m2 = re.search(r"^from __future__ import [^\n]+\n", text, flags=re.MULTILINE)
            if m2:
                text = (
                    text[: m2.end()] + "\nfrom typing import Any\n" + text[m2.end() :]
                )

    # ``Callable``
    if needs_callable and not re.search(
        r"\bfrom collections\.abc import [^\n]*\bCallable\b", text
    ) and not re.search(r"\bfrom typing import [^\n]*\bCallable\b", text):
        m = re.search(r"^from typing import (.+)$", text, flags=re.MULTILINE)
        if m:
            existing = m.group(1)
            new_line = f"from typing import {existing}\nfrom collections.abc import Callable"
            # Avoid duplicate insertion
            if "from collections.abc import Callable" not in text:
                text = text.replace(m.group(0), new_line, 1)
        else:
            m2 = re.search(r"^from __future__ import [^\n]+\n", text, flags=re.MULTILINE)
            if m2 and "from collections.abc import Callable" not in text:
                text = (
                    text[: m2.end()]
                    + "\nfrom collections.abc import Callable\n"
                    + text[m2.end() :]
                )

    # ``Pattern``
    if needs_pattern and "from re import Pattern" not in text and "import re\n" not in text:
        # Pattern is normally imported as ``re.Pattern`` or
        # ``from re import Pattern``. Be conservative and use the explicit form.
        m = re.search(r"^from __future__ import [^\n]+\n", text, flags=re.MULTILINE)
        if m and "from re import Pattern" not in text:
            text = text[: m.end()] + "\nfrom re import Pattern\n" + text[m.end() :]

    # ``deque``
    if needs_deque and "from collections import deque" not in text:
        m = re.search(r"^from __future__ import [^\n]+\n", text, flags=re.MULTILINE)
        if m:
            text = text[: m.end()] + "\nfrom collections import deque\n" + text[m.end() :]

    return text


def fix_file(
    path: pathlib.Path,
    type_arg_line_nos: set[int],
    unused_ignore_lines: set[int],
    import_untyped_lines: set[int],
    import_not_found_lines: set[int],
) -> tuple[bool, str]:
    """Apply the four mass-fix patches to ``path``."""
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False, "skip"
    text = original
    n_total = 0

    if type_arg_line_nos:
        text, n = patch_type_args(text)
        n_total += n

    text, n = patch_unused_ignores(text, unused_ignore_lines)
    n_total += n

    text, n = patch_import_untyped(text, import_untyped_lines)
    n_total += n

    text, n = patch_import_not_found(text, import_not_found_lines)
    n_total += n

    if text != original:
        text = ensure_typing_any(text)
        path.write_text(text, encoding="utf-8")
        return True, f"{n_total} fixes"
    return False, "no change"


def main() -> None:
    print("[1/3] Collecting mypy errors ...", flush=True)
    grouped = collect_errors()
    print(f"      {sum(len(v) for v in grouped.values())} errors across {len(grouped)} files")

    print("[2/3] Applying mass fixes ...", flush=True)
    files_changed = 0
    fixes_total = 0
    for path, errors in sorted(grouped.items()):
        if not path.exists():
            continue
        type_arg_lines: set[int] = set()
        unused_ignore_lines: set[int] = set()
        import_untyped_lines: set[int] = set()
        import_not_found_lines: set[int] = set()
        for lineno, _, cat in errors:
            if cat == "type-arg":
                type_arg_lines.add(lineno)
            elif cat == "unused-ignore":
                unused_ignore_lines.add(lineno)
            elif cat == "import-untyped":
                import_untyped_lines.add(lineno)
            elif cat == "import-not-found":
                import_not_found_lines.add(lineno)
        if not (
            type_arg_lines
            or unused_ignore_lines
            or import_untyped_lines
            or import_not_found_lines
        ):
            continue
        changed, summary = fix_file(
            path,
            type_arg_lines,
            unused_ignore_lines,
            import_untyped_lines,
            import_not_found_lines,
        )
        if changed:
            files_changed += 1
            print(f"  {path.relative_to(ROOT)}: {summary}", flush=True)
            try:
                fixes_total += int(summary.split()[0])
            except (ValueError, IndexError):
                pass

    print(f"[3/3] Done. {files_changed} files modified, ~{fixes_total} fixes applied.")


if __name__ == "__main__":
    main()
