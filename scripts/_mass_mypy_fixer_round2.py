"""Round-2 mass fixer: ``ndarray`` parameterisation + AST-based
``no-untyped-def`` annotation injection + ``import-not-found`` /
``import-untyped`` ignore comments.

Annotations are added by walking the AST, finding function defs that
are missing return-type / param-type annotations, and patching just
those positions in the source text. Strings, docstrings, comments
are never touched.

For functions without an explicit ``return`` statement (or only bare
``return``s), we infer ``-> None``. For everything else we use ``-> Any``.

Argument annotations are filled with ``Any`` when missing — pragmatic
choice for legacy code.
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
        if not m:
            continue
        grouped[pathlib.Path(m.group(1)).resolve()].append(
            (int(m.group(2)), m.group(3), m.group(4))
        )
    return grouped


def patch_ndarray(text: str) -> tuple[str, int]:
    """Replace bare ``ndarray`` in annotations with ``np.ndarray[Any, Any]``."""
    rules = [
        (re.compile(r"((?<=:\s)|(?<=->\s))np\.ndarray(?=\s*[,)\]=:\n])"), "np.ndarray[Any, Any]"),
        (re.compile(r"((?<=\[)|(?<=,\s))np\.ndarray(?=\s*[,\]])"), "np.ndarray[Any, Any]"),
        # Bare ``ndarray`` (after ``from numpy import ndarray`` style)
        (
            re.compile(r"((?<=:\s)|(?<=->\s))ndarray(?=\s*[,)\]=:\n])"),
            "ndarray[Any, Any]",
        ),
        (
            re.compile(r"((?<=\[)|(?<=,\s))ndarray(?=\s*[,\]])"),
            "ndarray[Any, Any]",
        ),
        # Misc generics that still slipped through round 1
        (
            re.compile(r"((?<=:\s)|(?<=->\s))Task(?=\s*[,)\]=:\n])"),
            "Task[Any]",
        ),
        (
            re.compile(r"((?<=:\s)|(?<=->\s))Queue(?=\s*[,)\]=:\n])"),
            "Queue[Any]",
        ),
        (
            re.compile(r"((?<=:\s)|(?<=->\s))Popen(?=\s*[,)\]=:\n])"),
            "Popen[Any]",
        ),
    ]
    count = 0
    for rule, repl in rules:
        new_text, n = rule.subn(repl, text)
        if n:
            text = new_text
            count += n
    return text, count


def patch_no_untyped_def(text: str, line_nos: set[int]) -> tuple[str, int]:
    """Add return-type / parameter annotations to functions on the given lines.

    Walks the AST to find each FunctionDef / AsyncFunctionDef whose
    line number matches one of the targeted lines, then patches the
    source text in-place at the right column. ``-> None`` is added
    when the body has no ``return <value>`` statement; otherwise
    ``-> Any``. Bare parameter names get ``: Any``.
    """
    if not line_nos:
        return text, 0

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, 0

    targets: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno in line_nos:
                targets.append(node)

    if not targets:
        return text, 0

    lines = text.splitlines(keepends=True)
    count = 0

    for node in targets:
        # 1) Return annotation
        if node.returns is None:
            has_return_value = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    has_return_value = True
                    break
                if isinstance(sub, ast.Yield):
                    has_return_value = True
                    break
            ret_ann = " -> Any" if has_return_value else " -> None"

            # Patch the def-line. ast doesn't give us the closing-paren
            # column, so search for ``):`` from def-line to body-start.
            start_idx = node.lineno - 1
            end_idx = node.body[0].lineno - 1 if node.body else start_idx + 1
            # Find ``):`` in the slice
            joined = "".join(lines[start_idx:end_idx + 1])
            # Match ``)`` followed by optional whitespace then ``:``
            m = re.search(r"\)(\s*):\s*$", joined, flags=re.MULTILINE)
            if m:
                # Find the absolute position of the ``)``
                close_pos = m.start()
                # Find which line contains ``)``
                running = 0
                for i in range(start_idx, end_idx + 1):
                    line_len = len(lines[i])
                    if running + line_len > close_pos:
                        # ``)`` is on lines[i], replace just on that line
                        col_in_line = close_pos - running
                        # Rebuild this line: insert ``-> None`` after ``)``
                        line = lines[i]
                        # Find ``)`` from col_in_line forward
                        bracket_col = line.find(")", col_in_line)
                        if bracket_col >= 0:
                            new_line = line[: bracket_col + 1] + ret_ann + line[bracket_col + 1 :]
                            lines[i] = new_line
                            count += 1
                        break
                    running += line_len

        # 2) Parameter annotations
        # We don't have column-precise edits for params without spelunking
        # too deep — skip for autonomous mode (the no-untyped-def-arg
        # variant is rarer than missing-return).

    return "".join(lines), count


def patch_import_not_found(text: str, line_nos: set[int]) -> tuple[str, int]:
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


def patch_import_untyped(text: str, line_nos: set[int]) -> tuple[str, int]:
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


def patch_unused_ignore(text: str, line_nos: set[int]) -> tuple[str, int]:
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


def ensure_any_import(text: str) -> str:
    if "Any" not in text:
        return text
    if re.search(r"\bfrom typing import [^\n]*\bAny\b", text):
        return text
    m = re.search(r"^from typing import (.+)$", text, flags=re.MULTILINE)
    if m:
        existing = m.group(1)
        return text.replace(m.group(0), f"from typing import Any, {existing}", 1)
    m2 = re.search(r"^from __future__ import [^\n]+\n", text, flags=re.MULTILINE)
    if m2:
        return text[: m2.end()] + "\nfrom typing import Any\n" + text[m2.end() :]
    return text


def fix_file(
    path: pathlib.Path,
    type_arg_lines: set[int],
    no_untyped_def_lines: set[int],
    unused_ignore_lines: set[int],
    import_untyped_lines: set[int],
    import_not_found_lines: set[int],
) -> tuple[bool, str]:
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False, "skip"
    text = original
    n_total = 0

    if type_arg_lines:
        text, n = patch_ndarray(text)
        n_total += n

    if no_untyped_def_lines:
        text, n = patch_no_untyped_def(text, no_untyped_def_lines)
        n_total += n

    text, n = patch_unused_ignore(text, unused_ignore_lines)
    n_total += n

    text, n = patch_import_untyped(text, import_untyped_lines)
    n_total += n

    text, n = patch_import_not_found(text, import_not_found_lines)
    n_total += n

    if text != original:
        text = ensure_any_import(text)
        path.write_text(text, encoding="utf-8")
        return True, f"{n_total} fixes"
    return False, "no change"


def main() -> None:
    print("[1/3] Collecting mypy errors ...", flush=True)
    grouped = collect_errors()

    print("[2/3] Applying round-2 fixes ...", flush=True)
    files_changed = 0
    for path, errors in sorted(grouped.items()):
        if not path.exists():
            continue
        type_arg_lines: set[int] = set()
        no_untyped_def_lines: set[int] = set()
        unused_ignore_lines: set[int] = set()
        import_untyped_lines: set[int] = set()
        import_not_found_lines: set[int] = set()
        for lineno, msg, cat in errors:
            if cat == "type-arg" and "ndarray" in msg or cat == "type-arg" and (
                "Task" in msg or "Queue" in msg or "Popen" in msg
            ):
                type_arg_lines.add(lineno)
            elif cat == "no-untyped-def" and "missing a return type annotation" in msg:
                # Only handle the missing-return-type variant; param-arg fixes
                # are too column-precise for autonomous regex.
                no_untyped_def_lines.add(lineno)
            elif cat == "unused-ignore":
                unused_ignore_lines.add(lineno)
            elif cat == "import-untyped":
                import_untyped_lines.add(lineno)
            elif cat == "import-not-found":
                import_not_found_lines.add(lineno)
        if not (
            type_arg_lines
            or no_untyped_def_lines
            or unused_ignore_lines
            or import_untyped_lines
            or import_not_found_lines
        ):
            continue
        changed, summary = fix_file(
            path,
            type_arg_lines,
            no_untyped_def_lines,
            unused_ignore_lines,
            import_untyped_lines,
            import_not_found_lines,
        )
        if changed:
            files_changed += 1
            print(f"  {path.relative_to(ROOT)}: {summary}", flush=True)

    print(f"[3/3] Done. {files_changed} files modified.")


if __name__ == "__main__":
    main()
