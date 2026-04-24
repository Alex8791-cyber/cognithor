"""Jinja2-based directory tree renderer for cognithor init templates.

Uses ``SandboxedEnvironment`` (not plain ``Environment``) so untrusted
template content cannot access Python internals via ``__class__`` /
``__mro__`` / ``__subclasses__`` tricks. HTML/XML autoescape is enabled for
future HTML template support. Path segments are validated per-render to
block traversal attacks like a filename literally containing
``{{ '../../etc/passwd' }}``. See NC3 in Round 3 review.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader, StrictUndefined, select_autoescape
from jinja2.sandbox import SandboxedEnvironment

_PROJECT_NAME_CLEAN = re.compile(r"[^a-zA-Z0-9_]")

# Windows reserved device names that must never appear as file or project
# basenames — touching ``CON``, ``NUL``, ``COM1``–``COM9``, ``LPT1``–``LPT9``
# on NTFS returns to the console device and corrupts anything that tries to
# read them. See NI4 in Round 3 review.
_WIN_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Path segments that must never appear AFTER rendering — blocks
# ``{{ '../../etc/passwd' }}`` in a template filename escaping dest_dir.
_FORBIDDEN_SEGMENTS = {"", ".", ".."}

# Language-specific template suffixes. ``README.md.jinja.de`` selects the DE
# variant when ``context["lang"] == "de"``; ``README.md.jinja.en`` is skipped.
_LANG_SUFFIXES = {".de", ".en", ".zh", ".ar"}
_LANG_FALLBACK = "en"

# Standard Python .gitignore auto-injected when the template doesn't ship
# one. Prevents new projects from accidentally committing secrets (.env),
# local virtual-envs, compiled bytecode, or Cognithor state DBs on the
# first ``git add .``.
_DEFAULT_GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual envs
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Env + secrets
.env
.env.local
.env.*.local

# Cognithor
.cognithor/
*.db
*.db-journal
"""


def sanitize_project_name(name: str) -> str:
    """Convert free-form name to a safe Python package identifier.

    Rejects empty strings and Windows reserved device names (``CON``, ``NUL``,
    ``COM1..9``, ``LPT1..9``, ``PRN``, ``AUX``) on ALL platforms — not just
    Windows — so projects scaffolded on Linux remain portable to Windows
    developers.
    """
    if not name or not name.strip():
        raise ValueError("project name cannot be empty")
    cleaned = _PROJECT_NAME_CLEAN.sub("_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"project name reduces to empty: {name!r}")
    if cleaned[0].isdigit():
        cleaned = f"project_{cleaned}"
    if cleaned.upper() in _WIN_RESERVED:
        raise ValueError(
            f"'{cleaned}' is a reserved Windows device name. "
            f"Please choose a different name (e.g. '{cleaned}_app')."
        )
    return cleaned


def _build_env(src_dir: Path) -> SandboxedEnvironment:
    """Construct a sandboxed Jinja2 environment with autoescape on."""
    return SandboxedEnvironment(
        loader=FileSystemLoader(str(src_dir)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(["html", "xml"]),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def _safe_join(dest_dir: Path, rendered_parts: list[str]) -> Path:
    """Validate every rendered segment, then join under dest_dir.

    Raises ``ValueError`` on any sign of path traversal — forbidden tokens
    (``.``, ``..``, empty), embedded separators (``/`` or ``\\``), or a
    rendered basename that lands outside ``dest_dir`` after ``resolve()``.
    """
    for seg in rendered_parts:
        if seg in _FORBIDDEN_SEGMENTS:
            raise ValueError(f"Forbidden path segment after render: {seg!r}")
        if "/" in seg or "\\" in seg or seg.startswith(".."):
            raise ValueError(f"Path traversal in rendered segment: {seg!r}")
    candidate = dest_dir.joinpath(*rendered_parts).resolve()
    try:
        candidate.relative_to(dest_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Rendered path {candidate} escapes dest_dir {dest_dir}") from exc
    return candidate


def _select_language_files(src_dir: Path, lang: str) -> set[Path]:
    """For each base path with language variants, pick exactly ONE to render.

    Fallback order per base:
      1. Requested ``lang`` variant if it exists
      2. ``en`` variant as universal fallback
      3. First (alphabetically) available variant as last resort
    """
    variants: dict[Path, dict[str, Path]] = {}
    for src in src_dir.rglob("*"):
        if not src.is_file() or src.suffix not in _LANG_SUFFIXES:
            continue
        base = src.with_suffix("")
        file_lang = src.suffix.lstrip(".")
        variants.setdefault(base, {})[file_lang] = src

    selected: set[Path] = set()
    for _base, by_lang in variants.items():
        if lang in by_lang:
            selected.add(by_lang[lang])
        elif _LANG_FALLBACK in by_lang:
            selected.add(by_lang[_LANG_FALLBACK])
        else:
            first = sorted(by_lang.keys())[0]
            selected.add(by_lang[first])
    return selected


def _resolve_language_variant(
    src_path: Path,
    lang: str,
    *,
    selected: set[Path] | None = None,
) -> Path | None:
    """Return ``src_path`` to keep, or ``None`` to skip."""
    if src_path.suffix not in _LANG_SUFFIXES:
        return src_path
    if selected is not None:
        return src_path if src_path in selected else None
    file_lang = src_path.suffix.lstrip(".")
    return src_path if file_lang == lang else None


def _strip_template_suffixes(rel: Path, lang: str) -> Path:
    """Strip language suffix (if any) + ``.jinja`` suffix from the OUTPUT path."""
    parts = list(rel.parts)
    if not parts:
        return rel
    last = parts[-1]
    for lang_suf in _LANG_SUFFIXES:
        if last.endswith(lang_suf):
            last = last[: -len(lang_suf)]
            break
    if last.endswith(".jinja"):
        last = last[: -len(".jinja")]
    parts[-1] = last
    return Path(*parts)


def render_tree(src_dir: Path, dest_dir: Path, *, context: dict[str, Any]) -> None:
    """Render every file under src_dir into dest_dir, applying Jinja2 to .jinja files."""
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise FileExistsError(f"dest exists and is not empty: {dest_dir}")

    env = _build_env(src_dir)
    lang = context.get("lang", "en")

    selected_variants = _select_language_files(src_dir, lang)

    for src_path in src_dir.rglob("*"):
        rel = src_path.relative_to(src_dir)

        if (
            src_path.is_file()
            and _resolve_language_variant(src_path, lang, selected=selected_variants) is None
        ):
            continue

        if src_path.is_file():
            out_rel = _strip_template_suffixes(rel, lang)
        else:
            out_rel = rel

        rendered_parts = [env.from_string(p).render(**context) for p in out_rel.parts]
        dest_path = _safe_join(dest_dir, rendered_parts)

        if src_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # A file is a "template" if it contains ``.jinja`` anywhere in its
        # suffix chain (e.g. ``README.md.jinja.de``, ``main.py.jinja``).
        is_template = "jinja" in src_path.name.split(".")[1:]
        if is_template:
            template = env.get_template(str(rel).replace("\\", "/"))
            dest_path.write_text(template.render(**context), encoding="utf-8")
        else:
            shutil.copy2(src_path, dest_path)

    # Auto-inject a standard .gitignore if the template didn't ship one.
    # Protects scaffolded projects from accidentally committing .env,
    # __pycache__/, or local Cognithor state on the first `git add .`.
    gitignore_dest = dest_dir / ".gitignore"
    if not gitignore_dest.exists():
        gitignore_dest.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")
