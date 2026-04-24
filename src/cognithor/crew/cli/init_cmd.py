"""cognithor init — template-based project scaffolder.

Creates a new Cognithor Crew project directory from a named template.
Renders Jinja2 templates via ``scaffolder.render_tree`` and prints a
localized next-command hint using ``cognithor.i18n``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cognithor.crew.cli.list_templates_cmd import TEMPLATES_ROOT, list_templates
from cognithor.crew.cli.scaffolder import render_tree, sanitize_project_name


class InitCommandError(Exception):
    """Raised when the init command cannot complete."""


def _render_folder_tree(root: Path, *, max_entries: int = 40) -> str:
    """Return an ASCII tree rendering of the scaffolded folder.

    Limited to ``max_entries`` lines so the preview never scrolls off screen
    for large templates. Entries are sorted (directories first, then files).
    """
    lines: list[str] = [root.name + "/"]

    def _walk(path: Path, prefix: str) -> None:
        if len(lines) >= max_entries:
            return
        entries = sorted(
            path.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        count = len(entries)
        for idx, entry in enumerate(entries):
            if len(lines) >= max_entries:
                lines.append(f"{prefix}... (truncated)")
                return
            is_last = idx == count - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension)

    if root.is_dir():
        _walk(root, "")
    return "\n".join(lines)


def _resolve_template_dir(template: str) -> Path:
    """Find the named template directory, or raise InitCommandError."""
    known = {t.name for t in list_templates()}
    if template not in known:
        available = ", ".join(sorted(known)) or "(none)"
        raise InitCommandError(f"unknown template '{template}'. Available: {available}")
    return TEMPLATES_ROOT / template


def run_init(
    *,
    name: str,
    template: str,
    directory: Path,
    lang: str = "de",
    force: bool = False,
) -> int:
    """Execute the ``cognithor init`` subcommand.

    Parameters
    ----------
    name:
        Free-form project name. Sanitized via ``sanitize_project_name`` to
        produce a valid Python package identifier.
    template:
        Name of the template directory under ``crew/templates/``.
    directory:
        Target output directory. Must not exist or must be empty unless
        ``force=True``.
    lang:
        Language code for localization (``"de"``, ``"en"``, ``"zh"``).
    force:
        If True, remove an existing non-empty target before rendering.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    directory = Path(directory)
    project_name = sanitize_project_name(name)
    template_dir = _resolve_template_dir(template)

    if directory.exists() and any(directory.iterdir()):
        if force:
            print(f"--force: removing existing {directory}")
            shutil.rmtree(directory)
        else:
            raise InitCommandError(
                f"target directory '{directory}' is not empty. Use --force to overwrite."
            )

    directory.mkdir(parents=True, exist_ok=True)

    render_tree(
        template_dir,
        directory,
        context={
            "project_name": project_name,
            "raw_name": name,
            "lang": lang,
        },
    )

    # Preview the scaffolded tree
    print(_render_folder_tree(directory))
    print()

    # Localized next-command hint. Falls back to raw key if i18n missing.
    try:
        from cognithor.i18n import set_locale
        from cognithor.i18n import t as _t

        set_locale(lang)
        hint = _t("crew.init.next_command", dest=str(directory))
    except Exception:
        hint = f"Next: cd {directory} && pip install -e .[dev] && cognithor run"
    print(hint)
    return 0
