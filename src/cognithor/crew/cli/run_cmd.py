"""Scaffolded-project-internal 'cognithor run' — loads the Crew defined in
the generated project and calls kickoff()."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path


def run_project_crew(project_dir: Path | None = None) -> int:
    project_dir = Path(project_dir) if project_dir else Path.cwd()
    src = project_dir / "src"
    if not src.is_dir():
        print("No src/ directory — not a scaffolded project?", file=sys.stderr)
        return 2
    pkg_dirs = [p for p in src.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
    if not pkg_dirs:
        print("No Python package under src/", file=sys.stderr)
        return 2
    pkg_name = pkg_dirs[0].name

    sys.path.insert(0, str(src))
    try:
        mod = importlib.import_module(f"{pkg_name}.main")
    except ModuleNotFoundError as exc:
        print(f"cannot import {pkg_name}.main: {exc}", file=sys.stderr)
        return 2

    if not hasattr(mod, "build_crew"):
        print(f"{pkg_name}.main does not define build_crew()", file=sys.stderr)
        return 2

    crew = mod.build_crew()
    result = asyncio.run(crew.kickoff_async())
    print(result.raw)
    return 0
