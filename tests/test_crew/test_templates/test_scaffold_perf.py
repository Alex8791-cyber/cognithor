"""Spec §8.5 / R4-I7: each template must scaffold in <500ms (POSIX).

Windows CI runners have ~2-3× slower filesystem ops (NTFS small-file
overhead + Defender scan) — a scaffold that takes 150ms on Linux can
land at 600-800ms on Windows. Raising the Windows budget to 1500ms keeps
the spec's §8.5 intent (catch template bloat / render_tree regressions)
while tolerating the platform reality.
"""

import sys
import time
from pathlib import Path

import pytest

from cognithor.crew.cli.init_cmd import run_init

_ALL_TEMPLATES = [
    "research",
    "customer-support",
    "data-analyst",
    "content",
    "versicherungs-vergleich",
]

# POSIX budget per spec §8.5; Windows gets a realistic ceiling (see docstring).
_BUDGET_MS = 1500 if sys.platform == "win32" else 500


@pytest.mark.parametrize("template_name", _ALL_TEMPLATES)
def test_template_generation_under_budget(template_name: str, tmp_path: Path) -> None:
    project = tmp_path / f"perf_{template_name.replace('-', '_')}"
    start = time.perf_counter()
    run_init(name=project.name, template=template_name, directory=project, lang="de")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < _BUDGET_MS, (
        f"{template_name} scaffolding took {elapsed_ms:.0f}ms "
        f"(budget: {_BUDGET_MS}ms on {sys.platform}). "
        f"Spec §8.5 budget violated — investigate render_tree or template bloat."
    )
