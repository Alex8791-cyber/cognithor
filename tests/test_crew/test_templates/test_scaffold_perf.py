"""Spec §8.5 / R4-I7: each template must scaffold in <500ms."""

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


@pytest.mark.parametrize("template_name", _ALL_TEMPLATES)
def test_template_generation_under_500ms(template_name: str, tmp_path: Path) -> None:
    project = tmp_path / f"perf_{template_name.replace('-', '_')}"
    start = time.perf_counter()
    run_init(name=project.name, template=template_name, directory=project, lang="de")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500, (
        f"{template_name} scaffolding took {elapsed_ms:.0f}ms (budget: 500ms). "
        f"Spec §8.5 budget violated — investigate render_tree or template bloat."
    )
