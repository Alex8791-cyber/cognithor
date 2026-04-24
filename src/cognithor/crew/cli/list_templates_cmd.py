"""cognithor init --list-templates: discover + print template metadata."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"


class TemplateMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description_de: str
    description_en: str = ""
    required_models: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Explicit listing order. Defaults to a large sentinel so legacy templates
    # without ``order`` fall to the bottom rather than competing with ordered ones.
    order: int = 999


def list_templates() -> list[TemplateMeta]:
    """Return metadata for every discoverable template, sorted by (order, name)."""
    if not TEMPLATES_ROOT.exists():
        return []
    out: list[TemplateMeta] = []
    for d in sorted(TEMPLATES_ROOT.iterdir()):
        meta_file = d / "template.yaml"
        if not meta_file.is_file():
            continue
        data = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        out.append(TemplateMeta(**data))
    out.sort(key=lambda t: (t.order, t.name))
    return out


def print_templates(*, lang: str = "de") -> int:
    """CLI handler — prints templates + descriptions. Returns exit code."""
    templates = list_templates()
    if not templates:
        print("Keine Templates gefunden." if lang == "de" else "No templates found.")
        return 1
    header = "Verfügbare Templates:" if lang == "de" else "Available templates:"
    print(header)
    for t in templates:
        desc = t.description_de if lang == "de" else (t.description_en or t.description_de)
        print(f"  - {t.name:25} {desc}")
    return 0
