"""Task 67 -- integrations catalog generator tests."""

import json
import subprocess
import sys
from pathlib import Path


def test_generator_produces_valid_catalog(tmp_path: Path):
    out = tmp_path / "catalog.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_integrations_catalog.py",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "tools" in data
    assert isinstance(data["tools"], list)
    for entry in data["tools"]:
        assert "name" in entry
        assert "module" in entry
        assert "category" in entry
        assert "description" in entry


def test_catalog_only_includes_real_tools(tmp_path: Path):
    """Wahrheitspflicht: no entry is listed that doesn't exist in the repo."""
    import importlib

    out = tmp_path / "catalog.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_integrations_catalog.py",
            "--output",
            str(out),
        ],
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    for entry in data["tools"]:
        importlib.import_module(entry["module"])
