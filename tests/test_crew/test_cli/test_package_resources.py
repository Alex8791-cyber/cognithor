"""Task 38 — template package resources ship in the wheel."""

from pathlib import Path

import cognithor.crew.templates as _t


def test_templates_package_has_files():
    """After install there must be at least one template/template.yaml.

    Until Task 39 lands the `research` template, this test is
    expected-to-fail; after Task 39 it asserts the real template yaml.
    """
    pkg_dir = Path(_t.__file__).parent
    yamls = list(pkg_dir.glob("*/template.yaml"))
    assert yamls, f"No template.yaml files shipped in package at {pkg_dir}"
