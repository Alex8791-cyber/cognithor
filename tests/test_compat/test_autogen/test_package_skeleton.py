"""Compat package — public surface skeleton."""

from __future__ import annotations

import importlib

import pytest


def test_compat_package_imports() -> None:
    importlib.import_module("cognithor.compat")


def test_autogen_subpackage_imports() -> None:
    importlib.import_module("cognithor.compat.autogen")


def test_autogen_import_emits_deprecation_warning() -> None:
    """Re-importing emits a DeprecationWarning pointing at the migration guide."""
    import importlib

    import cognithor.compat.autogen as ca

    with pytest.warns(DeprecationWarning, match="migration"):
        importlib.reload(ca)
