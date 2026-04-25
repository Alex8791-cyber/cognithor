"""Verify the pack installs as a standalone package."""

from __future__ import annotations


def test_package_imports() -> None:
    import insurance_agent_pack

    assert hasattr(insurance_agent_pack, "__version__")


def test_cli_module_exists() -> None:
    from insurance_agent_pack import cli

    assert hasattr(cli, "main")


def test_crew_module_exists() -> None:
    from insurance_agent_pack import crew

    assert hasattr(crew, "build_team")
